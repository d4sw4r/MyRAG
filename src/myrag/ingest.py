"""Ordner einlesen: nur neue/geänderte Dateien neu verarbeiten."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from . import store
from .chunking import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, chunk_text
from .embeddings import Embedder
from .loaders import file_hash, iter_files, load_text

if TYPE_CHECKING:
    from .config import Config


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    chunks_written: int = 0

    def summary(self) -> str:
        return (
            f"{len(self.added)} neu, {len(self.updated)} geändert, "
            f"{len(self.unchanged)} unverändert, {len(self.removed)} entfernt, "
            f"{len(self.failed)} Fehler, {self.chunks_written} Chunks geschrieben"
        )


def ingest_folder(
    folder: str | Path,
    collection,
    embedder: Embedder,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    prune: bool = True,
    on_file: Callable[[str, str], None] | None = None,
) -> IngestReport:
    """Liest folder rekursiv ein und hält die Collection synchron.

    prune=True entfernt Chunks von Dateien, die im Ordner nicht mehr existieren.
    on_file(status, source) wird pro Datei aufgerufen (für Fortschrittsausgabe).
    """
    root = Path(folder).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Kein Ordner: {folder}")

    report = IngestReport()
    known = store.indexed_files(collection)
    seen: set[str] = set()

    for path in iter_files(root):
        source = path.relative_to(root).as_posix()
        seen.add(source)
        try:
            digest = file_hash(path)
            if known.get(source) == digest:
                report.unchanged.append(source)
                _notify(on_file, "unchanged", source)
                continue

            chunks = chunk_text(load_text(path), chunk_size, overlap)
            embeddings = embedder.embed_documents(chunks)
            if source in known:
                store.delete_source(collection, source)
            store.add_chunks(collection, source, digest, chunks, embeddings)
        except Exception as exc:  # eine kaputte Datei soll den Lauf nicht abbrechen
            report.failed[source] = f"{type(exc).__name__}: {exc}"
            _notify(on_file, "failed", source)
            continue

        report.chunks_written += len(chunks)
        status = "updated" if source in known else "added"
        getattr(report, status).append(source)
        _notify(on_file, status, source)

    if prune:
        for source in sorted(set(known) - seen):
            store.delete_source(collection, source)
            report.removed.append(source)
            _notify(on_file, "removed", source)

    return report


def _notify(cb, status: str, source: str) -> None:
    if cb is not None:
        cb(status, source)


def run(folder: Path, config: "Config | None" = None) -> int:
    """Einstiegspunkt für die CLI (`myrag ingest <ordner>`). Gibt den Exit-Code zurück."""
    from .embeddings import get_embedder

    db_path = getattr(config, "db_path", None)
    collection = store.get_collection(db_path)
    embedder = get_embedder()

    def progress(status: str, source: str) -> None:
        labels = {"added": "neu", "updated": "geändert", "removed": "entfernt", "failed": "FEHLER"}
        if status in labels:
            print(f"  [{labels[status]}] {source}")

    report = ingest_folder(folder, collection, embedder, on_file=progress)
    for source, err in report.failed.items():
        print(f"  Fehler bei {source}: {err}")
    print(report.summary())
    return 1 if report.failed else 0
