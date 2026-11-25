"""Ingest a folder: only (re)process new or changed files."""

from __future__ import annotations

import sys
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
    empty: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    chunks_written: int = 0

    def summary(self) -> str:
        text = (
            f"{len(self.added)} added, {len(self.updated)} updated, "
            f"{len(self.unchanged)} unchanged, {len(self.removed)} removed, "
            f"{len(self.failed)} failed, {self.chunks_written} chunks written"
        )
        if self.empty:
            text += f", {len(self.empty)} without text"
        return text


def ingest_folder(
    folder: str | Path,
    collection,
    embedder: Embedder,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    prune: bool = True,
    on_file: Callable[[str, str], None] | None = None,
) -> IngestReport:
    """Read folder recursively and keep its entries in the index in sync.

    Other folders in the same index are left untouched. prune=True removes chunks
    of files that no longer exist in the folder.
    on_file(status, source) is called for each file (for progress output).
    """
    root_path = Path(folder).resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")
    root = str(root_path)

    report = IngestReport()
    known = store.indexed_files(collection, root)
    seen: set[str] = set()

    for path in iter_files(root_path):
        source = path.relative_to(root_path).as_posix()
        seen.add(source)
        try:
            digest = file_hash(path)
            if known.get(source) == digest:
                report.unchanged.append(source)
                _notify(on_file, "unchanged", source)
                continue

            chunks = chunk_text(load_text(path), chunk_size, overlap)
            if source in known:
                store.delete_source(collection, source, root)
            if not chunks:
                # e.g. scanned PDFs without a text layer
                report.empty.append(source)
                _notify(on_file, "empty", source)
                continue
            embeddings = embedder.embed_documents(chunks)
            store.add_chunks(collection, source, digest, chunks, embeddings, root)
        except Exception as exc:  # a broken file must not abort the run
            report.failed[source] = f"{type(exc).__name__}: {exc}"
            _notify(on_file, "failed", source)
            continue

        report.chunks_written += len(chunks)
        status = "updated" if source in known else "added"
        getattr(report, status).append(source)
        _notify(on_file, status, source)

    if prune:
        for source in sorted(set(known) - seen):
            store.delete_source(collection, source, root)
            report.removed.append(source)
            _notify(on_file, "removed", source)

    return report


def _notify(cb, status: str, source: str) -> None:
    if cb is not None:
        cb(status, source)


def run(folder: Path, config: "Config | None" = None) -> int:
    """CLI entry point (`myrag ingest <folder>`). Returns the exit code."""
    from .embeddings import get_embedder, model_name_from_env

    db_path = getattr(config, "db_path", None)
    print(f"Loading embedding model {model_name_from_env()} (downloaded on first use) ...",
          file=sys.stderr)
    try:
        embedder = get_embedder()
    except Exception as exc:
        print(f"Error: Could not load the embedding model: {exc}", file=sys.stderr)
        print("Tip: The first run needs an internet connection to huggingface.co.",
              file=sys.stderr)
        return 1
    collection = store.get_collection(db_path)

    def progress(status: str, source: str) -> None:
        labels = {"added": "added", "updated": "updated", "removed": "removed",
                  "empty": "no text", "failed": "FAILED"}
        if status in labels:
            print(f"  [{labels[status]}] {source}", flush=True)

    print(f"Ingesting {folder} ...")
    report = ingest_folder(folder, collection, embedder, on_file=progress)
    for source, err in report.failed.items():
        print(f"  Error in {source}: {err}")
    print(report.summary())
    return 1 if report.failed else 0
