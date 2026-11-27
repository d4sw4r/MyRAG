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
    known = store.indexed_versions(collection, root)
    chunking = f"{chunk_size}/{overlap}"
    seen: set[str] = set()

    for path in iter_files(root_path):
        source = path.relative_to(root_path).as_posix()
        seen.add(source)
        try:
            digest = file_hash(path)
            if known.get(source) == f"{digest}|{chunking}":
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
            store.add_chunks(collection, source, digest, chunks, embeddings, root, chunking)
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


def _chunk_settings() -> tuple[int, int]:
    import os

    try:
        size = int(os.environ.get("MYRAG_CHUNK_SIZE") or DEFAULT_CHUNK_SIZE)
        overlap = int(os.environ.get("MYRAG_CHUNK_OVERLAP") or DEFAULT_CHUNK_OVERLAP)
    except ValueError:
        return DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
    return size, overlap


def run(folder: Path, config: "Config | None" = None, rebuild: bool = False) -> int:
    """CLI entry point (`myrag ingest <folder>`). Returns the exit code.

    rebuild=True deletes the index and re-ingests every folder it contained plus
    `folder`, e.g. after switching the embedding model.
    """
    from .embeddings import get_embedder, model_name_from_env

    db_path = getattr(config, "db_path", None)
    model = model_name_from_env()
    chunk_size, overlap = _chunk_settings()
    collection = store.get_collection(db_path)
    folders = [Path(folder)]

    index_model = store.index_embed_model(collection, db_path)
    if index_model and index_model != model:
        if not rebuild:
            print(f"Error: {store.model_mismatch_message(index_model, model)}", file=sys.stderr)
            return 1
    if rebuild:
        old_roots = [Path(r) for r in store.stats(collection) if r != "(unknown)"]
        missing = [r for r in old_roots if not r.is_dir()]
        for r in missing:
            print(f"  Note: {r} no longer exists and will not be re-ingested.", file=sys.stderr)
        folders = sorted({Path(folder).resolve(), *(r for r in old_roots if r.is_dir())})
        store.reset(db_path)
        collection = store.get_collection(db_path)
        print(f"Rebuilding the index with {model} for {len(folders)} folder(s).", file=sys.stderr)

    print(f"Loading embedding model {model} (downloaded on first use) ...", file=sys.stderr)
    try:
        embedder = get_embedder()
    except Exception as exc:
        print(f"Error: Could not load the embedding model: {exc}", file=sys.stderr)
        print("Tip: The first run needs an internet connection to huggingface.co.",
              file=sys.stderr)
        return 1
    store.write_index_info({"embed_model": model}, db_path)

    max_tokens = getattr(embedder, "max_tokens", None)
    # Rough rule: one token is about 3 characters for German or English text.
    if max_tokens and chunk_size / 3 > max_tokens:
        print(f"Warning: chunks of {chunk_size} characters are longer than the {max_tokens} "
              f"tokens {model} reads; the end of each chunk is ignored. Lower MYRAG_CHUNK_SIZE.",
              file=sys.stderr)

    def progress(status: str, source: str) -> None:
        labels = {"added": "added", "updated": "updated", "removed": "removed",
                  "empty": "no text", "failed": "FAILED"}
        if status in labels:
            print(f"  [{labels[status]}] {source}", flush=True)

    failed = False
    for f in folders:
        print(f"Ingesting {f} ...")
        report = ingest_folder(f, collection, embedder, chunk_size, overlap, on_file=progress)
        for source, err in report.failed.items():
            print(f"  Error in {source}: {err}")
        print(report.summary())
        failed = failed or bool(report.failed)
    return 1 if failed else 0
