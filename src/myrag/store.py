"""ChromaDB access. Schema per chunk:

    id        = "<root_key>:<source>::<chunk_index>"
    document  = chunk text
    metadata  = {source: relative POSIX path, root: absolute folder path,
                 chunk_index: int, file_hash: sha256 hex}

Several folders can share one index; `root` separates them. The collection has
no embedding_function of its own; embeddings are always passed explicitly
(for queries: query_embeddings=...).
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

DEFAULT_DB_PATH = Path(".myrag") / "chroma"
COLLECTION_NAME = "myrag"
_PAGE = 5000


def db_path_from_env() -> Path:
    return Path(os.environ.get("MYRAG_DB_PATH", str(DEFAULT_DB_PATH)))


def get_client(db_path: str | Path | None = None):
    import chromadb
    from chromadb.config import Settings

    path = Path(db_path) if db_path is not None else db_path_from_env()
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(path), settings=Settings(anonymized_telemetry=False)
    )


def get_collection(db_path: str | Path | None = None):
    client = get_client(db_path)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )


def index_exists(db_path: str | Path | None = None) -> bool:
    path = Path(db_path) if db_path is not None else db_path_from_env()
    return path.is_dir() and any(path.iterdir())


def reset(db_path: str | Path | None = None) -> bool:
    """Delete the entire index. Returns True if anything was deleted."""
    path = Path(db_path) if db_path is not None else db_path_from_env()
    if not path.exists():
        return False
    shutil.rmtree(path)
    return True


def root_key(root: str) -> str:
    return hashlib.sha256(root.encode()).hexdigest()[:12]


def chunk_id(source: str, index: int, root: str | None = None) -> str:
    prefix = f"{root_key(root)}:" if root else ""
    return f"{prefix}{source}::{index}"


def _where(source: str | None = None, root: str | None = None) -> dict | None:
    conds = [{k: v} for k, v in (("source", source), ("root", root)) if v is not None]
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def iter_metadatas(collection, root: str | None = None):
    offset = 0
    where = _where(root=root)
    while True:
        kwargs = {"include": ["metadatas"], "limit": _PAGE, "offset": offset}
        if where:
            kwargs["where"] = where
        metas = collection.get(**kwargs).get("metadatas") or []
        yield from (m for m in metas if m)
        if len(metas) < _PAGE:
            return
        offset += _PAGE


def indexed_files(collection, root: str | None = None) -> dict[str, str]:
    """Return {source: file_hash} of the indexed files (optionally of one folder only)."""
    return {
        m["source"]: m.get("file_hash", "")
        for m in iter_metadatas(collection, root)
        if "source" in m
    }


def stats(collection) -> dict[str, dict[str, int]]:
    """{folder: {"files": n, "chunks": m}} across the whole index."""
    files: dict[str, set[str]] = {}
    chunks: dict[str, int] = {}
    for m in iter_metadatas(collection):
        root = m.get("root", "(unknown)")
        files.setdefault(root, set()).add(m.get("source", ""))
        chunks[root] = chunks.get(root, 0) + 1
    return {r: {"files": len(files[r]), "chunks": chunks[r]} for r in sorted(files)}


def delete_source(collection, source: str, root: str | None = None) -> None:
    collection.delete(where=_where(source, root))


def add_chunks(
    collection,
    source: str,
    file_hash: str,
    chunks: list[str],
    embeddings: list[list[float]],
    root: str | None = None,
) -> None:
    if not chunks:
        return
    meta = {"source": source, "file_hash": file_hash}
    if root:
        meta["root"] = root
    collection.add(
        ids=[chunk_id(source, i, root) for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{**meta, "chunk_index": i} for i in range(len(chunks))],
    )
