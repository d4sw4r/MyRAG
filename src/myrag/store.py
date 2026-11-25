"""ChromaDB-Zugriff. Schema pro Chunk:

    id        = "<source>::<chunk_index>"
    document  = Chunk-Text
    metadata  = {source: relativer POSIX-Pfad, chunk_index: int, file_hash: sha256-hex}

Die Collection hat keine eigene embedding_function; Embeddings werden immer
explizit übergeben (bei Abfragen: query_embeddings=...).
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DB_PATH = Path(".myrag") / "chroma"
COLLECTION_NAME = "myrag"


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


def chunk_id(source: str, index: int) -> str:
    return f"{source}::{index}"


def indexed_files(collection) -> dict[str, str]:
    """Liefert {source: file_hash} für alle bereits indexierten Dateien."""
    result: dict[str, str] = {}
    offset, page = 0, 5000
    while True:
        batch = collection.get(include=["metadatas"], limit=page, offset=offset)
        metas = batch.get("metadatas") or []
        for meta in metas:
            if meta and "source" in meta:
                result[meta["source"]] = meta.get("file_hash", "")
        if len(metas) < page:
            return result
        offset += page


def delete_source(collection, source: str) -> None:
    collection.delete(where={"source": source})


def add_chunks(
    collection,
    source: str,
    file_hash: str,
    chunks: list[str],
    embeddings: list[list[float]],
) -> None:
    if not chunks:
        return
    collection.add(
        ids=[chunk_id(source, i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[
            {"source": source, "chunk_index": i, "file_hash": file_hash}
            for i in range(len(chunks))
        ],
    )
