"""Shared embedding module for ingest and ask.

Both sides must use the same model, otherwise the vectors don't match.
The model can be overridden with MYRAG_EMBED_MODEL.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Protocol

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


class Embedder(Protocol):
    model_name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Local embeddings with sentence-transformers (runs offline after the first download)."""

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def model_name_from_env() -> str:
    return os.environ.get("MYRAG_EMBED_MODEL", DEFAULT_MODEL)


@lru_cache(maxsize=4)
def get_embedder(model_name: str | None = None) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(model_name or model_name_from_env())
