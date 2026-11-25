"""Shared embedding module for ingest and ask.

Both sides must use the same model, otherwise the vectors don't match.
The model can be overridden with MYRAG_EMBED_MODEL. The index remembers which
model it was built with (see store.read_index_info), so a mismatch is caught.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Protocol

# multilingual-e5-small reads up to 512 tokens, so a 1000-character chunk is
# embedded completely. The previous default only read 128 tokens and silently
# ignored roughly the second half of every chunk.
DEFAULT_MODEL = "intfloat/multilingual-e5-small"
# Model used by indexes created before the model name was stored (v0.1).
LEGACY_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

_E5_INSTRUCT_QUERY = "Instruct: Given a question, retrieve passages that answer the question\nQuery: "


class Embedder(Protocol):
    model_name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def prefixes_for(model_name: str) -> tuple[str, str]:
    """(query_prefix, passage_prefix) the model was trained with.

    E5 models expect "query: " / "passage: " prefixes; without them retrieval
    quality drops noticeably. Other models get no prefix.
    """
    name = model_name.lower().rsplit("/", 1)[-1]
    if "e5" in name and "instruct" in name:
        return _E5_INSTRUCT_QUERY, ""
    if "e5" in name:
        return "query: ", "passage: "
    return "", ""


class SentenceTransformerEmbedder:
    """Local embeddings with sentence-transformers (works offline after the first download)."""

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        self.model_name = model_name
        self.batch_size = batch_size
        self.query_prefix, self.passage_prefix = prefixes_for(model_name)
        self._model = SentenceTransformer(model_name)

    @property
    def max_tokens(self) -> int | None:
        """Number of tokens the model reads per text; longer texts are truncated."""
        return getattr(self._model, "max_seq_length", None)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode([self.passage_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.query_prefix + text])[0]


def model_name_from_env() -> str:
    return os.environ.get("MYRAG_EMBED_MODEL") or DEFAULT_MODEL


@lru_cache(maxsize=4)
def get_embedder(model_name: str | None = None) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(model_name or model_name_from_env())
