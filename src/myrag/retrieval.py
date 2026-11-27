"""Find the passages that best match a query.

Pipeline:
1. Vector search in ChromaDB (semantic similarity).
2. Keyword search with BM25 over all chunks (names, numbers, exact terms).
3. Both result lists are merged with reciprocal rank fusion (RRF).
4. Optionally a cross-encoder reranks the merged candidates.
5. Candidates below the relevance threshold are dropped.

Settings come from environment variables (see RetrievalSettings.from_env).
"""

from __future__ import annotations

import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

DEFAULT_RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RRF_K = 60  # standard constant for reciprocal rank fusion

_TOKEN = re.compile(r"\w+", re.UNICODE)


@dataclass
class Hit:
    text: str
    source: str
    chunk_index: int | None = None
    distance: float | None = None
    root: str | None = None
    id: str | None = None
    score: float | None = None  # reranker score (0..1) if a reranker ran
    keyword_score: float | None = None  # BM25 score if the keyword search found it

    @property
    def label(self) -> str:
        return self.source if self.chunk_index is None else f"{self.source} (section {self.chunk_index})"

    @property
    def similarity(self) -> float | None:
        """Cosine similarity from the vector search (1 = identical)."""
        return None if self.distance is None else 1.0 - self.distance


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class RetrievalSettings:
    hybrid: bool = True
    rerank: bool = True
    rerank_model: str = DEFAULT_RERANK_MODEL
    candidates: int = 30
    min_score: float = 0.1  # minimum reranker score (0..1)
    min_similarity: float = 0.0  # minimum cosine similarity without reranker (0 = off)

    @classmethod
    def from_env(cls) -> "RetrievalSettings":
        return cls(
            hybrid=_env_bool("MYRAG_HYBRID", True),
            rerank=_env_bool("MYRAG_RERANK", True),
            rerank_model=os.environ.get("MYRAG_RERANK_MODEL") or DEFAULT_RERANK_MODEL,
            candidates=max(1, int(_env_float("MYRAG_CANDIDATES", 30))),
            min_score=_env_float("MYRAG_MIN_SCORE", 0.1),
            min_similarity=_env_float("MYRAG_MIN_SIMILARITY", 0.0),
        )


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    """Okapi BM25 over a fixed list of documents."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lengths = [sum(tf.values()) for tf in self.tfs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter[str] = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = [t for t in set(tokenize(query)) if t in self.idf]
        out = []
        for tf, length in zip(self.tfs, self.lengths):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
            for t in terms:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            out.append(s)
        return out


class CrossEncoderReranker:
    """Scores (query, passage) pairs with a cross-encoder; higher = more relevant."""

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL):
        from sentence_transformers import CrossEncoder  # lazy: heavy import

        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        raw = self._model.predict([(query, t) for t in texts], show_progress_bar=False)
        values = [float(v) for v in raw]
        # Some models return logits, others probabilities. Map to 0..1 either way.
        if any(v < 0 or v > 1 for v in values):
            values = [1 / (1 + math.exp(-v)) for v in values]
        return values


@lru_cache(maxsize=2)
def get_reranker(model_name: str = DEFAULT_RERANK_MODEL) -> CrossEncoderReranker:
    return CrossEncoderReranker(model_name)


def _hit_from(id_, doc, meta, distance=None) -> Hit:
    meta = meta or {}
    return Hit(
        text=doc or "",
        source=str(meta.get("source", "unknown")),
        chunk_index=meta.get("chunk_index"),
        distance=distance,
        root=meta.get("root"),
        id=id_,
    )


class Retriever:
    """Hybrid search with optional reranking. Reuse one instance for many queries
    (e.g. in chat), so the keyword index is only built once."""

    def __init__(
        self,
        collection,
        embed: Callable[[str], list[float]] | None = None,
        reranker=None,
        settings: RetrievalSettings | None = None,
    ):
        self.collection = collection
        self.embed = embed
        self.reranker = reranker
        self.settings = settings or RetrievalSettings(rerank=reranker is not None)
        self._bm25: tuple[int, BM25, list[Hit]] | None = None

    # -- individual searches -------------------------------------------------

    def vector_search(self, query: str, n: int) -> list[Hit]:
        include = ["documents", "metadatas", "distances"]
        if self.embed is not None:
            res = self.collection.query(query_embeddings=[self.embed(query)], n_results=n, include=include)
        else:
            res = self.collection.query(query_texts=[query], n_results=n, include=include)
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0] or [None] * len(docs)
        return [_hit_from(i, d, m, dist) for i, d, m, dist in zip(ids, docs, metas, dists)]

    def keyword_search(self, query: str, n: int) -> list[Hit]:
        from . import store

        count = self.collection.count()
        if self._bm25 is None or self._bm25[0] != count:
            chunks = [_hit_from(i, d, m) for i, d, m in store.iter_chunks(self.collection)]
            self._bm25 = (count, BM25([h.text for h in chunks]), chunks)
        _, bm25, chunks = self._bm25
        scored = [(s, i) for i, s in enumerate(bm25.scores(query)) if s > 0]
        scored.sort(key=lambda x: -x[0])
        hits = []
        for s, i in scored[:n]:
            h = chunks[i]
            hits.append(Hit(h.text, h.source, h.chunk_index, None, h.root, h.id, None, s))
        return hits

    # -- full pipeline -------------------------------------------------------

    def search(self, query: str, k: int = 5) -> list[Hit]:
        count = self.collection.count()
        if count == 0:
            return []
        s = self.settings
        n = min(count, max(k, s.candidates if (s.hybrid or self.reranker) else k))

        lists = [self.vector_search(query, n)]
        if s.hybrid:
            lists.append(self.keyword_search(query, n))
        candidates = fuse(lists)

        if self.reranker is not None:
            scores = self.reranker.score(query, [h.text for h in candidates])
            for h, sc in zip(candidates, scores):
                h.score = sc
            candidates.sort(key=lambda h: -(h.score or 0.0))
            candidates = [h for h in candidates if (h.score or 0.0) >= s.min_score]
        elif s.min_similarity > 0:
            # Keyword-only hits have no similarity; they matched exact terms, so keep them.
            candidates = [h for h in candidates if h.similarity is None or h.similarity >= s.min_similarity]
        return candidates[:k]


def fuse(result_lists: list[list[Hit]]) -> list[Hit]:
    """Merge ranked lists with reciprocal rank fusion; keeps the best info per chunk."""
    scores: dict[str, float] = {}
    merged: dict[str, Hit] = {}
    for hits in result_lists:
        for rank, h in enumerate(hits):
            key = h.id or f"{h.root}:{h.source}::{h.chunk_index}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            if key not in merged:
                merged[key] = h
            else:
                m = merged[key]
                m.distance = m.distance if m.distance is not None else h.distance
                m.keyword_score = m.keyword_score if m.keyword_score is not None else h.keyword_score
    return [merged[key] for key in sorted(merged, key=lambda x: -scores[x])]


def retrieve(collection, question: str, k: int = 5, embed=None, reranker=None,
             settings: RetrievalSettings | None = None) -> list[Hit]:
    """One-off search; see Retriever.search."""
    return Retriever(collection, embed, reranker, settings).search(question, k)


class RetrievalError(ValueError):
    """Readable error for the command line."""


def from_config(config, collection=None) -> Retriever:
    """Build a Retriever for the CLI: checks the index, loads embedder and reranker."""
    from . import store
    from .embeddings import get_embedder, model_name_from_env

    db_path = config.db_path
    if collection is None:
        collection = store.get_collection(db_path)
    current = model_name_from_env()
    index_model = store.index_embed_model(collection, db_path)
    if index_model and index_model != current:
        raise RetrievalError(store.model_mismatch_message(index_model, current))

    settings = RetrievalSettings.from_env()
    embedder = get_embedder()
    reranker = None
    if settings.rerank:
        try:
            reranker = get_reranker(settings.rerank_model)
        except Exception as exc:  # reranking is an improvement, not a requirement
            print(f"Warning: reranker {settings.rerank_model} could not be loaded ({exc}); "
                  "continuing without reranking. Set MYRAG_RERANK=0 to silence this.",
                  file=sys.stderr)
    return Retriever(collection, embedder.embed_query, reranker, settings)
