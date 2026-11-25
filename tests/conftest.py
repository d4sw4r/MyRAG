import pytest


@pytest.fixture(autouse=True)
def _no_model_downloads(monkeypatch):
    """Tests never load the real reranker or a non-default embedding model."""
    monkeypatch.setenv("MYRAG_RERANK", "0")
    monkeypatch.delenv("MYRAG_EMBED_MODEL", raising=False)
    for var in ("MYRAG_HYBRID", "MYRAG_MIN_SCORE", "MYRAG_MIN_SIMILARITY", "MYRAG_CANDIDATES",
                "MYRAG_CHUNK_SIZE", "MYRAG_CHUNK_OVERLAP"):
        monkeypatch.delenv(var, raising=False)
