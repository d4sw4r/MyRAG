"""`myrag status`: overview of the index and configuration."""

from __future__ import annotations

from . import store
from .embeddings import model_name_from_env
from .retrieval import RetrievalSettings
from .llm import DEFAULT_MODELS, _has_anthropic_credentials


def run(config) -> int:
    print(f"Index:            {config.db_path.resolve()}")
    print(f"Embedding model:  {model_name_from_env()}")
    settings = RetrievalSettings.from_env()
    search = "hybrid (vector + keywords)" if settings.hybrid else "vector only"
    rerank = f"reranking with {settings.rerank_model}" if settings.rerank else "no reranking"
    print(f"Search:           {search}, {rerank}")
    model = config.model or DEFAULT_MODELS.get(config.llm, "?")
    line = f"Language model:   {config.llm} ({model})"
    if config.llm == "claude":
        line += ", API key " + ("found" if _has_anthropic_credentials() else "MISSING (ANTHROPIC_API_KEY)")
    elif config.llm == "ollama":
        line += f", {config.ollama_host}"
    print(line)

    if not store.index_exists(config.db_path):
        print("\nThe index is still empty. Start with: myrag ingest <folder>")
        return 0
    collection = store.get_collection(config.db_path)
    index_model = store.index_embed_model(collection, config.db_path)
    if index_model and index_model != model_name_from_env():
        print(f"\nWarning: {store.model_mismatch_message(index_model, model_name_from_env())}")
    stats = store.stats(collection)
    if not stats:
        print("\nThe index is still empty. Start with: myrag ingest <folder>")
        return 0
    files = sum(s["files"] for s in stats.values())
    chunks = sum(s["chunks"] for s in stats.values())
    print(f"\n{files} file(s), {chunks} chunk(s) in {len(stats)} folder(s):")
    for root, s in stats.items():
        print(f"  {root}: {s['files']} file(s), {s['chunks']} chunk(s)")
    return 0
