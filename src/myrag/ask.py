"""Ask questions about the documents: find matching chunks and answer with sources."""

from __future__ import annotations

from dataclasses import dataclass, field

from myrag.llm import LLM

SYSTEM_PROMPT = (
    "Answer questions using only the provided excerpts from the user's documents. "
    "Cite every statement with the number of the excerpt in square brackets, "
    "e.g. [1] or [2][3]. If the excerpts do not answer the question, say so openly "
    "and do not make anything up. Answer in the language of the question."
)


@dataclass
class Hit:
    text: str
    source: str
    chunk_index: int | None = None
    distance: float | None = None

    @property
    def label(self) -> str:
        return self.source if self.chunk_index is None else f"{self.source} (section {self.chunk_index})"


@dataclass
class Answer:
    text: str
    hits: list[Hit] = field(default_factory=list)


def retrieve(collection, question: str, k: int = 5, embed=None) -> list[Hit]:
    """Fetch the k most similar chunks from a ChromaDB collection.

    `embed` is an optional text -> vector function, for collections created
    without their own embedding function.
    """
    if embed is not None:
        res = collection.query(
            query_embeddings=[embed(question)], n_results=k,
            include=["documents", "metadatas", "distances"],
        )
    else:
        res = collection.query(
            query_texts=[question], n_results=k,
            include=["documents", "metadatas", "distances"],
        )
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0] or [None] * len(docs)
    hits = []
    for doc, meta, dist in zip(docs, metas, dists):
        meta = meta or {}
        hits.append(Hit(
            text=doc,
            source=str(meta.get("source", "unknown")),
            chunk_index=meta.get("chunk_index"),
            distance=dist,
        ))
    return hits


def build_prompt(question: str, hits: list[Hit]) -> str:
    parts = [f"[{i}] Source: {h.label}\n{h.text}" for i, h in enumerate(hits, 1)]
    return "Excerpts:\n\n" + "\n\n---\n\n".join(parts) + f"\n\nQuestion: {question}"


EMPTY_INDEX = "The index is empty. Ingest a folder first: myrag ingest <folder>"


def ask(question: str, collection, llm: LLM, k: int = 5, embed=None) -> Answer:
    if collection.count() == 0:
        return Answer(EMPTY_INDEX, [])
    hits = retrieve(collection, question, k=min(k, collection.count()), embed=embed)
    if not hits:
        return Answer("Nothing relevant was found in the index.", [])
    return Answer(llm.generate(SYSTEM_PROMPT, build_prompt(question, hits)), hits)


def format_answer(answer: Answer) -> str:
    if not answer.hits:
        return answer.text
    lines = [answer.text, "", "Sources:"]
    lines += [f"  [{i}] {h.label}" for i, h in enumerate(answer.hits, 1)]
    return "\n".join(lines)


def run(question: str, config, top_k: int = 5) -> int:
    """Entry point for `myrag ask`: retrieves, queries the language model and prints the answer."""
    import sys

    from myrag import store
    from myrag.embeddings import get_embedder
    from myrag.llm import LLMError, get_llm

    if not store.index_exists(config.db_path):
        print(f"Error: No index at {config.db_path}. {EMPTY_INDEX}", file=sys.stderr)
        return 1
    try:
        llm = get_llm(config.llm, config.model, getattr(config, "ollama_host", None))
        collection = store.get_collection(config.db_path)
        if collection.count() == 0:  # don't even load the embedding model
            print(f"Error: {EMPTY_INDEX}", file=sys.stderr)
            return 1
        answer = ask(question, collection, llm, k=top_k, embed=get_embedder().embed_query)
    except (LLMError, RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(format_answer(answer))
    return 0
