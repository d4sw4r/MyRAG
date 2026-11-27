"""Ask questions about the documents: find matching chunks and answer with sources."""

from __future__ import annotations

from dataclasses import dataclass, field

from myrag.llm import LLM
from myrag.retrieval import Hit, Retriever, retrieve  # noqa: F401  (re-exported)

SYSTEM_PROMPT = (
    "Answer questions using only the provided excerpts from the user's documents. "
    "Cite every statement with the number of the excerpt in square brackets, "
    "e.g. [1] or [2][3]. If the excerpts do not answer the question, say so openly "
    "and do not make anything up. Answer in the language of the question."
)


@dataclass
class Answer:
    text: str
    hits: list[Hit] = field(default_factory=list)


def build_prompt(question: str, hits: list[Hit]) -> str:
    parts = [f"[{i}] Source: {h.label}\n{h.text}" for i, h in enumerate(hits, 1)]
    return "Excerpts:\n\n" + "\n\n---\n\n".join(parts) + f"\n\nQuestion: {question}"


EMPTY_INDEX = "The index is empty. Ingest a folder first: myrag ingest <folder>"
NOTHING_FOUND = (
    "Nothing relevant was found in the index for this question. "
    "Try other words, or check with `myrag search` what the index contains."
)


def ask(question: str, collection, llm: LLM, k: int = 5, embed=None,
        retriever: Retriever | None = None) -> Answer:
    if collection.count() == 0:
        return Answer(EMPTY_INDEX, [])
    retriever = retriever or Retriever(collection, embed)
    hits = retriever.search(question, k=k)
    if not hits:
        return Answer(NOTHING_FOUND, [])
    return Answer(llm.generate(SYSTEM_PROMPT, build_prompt(question, hits)), hits)


def format_sources(hits: list[Hit]) -> str:
    return "\n".join(["Sources:"] + [f"  [{i}] {h.label}" for i, h in enumerate(hits, 1)])


def format_answer(answer: Answer) -> str:
    if not answer.hits:
        return answer.text
    return f"{answer.text}\n\n{format_sources(answer.hits)}"


def stream_answer(llm: LLM, system: str, messages: list[dict], out) -> str:
    """Print the answer while it is generated; returns the full text.

    Falls back to a single call for backends without streaming.
    """
    if not hasattr(llm, "stream"):
        text = llm.chat(system, messages) if hasattr(llm, "chat") else llm.generate(system, messages[-1]["content"])
        print(text, file=out)
        return text
    parts = []
    for piece in llm.stream(system, messages):
        parts.append(piece)
        print(piece, end="", file=out, flush=True)
    print(file=out)
    return "".join(parts).strip()


def open_index(config):
    """Open the collection for ask/search/chat. Raises ValueError with a readable message."""
    from myrag import store

    if not store.index_exists(config.db_path):
        raise ValueError(f"No index at {config.db_path}. {EMPTY_INDEX}")
    collection = store.get_collection(config.db_path)
    if collection.count() == 0:
        raise ValueError(EMPTY_INDEX)
    return collection


def run(question: str, config, top_k: int = 5) -> int:
    """Entry point for `myrag ask`: retrieves, queries the language model and prints the answer."""
    import sys

    from myrag import retrieval
    from myrag.llm import LLMError, get_llm

    try:
        llm = get_llm(config.llm, config.model, getattr(config, "ollama_host", None))
        collection = open_index(config)  # before loading any model
        retriever = retrieval.from_config(config, collection)
        hits = retriever.search(question, k=top_k)
        if not hits:
            print(NOTHING_FOUND)
            return 0
        prompt = build_prompt(question, hits)
        stream_answer(llm, SYSTEM_PROMPT, [{"role": "user", "content": prompt}], sys.stdout)
    except (LLMError, RuntimeError, ValueError) as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1
    print()
    print(format_sources(hits))
    return 0
