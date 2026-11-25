"""Fragen an die Dokumente stellen: passende Chunks suchen und mit Quellen beantworten."""

from __future__ import annotations

from dataclasses import dataclass, field

from myrag.llm import LLM

SYSTEM_PROMPT = (
    "Du beantwortest Fragen ausschließlich anhand der mitgelieferten Auszüge aus den "
    "Dokumenten der Person. Belege jede Aussage mit der Nummer des Auszugs in eckigen "
    "Klammern, z.B. [1] oder [2][3]. Wenn die Auszüge die Frage nicht beantworten, sag "
    "das offen und erfinde nichts. Antworte in der Sprache der Frage."
)


@dataclass
class Hit:
    text: str
    source: str
    chunk_index: int | None = None
    distance: float | None = None

    @property
    def label(self) -> str:
        return self.source if self.chunk_index is None else f"{self.source} (Abschnitt {self.chunk_index})"


@dataclass
class Answer:
    text: str
    hits: list[Hit] = field(default_factory=list)


def retrieve(collection, question: str, k: int = 5, embed=None) -> list[Hit]:
    """Holt die k ähnlichsten Chunks aus einer ChromaDB-Collection.

    `embed` ist optional eine Funktion Text -> Vektor, falls die Collection ohne
    eigene Embedding-Funktion angelegt wurde.
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
            source=str(meta.get("source", "unbekannt")),
            chunk_index=meta.get("chunk_index"),
            distance=dist,
        ))
    return hits


def build_prompt(question: str, hits: list[Hit]) -> str:
    parts = [f"[{i}] Quelle: {h.label}\n{h.text}" for i, h in enumerate(hits, 1)]
    return "Auszüge:\n\n" + "\n\n---\n\n".join(parts) + f"\n\nFrage: {question}"


EMPTY_INDEX = "Der Index ist leer. Lies zuerst einen Ordner ein: myrag ingest <ordner>"


def ask(question: str, collection, llm: LLM, k: int = 5, embed=None) -> Answer:
    if collection.count() == 0:
        return Answer(EMPTY_INDEX, [])
    hits = retrieve(collection, question, k=min(k, collection.count()), embed=embed)
    if not hits:
        return Answer("Im Index wurde nichts Passendes gefunden.", [])
    return Answer(llm.generate(SYSTEM_PROMPT, build_prompt(question, hits)), hits)


def format_answer(answer: Answer) -> str:
    if not answer.hits:
        return answer.text
    lines = [answer.text, "", "Quellen:"]
    lines += [f"  [{i}] {h.label}" for i, h in enumerate(answer.hits, 1)]
    return "\n".join(lines)


def run(question: str, config, top_k: int = 5) -> int:
    """Einstieg für `myrag ask`: sucht, fragt das Sprachmodell und gibt die Antwort aus."""
    import sys

    from myrag import store
    from myrag.embeddings import get_embedder
    from myrag.llm import LLMError, get_llm

    if not store.index_exists(config.db_path):
        print(f"Fehler: Kein Index unter {config.db_path}. {EMPTY_INDEX}", file=sys.stderr)
        return 1
    try:
        llm = get_llm(config.llm, config.model, getattr(config, "ollama_host", None))
        collection = store.get_collection(config.db_path)
        if collection.count() == 0:  # Embedding-Modell gar nicht erst laden
            print(f"Fehler: {EMPTY_INDEX}", file=sys.stderr)
            return 1
        answer = ask(question, collection, llm, k=top_k, embed=get_embedder().embed_query)
    except (LLMError, RuntimeError, ValueError) as e:
        print(f"Fehler: {e}", file=sys.stderr)
        return 1
    print(format_answer(answer))
    return 0
