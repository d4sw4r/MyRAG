import json
import types

import chromadb
import httpx
import pytest

from myrag import store
from myrag.ask import NOTHING_FOUND, ask
from myrag.chat import ChatSession, loop
from myrag.embeddings import prefixes_for
from myrag.llm import ClaudeLLM, OllamaLLM
from myrag.retrieval import BM25, Hit, Retriever, RetrievalSettings, fuse, tokenize
from myrag.search import format_hit


def vec(text):
    v = [0.0] * 26
    for c in text.lower():
        if "a" <= c <= "z":
            v[ord(c) - 97] += 1
    return v


class FakeLLM:
    name = model = "fake"

    def __init__(self, answers=None):
        self.calls = []
        self.answers = list(answers or [])

    def chat(self, system, messages):
        self.calls.append(("chat", system, messages))
        return self.answers.pop(0) if self.answers else "query"

    def generate(self, system, prompt):
        return self.chat(system, [{"role": "user", "content": prompt}])

    def stream(self, system, messages):
        self.calls.append(("stream", system, messages))
        yield "Answer "
        yield "[1]"


class FakeReranker:
    """Score = share of query words that occur in the text."""

    def score(self, query, texts):
        words = set(tokenize(query))
        return [len(words & set(tokenize(t))) / max(len(words), 1) for t in texts]


TEXTS = [
    "Katzen schlafen sechzehn Stunden am Tag",
    "Rechnung Nummer RE-2024-0815 vom Stromanbieter",
    "Rezept für Pizza mit Tomaten und Basilikum",
    "Urlaub in Norwegen im Sommer",
]


@pytest.fixture
def collection():
    col = chromadb.EphemeralClient().get_or_create_collection(f"t{id(object())}", metadata={"hnsw:space": "cosine"})
    for i, t in enumerate(TEXTS):
        store.add_chunks(col, f"notes/{i}.md", "h", [t], [vec(t)], root="/docs")
    return col


def test_e5_prefixes():
    assert prefixes_for("intfloat/multilingual-e5-small") == ("query: ", "passage: ")
    q, p = prefixes_for("intfloat/multilingual-e5-large-instruct")
    assert q.startswith("Instruct:") and p == ""
    assert prefixes_for("paraphrase-multilingual-MiniLM-L12-v2") == ("", "")


def test_bm25_ranks_exact_terms():
    bm = BM25(TEXTS)
    scores = bm.scores("RE-2024-0815")
    assert max(range(len(scores)), key=scores.__getitem__) == 1
    assert bm.scores("unbekanntes Wort") == [0.0] * 4


def test_fuse_merges_duplicates():
    a = [Hit("x", "a", id="1", distance=0.1), Hit("y", "b", id="2", distance=0.2)]
    b = [Hit("y", "b", id="2", keyword_score=3.0)]
    merged = fuse([a, b])
    assert [h.id for h in merged] == ["2", "1"]  # found by both lists -> first
    assert merged[0].distance == 0.2 and merged[0].keyword_score == 3.0


def test_hybrid_finds_invoice_number(collection):
    # Letter-frequency vectors can't match a number, the keyword search can.
    vector_only = Retriever(collection, vec, settings=RetrievalSettings(hybrid=False, rerank=False))
    hybrid = Retriever(collection, vec, settings=RetrievalSettings(rerank=False))
    assert vector_only.search("0815", k=1)[0].source != "notes/1.md"
    assert hybrid.search("0815", k=1)[0].source == "notes/1.md"


def test_reranker_orders_and_filters(collection):
    r = Retriever(collection, vec, FakeReranker(), RetrievalSettings(min_score=0.5))
    hits = r.search("Pizza Tomaten", k=3)
    assert [h.source for h in hits] == ["notes/2.md"]
    assert hits[0].score == 1.0
    assert r.search("Quantenphysik Vorlesung", k=3) == []


def test_ask_without_relevant_hits_skips_llm(collection):
    llm = FakeLLM()
    r = Retriever(collection, vec, FakeReranker(), RetrievalSettings(min_score=0.5))
    answer = ask("Quantenphysik Vorlesung", collection, llm, retriever=r)
    assert answer.text == NOTHING_FOUND and llm.calls == []


def test_min_similarity_without_reranker(collection):
    r = Retriever(collection, vec, settings=RetrievalSettings(hybrid=False, rerank=False, min_similarity=0.999))
    assert r.search("zzzz", k=4) == []


def test_keyword_index_refreshes_after_ingest(collection):
    r = Retriever(collection, vec, settings=RetrievalSettings(rerank=False))
    assert r.keyword_search("Fahrrad", 5) == []
    store.add_chunks(collection, "new.md", "h", ["Fahrrad reparieren"], [vec("Fahrrad reparieren")], root="/docs")
    assert r.keyword_search("Fahrrad", 5)[0].source == "new.md"


def test_chat_rewrites_follow_up_and_keeps_history(collection):
    llm = FakeLLM(answers=["Pizza Tomaten Basilikum"])
    retriever = Retriever(collection, vec, settings=RetrievalSettings(rerank=False))
    out = __import__("io").StringIO()
    session = ChatSession(llm, retriever, k=2, out=out)
    assert session.ask("Wie mache ich Pizza?") == "Answer [1]"
    assert len(llm.calls) == 1  # first question: no rewriting
    session.ask("Und was kommt drauf?")
    kind, system, messages = llm.calls[1]
    assert kind == "chat" and "standalone" in system  # rewrite step
    kind, _, messages = llm.calls[2]
    assert kind == "stream"
    assert messages[0] == {"role": "user", "content": "Wie mache ich Pizza?"}
    assert messages[1] == {"role": "assistant", "content": "Answer [1]"}
    assert "Question: Und was kommt drauf?" in messages[2]["content"]
    assert session.last_hits[0].source == "notes/2.md"


def test_chat_loop_commands(collection):
    import io

    out = io.StringIO()
    session = ChatSession(FakeLLM(), Retriever(collection, vec, settings=RetrievalSettings(rerank=False)), out=out)
    inputs = iter(["/sources", "Pizza Tomaten", "/sources", "/clear", "/exit"])
    assert loop(session, lambda _: next(inputs)) == 0
    text = out.getvalue()
    assert "No sources yet." in text and "Answer [1]" in text
    assert "[1] notes/2.md" in text and "Conversation cleared." in text
    assert session.history == []


def test_format_hit_shows_path_and_scores():
    h = Hit("Ein langer Text " * 50, "a/b.md", 3, distance=0.25, root="/docs", score=0.9, keyword_score=2.5)
    out = format_hit(1, h)
    assert out.startswith("1. /docs/a/b.md (section 3)")
    assert "relevance 0.90" in out and "similarity 0.75" in out and "keywords 2.5" in out
    assert out.endswith("...")


def test_search_cli(tmp_path, monkeypatch, capsys):
    from myrag.cli import main
    from myrag.embeddings import model_name_from_env

    db = tmp_path / "db"
    col = store.get_collection(db)
    for i, t in enumerate(TEXTS):
        store.add_chunks(col, f"notes/{i}.md", "h", [t], [vec(t)], root="/docs")
    store.write_index_info({"embed_model": model_name_from_env()}, db)
    monkeypatch.setenv("MYRAG_DB_PATH", str(db))
    monkeypatch.setattr("myrag.embeddings.get_embedder", lambda *a: types.SimpleNamespace(embed_query=vec))
    assert main(["search", "Rechnung 0815", "-k", "2"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("1. /docs/notes/1.md")


def test_ask_refuses_mismatched_model(tmp_path, monkeypatch, capsys):
    from myrag.cli import main

    db = tmp_path / "db"
    store.add_chunks(store.get_collection(db), "a.md", "h", ["x"], [[1.0, 0.0]], root="/docs")
    # no info file + chunks = v0.1 index built with the old default model
    monkeypatch.setenv("MYRAG_DB_PATH", str(db))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr("myrag.embeddings.get_embedder", lambda *a: pytest.fail("must not load"))
    assert main(["ask", "x"]) == 1
    assert "--rebuild" in capsys.readouterr().err


def test_claude_stream_yields_text():
    class Msg:
        stop_reason = "end_turn"
        content = []

    class Stream:
        text_stream = iter(["Hal", "lo"])
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get_final_message(self): return Msg()

    captured = {}

    class Messages:
        def stream(self, **kw):
            captured.update(kw)
            return Stream()

    class Client:
        class beta:
            messages = Messages()

    history = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
               {"role": "user", "content": "c"}]
    assert "".join(ClaudeLLM(client=Client()).stream("sys", history)) == "Hallo"
    assert captured["messages"] == history


def test_ollama_stream():
    lines = [{"message": {"content": "Hal"}}, {"message": {"content": "lo"}}, {"done": True}]
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, content="\n".join(json.dumps(x) for x in lines).encode())

    llm = OllamaLLM(model="m", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert "".join(llm.stream("sys", [{"role": "user", "content": "q"}])) == "Hallo"
    assert seen["stream"] is True and seen["messages"][1] == {"role": "user", "content": "q"}
