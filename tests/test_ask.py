import json

import chromadb
import httpx
import pytest

from myrag.ask import ask, build_prompt, format_answer, retrieve, Hit
from myrag.llm import OllamaLLM, get_llm, ClaudeLLM


class FakeLLM:
    name = "fake"
    model = "fake"

    def __init__(self):
        self.calls = []

    def generate(self, system, prompt):
        self.calls.append((system, prompt))
        return "Die Antwort steht in [1]."


def vec(text):
    # Mini-Embedding: Buchstabenhäufigkeit, reicht für deterministische Tests.
    v = [0.0] * 26
    for c in text.lower():
        if "a" <= c <= "z":
            v[ord(c) - 97] += 1
    return v


@pytest.fixture
def collection():
    col = chromadb.EphemeralClient().get_or_create_collection(f"t{id(object())}")
    texts = ["Katzen schlafen viel", "Rezept für Pizza mit Tomaten", "Steuererklärung bis Juli"]
    col.add(
        ids=[f"c{i}" for i in range(3)],
        documents=texts,
        embeddings=[vec(t) for t in texts],
        metadatas=[{"source": f"notes/{i}.md", "chunk_index": 0} for i in range(3)],
    )
    return col


def test_retrieve_returns_sources(collection):
    hits = retrieve(collection, "Pizza Tomaten Rezept", k=1, embed=vec)
    assert hits[0].source == "notes/1.md"
    assert hits[0].chunk_index == 0


def test_ask_passes_numbered_sources_to_llm(collection):
    llm = FakeLLM()
    answer = ask("Pizza Tomaten Rezept", collection, llm, k=2, embed=vec)
    _, prompt = llm.calls[0]
    assert "[1] Quelle: notes/1.md" in prompt
    assert "Frage: Pizza Tomaten Rezept" in prompt
    out = format_answer(answer)
    assert "Quellen:" in out and "[1] notes/1.md (Abschnitt 0)" in out


def test_ask_empty_index():
    col = chromadb.EphemeralClient().get_or_create_collection("leer")
    llm = FakeLLM()
    answer = ask("irgendwas", col, llm, embed=vec)
    assert llm.calls == []
    assert "ingest" in answer.text


def test_ollama_request_shape():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"message": {"content": " Hallo [1] "}})

    llm = OllamaLLM(model="m", host="localhost:11434", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert llm.generate("sys", "frage") == "Hallo [1]"
    assert seen["url"] == "http://localhost:11434/api/chat"
    assert seen["model"] == "m" and seen["stream"] is False
    assert seen["messages"][0] == {"role": "system", "content": "sys"}


def test_ollama_missing_model_message():
    llm = OllamaLLM(model="fehlt", client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404))))
    with pytest.raises(RuntimeError, match="ollama pull fehlt"):
        llm.generate("s", "p")


def test_get_llm_switch(monkeypatch):
    monkeypatch.delenv("MYRAG_LLM", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    assert get_llm().name == "claude"
    monkeypatch.setenv("MYRAG_LLM", "ollama")
    assert get_llm().name == "ollama"
    llm = get_llm("claude")
    assert isinstance(llm, ClaudeLLM) and llm.model == "claude-opus-5-5"
    assert get_llm("ollama", "qwen", "http://box:1").host == "http://box:1"
    with pytest.raises(ValueError):
        get_llm("gpt")


def test_claude_uses_stream_and_text_blocks():
    class Block:
        def __init__(self, t, text=""):
            self.type, self.text = t, text

    class Msg:
        stop_reason = "end_turn"
        content = [Block("thinking"), Block("text", "Antwort [1]")]

    class Stream:
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

    llm = ClaudeLLM(client=Client())
    assert llm.generate("sys", "p") == "Antwort [1]"
    assert captured["model"] == "claude-opus-5-5"
    assert captured["system"] == "sys"


def _patch_embedder(monkeypatch):
    import types
    monkeypatch.setattr("myrag.embeddings.get_embedder", lambda *a: types.SimpleNamespace(embed_query=vec))


def test_run_uses_store_and_embedder(tmp_path, monkeypatch, capsys):
    import types
    from myrag import ask as ask_mod, store

    col = store.get_collection(tmp_path / "db")
    texts = ["Katzen schlafen viel", "Rezept für Pizza mit Tomaten"]
    for i, t in enumerate(texts):
        store.add_chunks(col, f"notes/{i}.md", "h", [t], [vec(t)], root="/docs")
    _patch_embedder(monkeypatch)
    monkeypatch.setattr("myrag.llm.get_llm", lambda *a: FakeLLM())
    cfg = types.SimpleNamespace(db_path=tmp_path / "db", llm="claude", model=None, ollama_host=None)
    assert ask_mod.run("Pizza Tomaten", cfg, top_k=1) == 0
    out = capsys.readouterr().out
    assert "[1] notes/1.md" in out


def test_run_without_index(tmp_path, monkeypatch, capsys):
    import types
    from myrag import ask as ask_mod

    monkeypatch.setattr("myrag.llm.get_llm", lambda *a: FakeLLM())
    cfg = types.SimpleNamespace(db_path=tmp_path / "fehlt", llm="claude", model=None, ollama_host=None)
    assert ask_mod.run("x", cfg) == 1
    assert "myrag ingest" in capsys.readouterr().err
    assert not (tmp_path / "fehlt").exists()


def test_run_empty_index_skips_embedder(tmp_path, monkeypatch, capsys):
    import types
    from myrag import ask as ask_mod, store

    store.get_collection(tmp_path / "db")

    def boom(*a):
        raise AssertionError("Embedder darf nicht geladen werden")

    monkeypatch.setattr("myrag.embeddings.get_embedder", boom)
    monkeypatch.setattr("myrag.llm.get_llm", lambda *a: FakeLLM())
    cfg = types.SimpleNamespace(db_path=tmp_path / "db", llm="claude", model=None, ollama_host=None)
    assert ask_mod.run("x", cfg) == 1
    assert "leer" in capsys.readouterr().err


def test_missing_api_key(monkeypatch, tmp_path):
    from myrag.llm import LLMError

    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        ClaudeLLM()


def test_claude_auth_error_is_readable():
    import anthropic
    from myrag.llm import LLMError

    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    err = anthropic.AuthenticationError("bad key", response=httpx.Response(401, request=req), body=None)

    class Messages:
        def stream(self, **kw):
            raise err

    class Client:
        class beta:
            messages = Messages()

    with pytest.raises(LLMError, match="ungültig"):
        ClaudeLLM(client=Client()).generate("s", "p")
