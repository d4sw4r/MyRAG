import hashlib
from pathlib import Path

import docx
import pytest

from myrag import store
from myrag.chunking import chunk_text
from myrag.ingest import ingest_folder
from myrag.loaders import iter_files, load_text


class FakeEmbedder:
    model_name = "fake"

    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text):
        d = hashlib.sha256(text.encode()).digest()
        return [b / 255 for b in d[:8]]


def test_chunk_overlap_and_size():
    text = " ".join(f"Wort{i}." for i in range(500))
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    for a, b in zip(chunks, chunks[1:]):
        assert b.split()[0] in a  # Überlappung vorhanden


def test_chunk_empty_and_short():
    assert chunk_text("   ") == []
    assert chunk_text("kurz") == ["kurz"]


def test_chunk_invalid_params():
    with pytest.raises(ValueError):
        chunk_text("x", chunk_size=10, overlap=10)


def test_loaders(tmp_path: Path):
    (tmp_path / "a.txt").write_text("Hallo Welt", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.md").write_text("# Titel\nInhalt", encoding="utf-8")
    d = docx.Document()
    d.add_paragraph("Word-Inhalt")
    d.save(tmp_path / "sub" / "c.docx")
    (tmp_path / "ignore.bin").write_bytes(b"\x00")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "x.txt").write_text("geheim")
    (tmp_path / "latin.txt").write_bytes("Größe".encode("cp1252"))

    files = [p.relative_to(tmp_path).as_posix() for p in iter_files(tmp_path)]
    assert files == ["a.txt", "latin.txt", "sub/b.md", "sub/c.docx"]
    assert "Word-Inhalt" in load_text(tmp_path / "sub" / "c.docx")
    assert load_text(tmp_path / "latin.txt") == "Größe"


def test_pdf_loader(tmp_path: Path):
    pdf = pytest.importorskip("pypdf")
    w = pdf.PdfWriter()
    w.add_blank_page(width=100, height=100)
    w.write(tmp_path / "leer.pdf")
    assert load_text(tmp_path / "leer.pdf").strip() == ""


def test_incremental_ingest(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("Erster Text. " * 200)
    (docs / "b.md").write_text("Zweiter Text")
    col = store.get_collection(tmp_path / "db")
    emb = FakeEmbedder()

    r1 = ingest_folder(docs, col, emb, chunk_size=300, overlap=50)
    assert sorted(r1.added) == ["a.txt", "b.md"]
    total = col.count()
    assert total == r1.chunks_written > 2
    meta = col.get(where={"source": "b.md"}, include=["metadatas"])["metadatas"][0]
    assert meta["chunk_index"] == 0 and len(meta["file_hash"]) == 64

    # zweiter Lauf: nichts geändert -> keine Embeddings
    calls = emb.calls
    r2 = ingest_folder(docs, col, emb, chunk_size=300, overlap=50)
    assert sorted(r2.unchanged) == ["a.txt", "b.md"] and emb.calls == calls
    assert col.count() == total

    # Datei ändern (kürzer), neue hinzufügen, eine löschen
    (docs / "a.txt").write_text("Jetzt kurz")
    (docs / "c.txt").write_text("Neu")
    (docs / "b.md").unlink()
    r3 = ingest_folder(docs, col, emb, chunk_size=300, overlap=50)
    assert r3.updated == ["a.txt"] and r3.added == ["c.txt"] and r3.removed == ["b.md"]
    got = col.get(include=["metadatas", "documents"])
    assert sorted(m["source"] for m in got["metadatas"]) == ["a.txt", "c.txt"]
    assert "Jetzt kurz" in got["documents"]

    # Persistenz: neue Collection-Instanz sieht dieselben Daten
    assert store.get_collection(tmp_path / "db").count() == 2


def test_broken_file_does_not_abort(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "kaputt.pdf").write_bytes(b"kein pdf")
    (docs / "ok.txt").write_text("ok")
    col = store.get_collection(tmp_path / "db")
    r = ingest_folder(docs, col, FakeEmbedder())
    assert "kaputt.pdf" in r.failed and r.added == ["ok.txt"]
