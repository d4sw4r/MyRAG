import pytest

from myrag.cli import build_parser, main


def test_ingest_parses_folder(tmp_path):
    args = build_parser().parse_args(["ingest", str(tmp_path)])
    assert args.command == "ingest"


def test_ask_parses_question():
    args = build_parser().parse_args(["ask", "Was steht drin?", "-k", "3"])
    assert args.question == "Was steht drin?" and args.top_k == 3


def test_ingest_rejects_missing_folder(tmp_path):
    assert main(["ingest", str(tmp_path / "gibtsnicht")]) == 1


def test_no_command_fails():
    with pytest.raises(SystemExit):
        main([])


def test_top_k_must_be_positive():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["ask", "x", "-k", "0"])


def test_status_and_reset(tmp_path, monkeypatch, capsys):
    from myrag import store

    db = tmp_path / "db"
    monkeypatch.setenv("MYRAG_DB_PATH", str(db))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    assert main(["status"]) == 0
    assert "noch leer" in capsys.readouterr().out

    col = store.get_collection(db)
    store.add_chunks(col, "a.md", "h", ["x", "y"], [[1.0, 0.0], [0.0, 1.0]], root="/docs")
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert "1 Dateien, 2 Chunks" in out and "/docs" in out and "gefunden" in out

    assert main(["reset", "--yes"]) == 0
    assert not db.exists()
    assert main(["reset", "--yes"]) == 0
