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
