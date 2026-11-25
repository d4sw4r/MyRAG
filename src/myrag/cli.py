"""Kommandozeile: `myrag ingest`, `myrag ask`, `myrag status`, `myrag reset`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import Config


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("muss mindestens 1 sein")
    return n


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="myrag", description="Eigenes RAG: Ordner einlesen und mit KI befragen.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ordner einlesen und indexieren")
    p_ingest.add_argument("folder", type=Path, help="Ordner, dessen Dateien aufgenommen werden")

    p_ask = sub.add_parser("ask", help="Frage an die eingelesenen Dateien stellen")
    p_ask.add_argument("question", help="Die Frage in Anführungszeichen")
    p_ask.add_argument("-k", "--top-k", type=_positive_int, default=5, help="Anzahl der Fundstellen (Standard: 5)")

    sub.add_parser("status", help="Zeigt, was im Index liegt und welches Modell antwortet")

    p_reset = sub.add_parser("reset", help="Löscht den kompletten Index")
    p_reset.add_argument("-y", "--yes", action="store_true", help="Ohne Rückfrage löschen")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = Config.from_env()

    if args.command == "ingest":
        folder = args.folder.expanduser().resolve()
        if not folder.is_dir():
            print(f"Fehler: {folder} ist kein Ordner.", file=sys.stderr)
            return 1
        from .ingest import run
        return run(folder, config)

    if args.command == "ask":
        from .ask import run
        return run(args.question, config, top_k=args.top_k)

    if args.command == "status":
        from .status import run
        return run(config)

    if args.command == "reset":
        return _reset(config, args.yes)

    return 1


def _reset(config: Config, yes: bool) -> int:
    from . import store

    if not store.index_exists(config.db_path):
        print(f"Kein Index unter {config.db_path}, nichts zu tun.")
        return 0
    if not yes:
        answer = input(f"Index unter {config.db_path} wirklich löschen? [j/N] ").strip().lower()
        if answer not in {"j", "ja", "y", "yes"}:
            print("Abgebrochen.")
            return 1
    store.reset(config.db_path)
    print(f"Index unter {config.db_path} gelöscht.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
