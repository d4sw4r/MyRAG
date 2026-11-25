"""Kommandozeile: `myrag ingest <ordner>` und `myrag ask "<frage>"`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import Config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="myrag", description="Eigenes RAG: Ordner einlesen und mit KI befragen.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ordner einlesen und indexieren")
    p_ingest.add_argument("folder", type=Path, help="Ordner, dessen Dateien aufgenommen werden")

    p_ask = sub.add_parser("ask", help="Frage an die eingelesenen Dateien stellen")
    p_ask.add_argument("question", help="Die Frage in Anführungszeichen")
    p_ask.add_argument("-k", "--top-k", type=int, default=5, help="Anzahl der Fundstellen (Standard: 5)")
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

    return 1


if __name__ == "__main__":
    sys.exit(main())
