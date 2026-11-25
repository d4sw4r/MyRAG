"""Command line: `myrag ingest`, `ask`, `search`, `chat`, `status`, `reset`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import Config


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return n


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="myrag", description="Personal RAG: ingest a folder and query it with AI.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest and index a folder")
    p_ingest.add_argument("folder", type=Path, help="Folder whose files are added to the index")
    p_ingest.add_argument("--rebuild", action="store_true",
                          help="Delete the index and re-ingest all its folders (e.g. after changing the embedding model)")

    p_ask = sub.add_parser("ask", help="Ask a question about the ingested files")
    p_ask.add_argument("question", help="The question, in quotes")
    p_ask.add_argument("-k", "--top-k", type=_positive_int, default=5, help="Number of retrieved passages (default: 5)")

    p_search = sub.add_parser("search", help="Show the best matching passages, without asking an AI")
    p_search.add_argument("query", help="Search text, in quotes")
    p_search.add_argument("-k", "--top-k", type=_positive_int, default=5, help="Number of passages (default: 5)")

    p_chat = sub.add_parser("chat", help="Interactive chat with follow-up questions")
    p_chat.add_argument("-k", "--top-k", type=_positive_int, default=5, help="Passages per question (default: 5)")

    sub.add_parser("status", help="Show what is in the index and which model answers")

    p_reset = sub.add_parser("reset", help="Delete the entire index")
    p_reset.add_argument("-y", "--yes", action="store_true", help="Delete without confirmation")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = Config.from_env()

    if args.command == "ingest":
        folder = args.folder.expanduser().resolve()
        if not folder.is_dir():
            print(f"Error: {folder} is not a folder.", file=sys.stderr)
            return 1
        from .ingest import run
        return run(folder, config, rebuild=args.rebuild)

    if args.command == "ask":
        from .ask import run
        return run(args.question, config, top_k=args.top_k)

    if args.command == "search":
        from .search import run
        return run(args.query, config, top_k=args.top_k)

    if args.command == "chat":
        from .chat import run
        return run(config, top_k=args.top_k)

    if args.command == "status":
        from .status import run
        return run(config)

    if args.command == "reset":
        return _reset(config, args.yes)

    return 1


def _reset(config: Config, yes: bool) -> int:
    from . import store

    if not store.index_exists(config.db_path):
        print(f"No index at {config.db_path}, nothing to do.")
        return 0
    if not yes:
        answer = input(f"Really delete the index at {config.db_path}? [y/N] ").strip().lower()
        if answer not in {"j", "ja", "y", "yes"}:
            print("Aborted.")
            return 1
    store.reset(config.db_path)
    print(f"Deleted the index at {config.db_path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
