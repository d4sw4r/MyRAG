"""`myrag search`: show the best matching passages without asking a language model.

Fast, free and handy for checking what the index finds for a question.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import PurePath

from .retrieval import Hit

SNIPPET_CHARS = 300


def format_hit(i: int, hit: Hit, snippet_chars: int = SNIPPET_CHARS) -> str:
    path = str(PurePath(hit.root) / hit.source) if hit.root else hit.source
    section = "" if hit.chunk_index is None else f" (section {hit.chunk_index})"
    scores = []
    if hit.score is not None:
        scores.append(f"relevance {hit.score:.2f}")
    if hit.similarity is not None:
        scores.append(f"similarity {hit.similarity:.2f}")
    if hit.keyword_score is not None:
        scores.append(f"keywords {hit.keyword_score:.1f}")
    snippet = " ".join(hit.text.split())
    if len(snippet) > snippet_chars:
        snippet = snippet[:snippet_chars].rsplit(" ", 1)[0] + " ..."
    body = textwrap.indent(textwrap.fill(snippet, width=88), "   ")
    return f"{i}. {path}{section}  [{', '.join(scores) or 'no score'}]\n{body}"


def format_hits(hits: list[Hit]) -> str:
    return "\n\n".join(format_hit(i, h) for i, h in enumerate(hits, 1))


def run(query: str, config, top_k: int = 5) -> int:
    from . import retrieval
    from .ask import NOTHING_FOUND, open_index

    try:
        collection = open_index(config)
        hits = retrieval.from_config(config, collection).search(query, k=top_k)
    except (RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(format_hits(hits) if hits else NOTHING_FOUND)
    return 0
