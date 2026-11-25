"""Split text into overlapping chunks."""

from __future__ import annotations

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into pieces of at most chunk_size characters.

    Consecutive chunks overlap by roughly overlap characters. Cuts are made
    preferably at paragraph, sentence or word boundaries.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            end = _best_break(text, start, end)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        next_start = max(end - overlap, start + 1)
        # start the overlap at a word boundary
        space = text.find(" ", next_start, end)
        if space != -1:
            next_start = space + 1
        start = next_start
    return chunks


def _best_break(text: str, start: int, end: int) -> int:
    """Find the best cut position in the back half of the window."""
    min_pos = start + (end - start) // 2
    for sep in ("\n\n", "\n", ". ", "! ", "? ", " "):
        pos = text.rfind(sep, min_pos, end)
        if pos != -1:
            return pos + len(sep)
    return end
