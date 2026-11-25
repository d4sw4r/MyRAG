"""Ordner einlesen und indexieren.

Platzhalter: Die eigentliche Umsetzung (Loader, Chunking, Hash-Cache,
Embeddings, Vektorstore) folgt in einem eigenen Schritt.
"""

from __future__ import annotations

from pathlib import Path

from .config import Config


def run(folder: Path, config: Config) -> int:
    print(f"[myrag] ingest ist noch nicht implementiert (Ordner: {folder}, Index: {config.db_path}).")
    return 2
