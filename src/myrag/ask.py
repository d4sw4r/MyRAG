"""Frage an den Index stellen und mit einem LLM beantworten.

Platzhalter: Retrieval, Prompt und LLM-Anbindung (Claude/Ollama) folgen
in einem eigenen Schritt.
"""

from __future__ import annotations

from .config import Config


def run(question: str, config: Config, top_k: int = 5) -> int:
    print(f"[myrag] ask ist noch nicht implementiert (Frage: {question!r}, LLM: {config.llm}).")
    return 2
