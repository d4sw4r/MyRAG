"""Central configuration via environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # ASSUMPTION: the index lives in ./.myrag/chroma (current directory) by default.
    db_path: Path
    llm: str  # "claude" | "ollama"
    model: str | None
    ollama_host: str

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            db_path=Path(os.environ.get("MYRAG_DB_PATH", ".myrag/chroma")).expanduser(),
            llm=os.environ.get("MYRAG_LLM", "claude"),
            model=os.environ.get("MYRAG_MODEL") or None,
            ollama_host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        )
