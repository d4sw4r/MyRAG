"""Sprachmodell-Backends für `myrag ask`: Claude-API und Ollama, umschaltbar."""

from __future__ import annotations

import os
from typing import Protocol

import httpx

DEFAULT_BACKEND = "claude"
DEFAULT_MODELS = {
    "claude": "claude-opus-5-5",
    "ollama": "llama3.1",
}


class LLM(Protocol):
    name: str
    model: str

    def generate(self, system: str, prompt: str) -> str: ...


class ClaudeLLM:
    name = "claude"

    def __init__(self, model: str | None = None, client=None, max_tokens: int = 64000):
        self.model = model or DEFAULT_MODELS["claude"]
        self.max_tokens = max_tokens
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client

    def generate(self, system: str, prompt: str) -> str:
        with self.client.beta.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
            # Bei einer Ablehnung des Modells übernimmt automatisch ein Ersatzmodell.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            message = stream.get_final_message()
        if message.stop_reason == "refusal":
            raise RuntimeError("Das Modell hat die Anfrage abgelehnt.")
        return "".join(b.text for b in message.content if b.type == "text").strip()


class OllamaLLM:
    name = "ollama"

    def __init__(self, model: str | None = None, host: str | None = None, client: httpx.Client | None = None):
        self.model = model or DEFAULT_MODELS["ollama"]
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.client = client or httpx.Client(timeout=600)

    def generate(self, system: str, prompt: str) -> str:
        try:
            resp = self.client.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Ollama ist unter {self.host} nicht erreichbar. Läuft `ollama serve`?"
            ) from e
        if resp.status_code == 404:
            raise RuntimeError(
                f"Ollama-Modell '{self.model}' nicht gefunden. Lade es mit `ollama pull {self.model}`."
            )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()


def get_llm(backend: str | None = None, model: str | None = None, ollama_host: str | None = None) -> LLM:
    """Wählt das Backend: Argument > MYRAG_LLM > Standard (Claude)."""
    backend = (backend or os.environ.get("MYRAG_LLM") or DEFAULT_BACKEND).lower()
    model = model or os.environ.get("MYRAG_MODEL") or None
    if backend == "claude":
        return ClaudeLLM(model=model)
    if backend == "ollama":
        return OllamaLLM(model=model, host=ollama_host)
    raise ValueError(f"Unbekanntes LLM-Backend '{backend}'. Erlaubt: claude, ollama")
