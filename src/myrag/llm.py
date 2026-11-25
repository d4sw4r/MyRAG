"""Language model backends for `myrag ask`: Claude API and Ollama, switchable."""

from __future__ import annotations

import os
from typing import Protocol

import httpx

DEFAULT_BACKEND = "claude"
DEFAULT_MODELS = {
    "claude": "claude-opus-5-5",
    "ollama": "llama3.1",
}


class LLMError(RuntimeError):
    """Readable error message for the command line."""


def _has_anthropic_credentials() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    if os.environ.get("ANTHROPIC_PROFILE"):
        return True
    config_home = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.isdir(os.path.join(config_home, "anthropic"))


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
            if not _has_anthropic_credentials():
                raise LLMError(
                    "No Claude API key found. Set ANTHROPIC_API_KEY "
                    "(get a key at https://platform.claude.com/settings/keys) "
                    "or use a local model with MYRAG_LLM=ollama."
                )
            import anthropic

            client = anthropic.Anthropic()
        self.client = client

    def generate(self, system: str, prompt: str) -> str:
        import anthropic

        try:
            return self._generate(system, prompt)
        except anthropic.AuthenticationError as e:
            raise LLMError("The Claude API key is invalid. Check ANTHROPIC_API_KEY.") from e
        except anthropic.PermissionDeniedError as e:
            raise LLMError(f"No access to model {self.model}: {e.message}") from e
        except anthropic.NotFoundError as e:
            raise LLMError(f"Model {self.model} not found. Check MYRAG_MODEL.") from e
        except anthropic.RateLimitError as e:
            raise LLMError("Claude API rate limit reached. Please wait a moment and ask again.") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"Claude API error ({e.status_code}): {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError("Claude API unreachable. Is there an internet connection?") from e

    def _generate(self, system: str, prompt: str) -> str:
        with self.client.beta.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
            # If the model refuses, a fallback model takes over automatically.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            message = stream.get_final_message()
        if message.stop_reason == "refusal":
            raise LLMError("The model refused the request.")
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
            raise LLMError(
                f"Ollama is not reachable at {self.host}. Is `ollama serve` running?"
            ) from e
        if resp.status_code == 404:
            raise LLMError(
                f"Ollama model '{self.model}' not found. Download it with `ollama pull {self.model}`."
            )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()


def get_llm(backend: str | None = None, model: str | None = None, ollama_host: str | None = None) -> LLM:
    """Select the backend: argument > MYRAG_LLM > default (Claude)."""
    backend = (backend or os.environ.get("MYRAG_LLM") or DEFAULT_BACKEND).lower()
    model = model or os.environ.get("MYRAG_MODEL") or None
    if backend == "claude":
        return ClaudeLLM(model=model)
    if backend == "ollama":
        return OllamaLLM(model=model, host=ollama_host)
    raise ValueError(f"Unknown LLM backend '{backend}'. Allowed: claude, ollama")
