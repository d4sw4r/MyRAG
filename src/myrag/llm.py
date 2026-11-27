"""Language model backends for `myrag ask`: Claude API and Ollama, switchable."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Iterator, Protocol

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


Message = dict  # {"role": "user" | "assistant", "content": str}


class LLM(Protocol):
    name: str
    model: str

    def generate(self, system: str, prompt: str) -> str: ...

    def chat(self, system: str, messages: list[Message]) -> str: ...

    def stream(self, system: str, messages: list[Message]) -> Iterator[str]: ...


@contextmanager
def _claude_errors(model: str):
    """Translate Anthropic SDK errors into readable LLMErrors."""
    import anthropic

    try:
        yield
    except anthropic.AuthenticationError as e:
        raise LLMError("The Claude API key is invalid. Check ANTHROPIC_API_KEY.") from e
    except anthropic.PermissionDeniedError as e:
        raise LLMError(f"No access to the model {model}: {e.message}") from e
    except anthropic.NotFoundError as e:
        raise LLMError(f"Model {model} not found. Check MYRAG_MODEL.") from e
    except anthropic.RateLimitError as e:
        raise LLMError("Claude API rate limit reached. Please wait a moment and ask again.") from e
    except anthropic.APIStatusError as e:
        raise LLMError(f"Claude API error ({e.status_code}): {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise LLMError("Claude API is not reachable. Is there an internet connection?") from e


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

    def _open_stream(self, system: str, messages: list[Message]):
        return self.client.beta.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
            # If the model declines, a fallback model takes over automatically.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )

    @staticmethod
    def _check(message) -> None:
        if message.stop_reason == "refusal":
            raise LLMError("The model declined the request.")

    def generate(self, system: str, prompt: str) -> str:
        return self.chat(system, [{"role": "user", "content": prompt}])

    def chat(self, system: str, messages: list[Message]) -> str:
        with _claude_errors(self.model):
            with self._open_stream(system, messages) as stream:
                message = stream.get_final_message()
        self._check(message)
        return "".join(b.text for b in message.content if b.type == "text").strip()

    def stream(self, system: str, messages: list[Message]) -> Iterator[str]:
        """Yield the answer text piece by piece as it is generated."""
        with _claude_errors(self.model):
            with self._open_stream(system, messages) as stream:
                yield from stream.text_stream
                message = stream.get_final_message()
        self._check(message)


class OllamaLLM:
    name = "ollama"

    def __init__(self, model: str | None = None, host: str | None = None, client: httpx.Client | None = None):
        self.model = model or DEFAULT_MODELS["ollama"]
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.client = client or httpx.Client(timeout=600)

    def _payload(self, system: str, messages: list[Message], stream: bool) -> dict:
        return {
            "model": self.model,
            "stream": stream,
            "messages": [{"role": "system", "content": system}, *messages],
        }

    def _not_reachable(self, e: Exception) -> LLMError:
        return LLMError(f"Ollama is not reachable at {self.host}. Is `ollama serve` running?")

    def _check_status(self, resp: httpx.Response) -> None:
        if resp.status_code == 404:
            raise LLMError(
                f"Ollama model '{self.model}' not found. Download it with `ollama pull {self.model}`."
            )
        resp.raise_for_status()

    def generate(self, system: str, prompt: str) -> str:
        return self.chat(system, [{"role": "user", "content": prompt}])

    def chat(self, system: str, messages: list[Message]) -> str:
        try:
            resp = self.client.post(f"{self.host}/api/chat", json=self._payload(system, messages, False))
        except httpx.ConnectError as e:
            raise self._not_reachable(e) from e
        self._check_status(resp)
        return resp.json()["message"]["content"].strip()

    def stream(self, system: str, messages: list[Message]) -> Iterator[str]:
        try:
            with self.client.stream(
                "POST", f"{self.host}/api/chat", json=self._payload(system, messages, True)
            ) as resp:
                if resp.status_code >= 400:
                    resp.read()
                self._check_status(resp)
                for line in resp.iter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    piece = (data.get("message") or {}).get("content")
                    if piece:
                        yield piece
                    if data.get("done"):
                        return
        except httpx.ConnectError as e:
            raise self._not_reachable(e) from e


def get_llm(backend: str | None = None, model: str | None = None, ollama_host: str | None = None) -> LLM:
    """Select the backend: argument > MYRAG_LLM > default (Claude)."""
    backend = (backend or os.environ.get("MYRAG_LLM") or DEFAULT_BACKEND).lower()
    model = model or os.environ.get("MYRAG_MODEL") or None
    if backend == "claude":
        return ClaudeLLM(model=model)
    if backend == "ollama":
        return OllamaLLM(model=model, host=ollama_host)
    raise ValueError(f"Unknown LLM backend '{backend}'. Allowed: claude, ollama")
