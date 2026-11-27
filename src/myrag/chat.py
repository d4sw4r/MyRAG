"""`myrag chat`: interactive questions with conversation history and streamed answers.

Follow-up questions ("and when was that?") are first rewritten into a standalone
search query using the conversation, so retrieval finds the right passages.
"""

from __future__ import annotations

import sys
from typing import Callable

from .ask import NOTHING_FOUND, SYSTEM_PROMPT, build_prompt, format_sources, stream_answer
from .llm import LLM
from .retrieval import Hit, Retriever

MAX_TURNS = 6  # earlier turns are dropped to keep the prompt small

CHAT_SYSTEM_PROMPT = SYSTEM_PROMPT + (
    " This is a conversation: earlier questions and answers are context, but cite only "
    "the excerpts given with the latest question."
)

CONDENSE_PROMPT = (
    "Rewrite the user's latest question as a standalone search query, using the "
    "conversation for context (resolve pronouns and references). Keep the language "
    "of the question. Reply with the query only, nothing else."
)

HELP = (
    "Type a question and press Enter. Commands: /sources (sources of the last answer), "
    "/clear (forget the conversation), /help, /exit"
)


class ChatSession:
    def __init__(self, llm: LLM, retriever: Retriever, k: int = 5, out=None):
        self.llm = llm
        self.retriever = retriever
        self.k = k
        self.out = out or sys.stdout
        self.history: list[tuple[str, str]] = []  # (question, answer)
        self.last_hits: list[Hit] = []

    def _history_messages(self) -> list[dict]:
        msgs: list[dict] = []
        for q, a in self.history[-MAX_TURNS:]:
            msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        return msgs

    def search_query(self, question: str) -> str:
        """Standalone query for retrieval; the question itself if there is no history."""
        if not self.history:
            return question
        transcript = "\n".join(
            f"User: {q}\nAssistant: {a}" for q, a in self.history[-MAX_TURNS:]
        )
        prompt = f"Conversation:\n{transcript}\n\nLatest question: {question}"
        try:
            chat = getattr(self.llm, "chat", None)
            messages = [{"role": "user", "content": prompt}]
            query = chat(CONDENSE_PROMPT, messages) if chat else self.llm.generate(CONDENSE_PROMPT, prompt)
        except Exception:
            return question  # retrieval with the raw question is still useful
        query = (query or "").strip().splitlines()[0] if (query or "").strip() else ""
        return query or question

    def ask(self, question: str) -> str:
        query = self.search_query(question)
        hits = self.retriever.search(query, k=self.k)
        self.last_hits = hits
        if not hits:
            print(NOTHING_FOUND, file=self.out)
            return NOTHING_FOUND
        messages = self._history_messages() + [{"role": "user", "content": build_prompt(question, hits)}]
        answer = stream_answer(self.llm, CHAT_SYSTEM_PROMPT, messages, self.out)
        self.history.append((question, answer))
        return answer

    def clear(self) -> None:
        self.history.clear()
        self.last_hits = []


def loop(session: ChatSession, input_fn: Callable[[str], str] = input) -> int:
    out = session.out
    print(f"MyRAG chat. {HELP}", file=out)
    while True:
        try:
            line = input_fn("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=out)
            return 0
        if not line:
            continue
        cmd = line.lower()
        if cmd in {"/exit", "/quit", "/q", "exit", "quit"}:
            return 0
        if cmd == "/help":
            print(HELP, file=out)
            continue
        if cmd == "/clear":
            session.clear()
            print("Conversation cleared.", file=out)
            continue
        if cmd == "/sources":
            print(format_sources(session.last_hits) if session.last_hits else "No sources yet.", file=out)
            continue
        print(file=out)
        try:
            session.ask(line)
        except KeyboardInterrupt:
            print("\n(interrupted)", file=out)
        except (RuntimeError, ValueError) as e:  # LLMError is a RuntimeError
            print(f"\nError: {e}", file=sys.stderr)


def run(config, top_k: int = 5) -> int:
    from . import retrieval
    from .ask import open_index
    from .llm import LLMError, get_llm

    try:
        llm = get_llm(config.llm, config.model, getattr(config, "ollama_host", None))
        collection = open_index(config)
        retriever = retrieval.from_config(config, collection)
    except (LLMError, RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return loop(ChatSession(llm, retriever, k=top_k))
