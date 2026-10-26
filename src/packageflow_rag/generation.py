"""Answer generation with the local Ollama chat model (streaming)."""

import re
from collections.abc import Iterator

import ollama

from .embeddings import explain_ollama_error
from .models import RetrievedChunk
from .prompts import build_messages

CITE_RE = re.compile(r"\[(\d{1,2})\]")


class OllamaGenerator:
    def __init__(self, model: str, host: str = "http://localhost:11434", num_ctx: int = 8192,
                 temperature: float = 0.1, timeout: float = 300.0, max_tokens: int | None = None,
                 keep_alive: str | None = None, think: bool = False):
        self.model = model
        self.host = host
        self.options = {"num_ctx": num_ctx, "temperature": temperature}
        self.keep_alive = keep_alive
        self.think = think
        if max_tokens:
            self.options["num_predict"] = max_tokens
        self.client = ollama.Client(host=host, timeout=timeout)

    def stream(self, messages: list[dict]) -> Iterator[str]:
        try:
            for part in self.client.chat(model=self.model, messages=messages, stream=True, options=self.options,
                                         keep_alive=self.keep_alive, think=self.think):
                text = part["message"].get("content") or ""
                if text:
                    yield text
        except Exception as e:
            raise explain_ollama_error(e, self.model, self.host) from e

    def complete(self, messages: list[dict]) -> str:
        return "".join(self.stream(messages))


def cited_numbers(answer: str, n_sources: int) -> list[int]:
    """Source numbers the answer actually cites, in order of first use."""
    seen: list[int] = []
    for m in CITE_RE.finditer(answer):
        n = int(m.group(1))
        if 1 <= n <= n_sources and n not in seen:
            seen.append(n)
    return seen


def answer(question: str, chunks: list[RetrievedChunk], stream: bool = True, generator: OllamaGenerator | None = None,
           budget_tokens: int | None = None) -> Iterator[str]:
    """Build the prompt and stream tokens from Ollama. (Most callers use rag.RAG instead.)"""
    if generator is None:
        from .config import settings
        generator = OllamaGenerator(settings.ollama_chat_model, settings.ollama_host, settings.ollama_num_ctx,
                                    settings.llm_temperature, settings.ollama_timeout)
    messages, _ = build_messages(question, chunks, budget_tokens)
    yield from generator.stream(messages)
