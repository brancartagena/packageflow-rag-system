"""The whole question -> answer pipeline in one place.

Used by scripts/ask.py, the Streamlit app and scripts/evaluate.py.

    rag = RAG()                       # opens the index, checks the embed model
    resp = rag.ask("Who can cancel a package?")
    for token in resp:                # streams the answer
        print(token, end="")
    resp.sources                      # chunks given to the model, numbered like the citations
    resp.cited                        # source numbers the answer actually cited
"""

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from .chunking import approx_tokens
from .config import Settings, settings as default_settings
from .embeddings import OllamaEmbedder
from .generation import OllamaGenerator, cited_numbers
from .models import RetrievedChunk
from .prompts import NOT_COVERED, SYSTEM_PROMPT, build_messages
from .retrieval import RetrievalResult, Retriever
from .vectorstore import VectorStore


@dataclass
class Response:
    question: str
    retrieval: RetrievalResult
    sources: list[RetrievedChunk] = field(default_factory=list)
    text: str = ""
    cited: list[int] = field(default_factory=list)
    gated: bool = False               # True when the min_dense_score gate answered "not covered"
    _stream: Iterator[str] | None = None
    _on_done: object = None

    def __iter__(self) -> Iterator[str]:
        if self._stream is None:
            if self.text:
                yield self.text
            return
        t0 = time.perf_counter()
        parts = []
        for tok in self._stream:
            parts.append(tok)
            yield tok
        self.text = "".join(parts)
        self.cited = cited_numbers(self.text, len(self.sources))
        self.retrieval.timings_ms["generate"] = (time.perf_counter() - t0) * 1000
        self._stream = None
        if callable(self._on_done):
            self._on_done(self)

    def consume(self) -> "Response":
        for _ in self:
            pass
        return self

    @property
    def not_covered(self) -> bool:
        return self.gated or NOT_COVERED.lower().rstrip(".") in self.text.lower()


class RAG:
    def __init__(self, cfg: Settings | None = None, store: VectorStore | None = None, embedder=None,
                 generator: OllamaGenerator | None = None, check_model: bool = True):
        self.cfg = cfg or default_settings
        c = self.cfg
        self.store = store or VectorStore(c.resolve(c.index_dir), c.collection_name)
        self.embedder = embedder or OllamaEmbedder(c.embed_model, c.ollama_host, c.embed_batch_size,
                                                   c.embed_num_ctx, c.ollama_timeout)
        if check_model:
            self.store.check_model(getattr(self.embedder, "model", c.embed_model))
        self.generator = generator or OllamaGenerator(c.ollama_chat_model, c.ollama_host, c.ollama_num_ctx,
                                                      c.llm_temperature, c.ollama_timeout)
        self.retriever = Retriever(self.store, self.embedder, c.candidate_k, c.rrf_k)

    @property
    def context_budget(self) -> int:
        return max(500, self.cfg.ollama_num_ctx - self.cfg.answer_reserve_tokens - approx_tokens(SYSTEM_PROMPT) - 150)

    def retrieve(self, question: str, top_k: int | None = None, doc_types: list[str] | None = None,
                 roles: list[str] | None = None, mode: str = "hybrid") -> RetrievalResult:
        return self.retriever.retrieve(question, top_k or self.cfg.top_k, doc_types, roles, mode)

    def ask(self, question: str, top_k: int | None = None, doc_types: list[str] | None = None,
            roles: list[str] | None = None, history: list[dict] | None = None, mode: str = "hybrid") -> Response:
        result = self.retrieve(question, top_k, doc_types, roles, mode)
        resp = Response(question=question, retrieval=result, _on_done=self._log)

        best = result.best_dense_score
        threshold = self.cfg.min_dense_score
        if not result.chunks or (threshold > 0 and best is not None and best < threshold):
            resp.gated = True
            resp.text = NOT_COVERED
            self._log(resp)
            return resp

        messages, used = build_messages(question, result.chunks, self.context_budget, history)
        resp.sources = used
        resp._stream = self.generator.stream(messages)
        return resp

    # ---------- logging (Phase 6) ----------
    def _log(self, resp: Response) -> None:
        try:
            log_dir = self.cfg.resolve(self.cfg.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            rec = {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "question": resp.question,
                "chat_model": self.cfg.ollama_chat_model,
                "embed_model": getattr(self.embedder, "model", self.cfg.embed_model),
                "mode": resp.retrieval.mode,
                "retrieved": [rc.chunk.chunk_id for rc in resp.retrieval.chunks],
                "used_sources": [rc.chunk.chunk_id for rc in resp.sources],
                "cited": resp.cited,
                "best_dense_score": resp.retrieval.best_dense_score,
                "gated": resp.gated,
                "not_covered": resp.not_covered,
                "answer_chars": len(resp.text),
                "timings_ms": {k: round(v, 1) for k, v in resp.retrieval.timings_ms.items()},
            }
            with open(log_dir / "queries.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass  # logging must never break answering

    def close(self) -> None:
        self.store.close()
