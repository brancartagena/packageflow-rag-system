"""Embeddings via Ollama (/api/embed), batched.

Some models need prefixes to work well:
- nomic-embed-text: 'search_document: ' for chunks, 'search_query: ' for questions
- qwen3-embedding:  an instruction prefix on queries only
- embeddinggemma:   'title: none | text: ' for chunks, 'task: search result | query: ' for questions
- bge-m3, mxbai, arctic, minilm: no prefix needed (mxbai has an optional query prompt)
"""

from collections.abc import Callable, Sequence

import ollama

QUERY_PREFIX = {
    "nomic-embed-text": "search_query: ",
    "qwen3-embedding": "Instruct: Given a question about the PackageFlow project, retrieve documentation passages that answer it\nQuery: ",
    "embeddinggemma": "task: search result | query: ",
    "mxbai-embed-large": "Represent this sentence for searching relevant passages: ",
}
DOC_PREFIX = {
    "nomic-embed-text": "search_document: ",
    "embeddinggemma": "title: none | text: ",
}


def _base(model: str) -> str:
    return model.split(":")[0].split("/")[-1]


def query_prefix(model: str) -> str:
    return QUERY_PREFIX.get(_base(model), "")


def doc_prefix(model: str) -> str:
    return DOC_PREFIX.get(_base(model), "")


class OllamaError(RuntimeError):
    """Friendly wrapper for the usual Ollama problems (not running, model not pulled)."""


def explain_ollama_error(e: Exception, model: str, host: str) -> OllamaError:
    msg = str(e)
    if isinstance(e, ollama.ResponseError) and getattr(e, "status_code", None) == 404:
        return OllamaError(f"Model '{model}' is not pulled in Ollama. Run: ollama pull {model}")
    if isinstance(e, (ConnectionError, OSError)) or "connect" in msg.lower():
        return OllamaError(f"Can't reach Ollama at {host}. Is it running? Start the Ollama app or run: ollama serve")
    return OllamaError(f"Ollama error with model '{model}': {msg}")


class OllamaEmbedder:
    def __init__(self, model: str, host: str = "http://localhost:11434", batch_size: int = 16,
                 num_ctx: int = 8192, timeout: float = 300.0):
        self.model = model
        self.host = host
        self.batch_size = batch_size
        self.num_ctx = num_ctx
        self.client = ollama.Client(host=host, timeout=timeout)
        self._dim: int | None = None

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            resp = self.client.embed(model=self.model, input=list(texts), truncate=True,
                                     options={"num_ctx": self.num_ctx})
        except Exception as e:
            raise explain_ollama_error(e, self.model, self.host) from e
        vecs = [list(v) for v in resp["embeddings"]]
        if vecs:
            self._dim = len(vecs[0])
        return vecs

    def embed_documents(self, texts: Sequence[str], progress: Callable[[int, int], None] | None = None) -> list[list[float]]:
        pre = doc_prefix(self.model)
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [pre + t for t in texts[i:i + self.batch_size]]
            out.extend(self._embed(batch))
            if progress:
                progress(min(i + self.batch_size, len(texts)), len(texts))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._embed([query_prefix(self.model) + text])[0]

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self._embed(["dimension probe"])[0])
        return self._dim


# Back-compat helpers matching the original stubs
def embed_documents(texts: list[str], model: str, batch_size: int = 32) -> list[list[float]]:
    return OllamaEmbedder(model, batch_size=batch_size).embed_documents(texts)


def embed_query(text: str, model: str) -> list[float]:
    return OllamaEmbedder(model).embed_query(text)
