"""Hybrid retrieval: dense (Qdrant) + keyword (BM25), fused with Reciprocal Rank Fusion.

Why both: embeddings understand meaning ("who confirms delivery?") but are bad
at exact identifiers (`/packages/{packageId}/match`, `exception_case`,
`Idempotency-Key`). BM25 is the opposite. RRF combines the two rankings
without needing their scores to be on the same scale.
"""

import re
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from .models import Chunk, RetrievedChunk

STOPWORDS = set("""
a an and are as at be but by can do does for from has have how i if in into is it its of on or so
that the their them then there these they this to was what when where which who why will with you
your we our us my me should would could about any all also been being than via per
""".split())

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. Keeps identifiers whole AND split:
    'packageId' -> packageid, package, id ; 'exception_case' -> exception_case, exception, case
    """
    out: list[str] = []
    for w in _WORD_RE.findall(text):
        lw = w.lower()
        parts: list[str] = []
        for piece in w.split("_"):
            parts.extend(p.lower() for p in _CAMEL_RE.findall(piece))
        if len(parts) > 1 or lw not in parts:
            if lw not in STOPWORDS:
                out.append(lw)
        out.extend(p for p in parts if p not in STOPWORDS and len(p) > 1)
    return out


def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion over lists of chunk_ids (best first)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores


def _allowed(c: Chunk, doc_types: list[str] | None, roles: list[str] | None) -> bool:
    if doc_types and c.doc_type not in doc_types:
        return False
    if roles and not set(roles) & set(c.metadata.get("roles", [])):
        return False
    return True


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        corpus = [tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(corpus) if chunks else None

    def search(self, query: str, limit: int, doc_types: list[str] | None = None,
               roles: list[str] | None = None) -> list[tuple[Chunk, float]]:
        if not self.bm25:
            return []
        q = tokenize(query)
        if not q:
            return []
        scores = self.bm25.get_scores(q)
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in ranked:
            if scores[i] <= 0:
                break
            c = self.chunks[i]
            if _allowed(c, doc_types, roles):
                out.append((c, float(scores[i])))
                if len(out) >= limit:
                    break
        return out


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    best_dense_score: float | None = None
    mode: str = "hybrid"
    timings_ms: dict = field(default_factory=dict)


class Retriever:
    """Holds the BM25 index in memory and queries the vector store.

    store:    VectorStore (or anything with .search(vector, limit, doc_types, roles) and .all_chunks())
    embedder: OllamaEmbedder (or anything with .embed_query(text))
    """

    def __init__(self, store, embedder, candidate_k: int = 25, rrf_k: int = 60):
        self.store = store
        self.embedder = embedder
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.refresh()

    def refresh(self) -> None:
        self.chunks = self.store.all_chunks()
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.bm25 = BM25Index(self.chunks)

    def retrieve(self, question: str, top_k: int = 8, doc_types: list[str] | None = None,
                 roles: list[str] | None = None, mode: str = "hybrid") -> RetrievalResult:
        import time

        t0 = time.perf_counter()
        dense: list[tuple[Chunk, float]] = []
        sparse: list[tuple[Chunk, float]] = []
        timings = {}
        if mode in ("hybrid", "dense"):
            vec = self.embedder.embed_query(question)
            timings["embed"] = (time.perf_counter() - t0) * 1000
            dense = self.store.search(vec, self.candidate_k, doc_types, roles)
            timings["dense"] = (time.perf_counter() - t0) * 1000 - timings["embed"]
        if mode in ("hybrid", "bm25"):
            t1 = time.perf_counter()
            sparse = self.bm25.search(question, self.candidate_k, doc_types, roles)
            timings["bm25"] = (time.perf_counter() - t1) * 1000

        dense_rank = {c.chunk_id: i for i, (c, _) in enumerate(dense, 1)}
        dense_score = {c.chunk_id: s for c, s in dense}
        bm25_rank = {c.chunk_id: i for i, (c, _) in enumerate(sparse, 1)}

        if mode == "dense":
            order = [(c.chunk_id, s) for c, s in dense]
        elif mode == "bm25":
            order = [(c.chunk_id, s) for c, s in sparse]
        else:
            fused = rrf([[c.chunk_id for c, _ in dense], [c.chunk_id for c, _ in sparse]], k=self.rrf_k)
            order = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

        lookup = {c.chunk_id: c for c, _ in dense} | {c.chunk_id: c for c, _ in sparse}
        results = [
            RetrievedChunk(chunk=lookup[cid], score=score, dense_rank=dense_rank.get(cid),
                           bm25_rank=bm25_rank.get(cid), dense_score=dense_score.get(cid))
            for cid, score in order[:top_k]
        ]
        best = max(dense_score.values()) if dense_score else None
        timings["total"] = (time.perf_counter() - t0) * 1000
        return RetrievalResult(chunks=results, best_dense_score=best, mode=mode, timings_ms=timings)
