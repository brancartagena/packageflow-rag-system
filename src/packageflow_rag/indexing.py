"""Ingest helpers: docs -> chunks.jsonl -> embeddings -> vector store.

Incremental by default (Phase 6): only chunks whose text changed are re-embedded,
chunks that disappeared are deleted. A full rebuild happens when asked
(--rebuild), when there is no index yet, or when EMBED_MODEL changed.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .loaders import load_corpus
from .models import Chunk
from .vectorstore import VectorStore


@dataclass
class IndexStats:
    mode: str           # "rebuild" | "incremental"
    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0
    total: int = 0


def build_chunks(cfg: Settings) -> tuple[list[Chunk], list[str]]:
    return load_corpus(cfg.resolve(cfg.source_dir), cfg.resolve(cfg.derived_dir),
                       cfg.chunk_max_tokens, cfg.chunk_overlap_tokens)


def write_chunks_jsonl(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


def read_chunks_jsonl(path: Path) -> list[Chunk]:
    with open(path, encoding="utf-8") as f:
        return [Chunk.from_dict(json.loads(line)) for line in f if line.strip()]


def index_chunks(store: VectorStore, embedder, chunks: list[Chunk], rebuild: bool = False,
                 progress: Callable[[int, int], None] | None = None) -> IndexStats:
    model = embedder.model
    meta = store.meta() if store.exists() else None
    need_rebuild = rebuild or meta is None or meta.get("embed_model") != model

    if need_rebuild:
        vectors = embedder.embed_documents([c.text for c in chunks], progress=progress)
        dim = len(vectors[0]) if vectors else embedder.dim
        store.recreate(dim, model)
        store.upsert(chunks, vectors)
        return IndexStats("rebuild", added=len(chunks), total=store.count())

    existing = store.existing_hashes()
    new_ids = {c.chunk_id for c in chunks}
    to_embed = [c for c in chunks if existing.get(c.chunk_id) != c.content_hash]
    removed = [cid for cid in existing if cid not in new_ids]
    if to_embed:
        vectors = embedder.embed_documents([c.text for c in to_embed], progress=progress)
        store.upsert(to_embed, vectors)
    store.delete(removed)
    added = sum(1 for c in to_embed if c.chunk_id not in existing)
    return IndexStats("incremental", added=added, updated=len(to_embed) - added, removed=len(removed),
                      unchanged=len(chunks) - len(to_embed), total=store.count())
