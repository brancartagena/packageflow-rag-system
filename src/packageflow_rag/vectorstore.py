"""Qdrant in local mode (QdrantClient(path=INDEX_DIR)). No server needed.

- Cosine distance.
- A sidecar file index_meta.json records which embedding model built the index.
  Querying with a different model raises IndexMismatchError instead of
  silently returning garbage.
- Payload keeps the full chunk, plus flat doc_type / roles fields for filters.
- Local mode allows ONE process at a time per folder. Close the Streamlit app
  before running ingest, or you get IndexLockedError.
"""

import json
import time
import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models

from .models import Chunk

META_FILE = "index_meta.json"


class IndexMismatchError(RuntimeError):
    pass


class IndexLockedError(RuntimeError):
    pass


class IndexMissingError(RuntimeError):
    pass


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"packageflow:{chunk_id}"))


def build_filter(doc_types: list[str] | None = None, roles: list[str] | None = None) -> models.Filter | None:
    must = []
    if doc_types:
        must.append(models.FieldCondition(key="doc_type", match=models.MatchAny(any=list(doc_types))))
    if roles:
        must.append(models.FieldCondition(key="roles", match=models.MatchAny(any=list(roles))))
    return models.Filter(must=must) if must else None


class VectorStore:
    def __init__(self, index_dir: Path | str, collection: str = "packageflow_docs", in_memory: bool = False):
        self.index_dir = Path(index_dir) if not in_memory else None
        self.collection = collection
        if in_memory:
            self.client = QdrantClient(location=":memory:")
        else:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            try:
                self.client = QdrantClient(path=str(self.index_dir))
            except RuntimeError as e:
                if "already accessed" in str(e).lower():
                    raise IndexLockedError(
                        f"The index at {self.index_dir} is open in another program (probably the Streamlit app "
                        "or another ask/ingest window). Close it and try again.") from e
                raise
        self._meta_mem: dict | None = None

    # ---------- metadata ----------
    def _meta_path(self) -> Path | None:
        return self.index_dir / META_FILE if self.index_dir else None

    def meta(self) -> dict | None:
        p = self._meta_path()
        if p is None:
            return self._meta_mem
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def _write_meta(self, meta: dict) -> None:
        p = self._meta_path()
        if p is None:
            self._meta_mem = meta
        else:
            p.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def exists(self) -> bool:
        return self.client.collection_exists(self.collection) and self.meta() is not None

    def check_model(self, embed_model: str) -> None:
        if not self.exists():
            raise IndexMissingError("No index yet. Run: python scripts/ingest.py")
        built_with = self.meta().get("embed_model")
        if built_with != embed_model:
            raise IndexMismatchError(
                f"The index was built with '{built_with}' but EMBED_MODEL is '{embed_model}'. "
                f"Either set EMBED_MODEL={built_with} in .env, or rebuild: python scripts/ingest.py --rebuild")

    # ---------- write ----------
    def recreate(self, dim: int, embed_model: str) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
        self._write_meta({"embed_model": embed_model, "dim": dim, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                          "updated_at": None, "chunk_count": 0})

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")
        points = [
            models.PointStruct(
                id=point_id(c.chunk_id),
                vector=v,
                payload={"chunk": c.to_dict(), "chunk_id": c.chunk_id, "doc_type": c.doc_type,
                         "roles": c.metadata.get("roles", []), "content_hash": c.content_hash},
            )
            for c, v in zip(chunks, vectors)
        ]
        for i in range(0, len(points), 128):
            self.client.upsert(self.collection, points=points[i:i + 128])
        self._touch()

    def delete(self, chunk_ids: list[str]) -> None:
        if chunk_ids:
            self.client.delete(self.collection, points_selector=models.PointIdsList(
                points=[point_id(c) for c in chunk_ids]))
            self._touch()

    def _touch(self) -> None:
        meta = self.meta() or {}
        meta["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        meta["chunk_count"] = self.count()
        self._write_meta(meta)

    # ---------- read ----------
    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def all_chunks(self) -> list[Chunk]:
        out, offset = [], None
        while True:
            pts, offset = self.client.scroll(self.collection, limit=256, offset=offset,
                                             with_payload=True, with_vectors=False)
            out.extend(Chunk.from_dict(p.payload["chunk"]) for p in pts)
            if offset is None:
                break
        return out

    def existing_hashes(self) -> dict[str, str]:
        return {c.chunk_id: c.content_hash for c in self.all_chunks()}

    def search(self, vector: list[float], limit: int, doc_types: list[str] | None = None,
               roles: list[str] | None = None) -> list[tuple[Chunk, float]]:
        res = self.client.query_points(self.collection, query=vector, limit=limit,
                                       query_filter=build_filter(doc_types, roles), with_payload=True)
        return [(Chunk.from_dict(p.payload["chunk"]), float(p.score)) for p in res.points]

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass
