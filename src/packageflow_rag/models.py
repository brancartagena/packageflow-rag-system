"""Core data types shared across the pipeline."""

from dataclasses import asdict, dataclass, field


@dataclass
class Chunk:
    chunk_id: str                  # stable, readable id, e.g. "api:op:POST:/packages/{packageId}/match"
    text: str                      # what gets embedded and shown to the LLM (starts with the context header)
    source_file: str
    doc_type: str                  # brd | tech_arch | cloud_arch | api | erd | other
    doc_label: str = ""            # human label used in citations, e.g. "BRD v0.4"
    section_path: list[str] = field(default_factory=list)
    doc_version: str | None = None
    metadata: dict = field(default_factory=dict)  # api_method, api_path, api_tag, roles, ...
    content_hash: str = ""

    @property
    def citation(self) -> str:
        """Short label shown under answers: 'BRD v0.4 > 8. Functional Requirements'."""
        return " > ".join([self.doc_label or self.source_file, *self.section_path])

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float                   # fused (RRF) score, only meaningful for ranking
    dense_rank: int | None = None
    bm25_rank: int | None = None
    dense_score: float | None = None  # cosine similarity from the vector store
