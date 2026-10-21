"""Markdown loader: Cloud System Architecture doc.

Split on ## and ### headings; keep the parent heading chain in section_path.
"""

from pathlib import Path

from ...src.packageflow_rag.models import Chunk


def load(path: Path) -> list[Chunk]:
    raise NotImplementedError("Phase 1")
