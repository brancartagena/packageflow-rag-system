"""Word (.docx) loader: BRD and Technical Architecture docs.

Plan (PLAN.md section 3):
- Walk paragraphs and tables in document order.
- Split into sections on Heading 1 / Heading 2 (and Title).
- Each table row becomes a labeled sentence with column headers ("Risk: X. Impact: Y.").
- De-duplicate merged header cells (the tech stack table repeats headers).
- Pull doc_version from the status table if present (BRD: "DRAFT v0.4").
"""

from pathlib import Path

from ...src.packageflow_rag.models import Chunk


def load(path: Path) -> list[Chunk]:
    raise NotImplementedError("Phase 1")
