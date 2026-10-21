"""Splitting helpers, context headers and chunk construction.

Loaders turn a document into ordered "units" (a paragraph, a list item, a table
row, a whole small table, a code block). pack_units() then groups units into
chunks of up to max_tokens without ever cutting a unit in half (unless a single
unit is bigger than max_tokens, then it is split on sentences/lines).
"""

import hashlib
import re

from .models import Chunk


def context_header(doc_label: str, section_path: list[str]) -> str:
    """e.g. '[BRD v0.4 > 8. Functional Requirements]'"""
    return "[" + " > ".join([doc_label, *section_path]) + "]"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def approx_tokens(text: str) -> int:
    """Rough token count (about 0.75 words per token). Good enough for budgeting."""
    return int(len(text.split()) / 0.75) + 1


def slug(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:max_len].rstrip("-") or "section"


def _split_oversized(unit: str, max_tokens: int) -> list[str]:
    """Split one unit that is bigger than max_tokens on line breaks, then sentences."""
    pieces = unit.split("\n") if "\n" in unit else re.split(r"(?<=[.!?])\s+", unit)
    out, cur = [], ""
    for p in pieces:
        cand = f"{cur}\n{p}" if cur else p
        if cur and approx_tokens(cand) > max_tokens:
            out.append(cur)
            cur = p
        else:
            cur = cand
    if cur:
        out.append(cur)
    return out


def pack_units(units: list[str], max_tokens: int, overlap_tokens: int = 0) -> list[str]:
    """Greedily group units into chunks of at most max_tokens.

    overlap_tokens: when a chunk is closed, the last unit is repeated at the start
    of the next chunk if it is no bigger than overlap_tokens (keeps context
    across a split without duplicating big blocks).
    """
    flat: list[str] = []
    for u in units:
        u = u.strip()
        if not u:
            continue
        flat.extend(_split_oversized(u, max_tokens) if approx_tokens(u) > max_tokens else [u])

    chunks, cur = [], []
    for u in flat:
        if cur and approx_tokens("\n".join([*cur, u])) > max_tokens:
            chunks.append("\n".join(cur))
            last = cur[-1]
            cur = [last] if overlap_tokens and approx_tokens(last) <= overlap_tokens else []
        cur.append(u)
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def split_text(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split long text on paragraph boundaries with overlap."""
    return pack_units(re.split(r"\n\s*\n", text), max_tokens, overlap_tokens)


def make_chunk(
    *,
    chunk_id: str,
    body: str,
    source_file: str,
    doc_type: str,
    doc_label: str,
    section_path: list[str],
    doc_version: str | None = None,
    metadata: dict | None = None,
) -> Chunk:
    """Build a Chunk with the context header prepended and a content hash."""
    text = f"{context_header(doc_label, section_path)}\n{body.strip()}"
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        source_file=source_file,
        doc_type=doc_type,
        doc_label=doc_label,
        section_path=list(section_path),
        doc_version=doc_version,
        metadata=metadata or {},
        content_hash=content_hash(text),
    )


def row_to_text(headers: list[str], cells: list[str]) -> str:
    """'Risk: X. Impact: Y.' style line for one table row."""
    parts = []
    for h, c in zip(headers, cells):
        c = " ".join(c.split())
        if not c:
            continue
        parts.append(f"{h}: {c}" if h else c)
    return " | ".join(parts)


def dedupe_cells(cells: list[str]) -> list[str]:
    """Merged cells come back repeated (python-docx); drop consecutive duplicates."""
    out: list[str] = []
    for c in cells:
        if not out or c != out[-1]:
            out.append(c)
    return out


def table_units(headers: list[str], rows: list[list[str]], max_tokens: int, caption: str = "") -> tuple[list[str], bool]:
    """Turn a table into units.

    Small table (up to 75% of max_tokens): one unit with all rows -> stays with
    the section prose. Large table: one unit per row, and the caller should NOT
    pack rows together (returns is_large=True) so each row is its own chunk.
    """
    lines = [row_to_text(headers, r) for r in rows]
    lines = [l for l in lines if l]
    head = f"Table ({caption or ', '.join(h for h in headers if h)}):"
    whole = "\n".join([head, *[f"- {l}" for l in lines]])
    if approx_tokens(whole) <= int(max_tokens * 0.75):
        return [whole], False
    return lines, True
