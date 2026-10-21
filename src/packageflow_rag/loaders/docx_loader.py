"""Word (.docx) loader: BRD and Technical Architecture docs.

- Walks paragraphs and tables in document order.
- Sections split on Title / Heading 1 / Heading 2 / Heading 3.
- List items become "- item" lines. Numbered steps keep their numbers.
- Tables: header row gives the labels for every row ("Risk: X | Impact: Y").
  Merged header/body cells (python-docx repeats them) are de-duplicated.
  Small tables stay with the section text; big tables become one chunk per row.
- A key/value table before the first heading (the BRD status table) becomes a
  "Document info" chunk and is where doc_version ("v0.4") is read from.
- Embedded images are skipped. (The one in the Tech Architecture doc is the
  same ERD that is indexed from data/derived/erd.md.)
"""

import re
from pathlib import Path

import docx
from docx.table import Table
from docx.text.paragraph import Paragraph

from ..chunking import dedupe_cells, make_chunk, pack_units, slug, table_units
from ..models import Chunk

HEADING_LEVELS = {"title": 0, "heading 1": 1, "heading 2": 2, "heading 3": 3}
VERSION_RE = re.compile(r"\bv(\d+(?:\.\d+)+)\b", re.IGNORECASE)


def _cell_text(cell) -> str:
    return " ".join(cell.text.split())


def _table_rows(t: Table) -> list[list[str]]:
    return [dedupe_cells([_cell_text(c) for c in r.cells]) for r in t.rows]


def _iter_blocks(d):
    for el in d.element.body.iterchildren():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "p":
            yield Paragraph(el, d)
        elif tag == "tbl":
            yield Table(el, d)


def load(path: Path, info, max_tokens: int = 400, overlap_tokens: int = 50) -> list[Chunk]:
    d = docx.Document(str(path))

    # Pass 1: build sections: list of (path, [(kind, payload)])
    sections: list[tuple[list[str], list[tuple[str, object]]]] = []
    stack: list[tuple[int, str]] = []
    pre_heading: list[tuple[str, object]] = []
    current: list[tuple[str, object]] | None = None
    title = None

    for block in _iter_blocks(d):
        if isinstance(block, Paragraph):
            text = " ".join(block.text.split())
            if not text:
                continue
            style = (block.style.name or "").lower() if block.style is not None else ""
            level = HEADING_LEVELS.get(style)
            if level == 0:
                title = text
                continue
            if level:
                stack = [(lv, h) for lv, h in stack if lv < level] + [(level, text)]
                current = []
                sections.append(([h for _, h in stack], current))
                continue
            kind = "li" if "list" in style else "p"
            (current if current is not None else pre_heading).append((kind, text))
        else:
            rows = _table_rows(block)
            if rows:
                (current if current is not None else pre_heading).append(("table", rows))

    # Document info (status table + intro lines before the first heading)
    version = None
    info_lines: list[str] = []
    for kind, payload in pre_heading:
        if kind == "table":
            for row in payload:  # key/value rows
                if len(row) >= 2:
                    info_lines.append(f"{row[0]}: {' '.join(row[1:])}")
                    m = VERSION_RE.search(" ".join(row[1:]))
                    key = row[0].lower()
                    if m and version is None and ("status" in key or "version" in key):
                        version = f"v{m.group(1)}"
                elif row:
                    info_lines.append(row[0])
        else:
            info_lines.append(payload)
    if title:
        info_lines.insert(0, title)

    label = f"{info.label} {version}" if version else info.label
    common = dict(source_file=path.name, doc_type=info.doc_type, doc_label=label, doc_version=version)
    chunks: list[Chunk] = []

    if info_lines:
        chunks.append(make_chunk(chunk_id=f"{info.doc_type}:info", body="\n".join(info_lines),
                                 section_path=["Document info"], **common))

    # Pass 2: sections -> units -> chunks
    used_ids: set[str] = set()
    for sec_path, blocks in sections:
        base_id = f"{info.doc_type}:{slug(sec_path[-1])}"
        if base_id in used_ids:  # two sections with the same heading text
            base_id = f"{info.doc_type}:{slug('-'.join(sec_path))}"
        used_ids.add(base_id)
        units: list[str] = []
        row_chunks: list[str] = []
        for kind, payload in blocks:
            if kind == "p":
                units.append(payload)
            elif kind == "li":
                units.append(payload if re.match(r"^\d+[.)]", payload) else f"- {payload}")
            else:
                rows = payload
                headers, body = rows[0], rows[1:]
                if not body:
                    units.append(" | ".join(headers))
                    continue
                t_units, is_large = table_units(headers, body, max_tokens)
                if is_large:
                    row_chunks.extend(t_units)
                else:
                    units.extend(t_units)

        n = 0
        for text in pack_units(units, max_tokens, overlap_tokens):
            n += 1
            chunks.append(make_chunk(chunk_id=f"{base_id}:{n}", body=text, section_path=sec_path, **common))
        for i, text in enumerate(row_chunks, 1):
            chunks.append(make_chunk(chunk_id=f"{base_id}:row{i}", body=text, section_path=sec_path,
                                     metadata={"table_row": True}, **common))

    return chunks
