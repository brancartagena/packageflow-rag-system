"""Markdown loader: Cloud System Architecture doc and derived text (erd.md).

- YAML-style frontmatter (--- ... ---) is stripped. 'version:' in it sets doc_version.
- Sections split on #, ##, ### headings; section_path keeps the heading chain.
- Fenced code blocks (``` ... ```, e.g. the mermaid diagram) stay whole.
- Markdown tables: small ones stay with the section text, big ones become one
  chunk per row with the column names on every row.
"""

import re
from pathlib import Path

from ..chunking import make_chunk, pack_units, slug, table_units
from ..models import Chunk

HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$")
TABLE_SEP_RE = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")


def _strip_frontmatter(text: str) -> tuple[str, dict]:
    meta: dict = {}
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            for line in text[3:end].splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
            text = text[end + 4:]
    return text, meta


def _split_row(line: str) -> list[str]:
    line = line.strip().strip("|")
    return [c.strip().replace("**", "") for c in line.split("|")]


def _blocks(lines: list[str]):
    """Yield ('p'|'code'|'table', payload) blocks from section lines."""
    i, buf = 0, []

    def flush():
        nonlocal buf
        text = "\n".join(buf).strip()
        buf = []
        return text

    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            if (t := flush()):
                yield "p", t
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("```"):
                j += 1
            yield "code", "\n".join(lines[i:j + 1])
            i = j + 1
            continue
        if line.strip().startswith("|") and i + 1 < len(lines) and TABLE_SEP_RE.match(lines[i + 1].strip()):
            if (t := flush()):
                yield "p", t
            headers = _split_row(line)
            rows, j = [], i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append(_split_row(lines[j]))
                j += 1
            yield "table", (headers, rows)
            i = j
            continue
        if not line.strip():
            if (t := flush()):
                yield "p", t
        elif line.strip() in ("---", "***"):
            if (t := flush()):
                yield "p", t
        else:
            buf.append(line)
        i += 1
    if (t := flush()):
        yield "p", t


def load(path: Path, info, max_tokens: int = 400, overlap_tokens: int = 50) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    text, meta = _strip_frontmatter(raw)
    version = meta.get("version")
    label = f"{info.label} {version}" if version else info.label
    common = dict(source_file=path.name, doc_type=info.doc_type, doc_label=label, doc_version=version)
    return markdown_to_chunks(text, id_prefix=info.doc_type, common=common,
                              max_tokens=max_tokens, overlap_tokens=overlap_tokens)


def markdown_to_chunks(
    text: str,
    *,
    id_prefix: str,
    common: dict,
    max_tokens: int,
    overlap_tokens: int,
    section_prefix: list[str] | None = None,
    extra_metadata: dict | None = None,
) -> list[Chunk]:
    """Turn a markdown string into chunks. Also used for the OpenAPI info.description."""
    section_prefix = section_prefix or []
    extra_metadata = extra_metadata or {}

    # A single H1 at the top is the document title, not a section: keep it out of section paths
    h1s = [l for l in text.splitlines() if re.match(r"^#\s+", l)]
    title_line = h1s[0] if len(h1s) == 1 else None

    # Split into sections by heading
    sections: list[tuple[list[str], list[str]]] = []
    stack: list[tuple[int, str]] = []
    cur: list[str] = []
    sections.append(([], cur))
    in_code = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        m = None if in_code else HEADING_RE.match(line)
        if m and line == title_line:
            cur.append(m.group(2).replace("**", "").strip())  # keep the title text in the intro chunk
            continue
        if m:
            level, heading = len(m.group(1)), m.group(2).replace("**", "").strip()
            stack = [(lv, h) for lv, h in stack if lv < level] + [(level, heading)]
            cur = []
            sections.append(([h for _, h in stack], cur))
        else:
            cur.append(line)

    chunks: list[Chunk] = []
    used: set[str] = set()
    for sec_path, lines in sections:
        units, row_units, code_units = [], [], []
        for kind, payload in _blocks(lines):
            if kind == "p":
                units.append(payload)
            elif kind == "code":
                code_units.append(payload)
            else:
                headers, rows = payload
                t_units, is_large = table_units(headers, rows, max_tokens)
                (row_units if is_large else units).extend(t_units)
        if not (units or row_units or code_units):
            continue

        path_for_cite = [*section_prefix, *sec_path] if (section_prefix or sec_path) else ["Overview"]
        base = f"{id_prefix}:{slug(sec_path[-1]) if sec_path else 'intro'}"
        if base in used:
            base = f"{id_prefix}:{slug('-'.join(sec_path))}"
        used.add(base)

        for n, body in enumerate(pack_units(units, max_tokens, overlap_tokens), 1):
            chunks.append(make_chunk(chunk_id=f"{base}:{n}", body=body, section_path=path_for_cite,
                                     metadata=dict(extra_metadata), **common))
        for n, body in enumerate(code_units, 1):
            for k, part in enumerate(pack_units([body], max_tokens * 2), 1):
                cid = f"{base}:code{n}" + (f".{k}" if k > 1 else "")
                chunks.append(make_chunk(chunk_id=cid, body=part, section_path=path_for_cite,
                                         metadata={**extra_metadata, "code_block": True}, **common))
        for n, body in enumerate(row_units, 1):
            chunks.append(make_chunk(chunk_id=f"{base}:row{n}", body=body, section_path=path_for_cite,
                                     metadata={**extra_metadata, "table_row": True}, **common))
    return chunks
