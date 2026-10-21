"""Loaders turn one source file into a list of Chunks.

Each loader exposes: load(path: Path, info: DocInfo, max_tokens: int, overlap_tokens: int) -> list[Chunk]

load_corpus() is the entry point used by ingest: it walks SOURCE_DIR and
DERIVED_DIR, skips files it can't read (images are indexed through their
derived .md text instead), and keeps only the newest version of versioned
files like packageflow-api-v07 / -v08.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from ..models import Chunk

SUPPORTED = {".docx", ".md", ".yaml", ".yml"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


@dataclass
class DocInfo:
    doc_type: str      # brd | tech_arch | cloud_arch | api | erd | other
    label: str         # base label for citations, version gets appended by the loader if found


# (filename regex, doc_type, label). First match wins. Matching is on the lowercased file name.
DOC_TYPES: list[tuple[str, str, str]] = [
    (r"brd", "brd", "BRD"),
    (r"technical_architecture|data_model", "tech_arch", "Tech Architecture & Data Model"),
    (r"cloud_system_architecture|cloud.*architecture", "cloud_arch", "Cloud System Architecture"),
    (r"(^|[-_])api([-_]|\.|$)|openapi", "api", "API Spec"),
    (r"(^|[-_ ])erd([-_ .]|$)", "erd", "ERD"),
]


def detect_doc(path: Path) -> DocInfo:
    name = path.name.lower()
    for pattern, doc_type, label in DOC_TYPES:
        if re.search(pattern, name):
            return DocInfo(doc_type, label)
    return DocInfo("other", path.stem.replace("_", " ").replace("-", " "))


def load_file(path: Path, max_tokens: int = 400, overlap_tokens: int = 50) -> list[Chunk]:
    """Dispatch to the right loader by file extension."""
    info = detect_doc(path)
    ext = path.suffix.lower()
    if ext == ".docx":
        from .docx_loader import load
    elif ext == ".md":
        from .markdown_loader import load
    elif ext in (".yaml", ".yml"):
        from .openapi_loader import load
    else:
        raise ValueError(f"No loader for {path.name}. Images go through image_loader first.")
    return load(path, info, max_tokens, overlap_tokens)


_VERSION_RE = re.compile(r"[-_ ]v(\d+(?:[._]\d+)*)", re.IGNORECASE)


def _version_key(path: Path) -> tuple[str, tuple[int, ...]]:
    """('packageflow-api-wip.yaml', (8,)) for 'packageflow-api-v08-wip.yaml'."""
    m = _VERSION_RE.search(path.name)
    if not m:
        return path.name.lower(), ()
    family = (path.name[: m.start()] + path.name[m.end():]).lower()
    nums = tuple(int(n) for n in re.split(r"[._]", m.group(1)))
    return family, nums


def latest_versions(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    """Keep only the newest version per file family. Returns (kept, skipped)."""
    best: dict[str, tuple[tuple[int, ...], Path]] = {}
    for p in paths:
        fam, ver = _version_key(p)
        if fam not in best or ver > best[fam][0]:
            best[fam] = (ver, p)
    kept = sorted(v[1] for v in best.values())
    skipped = sorted(set(paths) - set(kept))
    return kept, skipped


def discover(source_dir: Path, derived_dir: Path) -> tuple[list[Path], list[str]]:
    """Files to index, plus human-readable notes about what was skipped and why."""
    notes: list[str] = []
    candidates: list[Path] = []
    derived_md = sorted(derived_dir.glob("*.md")) if derived_dir.exists() else []
    derived_sources = {_derived_source_name(p) for p in derived_md}

    for p in sorted(source_dir.iterdir()) if source_dir.exists() else []:
        if p.is_dir() or p.name.startswith((".", "~$")):
            continue
        ext = p.suffix.lower()
        if ext in SUPPORTED:
            candidates.append(p)
        elif ext in IMAGE_EXT:
            if p.name in derived_sources:
                notes.append(f"{p.name}: indexed through its text version in {derived_dir.name}/")
            else:
                notes.append(f"{p.name}: SKIPPED, image with no text version. Run scripts/describe_image.py on it.")
        else:
            notes.append(f"{p.name}: skipped, unsupported file type")

    kept, skipped = latest_versions(candidates)
    for s in skipped:
        notes.append(f"{s.name}: skipped, a newer version of this file exists")
    return kept + derived_md, notes


def _derived_source_name(md_path: Path) -> str | None:
    """Read 'source_file:' from the derived .md frontmatter."""
    try:
        head = md_path.read_text(encoding="utf-8")[:1000]
    except OSError:
        return None
    m = re.search(r"^source_file:\s*(.+)$", head, re.MULTILINE)
    return m.group(1).strip() if m else None


def load_corpus(source_dir: Path, derived_dir: Path, max_tokens: int = 400, overlap_tokens: int = 50) -> tuple[list[Chunk], list[str]]:
    """Load every indexable file. Returns (chunks, notes)."""
    files, notes = discover(source_dir, derived_dir)
    chunks: list[Chunk] = []
    for f in files:
        try:
            got = load_file(f, max_tokens, overlap_tokens)
        except Exception as e:  # one bad file should not kill the whole ingest
            notes.append(f"{f.name}: FAILED to load ({type(e).__name__}: {e})")
            continue
        notes.append(f"{f.name}: {len(got)} chunks")
        chunks.extend(got)

    seen: set[str] = set()
    for c in chunks:  # chunk ids must be unique across the corpus
        if c.chunk_id in seen:
            raise ValueError(f"Duplicate chunk_id {c.chunk_id}")
        seen.add(c.chunk_id)
    return chunks, notes
