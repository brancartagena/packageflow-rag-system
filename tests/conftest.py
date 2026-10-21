import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_ollama import FakeOllama  # noqa: E402

SOURCE = ROOT / "data" / "source"
DERIVED = ROOT / "data" / "derived"


def have_docs() -> bool:
    return SOURCE.exists() and any(SOURCE.glob("*.docx"))


needs_docs = pytest.mark.skipif(not have_docs(), reason="put the PackageFlow docs in data/source/ to run this")


@pytest.fixture
def fake_ollama():
    with FakeOllama() as f:
        yield f


@pytest.fixture(scope="session")
def corpus():
    if not have_docs():
        pytest.skip("no docs in data/source/")
    from packageflow_rag.loaders import load_corpus
    chunks, notes = load_corpus(SOURCE, DERIVED, 400, 50)
    return chunks, notes
