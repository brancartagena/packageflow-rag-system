from packageflow_rag.chunking import context_header
from packageflow_rag.config import settings
from packageflow_rag.retrieval import rrf


def test_settings_load():
    assert settings.top_k > 0


def test_context_header():
    assert context_header("BRD v0.4", ["8. Functional Requirements"]) == "[BRD v0.4 > 8. Functional Requirements]"


def test_rrf_prefers_items_ranked_high_in_both():
    scores = rrf([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert max(scores, key=scores.get) in {"a", "b"}
    assert scores["d"] < scores["a"]
