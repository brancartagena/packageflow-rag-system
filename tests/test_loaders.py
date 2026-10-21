from collections import Counter
from pathlib import Path

from conftest import needs_docs
from packageflow_rag.loaders import detect_doc, latest_versions


def test_detect_doc_types():
    assert detect_doc(Path("Package_Delivery_BRD_Draft.docx")).doc_type == "brd"
    assert detect_doc(Path("PackageFlow_Technical_Architecture_and_Data_Model.docx")).doc_type == "tech_arch"
    assert detect_doc(Path("PackageFlow_Cloud_System_Architecture.md")).doc_type == "cloud_arch"
    assert detect_doc(Path("packageflow-api-v08-wip.yaml")).doc_type == "api"
    assert detect_doc(Path("PackageFlow - ERD.png")).doc_type == "erd"
    assert detect_doc(Path("erd.md")).doc_type == "erd"


def test_latest_versions_keeps_newest_only():
    kept, skipped = latest_versions([Path("packageflow-api-v07-wip.yaml"), Path("packageflow-api-v08-wip.yaml"),
                                     Path("other.md")])
    assert Path("packageflow-api-v08-wip.yaml") in kept and Path("other.md") in kept
    assert skipped == [Path("packageflow-api-v07-wip.yaml")]


@needs_docs
def test_corpus_shape(corpus):
    chunks, notes = corpus
    by_type = Counter(c.doc_type for c in chunks)
    assert set(by_type) >= {"brd", "tech_arch", "cloud_arch", "api", "erd"}
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(c.text.startswith("[") and c.content_hash for c in chunks)
    assert not any("FAILED" in n for n in notes), notes


@needs_docs
def test_every_api_operation_is_a_chunk(corpus):
    chunks, _ = corpus
    ops = [c for c in chunks if c.metadata.get("api_kind") == "operation"]
    assert len(ops) >= 65
    match = next(c for c in ops if c.metadata["api_path"] == "/packages/{packageId}/match")
    assert "Package Receiver" in match.metadata["roles"] and "Facilities Lead" in match.metadata["roles"]
    assert "MatchPackageRequest" in match.text and "procurementOrderId" in match.text


@needs_docs
def test_brd_requirements_are_rows_and_version_found(corpus):
    chunks, _ = corpus
    fr = [c for c in chunks if c.chunk_id.startswith("brd:8-functional-requirements:row")]
    assert len(fr) == 10
    assert all(c.doc_version == "v0.4" for c in chunks if c.doc_type == "brd")


@needs_docs
def test_erd_has_13_entities(corpus):
    chunks, _ = corpus
    entities = {c.chunk_id for c in chunks if c.chunk_id.startswith("erd:entity-")}
    assert len(entities) == 13
    person = next(c for c in chunks if c.chunk_id == "erd:entity-person:1")
    assert "enterprise_user_id" in person.text
