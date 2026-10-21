from packageflow_rag.chunking import approx_tokens, dedupe_cells, pack_units, row_to_text, table_units
from packageflow_rag.retrieval import tokenize


def test_pack_units_respects_max_and_never_splits_small_units():
    units = [f"sentence number {i} " * 10 for i in range(20)]
    chunks = pack_units(units, max_tokens=100)
    assert len(chunks) > 1
    assert all(approx_tokens(c) <= 100 for c in chunks)
    assert sum(c.count("sentence number 0") for c in chunks) == 10  # unit kept whole, not duplicated


def test_pack_units_splits_oversized_unit():
    big = ". ".join(["word " * 20] * 30)
    assert len(pack_units([big], max_tokens=80)) > 1


def test_table_units_small_vs_large():
    small, large_flag = table_units(["Risk", "Impact"], [["A", "B"], ["C", "D"]], 400)
    assert not large_flag and len(small) == 1 and "Risk: A | Impact: B" in small[0]
    rows = [[f"Req {i}", "the solution must do a long list of things " * 4] for i in range(12)]
    units, large_flag = table_units(["Requirement", "Description"], rows, 400)
    assert large_flag and len(units) == 12 and units[0].startswith("Requirement: Req 0")


def test_dedupe_merged_cells_and_row_text():
    assert dedupe_cells(["Layer", "Tech", "Tech", "Why", "Why"]) == ["Layer", "Tech", "Why"]
    assert row_to_text(["A", "B"], ["x", ""]) == "A: x"


def test_tokenize_keeps_identifiers_and_parts():
    toks = tokenize("POST /packages/{packageId}/match exception_case")
    for t in ["packageid", "package", "id", "match", "exception_case", "exception", "case"]:
        assert t in toks
    assert "the" not in tokenize("the package")
