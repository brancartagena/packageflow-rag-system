"""End to end with a fake Ollama server: ingest -> retrieve -> answer -> log."""

import json

import pytest

from packageflow_rag.config import Settings
from packageflow_rag.embeddings import OllamaEmbedder, OllamaError
from packageflow_rag.generation import OllamaGenerator, cited_numbers
from packageflow_rag.indexing import index_chunks
from packageflow_rag.rag import RAG
from packageflow_rag.vectorstore import IndexMismatchError, VectorStore


def make_chunks():
    from packageflow_rag.chunking import make_chunk
    data = [
        ("cloud_arch:db:1", "cloud_arch", ["Design decisions"], "PostgreSQL is the transactional database for packages."),
        ("erd:entity-exception-case:1", "erd", ["Entity: exception_case"], "exception_case fields: exception_id, owner_id, severity."),
        ("api:op:POST /packages/{packageId}/cancel", "api", ["Package Receiver", "POST /packages/{packageId}/cancel"],
         "Cancel a package receipt entered by mistake. Who can use this: Package Receiver, Facilities Lead."),
        ("brd:risks:1", "brd", ["10. Risks"], "Adoption risk: the delivery team may see extra workload."),
    ]
    out = []
    for cid, dt, path, body in data:
        meta = {"roles": ["Package Receiver", "Facilities Lead"]} if dt == "api" else {}
        out.append(make_chunk(chunk_id=cid, body=body, source_file="x", doc_type=dt, doc_label=dt.upper(),
                              section_path=path, metadata=meta))
    return out


def cfg_for(tmp_path, host, **kw):
    return Settings(_env_file=None, ollama_host=host, embed_model="fake-embed", ollama_chat_model="fake-chat",
                    index_dir=tmp_path / "index", log_dir=tmp_path / "logs", ollama_timeout=10, **kw)


def build_index(tmp_path, host, chunks):
    store = VectorStore(tmp_path / "index")
    emb = OllamaEmbedder("fake-embed", host)
    stats = index_chunks(store, emb, chunks)
    return store, emb, stats


def test_ingest_incremental_and_model_guard(tmp_path, fake_ollama):
    chunks = make_chunks()
    store, emb, stats = build_index(tmp_path, fake_ollama.host, chunks)
    assert stats.mode == "rebuild" and stats.total == 4

    changed = list(chunks)
    from dataclasses import replace
    changed[0] = replace(changed[0], text=changed[0].text + " Updated.", content_hash="new")
    changed = changed[:3]  # one removed
    stats2 = index_chunks(store, emb, changed)
    assert (stats2.mode, stats2.updated, stats2.removed, stats2.unchanged, stats2.total) == ("incremental", 1, 1, 2, 3)

    store.check_model("fake-embed")
    with pytest.raises(IndexMismatchError):
        store.check_model("nomic-embed-text")
    store.close()


def test_hybrid_retrieval_finds_identifiers(tmp_path, fake_ollama):
    store, emb, _ = build_index(tmp_path, fake_ollama.host, make_chunks())
    rag = RAG(cfg_for(tmp_path, fake_ollama.host), store=store, embedder=emb)
    res = rag.retrieve("exception_case owner_id", top_k=2)
    assert res.chunks[0].chunk.chunk_id == "erd:entity-exception-case:1"
    assert res.chunks[0].bm25_rank == 1
    only_api = rag.retrieve("cancel", top_k=5, doc_types=["api"])
    assert {rc.chunk.doc_type for rc in only_api.chunks} == {"api"}
    lead = rag.retrieve("cancel", top_k=5, roles=["Facilities Lead"])
    assert all("Facilities Lead" in rc.chunk.metadata.get("roles", []) for rc in lead.chunks)
    rag.close()


def test_answer_streams_cites_and_logs(tmp_path, fake_ollama):
    store, emb, _ = build_index(tmp_path, fake_ollama.host, make_chunks())
    cfg = cfg_for(tmp_path, fake_ollama.host)
    rag = RAG(cfg, store=store, embedder=emb)
    resp = rag.ask("Which database stores packages?")
    tokens = list(resp)
    assert len(tokens) > 1 and "PostgreSQL" in resp.text
    assert resp.cited == [1, 2]
    chat_call = [r for p, r in fake_ollama.calls if p == "/api/chat"][-1]
    assert chat_call["options"]["num_ctx"] == cfg.ollama_num_ctx
    assert "[1]" in chat_call["messages"][-1]["content"]
    log = (tmp_path / "logs" / "queries.jsonl").read_text().strip().splitlines()
    assert json.loads(log[-1])["question"] == "Which database stores packages?"
    rag.close()


def test_min_dense_score_gate(tmp_path, fake_ollama):
    store, emb, _ = build_index(tmp_path, fake_ollama.host, make_chunks())
    rag = RAG(cfg_for(tmp_path, fake_ollama.host, min_dense_score=0.99), store=store, embedder=emb)
    resp = rag.ask("What is the weather forecast?").consume()
    assert resp.gated and resp.not_covered
    assert not [p for p, _ in fake_ollama.calls if p == "/api/chat"]
    rag.close()


def test_friendly_errors(fake_ollama):
    with pytest.raises(OllamaError, match="ollama pull"):
        OllamaEmbedder("not-pulled", fake_ollama.host).embed_query("hi")
    with pytest.raises(OllamaError, match="reach Ollama"):
        OllamaEmbedder("fake-embed", "http://127.0.0.1:9", timeout=2).embed_query("hi")
    with pytest.raises(OllamaError, match="ollama pull"):
        list(OllamaGenerator("not-pulled", fake_ollama.host).stream([{"role": "user", "content": "hi"}]))


def test_cited_numbers():
    assert cited_numbers("A [2] and B [1][2], C [9]", 3) == [2, 1]
