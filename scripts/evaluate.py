"""Score retrieval (and optionally answers) against eval/golden_questions.yaml.

Usage:
  python scripts/evaluate.py                                     current EMBED_MODEL, hybrid vs dense vs bm25
  python scripts/evaluate.py --models qwen3-embedding:0.6b nomic-embed-text   compare embedding models
  python scripts/evaluate.py --answers                           also run the chat model on every question

Model comparison builds a temporary in-memory index per model from
data/processed/chunks.jsonl, so your real index is not touched.

Metrics:
  hit@k  share of questions where at least one expected chunk is in the top k
  MRR    1 / rank of the first expected chunk (1.0 = always first)
  Out-of-scope questions are used to suggest a MIN_DENSE_SCORE threshold.
"""

import _path  # noqa: F401
import json
import statistics
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from ruamel.yaml import YAML

from packageflow_rag.config import PROJECT_ROOT, settings as cfg
from packageflow_rag.embeddings import OllamaEmbedder, OllamaError
from packageflow_rag.indexing import build_chunks, read_chunks_jsonl, write_chunks_jsonl
from packageflow_rag.retrieval import Retriever
from packageflow_rag.vectorstore import VectorStore

app = typer.Typer(add_completion=False)
console = Console()
GOLDEN = PROJECT_ROOT / "eval" / "golden_questions.yaml"


def load_golden(path: Path = GOLDEN) -> list[dict]:
    return YAML(typ="safe", pure=True).load(path.read_text(encoding="utf-8")) or []


def matches(chunk_id: str, expected: list[str]) -> bool:
    return any(chunk_id == e or (e.endswith("*") and chunk_id.startswith(e[:-1])) for e in expected)


def hit_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    return 1.0 if any(matches(c, expected) for c in retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: list[str], expected: list[str]) -> float:
    for i, cid in enumerate(retrieved, start=1):
        if matches(cid, expected):
            return 1.0 / i
    return 0.0


def score_retriever(retriever: Retriever, golden: list[dict], modes: list[str], k: int) -> dict:
    in_scope = [q for q in golden if q.get("expected_chunks")]
    out_scope = [q for q in golden if q.get("category") == "out_of_scope"]
    report: dict = {"modes": {}, "per_question": []}
    for mode in modes:
        hits, rrs, by_cat = [], [], {}
        for q in in_scope:
            res = retriever.retrieve(q["question"], top_k=max(k, 10), mode=mode)
            ids = [rc.chunk.chunk_id for rc in res.chunks]
            h, rr = hit_at_k(ids, q["expected_chunks"], k), reciprocal_rank(ids, q["expected_chunks"])
            hits.append(h)
            rrs.append(rr)
            by_cat.setdefault(q.get("category", "other"), []).append(h)
            report["per_question"].append({"id": q["id"], "mode": mode, "hit": h, "rr": round(rr, 3), "top3": ids[:3]})
        report["modes"][mode] = {
            "hit@k": statistics.mean(hits) if hits else 0.0,
            "mrr": statistics.mean(rrs) if rrs else 0.0,
            "by_category": {c: statistics.mean(v) for c, v in by_cat.items()},
        }
    if "dense" in modes or "hybrid" in modes:
        def best(qs):
            out = []
            for q in qs:
                r = retriever.retrieve(q["question"], top_k=1, mode="dense")
                out.append(r.best_dense_score or 0.0)
            return out
        ins, outs = best(in_scope), best(out_scope)
        report["dense_scores"] = {"in_scope_min": min(ins) if ins else None,
                                  "in_scope_mean": statistics.mean(ins) if ins else None,
                                  "out_scope_max": max(outs) if outs else None,
                                  "out_scope_mean": statistics.mean(outs) if outs else None}
    return report


def print_report(model: str, rep: dict, k: int) -> None:
    t = Table(title=f"Retrieval: {model}")
    t.add_column("mode")
    t.add_column(f"hit@{k}", justify="right")
    t.add_column("MRR", justify="right")
    cats = sorted({c for m in rep["modes"].values() for c in m["by_category"]})
    for c in cats:
        t.add_column(c, justify="right")
    for mode, m in rep["modes"].items():
        t.add_row(mode, f"{m['hit@k']:.2f}", f"{m['mrr']:.2f}", *[f"{m['by_category'].get(c, 0):.2f}" for c in cats])
    console.print(t)
    ds = rep.get("dense_scores")
    if ds and ds["in_scope_mean"] is not None:
        console.print(f"  best cosine score: in-scope mean {ds['in_scope_mean']:.3f} (min {ds['in_scope_min']:.3f}), "
                      f"out-of-scope mean {ds['out_scope_mean'] or 0:.3f} (max {ds['out_scope_max'] or 0:.3f})")
        if ds["out_scope_max"] is not None and ds["out_scope_max"] < ds["in_scope_min"]:
            console.print(f"  [green]suggested MIN_DENSE_SCORE={(ds['out_scope_max'] + ds['in_scope_min']) / 2:.3f}[/] "
                          "(cleanly separates in-scope and out-of-scope questions)")
        else:
            console.print("  [yellow]in-scope and out-of-scope scores overlap; leave MIN_DENSE_SCORE=0 and rely on the prompt[/]")
    misses = [p for p in rep["per_question"] if p["mode"] == "hybrid" and not p["hit"]]
    if misses:
        console.print("  hybrid misses: " + ", ".join(f"{p['id']} (got {p['top3'][0] if p['top3'] else '-'})" for p in misses))


def run_answers(golden: list[dict]) -> dict:
    from packageflow_rag.rag import RAG

    rag = RAG()
    rows, ok_in, ok_out, n_in, n_out = [], 0, 0, 0, 0
    try:
        for q in golden:
            t0 = time.perf_counter()
            resp = rag.ask(q["question"]).consume()
            secs = time.perf_counter() - t0
            if q.get("category") == "out_of_scope":
                n_out += 1
                good = resp.not_covered
                ok_out += good
            else:
                n_in += 1
                good = (not resp.not_covered) and bool(resp.cited)
                ok_in += good
            rows.append({"id": q["id"], "ok": bool(good), "secs": round(secs, 1), "cited": resp.cited,
                         "answer": resp.text[:300]})
            console.print(f"  {'[green]ok[/]' if good else '[red]x[/] '} {q['id']} ({secs:.1f}s) {resp.text[:90]!r}")
    finally:
        rag.close()
    summary = {"in_scope_answered_with_citations": f"{ok_in}/{n_in}", "out_of_scope_refused": f"{ok_out}/{n_out}"}
    console.print(summary)
    return {"summary": summary, "rows": rows}


@app.command()
def main(
    models: list[str] = typer.Option(None, "--models", "-m", help="Embedding models to compare (default: EMBED_MODEL)"),
    modes: str = typer.Option("hybrid,dense,bm25", help="Comma list of retrieval modes"),
    k: int = typer.Option(5, help="k for hit@k"),
    answers: bool = typer.Option(False, help="Also generate answers with the chat model (slow)"),
):
    golden = load_golden()
    chunks_path = cfg.resolve(cfg.processed_dir) / "chunks.jsonl"
    if chunks_path.exists():
        chunks = read_chunks_jsonl(chunks_path)
    else:
        chunks, _ = build_chunks(cfg)
        write_chunks_jsonl(chunks, chunks_path)
    known = {c.chunk_id for c in chunks}
    for q in golden:
        for e in q.get("expected_chunks") or []:
            if not any(matches(c, [e]) for c in known):
                console.print(f"[yellow]{q['id']}: expected chunk '{e}' does not exist (chunking changed?)[/]")

    mode_list = [m.strip() for m in modes.split(",") if m.strip()]
    all_reports = {}
    for model in models or [cfg.embed_model]:
        console.print(f"\n[bold]{model}[/]: embedding {len(chunks)} chunks into a temporary index...")
        emb = OllamaEmbedder(model, cfg.ollama_host, cfg.embed_batch_size, cfg.embed_num_ctx, cfg.ollama_timeout)
        try:
            t0 = time.perf_counter()
            vecs = emb.embed_documents([c.text for c in chunks])
            store = VectorStore(None, "eval", in_memory=True)
            store.recreate(len(vecs[0]), model)
            store.upsert(chunks, vecs)
            console.print(f"  embedded in {time.perf_counter() - t0:.1f}s")
            rep = score_retriever(Retriever(store, emb, cfg.candidate_k, cfg.rrf_k), golden, mode_list, k)
        except OllamaError as e:
            console.print(f"[red]{e}[/]")
            continue
        print_report(model, rep, k)
        all_reports[model] = rep

    if answers:
        console.print(f"\n[bold]Answers with {cfg.ollama_chat_model}[/] (uses your real index)")
        try:
            all_reports["answers"] = run_answers(golden)
        except Exception as e:
            console.print(f"[red]{e}[/]")

    out_dir = PROJECT_ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"eval-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(all_reports, indent=2), encoding="utf-8")
    console.print(f"\nSaved {out}")


if __name__ == "__main__":
    sys.exit(app())
