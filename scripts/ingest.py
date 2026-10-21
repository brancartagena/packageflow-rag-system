"""Load source docs -> data/processed/chunks.jsonl -> embed with Ollama -> Qdrant index.

Usage:
  python scripts/ingest.py                 incremental: only re-embeds chunks that changed
  python scripts/ingest.py --rebuild       drop the index and embed everything again
  python scripts/ingest.py --chunks-only   just write chunks.jsonl (no Ollama needed), good for checking chunking
"""

import _path  # noqa: F401
import sys
import time
from collections import Counter

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from packageflow_rag.config import settings
from packageflow_rag.embeddings import OllamaEmbedder, OllamaError
from packageflow_rag.indexing import build_chunks, index_chunks, write_chunks_jsonl
from packageflow_rag.vectorstore import IndexLockedError, VectorStore

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    rebuild: bool = typer.Option(False, help="Drop and rebuild the index"),
    chunks_only: bool = typer.Option(False, "--chunks-only", help="Only write chunks.jsonl, skip embedding"),
):
    cfg = settings
    console.print(f"[bold]Source:[/] {cfg.resolve(cfg.source_dir)}")
    console.print(f"[bold]Derived:[/] {cfg.resolve(cfg.derived_dir)}")

    chunks, notes = build_chunks(cfg)
    for n in notes:
        style = "red" if ("FAILED" in n or "SKIPPED" in n) else "dim"
        console.print(f"  [{style}]{n}[/]")
    if not chunks:
        console.print("[red]No chunks. Put the docs in data/source/ first.[/]")
        raise typer.Exit(1)

    out = cfg.resolve(cfg.processed_dir) / "chunks.jsonl"
    write_chunks_jsonl(chunks, out)

    t = Table(title=f"{len(chunks)} chunks -> {out}")
    t.add_column("doc_type")
    t.add_column("chunks", justify="right")
    for k, v in sorted(Counter(c.doc_type for c in chunks).items()):
        t.add_row(k, str(v))
    console.print(t)
    if chunks_only:
        return

    console.print(f"[bold]Embed model:[/] {cfg.embed_model}   [bold]Index:[/] {cfg.resolve(cfg.index_dir)}")
    embedder = OllamaEmbedder(cfg.embed_model, cfg.ollama_host, cfg.embed_batch_size, cfg.embed_num_ctx,
                              cfg.ollama_timeout)
    try:
        store = VectorStore(cfg.resolve(cfg.index_dir), cfg.collection_name)
    except IndexLockedError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)

    t0 = time.perf_counter()
    try:
        with Progress(TextColumn("Embedding"), BarColumn(), TextColumn("{task.completed}/{task.total}"),
                      TimeElapsedColumn(), console=console) as prog:
            task = prog.add_task("embed", total=len(chunks))
            stats = index_chunks(store, embedder, chunks, rebuild=rebuild,
                                 progress=lambda done, total: prog.update(task, completed=done, total=total))
    except OllamaError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    finally:
        store.close()

    console.print(
        f"[green]Done[/] in {time.perf_counter() - t0:.1f}s ({stats.mode}): "
        f"{stats.added} added, {stats.updated} updated, {stats.removed} removed, "
        f"{stats.unchanged} unchanged. Index now has {stats.total} chunks.")


if __name__ == "__main__":
    sys.exit(app())
