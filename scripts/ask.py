"""Ask questions from the terminal.

Usage:
  python scripts/ask.py "Who can cancel a package?"
  python scripts/ask.py "What fields does exception_case have?" --doc-type erd
  python scripts/ask.py "What can I do?" --role "Delivery Team Member"
  python scripts/ask.py "idempotency key" --retrieval-only        just show what search finds, no LLM
  python scripts/ask.py                                           interactive chat (type 'exit' to quit)
"""

import _path  # noqa: F401
import sys

import typer
from rich.console import Console
from rich.table import Table

from packageflow_rag.config import settings
from packageflow_rag.embeddings import OllamaError
from packageflow_rag.rag import RAG
from packageflow_rag.vectorstore import IndexLockedError, IndexMismatchError, IndexMissingError

app = typer.Typer(add_completion=False)
console = Console()
DOC_TYPES = ["brd", "tech_arch", "cloud_arch", "api", "erd", "other"]


def show_sources(resp, retrieval_only: bool = False) -> None:
    rows = resp.retrieval.chunks if retrieval_only else resp.sources
    if not rows:
        return
    t = Table(title="Sources" + ("" if retrieval_only else "  (* = cited in the answer)"), show_lines=False)
    t.add_column("#", justify="right")
    t.add_column("Source")
    t.add_column("dense", justify="right")
    t.add_column("bm25", justify="right")
    t.add_column("cos", justify="right")
    for i, rc in enumerate(rows, 1):
        mark = "*" if i in resp.cited else ""
        t.add_row(f"{mark}{i}", rc.chunk.citation, str(rc.dense_rank or "-"), str(rc.bm25_rank or "-"),
                  f"{rc.dense_score:.3f}" if rc.dense_score is not None else "-")
    console.print(t)


def run_one(rag: RAG, question: str, doc_type, role, top_k, retrieval_only, mode, history=None) -> str:
    if retrieval_only:
        result = rag.retrieve(question, top_k, doc_type, role, mode)
        from packageflow_rag.rag import Response
        resp = Response(question=question, retrieval=result)
        show_sources(resp, retrieval_only=True)
        if result.chunks:
            console.rule("Top chunk")
            console.print(result.chunks[0].chunk.text)
        return ""
    resp = rag.ask(question, top_k, doc_type, role, history, mode)
    console.print()
    for tok in resp:
        console.print(tok, end="", markup=False, highlight=False)
    console.print("\n")
    if resp.gated:
        console.print(f"[dim](best match score {resp.retrieval.best_dense_score:.3f} is below "
                      f"MIN_DENSE_SCORE={settings.min_dense_score})[/]")
    show_sources(resp)
    return resp.text


@app.command()
def main(
    question: str = typer.Argument(None, help="Your question. Leave empty for interactive mode."),
    doc_type: list[str] = typer.Option(None, "--doc-type", "-d", help="Limit to: " + ", ".join(DOC_TYPES)),
    role: list[str] = typer.Option(None, "--role", "-r", help='Only API operations this role can call, e.g. "Facilities Lead"'),
    top_k: int = typer.Option(None, help="How many chunks to give the model (default TOP_K from .env)"),
    retrieval_only: bool = typer.Option(False, "--retrieval-only", help="Show search results only, no LLM"),
    mode: str = typer.Option("hybrid", help="hybrid | dense | bm25"),
):
    try:
        rag = RAG()
    except (IndexMissingError, IndexMismatchError, IndexLockedError) as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)

    console.print(f"[dim]chat: {settings.ollama_chat_model} | embed: {settings.embed_model} | "
                  f"{len(rag.retriever.chunks)} chunks[/]")
    try:
        if question:
            run_one(rag, question, doc_type, role, top_k, retrieval_only, mode)
            return
        history: list[dict] = []
        while True:
            try:
                q = console.input("[bold cyan]You:[/] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("exit", "quit", "q"):
                break
            if not q:
                continue
            ans = run_one(rag, q, doc_type, role, top_k, retrieval_only, mode, history)
            history += [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]
    except OllamaError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    finally:
        rag.close()


if __name__ == "__main__":
    sys.exit(app())
