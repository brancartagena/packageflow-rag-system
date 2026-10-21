"""Check that everything is set up: Python packages, Ollama, models, docs, index.

Usage: python scripts/doctor.py
"""

import _path  # noqa: F401
import importlib
import sys

from rich.console import Console

from packageflow_rag.config import PROJECT_ROOT, settings as cfg

console = Console()
problems = 0


def ok(msg):
    console.print(f"[green]OK[/]    {msg}")


def bad(msg, fix=None):
    global problems
    problems += 1
    console.print(f"[red]FIX[/]   {msg}")
    if fix:
        console.print(f"      [dim]-> {fix}[/]")


def warn(msg, fix=None):
    console.print(f"[yellow]NOTE[/]  {msg}")
    if fix:
        console.print(f"      [dim]-> {fix}[/]")


def main() -> int:
    console.print(f"[bold]PackageFlow RAG doctor[/]  (project: {PROJECT_ROOT})\n")

    v = sys.version_info
    ok(f"Python {v.major}.{v.minor}.{v.micro}")

    for mod, pkg in [("ollama", "ollama"), ("qdrant_client", "qdrant-client"), ("rank_bm25", "rank-bm25"),
                     ("docx", "python-docx"), ("ruamel.yaml", "ruamel.yaml"), ("pydantic_settings", "pydantic-settings"),
                     ("typer", "typer"), ("rich", "rich"), ("streamlit", "streamlit")]:
        try:
            importlib.import_module(mod)
            ok(f"package {pkg}")
        except ImportError:
            bad(f"package {pkg} is missing", "pip install -r requirements.txt")

    if (PROJECT_ROOT / ".env").exists():
        ok(".env found")
    else:
        warn("no .env, using defaults", "copy .env.example .env")

    # Ollama
    try:
        import ollama
        client = ollama.Client(host=cfg.ollama_host, timeout=10)
        names = {m.model for m in client.list().models}
        ok(f"Ollama running at {cfg.ollama_host} ({len(names)} models)")

        def has(model):
            return model in names or f"{model}:latest" in names

        for label, model in [("embed", cfg.embed_model), ("chat", cfg.ollama_chat_model)]:
            if has(model):
                ok(f"{label} model {model} is pulled")
            else:
                bad(f"{label} model {model} is not pulled", f"ollama pull {model}")
        if cfg.ollama_chat_model.endswith(("1b", ":0.5b", "0.6b")):
            warn(f"chat model {cfg.ollama_chat_model} is very small; expect weak citations and some made-up answers",
                 "try llama3.2:3b or qwen2.5:7b when you can, and set OLLAMA_CHAT_MODEL in .env")
    except Exception as e:
        bad(f"can't reach Ollama at {cfg.ollama_host} ({type(e).__name__})", "start the Ollama app, or run: ollama serve")

    # Docs
    src = cfg.resolve(cfg.source_dir)
    files = [p for p in src.glob("*") if p.is_file() and not p.name.startswith(".")] if src.exists() else []
    if files:
        ok(f"{len(files)} files in {src}")
    else:
        bad(f"no documents in {src}", "move the PackageFlow docs into data/source/")
    if "onedrive" in str(cfg.resolve(cfg.index_dir)).lower():
        warn("INDEX_DIR is inside OneDrive; sync can lock or corrupt it", "set INDEX_DIR=~/.packageflow-rag/index")

    # Index
    try:
        from packageflow_rag.vectorstore import IndexLockedError, VectorStore
        store = VectorStore(cfg.resolve(cfg.index_dir), cfg.collection_name)
        if store.exists():
            meta = store.meta()
            if meta.get("embed_model") == cfg.embed_model:
                ok(f"index has {store.count()} chunks, built with {meta['embed_model']} (updated {meta.get('updated_at') or meta.get('created_at')})")
            else:
                bad(f"index was built with {meta.get('embed_model')} but EMBED_MODEL is {cfg.embed_model}",
                    "python scripts/ingest.py --rebuild")
        else:
            bad("no index yet", "python scripts/ingest.py")
        store.close()
    except IndexLockedError as e:
        warn(str(e))

    console.print()
    if problems:
        console.print(f"[red]{problems} thing(s) to fix.[/]")
        return 1
    console.print("[green]All good. Try: python scripts/ask.py \"Who can cancel a package?\"[/]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
