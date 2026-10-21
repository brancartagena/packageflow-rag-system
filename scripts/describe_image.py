"""Turn an image (diagram, screenshot, Figma export) into reviewed-able Markdown in data/derived/.

Usage:
  python scripts/describe_image.py "data/source/PackageFlow App - Overview.png"
  python scripts/describe_image.py diagram.png --model llama3.2-vision --out data/derived/diagram.md

Needs a vision model in Ollama first, e.g.: ollama pull qwen2.5vl:7b
Review the .md by hand, then run: python scripts/ingest.py
"""

import _path  # noqa: F401
import sys
from pathlib import Path

import typer
from rich.console import Console

from packageflow_rag.config import settings
from packageflow_rag.embeddings import OllamaError
from packageflow_rag.loaders.image_loader import describe_image

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(image: Path, out: Path = typer.Option(None), model: str = typer.Option(None)):
    model = model or settings.vision_model
    out = out or settings.resolve(settings.derived_dir) / (image.stem.replace(" ", "_") + ".md")
    console.print(f"Describing {image.name} with {model} (can take a few minutes on CPU)...")
    try:
        path = describe_image(image, out, model, settings.ollama_host)
    except OllamaError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    console.print(f"[green]Saved[/] {path}. Review it, then run: python scripts/ingest.py")


if __name__ == "__main__":
    sys.exit(app())
