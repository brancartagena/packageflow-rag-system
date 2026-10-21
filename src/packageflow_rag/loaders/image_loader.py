"""One-time image to text conversion (diagrams, screenshots, Figma screens).

Uses an Ollama vision model (VISION_MODEL in .env, e.g. qwen2.5vl:7b or
llama3.2-vision) to transcribe an image into Markdown saved under data/derived/.
The file gets a frontmatter 'source_file:' line so ingest knows the image is
covered. ALWAYS review the output by hand before indexing: vision models
misread small text.

The ERD is already done by hand in data/derived/erd.md, so you don't need to
run this on it.
"""

import time
from pathlib import Path

import ollama

from ..embeddings import explain_ollama_error

PROMPT = """Transcribe this diagram or screen into Markdown so it can be searched as text.
- Start with one sentence saying what the image shows.
- List every box, table, screen element or component with ALL of its text, exactly as written.
- For diagrams, list every arrow or line as "A -> B: label".
- Use Markdown headings and tables. Do not add anything that is not in the image."""


def describe_image(image_path: Path, out_path: Path, model: str = "qwen2.5vl:7b",
                   host: str = "http://localhost:11434", timeout: float = 600.0) -> Path:
    client = ollama.Client(host=host, timeout=timeout)
    try:
        resp = client.chat(model=model, messages=[{"role": "user", "content": PROMPT, "images": [str(image_path)]}],
                           options={"temperature": 0})
    except Exception as e:
        raise explain_ollama_error(e, model, host) from e
    body = resp["message"]["content"].strip()
    front = (
        "---\n"
        f"title: {image_path.stem}\n"
        f"source_file: {image_path.name}\n"
        f"converted: {time.strftime('%Y-%m-%d')}\n"
        f"method: Transcribed by {model}. REVIEW BY HAND before indexing.\n"
        "---\n\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(front + body + "\n", encoding="utf-8")
    return out_path
