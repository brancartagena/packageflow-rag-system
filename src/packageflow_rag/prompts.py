"""Prompt templates.

Kept short and explicit on purpose: small local models (1B to 8B) follow a few
clear rules much better than a long list. Sources are numbered [1], [2], ...
because small models copy numbers more reliably than long section labels.
The app maps the numbers back to document sections under each answer.
"""

from .chunking import approx_tokens
from .models import RetrievedChunk

NOT_COVERED = "That isn't covered in the PackageFlow docs I have."

SYSTEM_PROMPT = f"""You answer questions about the PackageFlow project using ONLY the numbered sources the user gives you.

Rules:
1. Use only the sources. Never use outside knowledge about PackageFlow.
2. After each fact, cite the source number in square brackets, like [1] or [2][3].
3. If the sources do not contain the answer, reply exactly: "{NOT_COVERED}"
4. If two sources disagree, say so and cite both.
5. Be short and direct. Use bullet points for lists."""

ANSWER_TEMPLATE = """Sources:

{context}

Question: {question}

Answer using only the sources above, with [number] citations."""


def format_context(chunks: list[RetrievedChunk], budget_tokens: int | None = None) -> tuple[str, list[RetrievedChunk]]:
    """Number the chunks and cut the list to fit the token budget.

    Returns (context text, chunks actually used). Chunk text already starts with
    its [Doc > Section] header, so the model also sees where each source is from.
    """
    parts, used, total = [], [], 0
    for rc in chunks:
        block = f"[{len(used) + 1}] {rc.chunk.text}"
        t = approx_tokens(block)
        if budget_tokens is not None and used and total + t > budget_tokens:
            break
        parts.append(block)
        used.append(rc)
        total += t
    return "\n\n".join(parts), used


def build_messages(question: str, chunks: list[RetrievedChunk], budget_tokens: int | None = None,
                   history: list[dict] | None = None) -> tuple[list[dict], list[RetrievedChunk]]:
    context, used = format_context(chunks, budget_tokens)
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-4:]:  # a little chat memory, without the old sources
        msgs.append({"role": turn["role"], "content": turn["content"]})
    msgs.append({"role": "user", "content": ANSWER_TEMPLATE.format(context=context, question=question)})
    return msgs, used
