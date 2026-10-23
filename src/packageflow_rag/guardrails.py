"""Guardrails around the model. Three layers, all plain Python (no extra model calls).

1. Input  (before search):  check_question()
   Drops sentences that ask for something other than PackageFlow docs (coding
   puzzles, poems, "ignore your instructions"). If a real question is left, it
   is answered and the user is told the rest was skipped. If nothing is left,
   the model is never called.

2. Relevance (after search, before the model):  has_term_overlap()
   If none of the question's meaningful words appear in any retrieved chunk,
   the question is not about the docs. This needs no tuning, unlike a cosine
   cut-off (MIN_DENSE_SCORE), and the two work together.

3. Output (after the model):  clean_answer()
   Small local models copy the sources, repeat themselves, cite sources that
   don't exist, or write an answer and then add "not covered". This fixes or
   rejects those answers so the user only sees a short, cited reply.
"""

import re
from dataclasses import dataclass, field

from .prompts import NOT_COVERED

MAX_QUESTION_CHARS = 600

OFF_TOPIC_REPLY = ("I can only answer questions about the PackageFlow docs (BRD, architecture, data model "
                   "and API spec). I can't help with that request.")
SKIPPED_NOTICE = "I skipped the part of your message that isn't about the PackageFlow docs."

# ---------- 1. input ----------

_INJECTION_RE = re.compile(
    r"ignore\s+(all\s+|any\s+|the\s+|your\s+)*(previous|prior|above|earlier|system)\s+(instructions?|rules?|prompts?)"
    r"|disregard\s+.{0,40}(instructions?|rules?|prompt)"
    r"|(reveal|show|print|repeat|output|tell me)\s+.{0,30}(system prompt|your (instructions|rules|prompt))"
    r"|\byou\s+are\s+now\b|\bfrom\s+now\s+on\s+you\b|\bact\s+as\s+(if|though)\b|\bpretend\s+(to\s+be|you)\b"
    r"|\bjailbreak\b|\bdeveloper\s+mode\b|\bDAN\s+mode\b",
    re.I,
)

_LANGS = r"python|javascript|typescript|java|c\+\+|c#|golang|ruby|php|kotlin"
_OFF_TOPIC_RE = re.compile(
    r"\b(leet\s?code|hacker\s?rank|code\s?wars|two[\s-]?sum|fizz\s?buzz|binary\s+search\s+tree)\b"
    rf"|\b(write|solve|implement|generate|code|debug)\b[^.?!]{{0,60}}\b({_LANGS})\b"
    rf"|\b({_LANGS})\s+(script|program|function|code|snippet|class|algorithm)\b"
    r"|\b(write|compose|tell)\b[^.?!]{0,30}\b(poem|story|essay|joke|song|haiku|limerick)\b"
    r"|\b(solve|calculate)\b[^.?!]{0,30}\b(equation|integral|derivative|math\s+problem)\b"
    r"|\b(translate|recipe|horoscope|weather\s+forecast|lottery)\b",
    re.I,
)
_LEAD_CONNECTOR_RE = re.compile(r"^\s*(but|and|also|then|first|so|ok(ay)?|please)\b[\s,]*(before doing so|after that|after)?[\s,]*", re.I)


@dataclass
class QuestionCheck:
    clean: str = ""                    # what is safe to search and send to the model
    blocked: bool = False              # True = do not call the model at all
    reply: str = ""                    # shown to the user when blocked
    notice: str = ""                   # shown above the answer when part of the message was skipped
    dropped: list[str] = field(default_factory=list)


def check_question(question: str) -> QuestionCheck:
    q = " ".join((question or "").split())
    if not any(ch.isalnum() for ch in q):
        return QuestionCheck(blocked=True, reply="Ask me something about the PackageFlow docs.")
    notice = ""
    if len(q) > MAX_QUESTION_CHARS:
        q = q[:MAX_QUESTION_CHARS]
        notice = "Your question was long, so I only read the first part."

    if _INJECTION_RE.search(q):        # an attempt to change the rules: don't try to salvage the rest
        return QuestionCheck(blocked=True, reply=OFF_TOPIC_REPLY, dropped=[q])
    kept, dropped = [], []
    for s in re.split(r"(?<=[.?!])\s+", q):
        (dropped if _OFF_TOPIC_RE.search(s) else kept).append(s)

    clean = " ".join(kept)
    clean = _LEAD_CONNECTOR_RE.sub("", clean).strip()
    if not any(ch.isalnum() for ch in clean):
        return QuestionCheck(blocked=True, reply=OFF_TOPIC_REPLY, dropped=dropped)
    if dropped:
        notice = (notice + " " + SKIPPED_NOTICE).strip()
    return QuestionCheck(clean=clean, notice=notice, dropped=dropped)


# ---------- 2. relevance ----------

_STOP = set("""
a an and are as at be but by can could did do does for from get give has have how i if in into is it its me my
of on or our please should so tell that the their them then there these they this to us was we were what when
where which who whom why will with would you your about any all also been being than via per
""".split())
# Words in almost every chunk of this project: matching on them alone proves nothing.
_COMMON = {"packageflow", "package", "packages", "doc", "docs", "document", "documents", "app", "system"}
_WORD = re.compile(r"[a-z0-9_]{3,}")


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def _terms(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP}


def has_term_overlap(question: str, chunk_texts: list[str]) -> bool:
    """True if at least one distinctive question word appears in the retrieved text.

    Questions made only of common words ("What is PackageFlow?") always pass.
    """
    q_terms = _terms(question)
    distinctive = {t for t in q_terms if t not in {_stem(c) for c in _COMMON}}
    if not distinctive:
        return True
    haystack: set[str] = set()
    for t in chunk_texts:
        haystack |= _terms(t)
    return bool(distinctive & haystack)


# ---------- 3. output ----------

_HEADER_RE = re.compile(r"\[[^\[\]\n]{0,200}>[^\[\]\n]{0,200}\]\s*")   # echoed "[Doc > Section]" headers
_CITE_RE = re.compile(r"\[(\d{1,2})\]")
_LEAD_CITE_RE = re.compile(r"^\s*((?:\[\d{1,2}\]\s*)+)(.*)$")
_NC_RE = re.compile(r"that isn'?t covered in the packageflow docs i have\.?", re.I)


def _valid_cites(text: str, n_sources: int) -> list[int]:
    out: list[int] = []
    for m in _CITE_RE.finditer(text):
        n = int(m.group(1))
        if 1 <= n <= n_sources and n not in out:
            out.append(n)
    return out


def clean_answer(text: str, n_sources: int, require_citations: bool = True) -> tuple[str, list[str]]:
    """Return (answer to show, list of fixes applied). Fixes are logged to help tune the prompt."""
    fixes: list[str] = []
    t = (text or "").strip()

    t2 = _HEADER_RE.sub("", t)
    if t2 != t:
        fixes.append("removed echoed source headers")
        t = t2

    def _keep(m):
        return m.group(0) if 1 <= int(m.group(1)) <= n_sources else ""
    t2 = _CITE_RE.sub(_keep, t)
    if t2 != t:
        fixes.append("removed citations to sources that don't exist")
        t = t2

    # "[1] sentence" lines -> "sentence [1]"; repeated lines are shown once with all their citations
    extra: dict[str, list[int]] = {}
    shown: dict[str, str] = {}
    for line in (ln.strip() for ln in t.splitlines()):
        m = _LEAD_CITE_RE.match(line)
        lead = [int(c) for c in _CITE_RE.findall(m.group(1))] if m else []
        body = (m.group(2) if m else line).strip()
        key = re.sub(r"(\s*\[\d{1,2}\])+", "", body).lower().strip(" .")
        if not key:
            continue
        if key in shown:
            fixes.append("merged repeated lines")
            lead += [int(c) for c in _CITE_RE.findall(body)]
        else:
            shown[key], extra[key] = body, []
        extra[key] += [c for c in lead if c not in extra[key]]
    if shown:
        lines = []
        for key, body in shown.items():
            present = {int(c) for c in _CITE_RE.findall(body)}
            more = "".join(f"[{n}]" for n in extra[key] if n not in present)
            lines.append(f"{body} {more}".strip())
        t = "\n".join(lines)

    cites = _valid_cites(t, n_sources)
    if _NC_RE.search(t):
        rest = _NC_RE.sub("", t).strip()
        if cites and rest:
            fixes.append("removed 'not covered' sentence that contradicted a cited answer")
            t = rest
        else:
            return NOT_COVERED, fixes
    if require_citations and not cites:
        fixes.append("answer had no citations, replaced with 'not covered'")
        return NOT_COVERED, fixes
    return t.strip(), fixes
