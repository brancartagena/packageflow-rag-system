"""PackageFlow docs chat (Phase 5).

Run from the project root:
    streamlit run app/streamlit_app.py

The app keeps the index open while it runs. Stop it (Ctrl+C) before running
scripts/ingest.py, otherwise ingest reports the index is locked.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st  # noqa: E402

from packageflow_rag.config import settings  # noqa: E402
from packageflow_rag.embeddings import OllamaError  # noqa: E402
from packageflow_rag.loaders.openapi_loader import KNOWN_ROLES  # noqa: E402
from packageflow_rag.rag import RAG  # noqa: E402
from packageflow_rag.vectorstore import IndexLockedError, IndexMismatchError, IndexMissingError  # noqa: E402

DOC_TYPE_LABELS = {
    "brd": "BRD",
    "tech_arch": "Tech Architecture & Data Model",
    "cloud_arch": "Cloud System Architecture",
    "api": "API Spec",
    "erd": "ERD",
    "other": "Other",
}
EXAMPLES = [
    "Who can cancel a package receipt?",
    "What columns are in the exception_case table?",
    "Can a package be marked Delivered without a photo?",
    "Which Azure service is proposed for the event bus?",
]

st.set_page_config(page_title="PackageFlow Docs", page_icon="📦", layout="centered",
                   initial_sidebar_state="expanded")

CSS = """
<style>
:root {
  --bg: #FAF9F5; --panel: #F0EEE6; --ink: #1F1E1D; --muted: #73726C;
  --line: #E4E1D6; --accent: #C96442; --accent-soft: #F5E6DF;
}
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] { background: var(--bg); color: var(--ink); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stDecoration"], footer, #MainMenu, [data-testid="stMainMenu"], [data-testid="stAppDeployButton"], .stDeployButton { display: none !important; }
/* keep the arrow that reopens the sidebar visible (hiding the whole toolbar used to hide it too) */
[data-testid="stExpandSidebarButton"], [data-testid="collapsedControl"] { display: flex !important; visibility: visible !important; }
html, body, .stMarkdown, p, li, label, button, input, textarea {
  font-family: -apple-system, "Segoe UI", Inter, Roboto, sans-serif;
}
h1, h2, h3, .pf-greeting { font-family: Georgia, "Iowan Old Style", "Times New Roman", serif; font-weight: 500; letter-spacing: -0.01em; }

/* centered reading column */
.block-container { max-width: 760px; padding-top: 2.5rem; padding-bottom: 7rem; }

/* sidebar */
[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
.pf-brand { display:flex; align-items:center; gap:.6rem; font-family: Georgia, serif; font-size:1.25rem; margin-bottom:.25rem; }
.pf-logo { width:28px; height:28px; border-radius:8px; background:var(--accent); color:#fff; display:grid; place-items:center; font-size:.95rem; }
.pf-meta { color: var(--muted); font-size:.78rem; line-height:1.5; margin-bottom:1rem; }

/* buttons: quiet, rounded */
.stButton > button, [data-testid="stBaseButton-secondary"] {
  border-radius: 12px; border: 1px solid var(--line); background: #fff; color: var(--ink);
  transition: background .15s, border-color .15s; box-shadow: none;
}
.stButton > button:hover { background: var(--accent-soft); border-color: var(--accent); color: var(--ink); }
.pf-new .stButton > button { background: var(--accent); color:#fff; border-color: var(--accent); font-weight:500; }
.pf-new .stButton > button:hover { background:#B5573A; border-color:#B5573A; color:#fff; }

/* empty state */
.pf-hero { text-align:center; margin: 12vh 0 1.5rem; }
.pf-spark { font-size: 2rem; color: var(--accent); line-height:1; }
.pf-greeting { font-size: 2.1rem; margin: .6rem 0 .3rem; }
.pf-sub { color: var(--muted); font-size: .95rem; }
.pf-chips .stButton > button { text-align:left; padding:.7rem 1rem; border-radius:14px; font-size:.9rem; min-height:3.2rem; }

/* messages: user = tinted bubble, assistant = plain text with small mark */
[data-testid="stChatMessage"] { background: transparent; padding: .5rem 0; gap: .75rem; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { justify-content: flex-end; }
[data-testid="stChatMessageAvatarUser"] { display:none; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
  background: var(--panel); border-radius: 18px; padding: .7rem 1.05rem; max-width: 85%; flex: 0 1 auto;
}
[data-testid="stChatMessageAvatarAssistant"] { background: var(--accent); color:#fff; border-radius: 50%; }
[data-testid="stChatMessageContent"] p { line-height: 1.65; font-size: 1rem; }

/* sources */
[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 12px; background:#fff; margin-top:.5rem; }
[data-testid="stExpander"] summary { font-size:.85rem; color: var(--muted); }
.pf-src { font-size:.88rem; margin-top:.6rem; }
.pf-badge { display:inline-block; font-size:.68rem; padding:.05rem .5rem; border-radius:999px; background:var(--accent-soft); color:var(--accent); margin-right:.4rem; vertical-align:middle; }
.pf-badge.off { background: var(--panel); color: var(--muted); }
.pf-score { color: var(--muted); font-size:.75rem; }

/* input: floating pill */
[data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] { background: var(--bg); }
[data-testid="stChatInput"] {
  border: 1px solid var(--line); border-radius: 22px; background:#fff;
  box-shadow: 0 2px 14px rgba(31,30,29,.07);
}
[data-testid="stChatInput"]:focus-within { border-color: var(--accent); }
[data-testid="stChatInput"] textarea { font-size: 1rem; }
[data-testid="stChatInput"] button { background: var(--accent); color:#fff; border-radius: 12px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)



@st.cache_resource(show_spinner="Opening the index...")
def get_rag() -> RAG:
    return RAG()


def source_payload(resp) -> list[dict]:
    out = []
    for i, rc in enumerate(resp.sources, 1):
        out.append({
            "n": i,
            "cited": i in resp.cited,
            "citation": rc.chunk.citation,
            "text": rc.chunk.text,
            "dense_rank": rc.dense_rank,
            "bm25_rank": rc.bm25_rank,
            "dense_score": rc.dense_score,
        })
    out.sort(key=lambda s: (not s["cited"], s["n"]))
    return out


def render_sources(sources: list[dict], show_all: bool) -> None:
    if not sources:
        return
    cited = [s for s in sources if s["cited"]]
    shown = sources if show_all or not cited else cited
    label = f"Sources ({len(cited)} cited, {len(sources)} retrieved)" if cited else f"Sources ({len(sources)} retrieved, none cited)"
    with st.expander(label, expanded=False):
        for s in shown:
            badge = '<span class="pf-badge">cited</span>' if s["cited"] else '<span class="pf-badge off">retrieved</span>'
            scores = []
            if s["dense_rank"]:
                scores.append(f"dense #{s['dense_rank']}")
            if s["bm25_rank"]:
                scores.append(f"keyword #{s['bm25_rank']}")
            if s["dense_score"] is not None:
                scores.append(f"cosine {s['dense_score']:.3f}")
            st.markdown(f'<div class="pf-src">{badge}<b>[{s["n"]}] {s["citation"]}</b><br>'
                        f'<span class="pf-score">{" · ".join(scores)}</span></div>', unsafe_allow_html=True)
            st.code(s["text"], language=None, wrap_lines=True)


# ---------- load ----------
try:
    rag = get_rag()
except (IndexMissingError, IndexMismatchError, IndexLockedError) as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Couldn't start: {e}\n\nRun `python scripts/doctor.py` to see what's missing.")
    st.stop()

# ---------- sidebar ----------
with st.sidebar:
    st.markdown('<div class="pf-brand"><span class="pf-logo">📦</span>PackageFlow Docs</div>', unsafe_allow_html=True)
    meta = rag.store.meta() or {}
    st.markdown(f'<div class="pf-meta">{settings.ollama_chat_model} · {meta.get("embed_model")}<br>'
                f'{len(rag.retriever.chunks)} chunks · indexed {meta.get("updated_at") or meta.get("created_at")}</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="pf-new">', unsafe_allow_html=True)
    if st.button("＋  New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("&nbsp;", unsafe_allow_html=True)

    present = sorted({c.doc_type for c in rag.retriever.chunks})
    doc_types = st.multiselect("Search only in", present, format_func=lambda d: DOC_TYPE_LABELS.get(d, d),
                               placeholder="All documents")
    roles = st.multiselect("API operations for role", KNOWN_ROLES, placeholder="Any role",
                           help="Only keeps API operations this role can call. Other docs are hidden while this is set.")
    top_k = st.slider("Sources given to the model", 3, 15, settings.top_k)
    with st.expander("Advanced"):
        mode = st.radio("Search mode", ["hybrid", "dense", "bm25"], horizontal=True,
                        help="hybrid = meaning + keywords (best). dense = meaning only. bm25 = keywords only.")
        show_all = st.toggle("Show uncited sources too", value=False)
        retrieval_only = st.toggle("Search only (no LLM)", value=False)
    st.caption("Stop this app before running `scripts/ingest.py`.")

# ---------- chat ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

question = st.chat_input("Ask about requirements, roles, API, data model, architecture")
if not question and st.session_state.get("pending"):
    question = st.session_state.pop("pending")

if not st.session_state.messages and not question:
    st.markdown('<div class="pf-hero"><div class="pf-spark">✺</div>'
                '<div class="pf-greeting">What do you want to know about PackageFlow?</div>'
                '<div class="pf-sub">Answers come only from the BRD, architecture docs, ERD and API spec, with numbered citations.</div></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="pf-chips">', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, use_container_width=True, key=f"ex{i}"):
            st.session_state.pending = ex
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar=":material/auto_awesome:" if m["role"] == "assistant" else None):
        if m.get("notice"):
            st.caption(m["notice"])
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_sources(m.get("sources", []), show_all)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant", avatar=":material/auto_awesome:"):
        try:
            if retrieval_only:
                result = rag.retrieve(question, top_k, doc_types or None, roles or None, mode)
                from packageflow_rag.rag import Response
                resp = Response(question=question, retrieval=result, sources=result.chunks)
                resp.text = f"Top {len(result.chunks)} search results (no LLM):"
                st.markdown(resp.text)
            else:
                history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
                resp = rag.ask(question, top_k, doc_types or None, roles or None, history, mode)
                if resp.notice:
                    st.caption(resp.notice)
                if resp.gated:                       # a guardrail answered, the model was not called
                    st.markdown(resp.text)
                    reasons = {
                        "no_term_overlap": "None of the key words in your question appear in the docs I searched.",
                        "no_chunks": "Nothing in the docs matched your question.",
                        "low_score": (f"Best match score {resp.retrieval.best_dense_score or 0:.3f} is below "
                                      f"MIN_DENSE_SCORE={settings.min_dense_score}."),
                    }
                    if resp.gate_reason in reasons:
                        st.caption(reasons[resp.gate_reason])
                else:
                    box = st.empty()
                    it = iter(resp)
                    with st.spinner("Thinking..."):
                        acc = next(it, "")
                    box.markdown(acc + "▌")
                    for tok in it:
                        acc += tok
                        box.markdown(acc + "▌")
                    box.markdown(resp.text)          # the cleaned answer replaces the raw stream
            sources = source_payload(resp) if not retrieval_only else [
                {"n": i, "cited": False, "citation": rc.chunk.citation, "text": rc.chunk.text,
                 "dense_rank": rc.dense_rank, "bm25_rank": rc.bm25_rank, "dense_score": rc.dense_score}
                for i, rc in enumerate(resp.sources, 1)]
            if not retrieval_only and resp.not_covered:
                sources = []                         # "not covered" answers shouldn't list unrelated sources
            render_sources(sources, show_all or retrieval_only)
            st.session_state.messages.append({"role": "assistant", "content": resp.text, "sources": sources,
                                              "notice": getattr(resp, "notice", "")})
        except OllamaError as e:
            st.error(str(e))
            st.session_state.messages.pop()
