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

st.set_page_config(page_title="PackageFlow Docs", page_icon="📦", layout="wide")


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
            mark = "Cited · " if s["cited"] else ""
            scores = []
            if s["dense_rank"]:
                scores.append(f"dense #{s['dense_rank']}")
            if s["bm25_rank"]:
                scores.append(f"keyword #{s['bm25_rank']}")
            if s["dense_score"] is not None:
                scores.append(f"cosine {s['dense_score']:.3f}")
            st.markdown(f"**{mark}[{s['n']}] {s['citation']}**  \n<small>{' · '.join(scores)}</small>",
                        unsafe_allow_html=True)
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
    st.header("📦 PackageFlow Docs")
    meta = rag.store.meta() or {}
    st.caption(f"Chat model: `{settings.ollama_chat_model}`  \nEmbeddings: `{meta.get('embed_model')}`  \n"
               f"{len(rag.retriever.chunks)} chunks · indexed {meta.get('updated_at') or meta.get('created_at')}")

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
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption("Stop this app before running `scripts/ingest.py`.")

# ---------- chat ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

question = st.chat_input("Ask about requirements, roles, API, data model, architecture...")
if not question and st.session_state.get("pending"):
    question = st.session_state.pop("pending")

if not st.session_state.messages and not question:
    st.markdown("### Ask anything about PackageFlow")
    st.caption("Answers come only from the BRD, architecture docs, ERD and API spec, with numbered citations.")
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex, use_container_width=True):
            st.session_state.pending = ex
            st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_sources(m.get("sources", []), show_all)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
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
                with st.spinner("Thinking..."):
                    first = None
                    it = iter(resp)
                    try:
                        first = next(it)
                    except StopIteration:
                        pass
                if first is not None:
                    def _rest():
                        yield first
                        yield from it
                    st.write_stream(_rest())
                else:
                    st.markdown(resp.text)
                if resp.gated:
                    st.caption(f"Best match score {resp.retrieval.best_dense_score:.3f} is below "
                               f"MIN_DENSE_SCORE={settings.min_dense_score}, so the model wasn't asked.")
            sources = source_payload(resp) if not retrieval_only else [
                {"n": i, "cited": False, "citation": rc.chunk.citation, "text": rc.chunk.text,
                 "dense_rank": rc.dense_rank, "bm25_rank": rc.bm25_rank, "dense_score": rc.dense_score}
                for i, rc in enumerate(resp.sources, 1)]
            render_sources(sources, show_all or retrieval_only)
            st.session_state.messages.append({"role": "assistant", "content": resp.text, "sources": sources})
        except OllamaError as e:
            st.error(str(e))
            st.session_state.messages.pop()
