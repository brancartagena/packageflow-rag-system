# PackageFlow RAG System: Implementation Plan

**Goal:** Ask plain-language questions about PackageFlow ("Who can cancel a package?", "What happens when a package has no procurement match?", "Which table stores proof of delivery?") and get answers grounded in the project docs, with citations, running 100% locally on Ollama.

**Status:** Plan v0.2, 2026-10-20. Phases 0 to 6 are implemented. What's left is running it against your real Ollama models and tuning (see section 10).

---

## 1. What we're indexing

I profiled the five source files before writing this. The corpus is small, but it's mixed, and each type needs its own handling.

| File | Type | Size / shape | What matters for RAG |
|---|---|---|---|
| `Package_Delivery_BRD_Draft.docx` | Word | ~1,000 words, 12 H1 sections, 6 tables | Requirements, user stories, risks live **in tables**. Doc status says DRAFT v0.4. |
| `PackageFlow_Technical_Architecture_and_Data_Model.docx` | Word | ~950 words, 15 H1 sections, 9 tables, 1 image | Tech stack, lifecycle states, entity list, jobs, environments are all tables. |
| `PackageFlow_Cloud_System_Architecture.md` | Markdown | ~2,000 words, clean `##`/`###` headings | Design decisions, CAP position, failure paths. Easiest file. |
| `packageflow-api-v08-wip.yaml` | OpenAPI 3.0.3 | 59 paths, 65 operations, 101 schemas, 13 tags (212 KB) | Biggest file. Rich descriptions with **"Who can use this:"** role lines. Heavy `$ref` use. |
| `PackageFlow - ERD.png` | Image | 13 entities with fields and FK relationships | Text is locked in pixels. Has to be converted to text once. |

**Actual chunks:** 355 (192 from the API spec, 48 tech arch, 39 cloud arch, 38 BRD, 38 ERD). That's tiny. It means:

- Any vector DB will be fast. Pick on features and ease of use, not scale.
- Ingestion takes seconds, so re-indexing from scratch on every change is fine.
- **Chunking quality matters way more than infrastructure.** Most of the effort goes there.

---

## 2. Architecture

```
               INGEST (scripts/ingest.py)                                  QUERY (scripts/ask.py / Streamlit)
┌─────────────┐   ┌──────────────┐   ┌────────────┐   ┌──────────┐        ┌──────────┐   ┌──────────────────┐
│ data/source │──▶│ Loaders      │──▶│ Chunker    │──▶│ Embedder │──┐     │ Question │──▶│ Hybrid retriever │
│ docx md yaml│   │ per file type│   │ + metadata │   │ (Ollama) │  │     └──────────┘   │ dense + BM25     │
│ png (ERD)   │   └──────────────┘   └─────┬──────┘   └──────────┘  │                    │ fused with RRF   │
└─────────────┘                            │                        ▼                    └────────┬─────────┘
                                           │               ┌──────────────┐                       │ top k chunks
                                           └──────────────▶│ Qdrant local │◀──────────────────────┘
                                         chunks.jsonl      │ + BM25 index │              ┌──────────────────┐
                                         (inspectable)     └──────────────┘              │ Prompt + Ollama  │
                                                                                         │ chat model       │──▶ Answer + citations
                                                                                         └──────────────────┘
```

**Why hybrid search (dense + keyword):** this corpus is full of exact identifiers like `/packages/{packageId}/match`, `matchPackage`, `exception_case`, `Idempotency-Key`, and lifecycle state names. Embeddings are bad at exact string matching. BM25 is great at it. Combining both with Reciprocal Rank Fusion (RRF) covers both "what does this mean" and "find this exact thing" questions. With ~300 chunks, BM25 runs in memory with `rank-bm25`, no extra service needed.

**Why a custom pipeline instead of LangChain / LlamaIndex:** the hard part here is format-specific chunking (OpenAPI operations, Word tables, ERD entities), and frameworks don't do that well out of the box. A thin pipeline of ~600 lines is easier to debug and understand, and has fewer dependencies. LlamaIndex can be added later if needed.

---

## 3. Chunking strategy (the important part)

Every chunk gets a **context header** prepended before embedding, so a chunk never loses where it came from:

```
[BRD v0.4 > 8. Functional Requirements > FR-03]
Requirement: ... Description: ...
```

### Per source

| Source | Chunk unit | Notes |
|---|---|---|
| Word docs | One chunk per H1/H2 section. Long sections split at ~400 tokens with ~50 token overlap. | **Tables:** convert each row into a labeled sentence (`Risk: X. Impact: Y.`), keep the column headers on every row. Big tables (entity list, lifecycle states) become one chunk per row or small row group. Merged/duplicate header cells (seen in the tech stack table) get de-duplicated. |
| Markdown | One chunk per `##` / `###` section | Keep the parent heading in the context header. |
| OpenAPI | **One chunk per operation** + **one chunk per schema** + one overview chunk per tag | Operation chunk = method, path, operationId, tag, summary, description (incl. "Who can use this"), parameters, request body fields, response codes. Resolve `$ref`s one level deep so field names are inline. Schema chunk = name, description, fields, types, enums, required. |
| ERD image | One chunk per entity + one "relationships" chunk | See ERD note below. |

### ERD conversion (one-time)

The ERD looks like it was rendered from PlantUML. Options, best first:

1. **Use the PlantUML source** if you have it. Perfect text, zero errors.
2. **Transcribe once with a vision model** (`ollama pull qwen2.5vl:7b` or `llama3.2-vision`) into `data/derived/erd.md`, then **review it by hand**. Vision models do misread small field names.
3. Write `erd.md` by hand. 13 entities, maybe 20 minutes.

Whichever path, the derived `erd.md` is what gets indexed, not the PNG. Same idea for the one image inside the Technical Architecture docx: extract it, describe it once, save the description.

### Metadata on every chunk

`chunk_id`, `source_file`, `doc_type` (brd / tech_arch / cloud_arch / api / erd), `doc_version`, `section_path`, `api_method`, `api_path`, `api_tag`, `roles` (parsed from "Who can use this"), `content_hash`.

This enables filters like "only search the API" or "only operations a Package Receiver can call", and lets ingestion skip unchanged chunks later.

---

## 4. Embedding model comparison

All of these run in Ollama with `ollama pull <name>`. Specs are from the model cards; double check the Ollama library page before pulling since tags change.

| Model (Ollama tag) | Size | Dims | Max context | Strengths | Weaknesses | Fit here |
|---|---|---|---|---|---|---|
| **`qwen3-embedding:0.6b`** | 0.6B | up to 1024 | 32K | Top of the MTEB leaderboard for its size. Instruction-aware queries. Long context handles big API schemas without truncation. Bigger 4b/8b versions if you want more quality. | Slower than nomic on CPU. Needs a query instruction prefix for best results. | **Recommended primary** |
| **`nomic-embed-text`** (v1.5) | 137M | 768 | 8K (trained) | The common default. Fast on CPU, small, solid quality. | Needs `search_document:` / `search_query:` prefixes or quality drops. Older than qwen3. | **Baseline / fast fallback** |
| `bge-m3` | 567M | 1024 | 8K | Strong retrieval, multilingual (useful if Spanish content ever shows up). | Its sparse and multi-vector modes are not exposed through Ollama, so you only get dense. Heavier. | Good alternative |
| `snowflake-arctic-embed2` | 568M | 1024 | 8K | Strong retrieval scores, multilingual. | Similar weight to bge-m3, less community usage. | Alternative |
| `embeddinggemma` | 308M | 768 (can truncate to 512/256/128) | 2K | Very efficient for its quality. Built for on-device use. | 2K context is tight for the larger OpenAPI schema chunks. | OK if hardware is weak |
| `mxbai-embed-large` | 335M | 1024 | 512 | Strong on short text. | **512 token limit** will silently truncate API and table chunks. | Not recommended |
| `all-minilm` | 23M to 33M | 384 | 256 to 512 | Fastest, tiny. | Lowest quality, very short context. | Testing only |

**Recommendation:** start with **`qwen3-embedding:0.6b`**, keep **`nomic-embed-text`** as the baseline, and let the eval set (Phase 2) pick the winner. With ~300 chunks, re-embedding with a different model takes under a minute, so switching costs nothing. The embedding model is one line in `.env`.

> **Rule:** the same model must be used for indexing and querying. Changing models means re-running ingest. The index stores the model name and refuses to query with a mismatched one.

---

## 5. Vector database comparison

| Option | How it runs | Metadata filtering | Hybrid search | Ops effort | Azure path later | Verdict |
|---|---|---|---|---|---|---|
| **Qdrant (local mode)** | In-process, files on disk (`QdrantClient(path=...)`). No Docker, no server. | Rich payload filters (match, range, nested) | Native sparse vectors + fusion when we outgrow `rank-bm25` | Very low | Same API as Qdrant server in Docker / Azure Container Apps | **Recommended** |
| Chroma | In-process, files on disk | Basic `where` filters | No native BM25 (do it yourself) | Very low | Server mode exists, less common in enterprise | Close second. Easiest to learn. |
| LanceDB | In-process, columnar files | SQL-like filters | Built-in full-text + hybrid | Very low | Can store on Azure Blob | Good alternative, smaller community |
| FAISS | Library only | None. You manage metadata yourself. | No | Low, but more custom code | None | Too bare for this |
| PostgreSQL + pgvector | Needs a Postgres server | Full SQL | Yes, with Postgres full-text search | Medium | **Azure Database for PostgreSQL** supports pgvector. Matches the Cloud Architecture doc's choice of PostgreSQL. | Great if this RAG grows into the real platform |
| Azure AI Search | Managed cloud service | Yes | Yes, plus semantic ranker | Low ops, costs money | It *is* the Azure path | Production target, not for local dev |
| Milvus | Docker / cluster | Yes | Yes | High | Self-host on AKS | Overkill for 300 chunks |

**Recommendation:** **Qdrant in local mode** for now. No server to run, real metadata filtering (doc_type, role, version), and the exact same client code works against a Qdrant server or swaps cleanly to Azure AI Search later. Chroma is the fallback if you want the simplest possible start.

**Watch out:** this folder is inside **OneDrive**. Database files that are being written while OneDrive syncs them can get locked or corrupted. The index lives **outside OneDrive** by default (`~/.packageflow-rag/index`, set by `INDEX_DIR` in `.env`). Source docs stay in OneDrive, which is fine since they're only read.

---

## 6. Generation (local LLM via Ollama)

- Chat model is set by `OLLAMA_CHAT_MODEL` in `.env`. Run `ollama list` to see what you have. 7B to 14B instruct models (Llama 3.1 8B, Qwen 2.5/3 7B to 14B, Mistral Nemo, Gemma) all work for this.
- **Set `num_ctx` explicitly** (default in config: 8192). Ollama's default context window is small, and it silently drops the start of the prompt when exceeded, which means losing retrieved context with no error.
- `temperature: 0.1`. We want facts, not creativity.
- Prompt rules:
  1. Answer only from the provided context.
  2. Cite every claim as `[source > section]`.
  3. If the answer isn't in the context, say so. Don't guess.
  4. If sources disagree, say which says what. (Example: the Cloud Architecture doc picks PostgreSQL, while the project's Azure direction lists Azure SQL. The bot should surface that, not pick one silently.)
- Stream tokens to the terminal / UI so it feels responsive on CPU.

---

## 7. Implementation phases

### Phase 0: Setup (30 min) (done by you)
- Python venv (3.13 or 3.15 both work), `pip install -r requirements.txt`
- `ollama pull qwen3-embedding:0.6b` and `ollama pull nomic-embed-text`
- Move the 5 source docs into `data/source/`
- Copy `.env.example` to `.env`, set `OLLAMA_CHAT_MODEL`
- **Done when:** `python -m src.packageflow_rag.config` prints settings and Ollama responds.

### Phase 1: Loaders and chunking (done)
- `docx_loader.py` (sections + table rows), `markdown_loader.py`, `openapi_loader.py` (operations, schemas, tags, `$ref` resolution), ERD to `data/derived/erd.md`
- Write all chunks to `data/processed/chunks.jsonl` so you can open and read them
- **Done when:** ~300 chunks exist, every BRD table row and every API operation is its own readable chunk, and tests pass.

### Phase 2: Embeddings, index, eval set (built, needs a run on your machine)
- `embeddings.py` (Ollama `/api/embed`, batched, prefixes per model), `vectorstore.py` (Qdrant local)
- Write `eval/golden_questions.yaml`: 30 to 40 real questions, each with the chunk(s) that should come back. Mix: requirements, roles/permissions, API lookups, data model, architecture decisions.
- `scripts/evaluate.py` reports **hit@5** and **MRR** per embedding model
- **Done when:** the embedding model is chosen based on numbers, not vibes.

### Phase 3: Retrieval (built, needs a run on your machine)
- Dense search + BM25, fused with RRF (k=60), top 6 to 8 chunks
- Optional metadata filters (`--doc-type api`, `--role "Package Receiver"`)
- Re-run eval. Hybrid should beat dense-only on identifier questions.
- **Done when:** hit@5 at or above 0.85 on the golden set.

### Phase 4: Generation (built)
- `prompts.py`, `generation.py` (Ollama chat, streaming, `num_ctx`)
- Show sources under each answer
- **Done when:** answers cite sources and say "not in the docs" for out-of-scope questions (include 5 trick questions in the eval set).

### Phase 5: Interface (built: CLI + Streamlit)
- CLI first: `python scripts/ask.py "who can cancel a package?"`
- Then `app/streamlit_app.py`: chat box, source panel with the retrieved chunks, filters sidebar
- **Done when:** you can demo it.

### Phase 6: Hardening (built)
- Incremental re-ingest using `content_hash`
- Logging of question, retrieved chunk IDs, latency
- Version handling: only the latest API spec (v0.8) is indexed; older versions are skipped by a filename rule
- **Done when:** adding a new doc to `data/source/` and re-running ingest just works.

### Phase 7: Later / optional
- Reranker (`bge-reranker-v2-m3` via `sentence-transformers`). Adds PyTorch (~2 GB), so only if eval shows ranking problems.
- Add the Figma screens and prototype PNGs (vision model descriptions, same as ERD)
- Expose as a FastAPI endpoint
- Azure version: Azure AI Search (hybrid + semantic ranker) + Azure OpenAI, or keep Ollama on a VM / Container Apps with GPU

---

## 8. Risks and how we handle them

| Risk | Impact | Mitigation |
|---|---|---|
| ERD info lost or misread | Wrong data model answers | PlantUML source or hand-reviewed `erd.md` |
| Table-heavy Word docs chunked as flat text | Requirements and risks become unsearchable mush | Row-level table chunking with headers on each row |
| Small local LLM hallucinates | Confident wrong answers | Strict prompt, low temp, citations required, "not in docs" trick questions in eval |
| Ollama context overflow | Context silently dropped | Explicit `num_ctx`, cap retrieved tokens |
| OneDrive sync corrupts index | Broken DB files | Index stored outside OneDrive |
| Doc versions drift (API v0.7 in the Claude project, v0.8 here) | Mixed or stale answers | `doc_version` metadata, index only latest |
| Embedding model swapped without re-ingest | Garbage results | Model name stored with index, checked at query time |

---

## 9. Open questions for you

1. Which chat model do you have in Ollama? (`ollama list`) And roughly what hardware: GPU or CPU only, how much RAM?
2. Do you have the PlantUML (or other) source for the ERD?
3. Who's using this: just you, or your team? (Decides whether Streamlit locally is enough or it needs a shared server.)
4. Should the Figma screen PNGs and the screen coverage doc from the Claude project be added to the corpus?

---

## 10. What was built (implementation notes)

### How to run it

```powershell
pip install -r requirements.txt                 # again: PyYAML was swapped for ruamel.yaml
python scripts\doctor.py                        # checks Ollama, models, docs, index
python scripts\ingest.py                        # docs -> chunks.jsonl -> embeddings -> index
python scripts\evaluate.py                      # retrieval scores on the 40 golden questions
python scripts\evaluate.py -m qwen3-embedding:0.6b -m nomic-embed-text    # pick the embedding model
python scripts\ask.py "Who can cancel a package receipt?"
python scripts\ask.py                           # interactive chat in the terminal
streamlit run app\streamlit_app.py              # browser chat with sources panel
python -m pytest                                # 20 tests, no Ollama needed (uses a fake Ollama server)
```

### Files

| File | What it does |
|---|---|
| `src/packageflow_rag/loaders/` | `docx_loader`, `markdown_loader`, `openapi_loader`, `image_loader`. `__init__.py` detects doc type from the file name, skips images that already have a text version in `data/derived/`, and keeps only the newest version of files like `-v07` / `-v08`. |
| `chunking.py` | Packs paragraphs, list items and tables into chunks of up to 400 tokens without cutting them in half. Adds the `[Doc > Section]` header to every chunk. |
| `embeddings.py` | Ollama embeddings with the right prefixes per model (nomic, qwen3, embeddinggemma, mxbai). Friendly errors for "Ollama not running" and "model not pulled". |
| `vectorstore.py` | Qdrant local mode. Stores which embed model built the index and refuses to query with a different one. |
| `indexing.py` | Incremental ingest: only changed chunks get re-embedded, deleted ones get removed. |
| `retrieval.py` | Dense + BM25 fused with RRF. Filters by doc type and by role (API operations only). |
| `prompts.py`, `generation.py` | Short rule-based prompt with numbered citations `[1]`, streaming answers, explicit `num_ctx`. |
| `rag.py` | The full pipeline used by the CLI, the app and the eval. Logs every question to `~/.packageflow-rag/logs/queries.jsonl`. |
| `scripts/` | `ingest.py`, `ask.py`, `evaluate.py`, `doctor.py`, `describe_image.py` |
| `app/streamlit_app.py` | Chat UI: doc type and role filters, number of sources, search mode, sources panel with the cited chunks first. |
| `eval/golden_questions.yaml` | 35 real questions with the chunks that should come back, plus 5 the docs can't answer. |
| `tests/` | 20 tests, including a fake Ollama server so the whole pipeline is tested without models. |

### Decisions made while building (changes from the plan above)

- **ruamel.yaml instead of PyYAML.** Your venv is Python 3.15, and PyYAML has no prebuilt wheel for 3.15 yet, so pip tries to compile it and fails. ruamel.yaml is pure Python and installs fine. Tested on 3.15.
- **Loaders live in `src/packageflow_rag/loaders/`**, not `data/derived/`. `data/derived/` is only for text made from images (like `erd.md`). The old `.py` copies in `data/derived/` aren't used and can be deleted.
- **Tables:** a table stays in one chunk if it's under 75% of the chunk size; bigger tables (BRD functional requirements, lifecycle states, tech stack, entity list, service mapping) become one chunk per row with the column names on every row.
- **Numbered citations `[1]`** instead of `[BRD > section]` in the answer. Small models copy short numbers much more reliably. The app and CLI map the numbers back to the sections.
- **The image inside the Tech Architecture docx is the same ERD PNG** (identical file), so it's covered by `erd.md` and skipped.
- **`MIN_DENSE_SCORE` gate (off by default).** If the best match is below this cosine score, the bot says "not covered" without calling the LLM. `evaluate.py` suggests a value once it sees real scores. Useful with a 1B chat model, which tends to answer anyway.
- **Logs and index live outside OneDrive** (`~/.packageflow-rag/`).

### What's verified and what isn't

- Verified here: all 20 tests pass on Python 3.13 and 3.15; ingest, ask, evaluate, doctor and the Streamlit app all run end to end against a fake Ollama server; BM25-only retrieval on the real docs scores hit@5 = 0.80 on the golden set.
- **Not verified yet:** real embedding quality and real answers, because Ollama can't run in my cloud sandbox. That's the first thing to do on your machine: `python scripts/evaluate.py -m qwen3-embedding:0.6b -m nomic-embed-text`.

### Next steps

1. Run the commands above and compare the two embedding models. Set the winner as `EMBED_MODEL`.
2. Run `python scripts/evaluate.py --answers` to see how `llama3.2:1b` handles citations and the 5 trick questions. If it struggles, try `llama3.2:3b` or `qwen2.5:7b`.
3. If hybrid hit@5 is under 0.85, look at the misses it prints and adjust chunking or the golden set.
