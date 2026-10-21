# PackageFlow RAG System

Local Q&A over the PackageFlow project docs (BRD, architecture, data model, API spec, ERD) using Ollama.
Answers come only from the docs, with numbered citations. See **PLAN.md** for the design and decisions.

## Setup (Windows PowerShell, from this folder)

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

ollama pull qwen3-embedding:0.6b       # embeddings (default)
ollama pull nomic-embed-text           # optional, to compare in the eval
ollama list                            # your chat model must be in this list and in .env

python scripts\doctor.py               # checks everything, tells you what to fix
```

## Use it

```powershell
python scripts\ingest.py                                   # build the index (rerun after docs change)
python scripts\ask.py "Who can cancel a package receipt?"  # one question
python scripts\ask.py                                      # chat in the terminal, type exit to quit
python scripts\ask.py "exception_case" --retrieval-only    # see what search finds, no LLM
python scripts\ask.py "What can I do?" --role "Delivery Team Member"
streamlit run app\streamlit_app.py                         # browser chat
```

Stop the Streamlit app before running `ingest.py` (the index can only be open in one program at a time).

## Check quality

```powershell
python scripts\evaluate.py                                        # hit@5 / MRR on 40 golden questions
python scripts\evaluate.py -m qwen3-embedding:0.6b -m nomic-embed-text   # compare embedding models
python scripts\evaluate.py --answers                              # also test the chat model (slow)
python -m pytest                                                  # unit + pipeline tests, no Ollama needed
```

## Add a document

Drop it in `data\source\` (.docx, .md, .yaml) and run `python scripts\ingest.py`. Only changed chunks get re-embedded.
For an image, run `python scripts\describe_image.py "data\source\file.png"`, review the .md it writes to
`data\derived\`, then ingest.

## Layout

```
PackageFlow-RAG-System/
├── PLAN.md                     plan, comparisons, implementation notes
├── requirements.txt
├── .env / .env.example         settings (models, paths, chunk size, top_k)
├── data/
│   ├── source/                 original docs
│   ├── derived/                text made from images (erd.md)
│   └── processed/              chunks.jsonl (exactly what gets indexed)
├── src/packageflow_rag/
│   ├── loaders/                docx, markdown, openapi, image loaders
│   ├── chunking.py             splitting + context headers
│   ├── embeddings.py           Ollama embeddings
│   ├── vectorstore.py          Qdrant local
│   ├── indexing.py             incremental ingest
│   ├── retrieval.py            dense + BM25 + RRF
│   ├── prompts.py              prompt + citations
│   ├── generation.py           Ollama chat (streaming)
│   └── rag.py                  full pipeline + query log
├── scripts/                    ingest, ask, evaluate, doctor, describe_image
├── app/streamlit_app.py        browser UI
├── eval/golden_questions.yaml  test questions
└── tests/                      pytest suite with a fake Ollama server
```

The vector index and query logs live in `~\.packageflow-rag\` (outside OneDrive on purpose).
