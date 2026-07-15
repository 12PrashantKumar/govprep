# GovPrep AI — Production RAG Assistant for Indian Government Exam Prep

Ask a government-exam question in plain language and get a grounded, source-cited answer drawn from NCERT study material — and a clear *"not in my sources"* when the answer isn't there, instead of a hallucinated guess.

Built end to end to understand production-grade retrieval-augmented generation (RAG): retrieval, evaluation, grounding, observability, and serving — not just a wrapper around an LLM API.

![GovPrep demo](screenshot.png)

**🔗 Live demo:** https://govprep-frontend-55025882120.us-central1.run.app
> ⏳ First load may take up to a minute while the server wakes from idle.

**💻 Code:** https://github.com/12PrashantKumar/govprep

---

## What it does

- Answers exam-prep questions (Polity, History, Geography) grounded **only** in the source material, with source attribution.
- Refuses to answer when the retrieved passages don't support it — reducing hallucination instead of guessing.
- Runs two modes: a fixed RAG pipeline (`/chat`) and an **agentic mode** (`/chat/agent`) that routes between corpus search and live web search.
- Served as a FastAPI backend with a Streamlit web frontend.

## Architecture

```
Streamlit frontend  ──HTTP──>  FastAPI backend  ──>  RAG pipeline  ──>  Postgres + pgvector
  (pg_app.py)                   (pg_api.py)          rewrite ->            (Neon, hybrid
                                /chat, /chat/agent    guardrails ->         dense + BM25)
                                Pydantic-validated    hybrid retrieve ->
                                                      generate
```

The frontend and backend are decoupled services — the UI sends questions over HTTP and renders validated JSON; the backend owns the pipeline.

- **Backend:** FastAPI — `/chat` (RAG pipeline) and `/chat/agent` (agentic ReAct mode)
- **Retrieval:** Hybrid search — dense (pgvector) + sparse (Postgres full-text / BM25) fused with **Reciprocal Rank Fusion (RRF)**
- **Pipeline:** query rewriting (resolves follow-ups) → security guardrails → hybrid retrieval → grounded generation with source citation
- **Agent:** LangGraph ReAct agent with tool-calling (corpus search, live web search, calculator), max-iteration limits, and **graceful fallback to the core RAG pipeline** when tool-calling fails — so the user always gets a grounded answer
- **Security:** red-team tested against OWASP LLM01 (Prompt Injection); rigid SystemMessage isolation blocks jailbreaks and persona overrides (see `SECURITY.md`)
- **Database:** PostgreSQL + pgvector (Neon)
- **Embeddings:** sentence-transformers `all-mpnet-base-v2` (local, 768-dim)
- **LLM:** Groq — Llama 3.3 70B
- **Observability:** LangFuse — every agent step, tool call, token cost, and latency is traced
- **Deployment:** Docker on GCP Cloud Run

## Evaluation

Retrieval quality was measured, not assumed — and generation quality too. Both layers are scored against a 37-question gold set spanning all three subjects, each tagged with a required keyword and expected subject.

| Layer | Metric | Score |
|-------|--------|-------|
| Retrieval | Hit Rate@3 | **0.811** |
| Retrieval | MRR | **0.748** |
| Generation | Faithfulness (LLM-as-a-judge) | **4.68 / 5** |

Retrieval uses **strict** matching — a hit counts only when the correct keyword *and* subject appear in a top-3 chunk — so the raw number understates semantically-correct retrievals. MRR of 0.748 means the correct passage is usually ranked **first**, not merely present. Faithfulness confirms generated answers are grounded in the context actually retrieved. Method and limitations in `EVALUATION.md`.

### What the eval harness caught

The eval loop isn't decoration — it found two real problems that were invisible from the outside:

**1. A quarter of the gold set was unanswerable.** Hit Rate@3 sat at 0.375 and the obvious conclusion was "the retriever is weak." It wasn't. Reading the failures showed the ancient-India questions (Harappa, Ashoka's edicts, Kalinga) had no corresponding source — the corpus held *Themes in World History*, not Indian history. Those questions could never be answered regardless of retrieval quality. Expanding the corpus to Class 12 took Hit Rate@3 from **0.375 → 0.811** and MRR from **0.243 → 0.748**, with no change to the retriever itself.

**2. The embedding model was reloading on every query.** The retriever constructed `HuggingFaceEmbeddings` inside the request path, reloading ~400MB per call. Locally this silently OOM-killed long eval runs; in production it added significant latency to every request. Moving it to module scope fixed both.

Improving retrieval also lifted faithfulness (**4.30 → 4.68**) — better context produces better-grounded answers. The two layers aren't independent, which is exactly why both are measured.

## How it works

**Ingestion** (run once): `PDFs → text extraction → scrubbing → chunking → embeddings → Postgres/pgvector`

Ingestion is **idempotent** — each source is checked before processing, so re-running to add new material never duplicates existing chunks.

**Query** (every question):
```
question + history
   -> rewrite to a self-contained query (resolves follow-ups)
   -> security guardrails (prompt-injection + PII checks)
   -> hybrid retrieve (dense + BM25 + RRF)
   -> build grounded prompt (passages + sources)
   -> generate answer, cited to source
```

## Current corpus

Indexed over NCERT Class 11 and Class 12 textbooks:

- **Polity** — *Indian Constitution at Work* (XI), *Politics in India Since Independence* (XII)
- **History** — *Themes in World History* (XI), *Themes in Indian History* (XII)
- **Geography** — *Fundamentals of Physical Geography* (XI)

The ingestion pipeline loads any text-layer document placed in the subject folders, so more subjects and sources can be added over time — and because ingestion is idempotent, expanding the corpus is a single re-run.

## Project structure

```
govprep/
├── pg_api.py                    # FastAPI entrypoint (/chat, /chat/agent)
├── pg_app.py                    # Streamlit frontend (calls the API over HTTP)
├── scripts/
│   ├── pg_hybrid_retriever.py   # dense (pgvector) + BM25 + RRF retrieval
│   ├── pg_rewriter.py           # query rewriting for follow-ups
│   ├── pg_guardrails.py         # prompt-injection + PII checks
│   ├── pg_ingest.py             # idempotent NCERT PDF ingestion pipeline
│   ├── agent.py                 # LangGraph agent + tools + graceful fallback
│   ├── pg_eval_retrieval.py     # Hit Rate@3 + MRR harness
│   └── pg_eval_ragas_hybrid.py  # generation-quality harness
├── eval/
│   └── gold_set.json            # 37-question evaluation set
├── data/  polity/ history/ geography/   # NCERT PDFs (not committed)
├── Dockerfile.backend
├── SECURITY.md                  # threat model + red-team report
├── EVALUATION.md                # eval method, results, limitations
├── ROADMAP.md                   # planned work
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/12PrashantKumar/govprep.git
cd govprep
python -m venv venv
venv\Scripts\activate          # Windows  (source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
# create a .env with DATABASE_URL, GROQ_API_KEY, TAVILY_API_KEY, and LangFuse keys
```

Ingest the corpus, then run backend + frontend:
```bash
python scripts/pg_ingest.py                 # load, chunk, embed, store (idempotent)
uvicorn pg_api:app --reload                 # terminal 1 — backend
streamlit run pg_app.py                     # terminal 2 — frontend
```

Run the evaluation harnesses:
```bash
python scripts/pg_eval_retrieval.py         # Hit Rate@3 + MRR
python scripts/pg_eval_ragas_hybrid.py      # faithfulness
```

## Roadmap

Planned next steps: further retrieval tuning (chunking strategy, reranking), semantic caching, an explicit agentic router, a CI/CD pipeline, and re-running RAGAS with an OpenAI judge (Groq is incompatible with RAGAS's `n>1` sampling). See `ROADMAP.md`.

## Notes

Source PDFs and API keys are not committed. Built as a learning project to understand production-grade RAG end to end — retrieval, evaluation, grounding, observability, and serving.