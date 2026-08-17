# Ask My Docs

**Verifiable Retrieval-Augmented Generation over GitHub's official API documentation.** A production-grade developer assistant that answers endpoint, auth, schema, permissions, and rate-limit questions — with every claim tied to a citable source.

This isn't a generic "chat with PDF" project. It solves a real developer pain point: navigating GitHub's REST API docs to find the right endpoint, the right auth scope, and the right request schema, fast. Ask My Docs goes further than a typical RAG demo: structural ingestion of GitHub's live OpenAPI spec *and* its conceptual documentation, hybrid retrieval, cross-encoder reranking, parent-child retrieval, multi-query and query expansion, enforced source citations that are validated (not just requested), and a real evaluation harness with actual measured numbers — not just claims.

**Full technical design:** see [`DESIGN.md`](DESIGN.md) for the complete architecture, chunking strategy, metadata schema, retrieval pipeline, evaluation framework, and a full log of real bugs found and fixed while building this.

---

## Results, as actually measured

Not aspirational numbers — these are from real eval runs against a 45-question hand-written golden set, with every `expected_endpoint_id` verified against the live corpus.

**Retrieval (`eval/run_retrieval_eval.py`, 40 scorable questions, 5 `not_covered` excluded):**

| Metric | Score |
|---|---|
| Hit Rate @1 | 62.5% |
| Hit Rate @3 | 85.0% |
| Hit Rate @5 | 85.0% |
| MRR | 0.725 |

Per-category Hit Rate@5: `rate_limit` 100%, `endpoint_lookup` 94.7%, `auth_scope` 83.3%, `parameter_schema` 66.7%, `error_code` 60%. Every remaining failure has a diagnosed, documented root cause (see `DESIGN.md`) — sibling-endpoint term collisions, deep BM25 ranks on abstract comparisons, and one golden-set question that turned out to target an unfairly generic detail. None are "it just doesn't work sometimes."

**Generation quality (`eval/run_ragas_eval.py`, LLM-judged by `openai/gpt-oss-120b`):**

Run twice, independently, at different scales — both landed in the same range:

| Metric | 10-question clean run | 45-question run* |
|---|---|---|
| Faithfulness | 0.833 | 0.838 |
| Answer Relevancy | 0.889 | 0.852 |
| Context Precision | 0.808 | 0.808 |

\* *The 45-question run hit a real, hard constraint: Groq's free tier caps `openai/gpt-oss-120b` at 200,000 tokens/day, and the run's judge calls exhausted that mid-evaluation (~30% through), leaving a majority of individual question scores as missing rather than low — a data-availability gap, not a quality signal. Where scores DID complete, they closely corroborate the clean 10-question run above. This is documented as a real, hit-and-understood free-tier limitation in `DESIGN.md`'s Build Log, not glossed over.*

**On the specific question that motivated running this eval:** does faithfulness hold up when retrieval isn't perfect, or does the model quietly fabricate to cover the gap? Answer Relevancy for retrieval-hit vs. retrieval-miss questions: **0.860 vs. 0.720** — a real but moderate gap, not a collapse. And for the 5 `not_covered` questions (things the corpus genuinely shouldn't answer, like GraphQL or 2FA setup) — **5/5 were correctly declined**, not hallucinated.

---

## Why this project exists

Anyone can wire an embedding store to an LLM. What's hard — and what separates a hobby project from a production system — is:

- Retrieval that doesn't miss exact-match terms embeddings blur past
- Answers you can actually trust, because every claim is tied to a real chunk
- A way to know, automatically, when a change makes the system worse

Ask My Docs is built to demonstrate all three, with the receipts to back it up.

---

## Architecture

![Ask My Docs Architecture](architecture/architecture_diagram.svg)

### 1. Ingestion Pipeline (offline)
Two sources, both real, both live: GitHub's **OpenAPI spec** (1,209 endpoints, structurally chunked into description + request-schema pairs) and GitHub's **conceptual documentation** (authentication, scopes, rate limits, troubleshooting, webhooks — parsed from real Liquid-templated markdown, chunked by heading with a hard size cap to avoid pathological oversized chunks). **1,594 total chunks.** Each is embedded (`bge-m3`) into **Qdrant**, and simultaneously indexed into a **BM25 index** for exact-term matching. Embeddings are cached to disk so re-indexing after a Qdrant reset takes seconds, not hours.

### 2. Query / Serving Pipeline (online)
Built as an explicit **LangGraph** state machine: `condense_query → retrieve → rerank → expand_context → generate → validate_citations`, with a conditional retry edge if a citation turns out to be hallucinated. Conversation history rewrites follow-up questions into standalone queries before retrieval; GitHub-specific shorthand (`pr`, `repo`, `org`) gets expanded alongside the original term. Hybrid search (BM25 + vector, RRF-fused) feeds a **cross-encoder reranker**, which feeds **parent-child expansion** (small chunks for retrieval precision, full endpoint docs for generation completeness) — capped to a token budget so a context block can't silently exceed the LLM's rate limit. Generation runs on **Groq's free tier** (`openai/gpt-oss-20b`) under a citation-enforced prompt; the **citation validator** checks every `[cite: ...]` tag actually points to something that was retrieved, and triggers one regeneration attempt if not.

### 3. Evaluation (continuous, currently manual — CI gating not yet wired)
A **45-question hand-written golden set**, every expected answer verified against the real corpus, split across `endpoint_lookup`, `parameter_schema`, `auth_scope`, `error_code`, `rate_limit`, and `not_covered` (questions the corpus genuinely shouldn't answer — testing honest refusal, not just hit rate). Two eval harnesses: a deterministic retrieval-accuracy script (Hit Rate/MRR, no LLM needed) and a Ragas-based faithfulness/relevancy/context-precision harness using a stronger Groq model (`openai/gpt-oss-120b`) as judge.

---

## Tech Stack — 100% Free, Zero API Cost

Every component runs locally or on a genuinely free tier (no credit card). No OpenAI, Anthropic, Cohere, or Voyage API calls anywhere.

| Layer | Choice | Why |
|---|---|---|
| Embeddings | `bge-m3` (local, `sentence-transformers`) | Free, self-hosted, natively supports dense+sparse — no per-call cost |
| LLM (generation) | `openai/gpt-oss-20b` via [Groq](https://console.groq.com) free tier | Free, fast, no local GPU needed — see note below on why not local Ollama |
| LLM (eval judge) | `openai/gpt-oss-120b` via Groq free tier | Deliberately stronger/slower than the generation model — eval runs are occasional, not per-request, so judge quality matters more than judge speed |
| Reranker | `bge-reranker-base` (local, `sentence-transformers`) | Free cross-encoder reranking, no Cohere dependency |
| Vector DB | Qdrant (self-hosted, Docker) | Free, open-source, strong metadata filtering + native hybrid search |
| Keyword search | BM25 (`rank_bm25`) | Pure Python, no infra, catches exact-term misses |
| Orchestration | LangChain / LangGraph | Explicit state machine for the retrieve→generate pipeline, with a real conditional retry edge |
| API | FastAPI | Async-friendly, typed, production-standard |
| Evaluation | Ragas (in an isolated venv — see below) | Purpose-built RAG eval metrics with an LLM judge |
| Containerization | Docker / docker-compose | One-command local Qdrant spin-up |

**Why Groq instead of local Ollama for generation:** the original design used local Ollama models for a fully offline, zero-network stack. Mid-build, the project switched to Groq's free API tier instead — still genuinely free and rate-limited rather than metered, but faster and removes the local hardware dependency. This is documented as a real, deliberate pivot, not silently changed.

**Why a separate environment for evaluation:** Ragas' dependency chain currently conflicts with the main app's LangChain 1.x stack (a confirmed, documented version conflict — see `eval/requirements-eval.txt` for the full explanation and the exact pinned versions that work together). Evaluation tooling runs in its own venv, which is also just good practice — eval dependencies are often heavier and faster-moving than the production app they're evaluating.

See [`DESIGN.md`](DESIGN.md) for the full model comparison, trade-offs, and a running log of real issues found while building this.

---

## Project Structure

```
ask-my-docs/
├── architecture/
│   └── architecture_diagram.svg
├── data/
│   ├── raw/                          # cloned OpenAPI spec (gitignored)
│   └── processed/                    # parsed/chunked/embedded outputs (gitignored)
├── src/
│   ├── ingestion/
│   │   ├── openapi_parser.py         # parses GitHub's live OpenAPI spec, resolves $refs
│   │   ├── chunker.py                # endpoint-level chunking (description + request_schema)
│   │   ├── markdown_parser.py        # parses GitHub's conceptual docs (Liquid template aware)
│   │   ├── merge_chunks.py           # combines endpoint + conceptual chunks
│   │   └── indexer.py                # embeds (bge-m3, cached) + builds BM25 + Qdrant indexes
│   ├── retrieval/
│   │   ├── vector_search.py
│   │   ├── bm25_search.py
│   │   ├── fusion.py                 # Reciprocal Rank Fusion
│   │   ├── metadata_filter.py
│   │   ├── multi_query.py            # LLM-paraphrased query variants
│   │   ├── query_expansion.py        # GitHub/dev shorthand expansion
│   │   └── parent_child.py           # small-chunk retrieval -> full-endpoint generation, budget-capped
│   ├── reranking/
│   │   └── cross_encoder.py
│   ├── generation/
│   │   ├── prompt_templates.py
│   │   ├── generator.py              # Groq (openai/gpt-oss-20b) via langchain-groq
│   │   ├── citation_validator.py     # validity + sparse-citation detection
│   │   ├── conversation_manager.py   # follow-up question condensation
│   │   ├── rag_graph.py              # LangGraph pipeline with citation-retry loop
│   │   └── chat_session.py           # interactive multi-turn CLI
│   └── api/
│       └── main.py                   # FastAPI /query endpoint
├── eval/
│   ├── golden_set.jsonl              # 45 hand-written, corpus-verified questions
│   ├── run_retrieval_eval.py         # deterministic Hit Rate/MRR
│   ├── run_ragas_eval.py             # LLM-judged faithfulness/relevancy/precision
│   ├── diagnose_retrieval_miss.py    # traces a query through every pipeline stage
│   ├── test_multiquery_recovery.py   # targeted re-test of known retrieval failures
│   ├── requirements-eval.txt         # isolated, pinned, Ragas-compatible dependency set
│   └── report/                       # eval output JSON (gitignored)
├── tests/
├── verify_pipeline.py                # one-command sanity check for the ingestion pipeline
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Observability — LangSmith tracing (optional)

Since the pipeline is already built on LangChain/LangGraph, tracing is nearly free to add — no code restructuring needed, just environment variables.

**What it shows:** every node in the LangGraph pipeline (`condense_query → retrieve → rerank → expand_context → generate → validate_citations`) as a step in a visual trace, including the exact prompt sent to Groq, the retrieved chunks, latency per step, and whether the citation-retry loop fired. This is genuinely useful for debugging the exact kind of retrieval-drift issues found during this build (see `DESIGN.md`'s Build Log) — instead of adding print statements, you can see the real intermediate state in the LangSmith UI.

**Setup:**
1. Free account at [smith.langchain.com](https://smith.langchain.com) — Developer tier, no credit card.
2. Set three environment variables:
```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="your-key-here"
export LANGSMITH_PROJECT="ask-my-docs"   # optional, groups traces under this name
```
3. Run anything that calls `run_rag_pipeline()` (`chat_session.py`, the FastAPI `/query` endpoint) — traces appear automatically at smith.langchain.com.

**Honest budget note, same spirit as the Groq rate-limit callouts elsewhere in this doc:** the free Developer tier includes 5,000 traces/month. A single RAG query generates multiple trace entries (one per LangGraph node plus each underlying LLM call), so budget roughly 5-10 trace units per query — meaning realistically a few hundred to ~1,000 real user queries per month before hitting the free tier ceiling. Fine for development and demoing; something to be aware of before pointing real traffic at it.

Tracing is fully optional — if the environment variables aren't set, the app runs exactly as before with zero LangSmith dependency at runtime.

---

## Getting Started

```bash
git clone <repo-url>
cd ask-my-docs
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

docker-compose up -d                                     # starts Qdrant locally

# one-time ingestion (downloads GitHub's live spec + conceptual docs)
python src/ingestion/openapi_parser.py --spec-file data/raw/api.github.com.json --out data/processed/endpoints.json
python src/ingestion/chunker.py
python -m src.ingestion.markdown_parser
python -m src.ingestion.merge_chunks
python -m src.ingestion.indexer --chunks-file data/processed/chunks.json   # embeds with bge-m3, caches to disk

$env:GROQ_API_KEY = "your-free-key-here"                 # console.groq.com/keys, no card needed

python -m src.generation.chat_session                     # interactive multi-turn CLI
# or:
uvicorn src.api.main:app --reload                         # REST API, docs at /docs
```

**Sanity check anytime:** `python verify_pipeline.py` — one-command check that ingestion is producing correct, non-orphaned, deduplicated chunks.

**Running evaluation** requires a *separate* venv (see the Tech Stack section above for why):
```bash
python -m venv .venv-eval && source .venv-eval/bin/activate
pip install -r eval/requirements-eval.txt
python -m eval.run_retrieval_eval
python -m eval.run_ragas_eval --limit 10   # sanity check before the full 45-question run
```

---

## Evaluation Methodology

Two complementary harnesses, both against the same 45-question golden set:

- **Retrieval accuracy** (`run_retrieval_eval.py`) — deterministic Hit Rate@1/3/5 and MRR, no LLM judge needed, fast and free. Includes a targeted diagnostic tool (`diagnose_retrieval_miss.py`) that traces exactly which pipeline stage (BM25, vector, fusion, or rerank) drops a given chunk for a given query — this is how a real eval-harness bug (a too-narrow candidate window truncating results *before* reranking ever saw them) was found and fixed, rather than just observed.
- **Generation quality** (`run_ragas_eval.py`) — Faithfulness, Answer Relevancy, and Context Precision, LLM-judged. Explicitly reports faithfulness scores split by **retrieval-hit vs. retrieval-miss** questions — the point isn't just "what's the average," it's "does answer quality hold up when retrieval isn't perfect, or does the model quietly fabricate to fill the gap." Also tracks whether `not_covered` questions get honestly declined rather than hallucinated.

See `DESIGN.md` for the full list of real bugs found while building and running these harnesses — several were genuinely instructive (a stale local index giving a false-negative diagnosis, an eval script's own truncation bug, a Qdrant version-specific panic, Ragas/Groq API incompatibilities).

---

## Build Roadmap

- [x] Phase 1 — Ingestion: OpenAPI spec parsing, conceptual doc parsing, chunking, embedding, vector + BM25 indexing
- [x] Phase 2 — Hybrid retrieval (RRF), metadata filtering, cross-encoder reranking, parent-child retrieval
- [x] Phase 3 — Multi-query retrieval, query expansion, conversational RAG
- [x] Phase 4 — Citation-enforced generation + validator with regeneration loop, LangGraph orchestration
- [x] Phase 5 — Golden set (45 questions) + retrieval eval + Ragas evaluation harness
- [ ] Phase 6 — CI gating via GitHub Actions (eval harness exists; not yet wired to auto-block PRs)
- [ ] Phase 7 — Deployment, latency/cost dashboards, full 45-question Ragas run

---

## License

MIT