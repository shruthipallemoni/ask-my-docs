# Ask My Docs — GitHub API Documentation Assistant
### Full Production Design Document

**Corpus:** Official GitHub REST + GraphQL API documentation
**Problem solved:** Developers waste time hunting through docs for endpoints, auth flows, schemas, permissions, rate limits, and working code examples. This system answers those questions directly, with citations back to the exact doc section.

This document is the technical design reference for the build. Each section maps to a real engineering decision with trade-offs — this is what you'd walk an interviewer through.

> **Zero-cost stack, at a glance** — no paid API calls anywhere in this design:
> - **Embeddings:** `bge-m3` (local, via `sentence-transformers`)
> - **Reranker:** `bge-reranker-base` (local, via `sentence-transformers`/`FlagEmbedding`)
> - **LLM (generation):** `openai/gpt-oss-20b` via [Groq](https://console.groq.com)'s free tier — see note below
> - **LLM (eval judge):** `openai/gpt-oss-120b` via Groq free tier — deliberately stronger/slower than generation, since eval runs are occasional
> - **Vector DB:** Qdrant (self-hosted via Docker, free)
> - **Keyword search:** BM25 (`rank_bm25`, pure Python, free)
> - **Eval:** Ragas + a 45-question hand-written golden set (open-source)
> - **Observability:** Langfuse self-hosted or OpenTelemetry (open-source) — not yet wired in
> - **CI:** GitHub Actions (free tier is plenty for this project's scale) — not yet wired in
>
> No paid API calls, but not zero network dependency either — see the note below on the Ollama→Groq pivot.
>
> **Status: this document was written before the build started. Sections below have been updated as real results came in — see the "Build Log" section near the end for what actually happened, including a documented pivot away from local Ollama, real bugs found, and real measured eval numbers.**

---

## 1. System Architecture

Three pipelines, same as before, now specialized for API documentation:

```
┌─────────────── INGESTION (offline) ───────────────┐
GitHub Docs (OpenAPI spec + markdown) 
   → Structural Parser (endpoint/section aware)
   → Chunker (schema-aware, code-block-preserving)
   → Metadata Extractor (method, path, auth, params)
   → Embedder (dual: text + code-aware)
   → Vector Index (Qdrant) + BM25 Index + Metadata Store

┌─────────────── QUERY (online) ─────────────────────┐
User Query 
   → Query Router (is this a "how do I" vs "what does X return" vs code question?)
   → Query Expansion / Multi-Query
   → Hybrid Retrieval (BM25 + Vector) with Metadata Filters
   → Parent-Child Retrieval (expand matched chunk to full endpoint doc)
   → Cross-Encoder Reranker
   → Context Compression
   → LLM Generation (citation-enforced, schema-aware prompt)
   → Citation Validator
   → Response + Sources

┌─────────────── EVAL / CI (continuous) ──────────────┐
Golden Set (real dev questions + expected endpoints/answers)
   → Retrieval metrics + Ragas (faithfulness, relevance)
   → Regression scorecard → GitHub Actions merge gate
```

**Key architectural decision vs. a generic RAG:** the ingestion layer is *structure-aware*. GitHub's docs aren't prose — they're organized around endpoints, each with a method, path, params table, and response schema. Throwing that into a generic 500-token sliding-window chunker destroys the structure that makes the docs useful. Everything downstream (metadata, chunking, retrieval) is built around this.

---

## 2. Data Collection & Preprocessing

**Sources (in priority order):**
1. GitHub's [OpenAPI description](https://github.com/github/rest-api-description) (JSON/YAML) — this is the *ground truth* for every REST endpoint: paths, methods, parameters, request/response schemas, auth scopes. Parsing this directly is far more reliable than scraping rendered HTML.
2. GitHub Docs markdown source (the `github/docs` repo) — for prose explanations, guides, and conceptual pages (auth flows, rate limiting, webhooks) that don't map to a single endpoint.
3. GraphQL schema (`.graphql` SDL / introspection JSON) if you extend to GraphQL.

**Why parse the OpenAPI spec instead of scraping the website:**
| Approach | Pros | Cons |
|---|---|---|
| Scrape rendered docs site | Matches what a human sees | Fragile to HTML changes, loses structured schema, harder to extract exact param types |
| Parse OpenAPI spec directly | Structured, versioned, machine-accurate, includes every param/schema/example | Prose/conceptual pages aren't in the spec — need markdown too |

**Preprocessing steps:**
- Pull spec + docs repo via `git clone` (respects licensing, avoids scraping ToS issues, and gives you clean version history — you can pin to a spec version for reproducible evals).
- Normalize markdown (strip nav/sidebar cruft, keep code fences intact).
- Resolve `$ref` pointers in the OpenAPI spec so each endpoint is a fully self-contained object before chunking.
- Preserve code blocks as atomic units — never split a code example mid-block.

---

## 3. Chunking Strategy for Technical/Code-Heavy Docs

Generic fixed-token chunking is the wrong default here. Use a **hybrid structural strategy**:

**A. Endpoint-level chunking (primary, for the OpenAPI spec)**
One chunk = one endpoint operation (e.g., `POST /repos/{owner}/{repo}/issues`), containing: description, parameters, request body schema, response schema, one representative code example. This keeps everything a developer needs about *one* endpoint together — no split-context problem.

If an endpoint's full documentation exceeds your embedding/context budget, split into logical sub-chunks (Description+Params / Request+Response schema / Code examples) but keep them linked via a shared `endpoint_id` — this is what parent-child retrieval (section 7) is for.

**B. Semantic chunking (for prose/conceptual pages)**
For guides like "Authenticating with GitHub Apps," split by heading hierarchy (H2/H3 sections), not fixed token count. A section is a complete thought; a 512-token window often isn't.

**C. Code block preservation rule**
Never let a chunk boundary fall inside a code fence. If a code example is long, keep it as its own chunk with a back-reference to the parent endpoint, tagged `chunk_type: code_example`.

**Trade-off:** structural chunking requires custom parsing logic upfront (more engineering than `RecursiveCharacterTextSplitter`), but it's the difference between "retrieves a relevant paragraph" and "retrieves the exact endpoint with its full schema" — which is the entire value proposition here.

---

## 4. Metadata Schema per Chunk

This is what makes metadata filtering (section 7) possible. Store per chunk:

```json
{
  "chunk_id": "repos-create-issue-desc",
  "endpoint_id": "POST /repos/{owner}/{repo}/issues",
  "http_method": "POST",
  "api_path": "/repos/{owner}/{repo}/issues",
  "api_category": "Issues",
  "chunk_type": "description | parameters | request_schema | response_schema | code_example | conceptual",
  "auth_required": true,
  "auth_scopes": ["repo", "public_repo"],
  "rate_limit_tier": "primary",
  "required_params": ["owner", "repo", "title"],
  "optional_params": ["body", "assignees", "labels"],
  "response_codes": [201, 404, 410, 422],
  "language": "curl | javascript | python | null",
  "source_url": "https://docs.github.com/en/rest/issues/issues#create-an-issue",
  "spec_version": "2022-11-28",
  "last_updated": "2026-06-01"
}
```

This metadata is what lets you answer "which endpoints don't require auth" or "show me POST endpoints under Issues" without relying on the LLM to infer it from raw text — it's a structured filter, which is more reliable and much cheaper than another retrieval pass.

---

## 5. Embedding Model Selection — Zero-Cost Stack

Every model below runs **locally, free, no API key, no usage cap** — via `sentence-transformers` or Ollama. No OpenAI, Cohere, or Voyage calls anywhere in this design.

| Model | Strength | Trade-off | Run via |
|---|---|---|---|
| **`bge-m3`** (BAAI, recommended) | Natively supports dense + sparse + multi-vector in one model — pairs directly with your hybrid search design; strong on technical text | ~2GB, needs a decent CPU or small GPU for good throughput | `sentence-transformers` (`BAAI/bge-m3`) |
| `bge-large-en-v1.5` | Slightly lighter than bge-m3, very strong general retrieval | Dense-only — you'd run BM25 fully separately instead of getting sparse vectors for free | `sentence-transformers` |
| `nomic-embed-text` | Fast, small, good general quality, easy via Ollama | Slightly weaker than bge-m3 on technical/code text in benchmarks | `ollama pull nomic-embed-text` |
| `jina-embeddings-v2-base-code` | Open-source, code-specialized — useful for the code-example chunks | Narrower domain focus, not ideal for prose chunks | `sentence-transformers` |

**Recommendation:** run **`bge-m3`** as your single embedding model for everything (all local, all free). It's strong enough on both prose and code-adjacent text that you don't need a second specialized model to hit production quality — and it removes the dual-model complexity from Phase 1. If you want a genuinely interesting Phase 3 experiment, add `jina-embeddings-v2-base-code` for the `code_example` chunk type only and benchmark the split-model setup against the single-model baseline on your eval set. Either way it costs $0 — the only resource cost is local compute/time.

**Hardware note:** these models run fine on CPU for a corpus this size (GitHub's API docs are a few thousand chunks, not millions) — you don't need a GPU, just patience during the one-time indexing pass. If you have any GPU available, indexing will just be faster.

### Generation LLM — pivoted from local Ollama to Groq's free API tier (real, documented change)

This section originally recommended local Ollama models (`llama3.1:8b`). **That changed mid-build.** Keeping the original reasoning below for context, followed by what actually happened and why.

**Original plan — local via Ollama:**

| Model | Strength | Trade-off |
|---|---|---|
| `llama3.1:8b` | Strong instruction-following, reliable at structured/citation-enforced output | Needs a reasonably capable CPU; slower than a hosted API |
| `qwen2.5-coder:7b` | Strong on code/schema-heavy content | Larger variant needs more RAM/VRAM |

**What actually happened:** the project pivoted to **Groq's free API tier** instead. Groq is still genuinely free (no credit card, rate-limited rather than metered) but removes the local hardware dependency and is dramatically faster for iterative testing — every retrieval/generation example documented in this README and in the build log below ran through Groq, not a local model.

**Models used, and why two different ones:**
- **Generation: `openai/gpt-oss-20b`** — Groq's fastest general-purpose model, with the most free-tier token-per-minute headroom of the strong options. Good instruction-following for citation-enforced output, though notably looser about *density* of citations than expected (see build log — it tends toward one summary citation rather than one per claim, which motivated `citation_validator.py`'s "sparse citation" tracking separate from "invalid citation").
- **Eval judge: `openai/gpt-oss-120b`** — deliberately a different, stronger, slower model than generation. Eval runs are occasional, not per-user-request, so judge reliability matters more than judge speed, and it shares the same free-tier token budget as the 20b model.

**Important, time-sensitive note found during the build:** Groq deprecates models on a rolling basis. `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` were announced deprecated (June 17, 2026) with full decommissioning by August 2026 — i.e., around when this was being built. Don't default to whatever a tutorial or old blog post recommends; check Groq's current model list before picking one, and prefer the actively-recommended `openai/gpt-oss-*` family.

**Free-tier constraint that caused real bugs (see build log):** `gpt-oss-20b`/`120b` are capped at **8,000 tokens/minute** on the free tier. This isn't a theoretical limit — it caused an actual `HTTP 413` failure once parent-child expansion started pulling in multiple large endpoints' worth of context, and a similar failure in the eval harness when the LLM judge was handed an uncapped context list. Both were fixed with an explicit context-size budget (`cap_expanded_chunks()` in `parent_child.py`) — a good example of context compression being motivated by a real production error rather than built proactively.

**Everything runs through Groq's API** for generation and eval judging; embeddings and reranking stay fully local via `sentence-transformers`. No local LLM hosting, no GPU required, still $0 cost — just a documented trade-off of "free hosted API" over "fully offline local model."

---

## 6. Vector Database Choice

| Option | Why it fits / doesn't |
|---|---|
| **Qdrant** (recommended) | Open-source, excellent metadata filtering (critical here — you need filters on `http_method`, `auth_required`, `api_category`), native hybrid search support, easy local Docker + free cloud tier for demo deployment |
| Weaviate | Also strong metadata filtering + hybrid search; heavier to self-host |
| Pinecone | Fully managed, zero ops, but closed-source and costs money at scale — weaker story for a "prefer open-source" portfolio project |
| Chroma | Great for quick prototyping, weaker at production-scale metadata filtering and horizontal scaling |
| pgvector | If you want "just use Postgres" simplicity and already have relational metadata — very defensible choice, but filtering performance and hybrid search require more manual work |

**Recommendation: Qdrant.** The metadata filtering story (section 4) is central to this project's design, and Qdrant's filter engine plus native BM25+vector hybrid support directly serves that. It's also trivial to run locally via Docker — **free, no cloud account needed** — which keeps your project fully reproducible for reviewers with zero infra cost.

---

## 7. Advanced RAG Techniques — Implementation Notes

**Hybrid Search (BM25 + Vector)**
Run both retrievers in parallel, combine with **Reciprocal Rank Fusion (RRF)**: `score = Σ 1/(k + rank_i)` across retrievers. BM25 alone catches exact matches (`repos/{owner}/{repo}/issues`, error code `422`) that embeddings blur; vector search catches semantic matches ("how do I open a ticket" → issues endpoint). Qdrant supports this natively via its query API.

**Metadata Filtering**
Apply structured filters *before or alongside* vector search — e.g., if the query mentions "without authentication," filter `auth_required: false` first, then search within that subset. This turns fuzzy retrieval into a much sharper problem.

**Parent-Child Retrieval**
Retrieve at the sub-chunk level (e.g., matched the `request_schema` chunk) but return the *full endpoint document* (all sibling chunks sharing `endpoint_id`) to the generator. Small chunks improve retrieval precision; full parent context improves answer completeness. Store a parent→children mapping in your metadata store (or just query by `endpoint_id`).

**Multi-Query Retrieval**
Use the LLM to generate 3–4 paraphrases of the user's query (different phrasings, different technical vocabulary) and retrieve for each, then dedupe/merge results. Helps when a developer's phrasing ("how do I star a repo") doesn't match doc vocabulary ("PUT /user/starred/{owner}/{repo}").

**Query Expansion**
Expand abbreviations and GitHub-specific shorthand (PR → pull request, gh → GitHub CLI) before retrieval — a small synonym/expansion layer specific to this domain adds real value cheaply.

**Reranking**
After hybrid retrieval + RRF, take top ~20 candidates and rerank with **`bge-reranker-base`** (or the larger `bge-reranker-v2-m3` if your hardware allows) — free, self-hosted via `sentence-transformers`/`FlagEmbedding`, no API. Cross-encoders score query+doc jointly, catching precision that bi-encoder similarity misses. Keep top 5 after reranking.

**Context Compression**
Before sending to the LLM, strip retrieved chunks down to the sentences/fields actually relevant to the query (e.g., an LLM-based or extractive compressor). Reduces token cost and reduces the chance the model latches onto irrelevant nearby text.

**Conversational RAG**
Maintain conversation history and rewrite follow-up questions into standalone queries before retrieval (e.g., "what about DELETE?" → "What does the DELETE method do for /repos/{owner}/{repo}/issues/{issue_number}?" using prior turn context). This "query condensation" step is what makes multi-turn RAG actually work.

**Source Citations**
Every generated claim must cite a `chunk_id`/`source_url`. Validate post-generation that cited IDs exist in the retrieved set (see your existing citation validator design) — reject and regenerate if not.

**Agentic RAG (optional, Phase 3 stretch)**
For complex questions ("how do I create a repo and then add a webhook to it"), an agent can decompose into sub-queries, retrieve for each independently, and compose a multi-step answer — potentially even validating against the live GitHub API in a sandboxed call. This is genuinely advanced and worth having as a stretch goal, not a Phase 1 requirement.

---

## 8. End-to-End Retrieval Pipeline

```
1. User query received (+ conversation history)
2. Query condensation (if follow-up) → standalone query
3. Query expansion (GitHub shorthand → full terms)
4. Multi-query generation (2-3 paraphrases)
5. For each query variant:
     a. Vector search (top 20) 
     b. BM25 search (top 20)
     c. Apply metadata filters if query implies them
6. Merge all variant results via RRF → top ~20 unique chunks
7. Parent expansion (pull full endpoint doc for matched chunks)
8. Cross-encoder rerank → top 5
9. Context compression (trim to relevant spans)
10. Assemble prompt: system instructions (cite sources) + compressed context + query
11. LLM generates answer with inline citation markers
12. Citation validator checks markers resolve to real chunk_ids
    → if invalid: regenerate once, then fall back to "insufficient context" response
13. Return answer + resolved source links to user
```

---

## 9. Evaluation Framework

| Metric | How to measure | Tooling |
|---|---|---|
| Retrieval accuracy | Hit rate / MRR / NDCG against a golden set of (query → correct endpoint) pairs | Custom script + your golden set |
| Context relevance | Ragas `context_precision` | Ragas |
| Faithfulness | Ragas `faithfulness` (claims supported by retrieved context) | Ragas |
| Hallucination rate | % of generated citations that fail validation, or claims unsupported by any retrieved chunk | Your citation validator, logged as a metric |
| Latency | p50/p95 for retrieval, reranking, and generation stages separately | OpenTelemetry tracing / Langfuse |
| Answer quality | LLM-as-judge scoring (correctness, completeness) against golden answers, spot-checked by you | Custom eval harness + Ragas `answer_relevancy` |

**Golden set construction:** hand-write ~50–100 realistic developer questions ("how do I list open issues assigned to me," "what scope do I need to delete a repo," "what does a 422 mean on create issue") with the correct endpoint(s) and a reference answer. This is real, tedious work — but it's the single most interview-impressive artifact in the project, because it shows you understand that RAG systems need ground truth to be trustworthy, not just "it looked right when I tried it."

---

## 10. Project Folder Structure

```
ask-my-docs/
├── architecture/
│   └── architecture_diagram.svg
├── data/
│   ├── raw/                    # cloned OpenAPI spec + docs markdown
│   └── processed/              # parsed, chunked, metadata-tagged
├── src/
│   ├── ingestion/
│   │   ├── openapi_parser.py
│   │   ├── markdown_parser.py
│   │   ├── chunker.py
│   │   ├── metadata_extractor.py
│   │   └── indexer.py
│   ├── retrieval/
│   │   ├── vector_search.py
│   │   ├── bm25_search.py
│   │   ├── fusion.py
│   │   ├── metadata_filter.py
│   │   ├── multi_query.py
│   │   └── query_expansion.py
│   ├── reranking/
│   │   └── cross_encoder.py
│   ├── generation/
│   │   ├── prompt_templates.py
│   │   ├── context_compression.py
│   │   ├── generator.py
│   │   ├── citation_validator.py
│   │   └── conversation_manager.py
│   ├── agent/                  # Phase 3
│   │   └── query_planner.py
│   └── api/
│       ├── main.py
│       └── routes/
├── eval/
│   ├── golden_set.jsonl
│   ├── run_retrieval_eval.py
│   ├── run_ragas_eval.py
│   └── report/
├── .github/workflows/eval-gate.yml
├── tests/
│   ├── unit/
│   └── integration/
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 11. Week-by-Week Roadmap

| Week | Focus |
|---|---|
| 1 | Data collection: clone OpenAPI spec + docs repo, build parsers, validate structure |
| 2 | Chunking + metadata extraction pipeline, load into Qdrant + BM25 index |
| 3 | Baseline retrieval: vector search only, basic FastAPI endpoint, manual sanity checks |
| 4 | Hybrid search (BM25 + RRF) + metadata filtering, build first 30-question golden set |
| 5 | Cross-encoder reranking + parent-child retrieval, run first retrieval eval |
| 6 | Citation-enforced generation + citation validator, conversational query condensation |
| 7 | Multi-query retrieval + query expansion + context compression |
| 8 | Full Ragas evaluation harness, CI gating via GitHub Actions |
| 9 | Latency instrumentation (tracing, p50/p95), performance tuning |
| 10 | Agentic decomposition (stretch), deployment, polish README with real benchmark numbers |

---

## 12. Phased Feature Plan

**Phase 1 — Core (must-ship, weeks 1–4)**
Structural ingestion, endpoint-level + semantic chunking, metadata schema, hybrid search (BM25 + vector), metadata filtering, basic generation with citations, minimal golden set.

**Phase 2 — Production hardening (weeks 5–8)**
Cross-encoder reranking, parent-child retrieval, citation validation + regeneration, conversational RAG, multi-query + query expansion, full Ragas eval harness, CI regression gating.

**Phase 3 — Differentiators (weeks 9–10, stretch)**
Context compression, agentic multi-step query decomposition, latency/cost dashboards, live deployment with monitoring, optional dual embedding-model comparison writeup.

This phasing matters for interviews: it shows you can ship a working core fast and layer in sophistication deliberately, rather than over-engineering from day one — which is itself a signal of production judgment.

---

## 13. Deployment & Scalability

- **Containerize** everything with Docker Compose (API + Qdrant) for one-command local reproducibility — this alone matters more than people think for reviewers actually trying your project.
- **Deploy** the API on a small managed host (Fly.io, Railway, or a single EC2/DigitalOcean box) with Qdrant either self-hosted alongside it or on Qdrant Cloud's free tier.
- **Scalability considerations to document (even if you don't fully implement them):**
  - Horizontal scaling of the FastAPI layer behind a load balancer (stateless by design)
  - Qdrant sharding/replication for larger corpora
  - Async ingestion pipeline (queue-based) so re-indexing the doc corpus on GitHub spec updates doesn't block serving
  - Caching layer (e.g., Redis) for repeated/common queries to cut latency and LLM cost
- **Cost control:** track cost-per-query (embedding + rerank + generation) as a first-class metric, not an afterthought — ties back to your observability project if you build both.

---

## 14. Making This Stand Out in Interviews

What actually differentiates this from the hundreds of "chat with PDF" projects reviewers see:

1. **You solved a real, specific problem** — not "chat with any document," but a scoped, well-understood developer pain point with a clear success criterion (correct endpoint, correct auth scope, working code example).
2. **You built retrieval from structured source data**, not scraped prose — parsing the OpenAPI spec directly is a detail that signals real engineering maturity.
3. **You have a golden evaluation set and real numbers** — "faithfulness improved from 0.71 to 0.89 after adding reranking" is a sentence that wins interviews. Vague claims of "it works well" don't.
4. **You made and justified trade-offs** — embedding model choice, vector DB choice, chunking strategy — each with a documented reason, not a default. Be ready to explain what you'd do differently at 10x the corpus size.
5. **You built in failure handling** — citation validation with reject/regenerate, graceful "insufficient context" fallback — shows you designed for the system being wrong, not just for the happy path.
6. **You can talk about the regression gate** — CI blocking a merge because faithfulness dropped is a concrete, memorable story that demonstrates you think about RAG as a system with ongoing quality risk, not a one-time build.

**For your resume:** lead with the outcome and the technique, not the tech stack — e.g., *"Built a hybrid-retrieval RAG system over GitHub's API docs with cross-encoder reranking and citation validation, improving retrieval MRR by X% and reducing hallucinated citations to <Y% on a 100-question eval set."* Numbers beat adjectives every time.

---

## 15. Build Log: Real Issues Found and Fixed

Everything below actually happened during this build, in roughly chronological order. This is deliberately kept as a log, not smoothed over — the debugging is often more interview-worthy than a clean run.

**Ingestion**
- Parsed GitHub's live OpenAPI spec directly (not scraped HTML): 1,209 real endpoints, 51 categories, zero unresolved `$ref` pointers.
- Found early that the spec has **no machine-readable OAuth scope data** — `auth_required` is a documented heuristic (method + path pattern), not ground truth. Flagged explicitly rather than faked.

**BM25 vs. vector search — the case for hybrid, with real numbers**
- `"delete a repository"` ranked the correct endpoint (`DELETE /repos/{owner}/{repo}`) **21st out of 1,529** with BM25 alone — beaten by unrelated endpoints that just repeat "repository" often.
- The same query with **vector search** (`bge-m3`) put it at **#1, score 0.72** — concrete evidence for the hybrid design, not just a design-doc assumption.

**Infra bugs (Qdrant, CPU)**
- Qdrant panicked (`OffsetZero`) when payload indexes were created *after* data was already inserted, on some server versions. Fixed by creating indexes at collection-creation time, before any upsert.
- Local embedding took ~3 hours for 1,529 chunks — diagnosed as PyTorch defaulting to only 2 CPU threads. Fixed with `torch.set_num_threads(cpu_count())`, plus an on-disk embedding cache so a Qdrant rebuild never re-runs the slow step again.

**Reranking earns its complexity**
- RRF fusion (rank-based, not score-based) buried the correct `DELETE /repos/{owner}/{repo}` result outside the top 5 despite it being #1 in raw vector search, because of a middling BM25 rank dragging down its combined score.
- Cross-encoder reranking correctly restored it to **#1 (score 0.9985)** — real before/after evidence that reranking isn't redundant with fusion.

**A real bug in the retrieval code itself**
- `vector_search()` and `bm25_search()` result dicts were missing `endpoint_id` (only had `chunk_id`) — an oversight that only surfaced as a `KeyError` once `parent_child.py` needed it. Fixed at the source in both files.

**Query expansion, measured**
- `"how do I merge a pr"` — BM25 alone was distracted by unrelated `fork-pr-contributor-approval` permission endpoints (score 8.19). With shorthand expansion (`pr` → `pr pull request`), the correct merge endpoint jumped to **#1, score 13.33**.

**Multi-query retrieval — a truncation bug in the LLM call**
- Early testing showed a generated "query variant" that was literally the single word `"How"` — a response truncated mid-sentence by too-low `max_tokens` (200), naively treated as a real paraphrase. Fixed by raising `max_tokens` to 400 and filtering out any variant under 4 words.

**The conceptual-docs gap — found live, root-caused, fixed**
- Asked `"what scopes do I need to create a repository"` twice in the same session. First answer: *"doesn't cover this"* (technically honest, but wrong — the info existed in a retrieved endpoint's prose). Second identical question: a **confidently wrong answer about unrelated Copilot billing scopes** — and it passed citation validation cleanly, because the citation genuinely pointed to a real retrieved chunk, just the *wrong* one. This exposed a real, worth-understanding limitation: **citation validity checks existence, not topical relevance.**
- Root cause: the corpus only had endpoint-level OpenAPI chunks — zero conceptual/guide pages. A generic "what scopes" question had no dedicated home to retrieve.
- Fix: built `markdown_parser.py` against GitHub's real conceptual docs (auth, scopes, rate limits, troubleshooting, webhooks), handling actual Liquid templating (`{% ifversion %}`, `{% data variables.* %}`, `AUTOTITLE` links). Found and fixed a regex bug along the way (hyphens in tag paths weren't matched, silently leaving two duplicate empty chunks).
- Found and fixed a second real bug during this: one conceptual section was **18,090 characters** — encoding it in a batch with normal-sized chunks caused a **5.1GB memory allocation attempt** (attention scales quadratically with sequence length) and crashed. Fixed with paragraph-aware size-capped sub-chunking (2,500-char cap, never splits mid-paragraph or mid-code-block).
- Retested the same failing query after the fix: correct answer, correct scopes, no hallucination.

**An eval harness bug, found via a targeted diagnostic (not a guess)**
- `run_retrieval_eval.py` called `hybrid_search(query, top_k=10)` — truncating the RRF-fused candidate list to 10 *before* the reranker ever got a chance to see a relevant chunk that was ranked 15th in fusion. Built `diagnose_retrieval_miss.py` to trace exactly which pipeline stage dropped a known failure, confirmed the truncation as root cause, and widened the window to 25.
- Result was a genuine, honest trade-off, not a clean win: `auth_scope` Hit Rate@5 improved 66.7% → 83.3%, but `parameter_schema` dropped 83.3% → 66.7% — a wider candidate pool gave the reranker more near-duplicate sibling endpoints to get confused by. Both effects are documented, not just the flattering one.

**Multi-query, tested against real remaining failures**
- Ran the 6 questions still failing after the above fix through multi-query retrieval: **2/6 recovered.** The other 4 each had a distinct, diagnosed cause — a genuine term collision (`/issues` vs. `/issue-fields`), a golden-set question that (in hindsight) targeted an unfairly generic detail shared by dozens of endpoints, a query whose target ranked ~34th even in BM25 alone, and a common status code (`201`) with too little discriminative signal to retrieve on. Multi-query was kept as an optional, user-triggered mode rather than a default, based on this measured 33% recovery rate against its real latency/cost cost.

**Ragas + Groq compatibility — several real, fixed issues**
- Ragas 0.4.x has a broken import against current `langchain-community` (confirmed upstream bug); pinned `ragas==0.3.9`.
- That pin needs an older `langchain-core`, incompatible with the main app's LangGraph-based stack — required building and documenting a **fully separate, pinned eval virtual environment** (`eval/requirements-eval.txt`), and designing `run_ragas_eval.py` to avoid importing `rag_graph.py` at all, composing the same retrieve→rerank→generate functions directly instead.
- Ragas defaults to `n>1` self-consistency sampling per judge call; Groq's API flatly rejects `n>1`. Fixed by forcing `reproducibility=1` / `strictness=1` on every metric.
- The LLM judge's `retrieved_contexts` were built from the *uncapped* chunk list, independently hitting the same 8K-tokens/minute wall that generation had already been fixed for. Fixed by extracting the budget logic into a shared `cap_expanded_chunks()` function used by both the generator and the eval harness — one source of truth for the size cap.
- Caught, before it caused a failure: the first working eval config used `llama-3.3-70b-versatile` as judge — a live search check confirmed Groq had announced (June 17, 2026) full decommissioning of that model by August 2026, i.e. imminently. Switched to `openai/gpt-oss-120b`.
- After all of the above: a 10-question sanity run completed with **zero missing scores** (down from 13/30 individual judge-call failures on the very first attempt) — `Faithfulness: 0.833`, `Answer Relevancy: 0.889`, `Context Precision: 0.808`.

**Scaling to the full 45-question run — two more real constraints, both hit and fixed/understood**
- **A rolling-window (TPM) failure, not a per-request one.** Even with every individual request correctly capped, running 30+ sequential `generate_answer()` calls with zero pacing let *cumulative* usage within Groq's 60-second window exceed the 8K TPM ceiling — the per-request cap was necessary but not sufficient. Fixed with an 8-second pacing delay between pipeline-building calls, plus a genuine retry-with-exponential-backoff wrapper (`_generate_answer_with_backoff`) as a reactive safety net — tested with a simulated failure-then-recovery scenario before trusting it on a real run.
- **A second, sneakier context-budget bug**, found because the retry-with-backoff above kept failing identically across 5 attempts and ~5 minutes of backoff — a strong signal it wasn't transient contention but one specific oversized request. Root cause: `markdown_parser.py`'s earlier oversized-chunk fix split one long conceptual page into 22 small parts, but all 22 kept the *same* synthetic `endpoint_id` (same source doc). `parent_child.py`'s expansion logic treated everything sharing an `endpoint_id` as one inseparable group and — via its own "always keep the top-ranked match in full" exception — let that 22-part group (41,182 chars, ~10,295 tokens) through completely uncapped for exactly the query where it ranked #1. Fixed by capping **per-group**, not just across groups, in `cap_expanded_chunks()` — even the top-ranked match now only contributes as many of its own siblings as fit the budget. Verified directly against the real 22-chunk data that caused the crash: 41,182 chars → 5,492 chars, correctly capped.
- **A third, harder constraint that couldn't be code-fixed**: after both of the above were resolved, a full 45-question × 3-metric run (135 judge calls) completed end-to-end for the first time — but partway through (~30%), Groq's **200,000 tokens/day** cap for `openai/gpt-oss-120b` was exhausted, causing the remaining ~70% of judge calls to fail permanently for that day (not a transient wait — the daily budget simply doesn't reset until the next day). This is a genuine free-tier ceiling, not a bug to fix.
- **Decision:** rather than force a "perfectly clean" full run (which would require either spreading the workload across multiple days or reducing judge model quality), the honest engineering call was to treat the partial 45-question run as **corroborating evidence** alongside the clean 10-question run, since both landed in the same range (Faithfulness 0.833 vs. 0.838, Context Precision 0.808 vs. 0.808 exactly) — two independent samples agreeing is meaningful signal even without 100% coverage. The `not_covered` honesty check (5/5 correctly declined) and the retrieval-hit-vs-miss Answer Relevancy comparison (0.860 vs. 0.720 — a real but moderate gap, not a collapse) both came from the larger partial run and held up. Documented as a real free-tier constraint rather than quietly working around it or pretending the numbers are more complete than they are.