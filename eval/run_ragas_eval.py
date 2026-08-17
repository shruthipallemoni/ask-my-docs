"""
run_ragas_eval.py

Runs the full retrieve -> rerank -> expand -> generate pipeline against the
golden set, then scores it with Ragas metrics using an LLM judge (Groq)
and local bge-m3 embeddings.

IMPORTANT — environment: this script must run in a SEPARATE virtual
environment (see eval/requirements-eval.txt), not the main app's venv.
Ragas' dependency chain conflicts with the main app's newer LangChain 1.x
stack (confirmed by testing — see requirements-eval.txt for the full
explanation). Because of this, we deliberately do NOT import rag_graph.py
here (it imports LangGraph, which needs a newer langchain-core than Ragas
currently supports). Instead, this script composes the same underlying
functions directly — hybrid_search, rerank, expand_to_parent,
generate_answer — which is functionally identical to what the LangGraph
pipeline does, just invoked as a plain sequence instead of through the
state machine. The retrieval/generation LOGIC being evaluated is exactly
the same either way.

Metrics used:
  - Faithfulness: are the answer's claims actually supported by the
    retrieved context? (catches hallucination beyond what citation_validator
    can check, since it doesn't just verify a citation POINTS somewhere
    real — it checks whether the claim is actually TRUE given the context)
  - ResponseRelevancy: does the answer actually address the question asked?
  - LLMContextPrecisionWithReference: are the retrieved chunks relevant to
    the question, using the golden set's reference_answer as ground truth?

Known issues found + fixed during testing:
  1. Ragas' Faithfulness/other metrics default to a "reproducibility"
     setting > 1 (self-consistency: sample the judge N times, take
     majority). Groq's API rejects any n > 1 ('n must be at most 1'),
     causing ~30-40% of judge calls to fail outright. Fixed by forcing
     reproducibility=1 (and strictness=1) on every metric below.
  2. The default RunConfig ran too many concurrent + under-provisioned
     calls, causing LLMDidNotFinishException (max_tokens too low) and
     timeouts under Groq's free-tier rate limits. Fixed with a fully
     serial (max_workers=1), higher max_tokens, longer timeout/retry config.
  3. Do NOT use llama-3.3-70b-versatile as the judge — Groq announced
     (June 17, 2026) it's being fully decommissioned by August 2026 (i.e.
     now). Use openai/gpt-oss-120b instead (see JUDGE_MODEL below).

Rate limit note: Groq's free tier is rate-limited (requests/min AND
requests/day). Faithfulness alone makes several LLM calls per sample
(claim extraction + per-claim verification), so evaluating all 45 golden
set questions across 3 metrics can add up to several hundred calls. Test
on a small --limit first before running the full set.

Usage (from the SEPARATE eval venv, Qdrant running, GROQ_API_KEY set):
    python -m eval.run_ragas_eval --limit 5      # sanity check first
    python -m eval.run_ragas_eval                # full golden set
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from groq import APIStatusError
from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from src.retrieval.query_expansion import expand_query
from src.retrieval.fusion import hybrid_search
from src.reranking.cross_encoder import rerank
from src.retrieval.parent_child import expand_to_parent, format_for_generation, cap_expanded_chunks
from src.generation.generator import generate_answer
from src.generation.citation_validator import validate_citations, strip_citation_tags

# Small pacing delay BETWEEN pipeline-building calls to Groq. TPM (tokens
# per minute) is a ROLLING WINDOW limit across ALL requests, not a
# per-request cap — a real failure found in testing: even with each
# individual request correctly capped under the size budget, running 30+
# sequential generate_answer() calls with zero pacing let the CUMULATIVE
# usage within a 60-second window tip over the 8,000 TPM ceiling, crashing
# the whole run partway through. This delay is a proactive measure;
# _generate_answer_with_backoff below is the reactive safety net for
# whenever pacing alone isn't quite enough.
PACING_DELAY_SECONDS = 8


def _generate_answer_with_backoff(query: str, context_text: str, max_retries: int = 6, base_wait: int = 20) -> str:
    """
    Wraps generate_answer with retry + exponential backoff specifically for
    Groq TPM rate-limit errors (HTTP 413/429). Unlike Ragas' own judge
    calls (which already retry via RunConfig), our own pipeline-building
    calls to generate_answer had ZERO retry logic — a single transient
    rate-limit hit crashed the entire 45-question run. This is the fix.
    """
    for attempt in range(max_retries):
        try:
            return generate_answer(query, context_text)
        except APIStatusError as e:
            is_rate_limit = getattr(e, "status_code", None) in (413, 429) or "rate_limit" in str(e).lower()
            if not is_rate_limit or attempt == max_retries - 1:
                raise
            wait = base_wait * (attempt + 1)
            print(f"    (rate limited — waiting {wait}s before retry {attempt + 1}/{max_retries})")
            time.sleep(wait)
    raise RuntimeError(f"generate_answer failed after {max_retries} retries due to persistent rate limiting")

# Judge model — deliberately DIFFERENT from the generation model
# (openai/gpt-oss-20b). Eval runs are occasional, not per-user-request, so
# we can afford the slower but stronger openai/gpt-oss-120b for more
# reliable scoring — both share the same 8K TPM free-tier budget, so this
# costs nothing extra. IMPORTANT: do not use llama-3.3-70b-versatile —
# Groq announced (June 17, 2026) that it's being fully decommissioned by
# August 2026, i.e. now. It could stop responding at any point.
JUDGE_MODEL = "openai/gpt-oss-120b"

# Phrases that indicate the assistant correctly declined an unanswerable
# question, rather than hallucinating. Used only for the "not_covered"
# category, where there's no expected_endpoint_id to check retrieval
# against — the right behavior IS the answer, not a retrieved chunk.
DECLINE_PHRASES = ["doesn't cover", "does not cover", "not covered", "insufficient context", "cannot find this"]


def _load_retrieval_hits(path: str = "eval/report/retrieval_eval_results.json") -> dict[str, bool]:
    """
    Loads hit_at_5 per golden_id from a prior run_retrieval_eval.py run, so
    we can compare faithfulness/relevancy scores for questions where
    retrieval SUCCEEDED vs FAILED — this is the whole point of checking
    whether generation quality holds up when context is imperfect, not
    just what it looks like on average.
    Returns an empty dict (with a warning) if no prior retrieval eval was found.
    """
    p = Path(path)
    if not p.exists():
        print(f"  (note: no prior retrieval eval found at {path} — run_retrieval_eval.py first "
              f"for a retrieval-hit vs retrieval-miss breakdown. Continuing without it.)")
        return {}
    data = json.loads(p.read_text())
    return {r["id"]: r["hit_at_5"] for r in data["detailed_results"]}


def load_golden_set(path: str = "eval/golden_set.jsonl") -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_pipeline_for_eval(query: str, retrieve_k: int = 25, rerank_top_k: int = 3) -> dict:
    """
    Runs retrieve -> rerank -> expand -> generate for one query, WITHOUT
    LangGraph (see module docstring for why). Returns everything Ragas
    needs plus our own citation validation result for a deterministic
    cross-check alongside the LLM-judged metrics.
    """
    search_query = expand_query(query)
    candidates = hybrid_search(search_query, top_k=retrieve_k)
    reranked = rerank(query, candidates, top_k=rerank_top_k)
    expanded = expand_to_parent(reranked)

    # IMPORTANT: use the SAME budget-capped chunk set the generator actually
    # saw, not the raw uncapped `expanded` list. A real bug found in
    # testing: retrieved_contexts was built from the raw list, so the LLM
    # judge sometimes received MORE context than the generator did — large
    # enough to independently hit Groq's 8K TPM limit and fail outright,
    # even after the generator's own prompt was already fixed to fit.
    capped_chunks = cap_expanded_chunks(expanded)
    context_text = format_for_generation(expanded)

    raw_answer = _generate_answer_with_backoff(query, context_text)
    endpoint_ids = {c["endpoint_id"] for c in expanded}
    citation_result = validate_citations(raw_answer, endpoint_ids)

    return {
        "retrieved_contexts": [c["text"] for c in capped_chunks],
        "response": strip_citation_tags(raw_answer) if citation_result.is_valid else raw_answer,
        "citation_valid": citation_result.is_valid,
        "citation_sparse": citation_result.is_sparsely_cited,
    }


def build_evaluation_dataset(golden_set: list[dict], retrieval_hits: dict[str, bool]) -> tuple[EvaluationDataset, list[dict]]:
    rows = []
    supplementary = []  # our own deterministic checks, kept alongside Ragas's LLM-judged scores

    for entry in golden_set:
        print(f"  running pipeline for {entry['id']}: {entry['query']!r}")
        result = run_pipeline_for_eval(entry["query"])
        time.sleep(PACING_DELAY_SECONDS)  # stay under Groq's rolling 8K TPM window — see module docstring

        rows.append(
            {
                "user_input": entry["query"],
                "retrieved_contexts": result["retrieved_contexts"],
                "response": result["response"],
                "reference": entry["reference_answer"],
            }
        )

        is_not_covered = entry["category"] == "not_covered"
        declined_correctly = None
        if is_not_covered:
            answer_lower = result["response"].lower()
            declined_correctly = any(phrase in answer_lower for phrase in DECLINE_PHRASES)

        supplementary.append(
            {
                "id": entry["id"],
                "category": entry["category"],
                "citation_valid": result["citation_valid"],
                "citation_sparse": result["citation_sparse"],
                "retrieval_hit_at_5": retrieval_hits.get(entry["id"]),  # None if not_covered or no prior eval run
                "declined_correctly": declined_correctly,  # None unless category == not_covered
            }
        )

    return EvaluationDataset.from_list(rows), supplementary


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Run Ragas evaluation against the golden set.")
    ap.add_argument("--golden-set", default="eval/golden_set.jsonl")
    ap.add_argument("--out", default="eval/report/ragas_eval_results.json")
    ap.add_argument("--limit", type=int, default=None, help="Only evaluate the first N golden set entries (for a quick, cheap sanity check)")
    ap.add_argument("--max-workers", type=int, default=1, help="Ragas concurrency — keep at 1 to respect Groq's free-tier TPM limit")
    args = ap.parse_args()

    golden_set = load_golden_set(args.golden_set)
    if args.limit:
        golden_set = golden_set[: args.limit]
        print(f"LIMITED RUN: evaluating only the first {args.limit} golden set entries.\n")

    retrieval_hits = _load_retrieval_hits()

    print(f"Running pipeline for {len(golden_set)} questions...")
    dataset, supplementary = build_evaluation_dataset(golden_set, retrieval_hits)

    print(f"\nScoring with Ragas using {JUDGE_MODEL} as judge (this makes several LLM calls per sample)...")
    evaluator_llm = LangchainLLMWrapper(ChatGroq(model=JUDGE_MODEL, temperature=0, max_tokens=3072))
    evaluator_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="BAAI/bge-m3"))

    metrics = [Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference()]
    for m in metrics:
        # Forces the judge to make exactly ONE call per sub-check instead of
        # Ragas' default self-consistency sampling (n>1), which Groq's API
        # flatly rejects ('n must be at most 1'). This was causing ~30-40%
        # of judge calls to fail outright in earlier testing.
        if hasattr(m, "reproducibility"):
            m.reproducibility = 1
        if hasattr(m, "strictness"):
            m.strictness = 1

    # Fully serial (max_workers=1) so requests don't stack up within the
    # same minute against Groq's free-tier 8K TPM cap; generous timeout and
    # retries so a transient rate-limit or slow response doesn't just drop
    # the job (which was silently degrading the aggregate score earlier).
    run_config = RunConfig(max_workers=args.max_workers, timeout=300, max_retries=10, max_wait=90)

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config,
    )

    result_df = result.to_pandas()

    # Surface any judge-call failures explicitly rather than letting them
    # silently pull down the aggregate — a missing score is a DIFFERENT
    # signal than a low score, and conflating them was the root problem
    # with trusting the very first run's numbers.
    for col in ("faithfulness", "answer_relevancy", "llm_context_precision_with_reference"):
        if col in result_df.columns:
            n_missing = result_df[col].isna().sum()
            if n_missing:
                print(f"  WARNING: {col} has {n_missing}/{len(result_df)} missing scores (judge call failures)")

    # merge in our deterministic citation checks alongside Ragas's LLM-judged scores
    for i, supp in enumerate(supplementary):
        result_df.loc[i, "citation_valid"] = supp["citation_valid"]
        result_df.loc[i, "citation_sparse"] = supp["citation_sparse"]
        result_df.loc[i, "category"] = supp["category"]
        result_df.loc[i, "golden_id"] = supp["id"]
        result_df.loc[i, "retrieval_hit_at_5"] = supp["retrieval_hit_at_5"]
        result_df.loc[i, "declined_correctly"] = supp["declined_correctly"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_json(out_path, orient="records", indent=2)

    print("\n" + "=" * 60)
    print("RAGAS EVAL SUMMARY")
    print("=" * 60)
    numeric_cols = [c for c in result_df.columns if c not in
                    ("user_input", "retrieved_contexts", "response", "reference", "category", "golden_id",
                     "retrieval_hit_at_5", "declined_correctly")]
    for col in numeric_cols:
        if result_df[col].dtype in ("float64", "bool"):
            print(f"  {col}: {result_df[col].mean():.3f}")

    print("\nPer-category breakdown:")
    for cat in sorted(result_df["category"].unique()):
        cat_df = result_df[result_df["category"] == cat]
        print(f"  {cat} (n={len(cat_df)}):")
        for col in numeric_cols:
            if result_df[col].dtype in ("float64", "bool"):
                print(f"    {col}: {cat_df[col].mean():.3f}")

    # --- The comparison the user specifically asked for: does answer
    # quality hold up when retrieval was imperfect, or does it quietly
    # degrade? Reporting this explicitly rather than burying it in an
    # overall average, since a good overall score can hide a real gap here. ---
    scorable = result_df[result_df["retrieval_hit_at_5"].notna()]
    if len(scorable) > 0:
        hits = scorable[scorable["retrieval_hit_at_5"] == True]
        misses = scorable[scorable["retrieval_hit_at_5"] == False]
        print("\n" + "=" * 60)
        print("FAITHFULNESS/QUALITY: RETRIEVAL-HIT vs RETRIEVAL-MISS QUESTIONS")
        print("=" * 60)
        print(f"  Retrieval succeeded (hit@5=True):  n={len(hits)}")
        print(f"  Retrieval failed    (hit@5=False): n={len(misses)}")
        for col in numeric_cols:
            if result_df[col].dtype == "float64" and len(hits) > 0 and len(misses) > 0:
                print(f"    {col}:  hit={hits[col].mean():.3f}   miss={misses[col].mean():.3f}   "
                      f"(diff={hits[col].mean() - misses[col].mean():+.3f})")
        if len(misses) > 0:
            print("\n  Interpretation: a SMALL gap here means the citation validator + faithfulness")
            print("  metric are doing their job — the system stays honest even when retrieval")
            print("  isn't perfect. A LARGE gap (faithfulness collapsing on misses) would mean")
            print("  the model is filling retrieval gaps with confident-sounding fabrication.")
    else:
        print("\n(No prior retrieval eval results found, or no misses in this subset — run "
              "eval.run_retrieval_eval first and pick a --limit that includes some known "
              "failures for a meaningful hit-vs-miss comparison.)")

    # not_covered: did the system correctly decline rather than hallucinate?
    not_covered_df = result_df[result_df["category"] == "not_covered"]
    if len(not_covered_df) > 0:
        n_declined = not_covered_df["declined_correctly"].sum()
        print(f"\n'not_covered' questions correctly declined: {n_declined}/{len(not_covered_df)}")
        if n_declined < len(not_covered_df):
            print("  Questions that did NOT decline correctly (potential hallucination risk):")
            for _, row in not_covered_df[not_covered_df["declined_correctly"] == False].iterrows():
                print(f"    [{row['golden_id']}] {row['user_input']!r}")
                print(f"      answered: {row['response'][:150]}...")

    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()