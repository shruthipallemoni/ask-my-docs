"""
run_retrieval_eval.py

Measures RETRIEVAL accuracy against the golden set — does the pipeline
(hybrid search + rerank) actually surface the correct endpoint(s) for each
question? This is deterministic (no LLM judge needed), so it's fast, free,
and fully reproducible — a good first eval to run before the heavier,
LLM-judged Ragas eval.

Metrics:
  - Hit Rate@k: fraction of questions where AT LEAST ONE expected endpoint
    appears in the top-k results (k = 1, 3, 5)
  - MRR (Mean Reciprocal Rank): rewards ranking the correct endpoint HIGHER,
    not just "somewhere in the top 5" — 1/rank of the first correct hit,
    averaged across all questions, 0 if never found.

"not_covered" category entries (expected_endpoint_ids == []) are excluded
from these retrieval metrics — there's no "correct endpoint" to rank
against for a question the corpus genuinely shouldn't answer. They're
counted separately, just for visibility.

Usage (from project root, Qdrant running):
    python -m eval.run_retrieval_eval
"""

from __future__ import annotations

import json
from pathlib import Path

from src.retrieval.fusion import hybrid_search
from src.reranking.cross_encoder import rerank


def load_golden_set(path: str = "eval/golden_set.jsonl") -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_one(entry: dict, retrieve_k: int = 25, rerank_top_k: int = 5) -> dict:
    """
    Runs hybrid search + rerank for one golden set question, and computes
    where (if anywhere) an expected endpoint appears in the results.

    retrieve_k controls how many FUSED candidates survive before reranking
    — this needs to be generously wide (25, not something narrower like 10).
    A real bug found during testing: an earlier version passed a narrow
    top_k straight into hybrid_search's fusion step, truncating the
    candidate list BEFORE the reranker ever got a chance to see and
    correctly re-rank a relevant-but-not-top-10 chunk. The reranker isn't
    the bottleneck here — a too-narrow pre-rerank window is.
    """
    query = entry["query"]
    expected = set(entry["expected_endpoint_ids"])

    candidates = hybrid_search(query, top_k=retrieve_k, retrieve_k=retrieve_k)
    reranked = rerank(query, candidates, top_k=rerank_top_k)

    ranked_endpoint_ids = [r["endpoint_id"] for r in reranked]

    rank_of_first_hit = None
    for i, eid in enumerate(ranked_endpoint_ids, start=1):
        if eid in expected:
            rank_of_first_hit = i
            break

    return {
        "id": entry["id"],
        "query": query,
        "category": entry["category"],
        "expected": sorted(expected),
        "retrieved": ranked_endpoint_ids,
        "rank_of_first_hit": rank_of_first_hit,  # None if not found in top rerank_top_k
        "hit_at_1": rank_of_first_hit == 1,
        "hit_at_3": rank_of_first_hit is not None and rank_of_first_hit <= 3,
        "hit_at_5": rank_of_first_hit is not None and rank_of_first_hit <= 5,
        "reciprocal_rank": (1.0 / rank_of_first_hit) if rank_of_first_hit else 0.0,
    }


def run_eval(golden_set_path: str = "eval/golden_set.jsonl") -> dict:
    golden_set = load_golden_set(golden_set_path)

    scorable = [e for e in golden_set if e["expected_endpoint_ids"]]
    not_covered = [e for e in golden_set if not e["expected_endpoint_ids"]]

    results = []
    for entry in scorable:
        print(f"  evaluating {entry['id']}: {entry['query']!r}")
        results.append(evaluate_one(entry))

    n = len(results)
    hit_rate_1 = sum(r["hit_at_1"] for r in results) / n
    hit_rate_3 = sum(r["hit_at_3"] for r in results) / n
    hit_rate_5 = sum(r["hit_at_5"] for r in results) / n
    mrr = sum(r["reciprocal_rank"] for r in results) / n

    # per-category breakdown
    categories = sorted({r["category"] for r in results})
    per_category = {}
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        cn = len(cat_results)
        per_category[cat] = {
            "count": cn,
            "hit_rate_at_5": sum(r["hit_at_5"] for r in cat_results) / cn,
            "mrr": sum(r["reciprocal_rank"] for r in cat_results) / cn,
        }

    summary = {
        "total_scorable_questions": n,
        "not_covered_questions_excluded": len(not_covered),
        "hit_rate_at_1": round(hit_rate_1, 3),
        "hit_rate_at_3": round(hit_rate_3, 3),
        "hit_rate_at_5": round(hit_rate_5, 3),
        "mrr": round(mrr, 3),
        "per_category": per_category,
        "failures": [
            {"id": r["id"], "query": r["query"], "expected": r["expected"], "retrieved": r["retrieved"]}
            for r in results
            if not r["hit_at_5"]
        ],
    }

    return {"summary": summary, "detailed_results": results}


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Run retrieval accuracy eval against the golden set.")
    ap.add_argument("--golden-set", default="eval/golden_set.jsonl")
    ap.add_argument("--out", default="eval/report/retrieval_eval_results.json")
    args = ap.parse_args()

    print("Running retrieval evaluation...\n")
    results = run_eval(args.golden_set)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    summary = results["summary"]
    print("\n" + "=" * 60)
    print("RETRIEVAL EVAL SUMMARY")
    print("=" * 60)
    print(f"Questions scored: {summary['total_scorable_questions']} "
          f"(+ {summary['not_covered_questions_excluded']} not_covered, excluded)")
    print(f"Hit Rate @1: {summary['hit_rate_at_1']:.1%}")
    print(f"Hit Rate @3: {summary['hit_rate_at_3']:.1%}")
    print(f"Hit Rate @5: {summary['hit_rate_at_5']:.1%}")
    print(f"MRR:         {summary['mrr']:.3f}")
    print("\nPer-category (Hit Rate @5 / MRR):")
    for cat, stats in summary["per_category"].items():
        print(f"  {cat:20s} n={stats['count']:2d}  hit@5={stats['hit_rate_at_5']:.1%}  mrr={stats['mrr']:.3f}")

    if summary["failures"]:
        print(f"\n{len(summary['failures'])} question(s) with NO correct endpoint in top 5:")
        for f in summary["failures"]:
            print(f"  [{f['id']}] {f['query']!r}")
            print(f"      expected: {f['expected']}")
            print(f"      got:      {f['retrieved'][:3]}...")

    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()