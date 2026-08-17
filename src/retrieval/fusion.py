"""
fusion.py

Combines BM25 (keyword) and vector (semantic) search results into one
ranked list using Reciprocal Rank Fusion (RRF).

Why RRF instead of averaging raw scores:
BM25 scores and cosine similarity scores live on completely different
scales (BM25: unbounded, often 0-15+; cosine: 0-1). Averaging them directly
would let whichever one has bigger numbers dominate, for no principled
reason. RRF sidesteps this entirely by using each result's *rank* instead
of its raw score:

    RRF_score(doc) = sum over each retriever of  1 / (k + rank_in_that_retriever)

A doc that appears near the top of BOTH lists gets a high combined score.
A doc that only one retriever found still gets credit, just less of it.
`k` (typically 60) softens the effect of rank differences further down
each list, so #1 vs #2 matters more than #40 vs #41.

Usage (from project root):
    python -m src.retrieval.fusion "delete a repository"
"""

from __future__ import annotations

from src.retrieval.bm25_search import bm25_search
from src.retrieval.vector_search import vector_search

RRF_K = 60  # standard default from the original RRF paper (Cormack et al.)


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = RRF_K,
    top_k: int = 5,
    labels: list[str] | None = None,
) -> list[dict]:
    """
    ranked_lists: a list of ranked result lists, each a list of dicts with
    at least a 'chunk_id' key, already sorted best-first by that retriever.

    labels: optional name for each list (shown in 'found_by'), e.g.
    ["bm25", "vector"] for plain hybrid search, or
    ["bm25_q1", "vector_q1", "bm25_q2", "vector_q2", ...] for multi-query.
    Defaults to ["bm25", "vector"] when there are exactly 2 lists (backward
    compatible with plain hybrid search), otherwise "source_0", "source_1", etc.

    Returns a single fused ranking, deduped by chunk_id, with an added
    'rrf_score' and 'found_by' field showing which retriever(s) contributed.
    """
    if labels is None:
        labels = ["bm25", "vector"] if len(ranked_lists) == 2 else [f"source_{i}" for i in range(len(ranked_lists))]

    rrf_scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}
    found_by: dict[str, set[str]] = {}

    for list_idx, ranked_list in enumerate(ranked_lists):
        source_name = labels[list_idx]
        for rank, result in enumerate(ranked_list):  # rank is 0-indexed
            chunk_id = result["chunk_id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
            chunk_data[chunk_id] = result  # last writer wins for display fields, fine since text is identical
            found_by.setdefault(chunk_id, set()).add(source_name)

    fused = sorted(rrf_scores.items(), key=lambda kv: -kv[1])[:top_k]

    results = []
    for chunk_id, score in fused:
        r = dict(chunk_data[chunk_id])
        r["rrf_score"] = score
        r["found_by"] = sorted(found_by[chunk_id])
        results.append(r)
    return results


def hybrid_search(
    query: str,
    top_k: int = 5,
    retrieve_k: int = 20,
    qdrant_url: str = "http://localhost:6333",
) -> list[dict]:
    """
    Run BM25 and vector search in parallel (conceptually — sequential here
    for simplicity), fuse with RRF, return the top_k combined results.
    retrieve_k controls how deep each individual retriever searches before
    fusion — wider than top_k so RRF has enough overlap signal to work with.
    """
    bm25_results = bm25_search(query, top_k=retrieve_k)
    vector_results = vector_search(query, top_k=retrieve_k, qdrant_url=qdrant_url)

    return reciprocal_rank_fusion([bm25_results, vector_results], top_k=top_k)


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Run hybrid (BM25 + vector, RRF-fused) search.")
    ap.add_argument("query")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--retrieve-k", type=int, default=20)
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    args = ap.parse_args()

    results = hybrid_search(args.query, top_k=args.top_k, retrieve_k=args.retrieve_k, qdrant_url=args.qdrant_url)

    print(f"\nQuery: {args.query!r}\n")
    for i, r in enumerate(results, 1):
        found = "+".join(r["found_by"])
        print(f"{i}. rrf={r['rrf_score']:.4f}  ({found:11s})  {r['http_method']} {r['api_path']}  [{r['chunk_type']}]")


if __name__ == "__main__":
    main()