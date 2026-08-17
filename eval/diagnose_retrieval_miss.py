"""
diagnose_retrieval_miss.py

Targeted diagnostic for a specific retrieval failure: traces exactly where
a given expected chunk/endpoint ranks at EACH stage of the pipeline
(BM25 alone, vector alone, RRF-fused, after rerank) for a given query.

This is the difference between "the pipeline missed it" and "we know
EXACTLY which stage dropped it" — necessary before deciding whether the
fix is a bigger retrieve_k, weighted fusion, better chunking, or something
else entirely.

Usage (from project root, Qdrant running):
    python -m eval.diagnose_retrieval_miss "What scopes are needed to create a private repository?" "apps/oauth-apps/building-oauth-apps/scopes-for-oauth-apps"
"""

from src.retrieval.bm25_search import bm25_search
from src.retrieval.vector_search import vector_search
from src.retrieval.fusion import reciprocal_rank_fusion
from src.reranking.cross_encoder import rerank


def find_rank(results: list[dict], target_substring: str) -> list[tuple[int, str, float]]:
    """Returns (rank, chunk_id, score) for every result whose chunk_id/endpoint_id contains target_substring."""
    hits = []
    for i, r in enumerate(results, 1):
        haystack = r.get("endpoint_id", "") + r.get("chunk_id", "")
        if target_substring in haystack:
            score = r.get("score") or r.get("rrf_score") or r.get("rerank_score") or 0.0
            hits.append((i, r["chunk_id"], score))
    return hits


def diagnose(query: str, target_substring: str, retrieve_k: int = 25):
    print(f"Query: {query!r}")
    print(f"Looking for: chunks matching {target_substring!r}\n")

    print("--- Stage 1: BM25 alone (top 100) ---")
    bm25_results = bm25_search(query, top_k=100)
    hits = find_rank(bm25_results, target_substring)
    print(f"  Found at: {hits if hits else 'NOT in top 100'}\n")

    print(f"--- Stage 2: Vector search alone (top {retrieve_k}) ---")
    vector_results = vector_search(query, top_k=retrieve_k)
    hits = find_rank(vector_results, target_substring)
    print(f"  Found at: {hits if hits else f'NOT in top {retrieve_k}'}\n")

    print(f"--- Stage 3: RRF-fused (BM25 top {retrieve_k} + vector top {retrieve_k}) ---")
    bm25_for_fusion = bm25_search(query, top_k=retrieve_k)
    fused = reciprocal_rank_fusion([bm25_for_fusion, vector_results], top_k=retrieve_k)
    hits = find_rank(fused, target_substring)
    print(f"  Found at: {hits if hits else 'NOT in fused results'}\n")

    print("--- Stage 4: After cross-encoder rerank (top 5) ---")
    reranked = rerank(query, fused, top_k=5)
    hits = find_rank(reranked, target_substring)
    print(f"  Found at: {hits if hits else 'NOT in final top 5'}\n")

    print("Top 5 final results for reference:")
    for i, r in enumerate(reranked, 1):
        print(f"  {i}. {r['http_method']} {r['api_path']}  [{r['chunk_type']}]")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("target_substring", help="Substring to search for in chunk_id/endpoint_id, e.g. a doc path")
    ap.add_argument("--retrieve-k", type=int, default=25)
    args = ap.parse_args()

    diagnose(args.query, args.target_substring, retrieve_k=args.retrieve_k)