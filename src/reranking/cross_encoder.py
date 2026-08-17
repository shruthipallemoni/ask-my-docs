"""
cross_encoder.py

Reranks the fused (BM25 + vector) candidates using a cross-encoder model.

Why this fixes the RRF ordering problem we just observed:
A cross-encoder takes the query and ONE candidate document TOGETHER as a
single input, and the model's attention layers directly compare them,
token against token. This is much more precise than comparing two
separately-computed vectors (what bi-encoders / plain vector search do) —
but it's also much slower, which is exactly why we don't run it over all
1,529 chunks. We only rerank the ~20-30 survivors that already made it
through hybrid fusion.

Model: bge-reranker-base (BAAI) — free, local, via sentence-transformers'
CrossEncoder class. No API key, no cost.

Usage (from project root):
    python -m src.reranking.cross_encoder "delete a repository" --http-method DELETE
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    candidates: list of chunk dicts (with at least 'text'), already fused
    by BM25+vector search — typically ~20-30 of them.

    Returns the top_k candidates re-sorted by cross-encoder relevance score,
    with a 'rerank_score' field added. This score is NOT comparable to RRF
    scores or cosine similarity — it's the cross-encoder's own scale.
    """
    if not candidates:
        return []

    model = _get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda pair: -pair[1])

    results = []
    for candidate, score in scored[:top_k]:
        r = dict(candidate)
        r["rerank_score"] = float(score)
        results.append(r)
    return results


def main():
    import argparse
    from src.retrieval.metadata_filter import filtered_hybrid_search
    from src.retrieval.fusion import hybrid_search

    ap = argparse.ArgumentParser(description="Run hybrid search + cross-encoder reranking end to end.")
    ap.add_argument("query")
    ap.add_argument("--http-method")
    ap.add_argument("--api-category")
    ap.add_argument("--retrieve-k", type=int, default=25, help="How many fused candidates to rerank")
    ap.add_argument("--top-k", type=int, default=5, help="How many to show after reranking")
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    args = ap.parse_args()

    criteria = {}
    if args.http_method:
        criteria["http_method"] = args.http_method.upper()
    if args.api_category:
        criteria["api_category"] = args.api_category

    if criteria:
        candidates = filtered_hybrid_search(
            args.query, criteria=criteria, top_k=args.retrieve_k, qdrant_url=args.qdrant_url
        )
    else:
        candidates = hybrid_search(args.query, top_k=args.retrieve_k, qdrant_url=args.qdrant_url)

    print(f"\nFused {len(candidates)} candidates, reranking with {RERANKER_MODEL_NAME}...\n")
    reranked = rerank(args.query, candidates, top_k=args.top_k)

    print(f"Query: {args.query!r}  Filters: {criteria or '(none)'}\n")
    for i, r in enumerate(reranked, 1):
        print(f"{i}. rerank_score={r['rerank_score']:.4f}  {r['http_method']} {r['api_path']}  [{r['chunk_type']}]")


if __name__ == "__main__":
    main()