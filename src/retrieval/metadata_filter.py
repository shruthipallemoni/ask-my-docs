"""
metadata_filter.py

Structured filtering on top of hybrid search — e.g. "only DELETE endpoints",
"only endpoints in the 'issues' category", "only endpoints that don't
require auth". This is the difference between hoping the LLM infers these
constraints from fuzzy text matches, and guaranteeing them with an actual
filter (see DESIGN.md section 7).

Two different retrievers, two different filtering mechanisms:
  - Qdrant (vector search) supports filtering natively at query time —
    the filter is applied *during* the vector search itself, which is both
    correct and efficient.
  - BM25 (rank_bm25) has no concept of filtering. We apply the filter
    *after* scoring, by dropping any result whose metadata doesn't match.
    This works fine at this corpus size (~1,500 chunks — scoring the whole
    corpus is cheap), but wouldn't scale to a much larger corpus without
    a real inverted-index-with-filters engine (e.g. OpenSearch/Elasticsearch).
    Worth knowing as a documented limitation, not a silent gap.

Usage (from project root):
    python -m src.retrieval.metadata_filter "delete a repository" --http-method DELETE
"""

from __future__ import annotations

from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.retrieval.bm25_search import bm25_search
from src.retrieval.vector_search import vector_search
from src.retrieval.fusion import reciprocal_rank_fusion


def build_qdrant_filter(criteria: dict) -> Filter | None:
    """
    criteria: simple {field: value} dict, e.g.
        {"http_method": "DELETE", "api_category": "issues", "auth_required": True}
    Every key becomes an exact-match "must" condition (AND'ed together).
    """
    if not criteria:
        return None
    conditions = [
        FieldCondition(key=field_name, match=MatchValue(value=value))
        for field_name, value in criteria.items()
    ]
    return Filter(must=conditions)


def _matches_criteria(chunk: dict, criteria: dict) -> bool:
    """Same AND-of-exact-matches semantics as build_qdrant_filter, applied in Python."""
    return all(chunk.get(field_name) == value for field_name, value in criteria.items())


def filtered_hybrid_search(
    query: str,
    criteria: dict | None = None,
    top_k: int = 5,
    retrieve_k: int = 30,
    qdrant_url: str = "http://localhost:6333",
) -> list[dict]:
    """
    Hybrid search (BM25 + vector, RRF-fused) with an optional structured
    metadata filter applied on both sides before fusion.

    retrieve_k is larger than in plain fusion.hybrid_search — filtering can
    remove a lot of the top results, so we cast a wider net upstream to make
    sure top_k good matches survive the filter.
    """
    qdrant_filter = build_qdrant_filter(criteria) if criteria else None

    vector_results = vector_search(
        query, top_k=retrieve_k, qdrant_url=qdrant_url, metadata_filter=qdrant_filter
    )

    bm25_raw = bm25_search(query, top_k=retrieve_k * 3)  # overfetch since we filter after
    bm25_results = (
        [r for r in bm25_raw if _matches_criteria(r, criteria)][:retrieve_k]
        if criteria
        else bm25_raw[:retrieve_k]
    )

    return reciprocal_rank_fusion([bm25_results, vector_results], top_k=top_k)


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Run filtered hybrid search.")
    ap.add_argument("query")
    ap.add_argument("--http-method", help="e.g. GET, POST, DELETE")
    ap.add_argument("--api-category", help="e.g. issues, repos, pulls")
    ap.add_argument("--chunk-type", help="description | request_schema")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    args = ap.parse_args()

    criteria = {}
    if args.http_method:
        criteria["http_method"] = args.http_method.upper()
    if args.api_category:
        criteria["api_category"] = args.api_category
    if args.chunk_type:
        criteria["chunk_type"] = args.chunk_type

    results = filtered_hybrid_search(args.query, criteria=criteria, top_k=args.top_k, qdrant_url=args.qdrant_url)

    print(f"\nQuery: {args.query!r}  Filters: {criteria or '(none)'}\n")
    if not results:
        print("No results matched both the query and the filter criteria.")
        return
    for i, r in enumerate(results, 1):
        found = "+".join(r["found_by"])
        print(f"{i}. rrf={r['rrf_score']:.4f}  ({found:11s})  {r['http_method']} {r['api_path']}  [{r['chunk_type']}]")


if __name__ == "__main__":
    main()