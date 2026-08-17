"""
parent_child.py

After reranking gives us the best-matching chunk(s), this expands each one
into the FULL endpoint document — pulling in sibling chunks (e.g. if we
matched the "description" chunk for an endpoint, also grab its
"request_schema" chunk, if one exists) using the shared `endpoint_id` field
set back in chunker.py.

Why this matters: chunks were split small on purpose (better retrieval
precision — see DESIGN.md section 3), but a real answer often needs the
FULL picture. E.g. "how do I create an issue" needs both the description
(what the endpoint does, required params) AND the request body schema
(exact field names/types) to generate a complete, correct answer. Splitting
for retrieval and reassembling for generation is the whole point of
parent-child retrieval.

Usage (from project root):
    python -m src.retrieval.parent_child "how do I create an issue"
"""

from __future__ import annotations

import json
from pathlib import Path

CHUNKS_PATH = "data/processed/chunks.json"

_chunks_by_endpoint: dict[str, list[dict]] | None = None


def _load_chunks_by_endpoint(chunks_path: str = CHUNKS_PATH) -> dict[str, list[dict]]:
    global _chunks_by_endpoint
    if _chunks_by_endpoint is not None:
        return _chunks_by_endpoint

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    by_endpoint: dict[str, list[dict]] = {}
    for c in chunks:
        by_endpoint.setdefault(c["endpoint_id"], []).append(c)
    _chunks_by_endpoint = by_endpoint
    return by_endpoint


def expand_to_parent(matched_chunks: list[dict], chunks_path: str = CHUNKS_PATH) -> list[dict]:
    """
    Given a list of matched (reranked) chunks, return the full set of
    sibling chunks per unique endpoint_id, in a sensible order
    (description first, then request_schema, then anything else).
    Deduplicates so an endpoint matched via multiple chunks isn't repeated.
    """
    by_endpoint = _load_chunks_by_endpoint(chunks_path)

    seen_endpoints: set[str] = set()
    expanded: list[dict] = []

    type_priority = {"description": 0, "request_schema": 1, "code_example": 2, "conceptual": 3}

    for matched in matched_chunks:
        endpoint_id = matched["endpoint_id"]
        if endpoint_id in seen_endpoints:
            continue
        seen_endpoints.add(endpoint_id)

        siblings = by_endpoint.get(endpoint_id, [matched])
        siblings_sorted = sorted(siblings, key=lambda c: type_priority.get(c["chunk_type"], 99))
        expanded.extend(siblings_sorted)

    return expanded


def cap_expanded_chunks(expanded_chunks: list[dict], max_chars: int = 6000) -> list[dict]:
    """
    Returns the subset of expanded_chunks that fits within max_chars total
    (measuring each endpoint's combined chunk text), keeping COMPLETE
    endpoint groups in reranked-priority order and dropping lower-ranked
    endpoints entirely once the budget is used up.

    This is the single source of truth for the size budget — used by
    format_for_generation() (for the LLM prompt) AND by anything that needs
    the same capped set of chunks for other purposes (e.g. eval harnesses
    building a 'retrieved_contexts' list for an LLM judge). Both need the
    SAME cap, since a judge prompt can hit the exact same token-limit
    failure the generator did if it's handed the uncapped chunk list.

    IMPORTANT — a real bug found in testing: a single endpoint_id can span
    MANY chunks when a large conceptual doc got split by markdown_parser.py
    (all its parts share one endpoint_id, since they're the same source
    page). One real page split into 22 parts totaling ~41,000 characters —
    and the original version of this function only capped the TOTAL across
    groups, with an exception that always kept the first (highest-priority)
    group in FULL even if it alone blew past the budget. That exception let
    a 22-part group through completely uncapped, building a >10,000-token
    prompt that got rejected outright by Groq's rate limiter. Fixed by
    capping the size of EACH group individually too — even the top-ranked
    match only contributes up to max_chars of its own siblings, not all of
    them.
    """
    by_endpoint: dict[str, list[dict]] = {}
    order: list[str] = []
    for c in expanded_chunks:
        eid = c["endpoint_id"]
        if eid not in by_endpoint:
            by_endpoint[eid] = []
            order.append(eid)
        by_endpoint[eid].append(c)

    kept_chunks: list[dict] = []
    total_chars = 0
    n_dropped_endpoints = 0
    n_dropped_sibling_chunks = 0

    for eid in order:
        chunks = by_endpoint[eid]

        # --- per-group cap: this endpoint's OWN contribution is capped at
        # max_chars, regardless of how many sibling chunks it actually has.
        # Always keep at least the first chunk of the group even if it
        # alone exceeds max_chars (some content beats none), but every
        # sibling after that respects the group's own budget.
        group_chunks: list[dict] = []
        group_chars = 0
        for c in chunks:
            if group_chunks and group_chars + len(c["text"]) > max_chars:
                n_dropped_sibling_chunks += len(chunks) - len(group_chunks)
                break
            group_chunks.append(c)
            group_chars += len(c["text"])

        # --- across-groups cap: same as before — always keep the first
        # (highest-priority) group, respect the remaining total budget for
        # every group after that.
        if kept_chunks and total_chars + group_chars > max_chars:
            n_dropped_endpoints += 1
            continue

        kept_chunks.extend(group_chunks)
        total_chars += group_chars

    if n_dropped_endpoints:
        print(f"  (context budget: dropped {n_dropped_endpoints} lower-ranked endpoint(s) to stay under {max_chars} chars)")
    if n_dropped_sibling_chunks:
        print(f"  (context budget: dropped {n_dropped_sibling_chunks} sibling chunk(s) from an oversized "
              f"multi-part document to stay under {max_chars} chars per endpoint)")

    return kept_chunks


def format_for_generation(expanded_chunks: list[dict], max_chars: int = 6000) -> str:
    """
    Combine expanded chunks into one clean text block per endpoint, ready
    to hand to the LLM as context. Groups sibling chunks under one header
    instead of repeating the endpoint line for every chunk.

    max_chars is a hard budget on the TOTAL context size (~4 chars/token,
    so 6000 chars is roughly 1500 tokens). This exists because of a real
    failure found in testing: with rerank_top_k=3, parent-child expansion
    could pull in 3 full endpoints' worth of description + schema text,
    building a prompt that hit Groq's free-tier 8,000 tokens/minute limit
    and got rejected outright (HTTP 413). The actual capping logic lives in
    cap_expanded_chunks() above so other consumers (like an eval harness's
    LLM judge) can apply the exact same budget.
    """
    capped_chunks = cap_expanded_chunks(expanded_chunks, max_chars=max_chars)
    n_dropped_endpoints = len({c["endpoint_id"] for c in expanded_chunks}) - len({c["endpoint_id"] for c in capped_chunks})

    by_endpoint: dict[str, list[dict]] = {}
    order: list[str] = []
    for c in capped_chunks:
        eid = c["endpoint_id"]
        if eid not in by_endpoint:
            by_endpoint[eid] = []
            order.append(eid)
        by_endpoint[eid].append(c)

    blocks = []
    for eid in order:
        chunks = by_endpoint[eid]
        source_url = chunks[0].get("source_url", "")
        block_lines = [f"=== {eid} ===", f"(source: {source_url})", ""]
        for c in chunks:
            block_lines.append(c["text"])
            block_lines.append("")
        blocks.append("\n".join(block_lines))

    result = "\n---\n".join(blocks)
    if n_dropped_endpoints:
        result += (
            f"\n---\n(Note: {n_dropped_endpoints} additional lower-ranked result(s) omitted "
            f"to keep the context within budget.)"
        )
    return result


def main():
    import argparse
    from src.reranking.cross_encoder import rerank
    from src.retrieval.fusion import hybrid_search

    ap = argparse.ArgumentParser(description="Run full retrieval pipeline through parent-child expansion.")
    ap.add_argument("query")
    ap.add_argument("--retrieve-k", type=int, default=25)
    ap.add_argument("--rerank-top-k", type=int, default=3, help="How many top chunks to expand into full docs")
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    args = ap.parse_args()

    candidates = hybrid_search(args.query, top_k=args.retrieve_k, qdrant_url=args.qdrant_url)
    reranked = rerank(args.query, candidates, top_k=args.rerank_top_k)

    print(f"Top {len(reranked)} reranked chunks (before expansion):")
    for r in reranked:
        print(f"  {r['http_method']} {r['api_path']}  [{r['chunk_type']}]  score={r['rerank_score']:.4f}")

    expanded = expand_to_parent(reranked)
    print(f"\nExpanded to {len(expanded)} total chunks across {len(set(c['endpoint_id'] for c in expanded))} unique endpoints")

    context_text = format_for_generation(expanded)
    print("\n" + "=" * 70)
    print("FULL CONTEXT THAT WOULD BE SENT TO THE LLM:")
    print("=" * 70)
    print(context_text[:3000])
    if len(context_text) > 3000:
        print(f"\n... ({len(context_text) - 3000} more characters truncated for display)")


if __name__ == "__main__":
    main()