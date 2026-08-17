"""
multi_query.py

Generates a few paraphrased variants of the user's query using the LLM,
runs hybrid search for EACH variant, then fuses all the resulting ranked
lists together with RRF. This helps when a developer's phrasing doesn't
match the vocabulary GitHub's docs use — e.g. "how do I star a repo" vs.
the doc's own wording, or casual phrasing vs. formal API terminology.

Trade-off worth being explicit about: this costs one extra LLM call (to
generate the variants) plus N extra retrieval passes instead of one. For
an interactive assistant, that's a real latency cost — worth it when
recall matters more than raw speed, which is why this is offered as an
optional retrieval mode rather than baked into every query by default.

Usage (from project root, Qdrant running + GROQ_API_KEY set):
    python -m src.retrieval.multi_query "how do I star a gist"
"""

from __future__ import annotations

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from src.retrieval.bm25_search import bm25_search
from src.retrieval.vector_search import vector_search
from src.retrieval.fusion import reciprocal_rank_fusion

import os
from dotenv import load_dotenv
load_dotenv()

MULTI_QUERY_MODEL = "openai/gpt-oss-20b"  # same fast/free model as generation

MULTI_QUERY_SYSTEM_PROMPT = """You generate alternate phrasings of a developer's question about GitHub's REST API, to improve document retrieval. Given one question, produce {n} different ways a developer might phrase the SAME underlying question — vary vocabulary and phrasing, but do not change what's being asked. Output ONLY the {n} variants, one per line, no numbering, no extra commentary."""

_llm: ChatGroq | None = None


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        # max_tokens raised from 200 -> 400: at 200, longer variants could get
        # cut off mid-sentence, producing garbage fragments (e.g. a variant
        # that's literally just "How") that then got treated as real queries.
        _llm = ChatGroq(model=MULTI_QUERY_MODEL, temperature=0.7, max_tokens=400)
    return _llm


def generate_query_variants(query: str, n: int = 3) -> list[str]:
    """Ask the LLM for n paraphrased variants of the query. Always includes the original query too."""
    llm = _get_llm()
    messages = [
        SystemMessage(content=MULTI_QUERY_SYSTEM_PROMPT.format(n=n)),
        HumanMessage(content=query),
    ]
    response = llm.invoke(messages)
    raw_lines = [line.strip() for line in response.content.strip().split("\n") if line.strip()]

    # Guard against truncated/fragment lines (e.g. a response cut off mid-word,
    # or the model echoing a bare word). A real paraphrased question should be
    # a handful of words at minimum and should end with sentence-ending
    # punctuation OR at least look like a complete clause. This is a
    # heuristic, not a perfect grammar check, but it reliably filters out
    # the kind of truncation fragment we saw in testing ("How").
    valid_variants = [
        line for line in raw_lines
        if len(line.split()) >= 4
    ]

    if len(valid_variants) < len(raw_lines):
        print(f"  (dropped {len(raw_lines) - len(valid_variants)} incomplete/fragment variant(s) from LLM output)")

    return [query] + valid_variants[:n]


def multi_query_search(
    query: str,
    n_variants: int = 3,
    top_k: int = 5,
    retrieve_k_per_variant: int = 15,
    qdrant_url: str = "http://localhost:6333",
) -> list[dict]:
    """
    Generate query variants, run hybrid search per variant, fuse everything
    together with RRF into one final ranked list.
    """
    variants = generate_query_variants(query, n=n_variants)

    all_ranked_lists = []
    labels = []
    for i, variant in enumerate(variants):
        bm25_results = bm25_search(variant, top_k=retrieve_k_per_variant)
        vector_results = vector_search(variant, top_k=retrieve_k_per_variant, qdrant_url=qdrant_url)
        all_ranked_lists.append(bm25_results)
        labels.append(f"bm25_q{i}")
        all_ranked_lists.append(vector_results)
        labels.append(f"vector_q{i}")

    fused = reciprocal_rank_fusion(all_ranked_lists, top_k=top_k, labels=labels)
    return fused, variants


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Run multi-query hybrid search.")
    ap.add_argument("query")
    ap.add_argument("--n-variants", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    args = ap.parse_args()

    results, variants = multi_query_search(
        args.query, n_variants=args.n_variants, top_k=args.top_k, qdrant_url=args.qdrant_url
    )

    print(f"Original query: {args.query!r}")
    print("Generated variants:")
    for v in variants[1:]:
        print(f"  - {v!r}")
    print()

    print("Fused results:")
    for i, r in enumerate(results, 1):
        found = "+".join(r["found_by"])
        print(f"{i}. rrf={r['rrf_score']:.4f}  ({found})  {r['http_method']} {r['api_path']}  [{r['chunk_type']}]")


if __name__ == "__main__":
    main()