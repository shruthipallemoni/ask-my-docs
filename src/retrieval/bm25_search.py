"""
bm25_search.py

Keyword-based search over the chunks, using the BM25 index built by
indexer.py. This is the "exact match" half of hybrid search — it catches
things vector search sometimes blurs past, like exact path segments
(`/repos/{owner}/{repo}`), status codes (`422`), or parameter names.

Usage (from project root):
    python -m src.retrieval.bm25_search "delete a repository"
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

# Reuse the exact same tokenizer as indexing time — if these ever drift apart,
# BM25 scores become meaningless (query tokens won't match index tokens the
# same way). Keeping one source of truth for tokenization matters.
from src.ingestion.indexer import _tokenize

_bm25: BM25Okapi | None = None
_chunk_ids: list[str] | None = None
_chunks_by_id: dict[str, dict] | None = None


def _load(bm25_index_path: str, chunks_path: str) -> None:
    global _bm25, _chunk_ids, _chunks_by_id
    if _bm25 is not None:
        return  # already loaded for this process

    with open(bm25_index_path, "rb") as f:
        data = pickle.load(f)
    _bm25 = data["bm25"]
    _chunk_ids = data["chunk_ids"]

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    _chunks_by_id = {c["chunk_id"]: c for c in chunks}


def bm25_search(
    query: str,
    top_k: int = 5,
    bm25_index_path: str = "data/processed/bm25_index.pkl",
    chunks_path: str = "data/processed/chunks.json",
) -> list[dict]:
    """Return the top_k chunks ranked by BM25 score for the given query."""
    _load(bm25_index_path, chunks_path)

    scores = _bm25.get_scores(_tokenize(query))
    ranked_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]

    results = []
    for idx in ranked_idx:
        chunk_id = _chunk_ids[idx]
        chunk = _chunks_by_id[chunk_id]
        results.append(
            {
                "score": float(scores[idx]),
                "chunk_id": chunk_id,
                "endpoint_id": chunk["endpoint_id"],
                "http_method": chunk["http_method"],
                "api_path": chunk["api_path"],
                "chunk_type": chunk["chunk_type"],
                "text": chunk["text"],
            }
        )
    return results


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Run a BM25 keyword search against the chunk index.")
    ap.add_argument("query")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    results = bm25_search(args.query, top_k=args.top_k)

    print(f"\nQuery: {args.query!r}\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. score={r['score']:.4f}  {r['http_method']} {r['api_path']}  [{r['chunk_type']}]")


if __name__ == "__main__":
    main()