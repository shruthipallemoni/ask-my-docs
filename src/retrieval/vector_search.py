"""
vector_search.py

Semantic search over the chunks we embedded and stored in Qdrant. This is
the "meaning-based" half of hybrid search — where BM25 matches exact words,
this matches intent, so "delete a repository" can find the right endpoint
even if the wording doesn't overlap much with the doc text.

Usage (from project root, with Qdrant running via docker-compose):
    python -m src.retrieval.vector_search "delete a repository"
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
COLLECTION_NAME = "ask_my_docs_chunks"

# Loaded once per process — reused across calls so we're not reloading
# the ~2GB model on every single search.
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def vector_search(
    query: str,
    top_k: int = 5,
    qdrant_url: str = "http://localhost:6333",
    metadata_filter: dict | None = None,
) -> list[dict]:
    """
    Embed the query and return the top_k most similar chunks from Qdrant.

    metadata_filter (optional): a Qdrant filter dict, e.g.
        {"must": [{"key": "http_method", "match": {"value": "DELETE"}}]}
    to combine semantic search with structured filtering (section 7 of DESIGN.md).
    """
    model = _get_model()
    client = QdrantClient(url=qdrant_url)

    query_vector = model.encode(query, normalize_embeddings=True).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        query_filter=metadata_filter,
    ).points

    return [
        {
            "score": r.score,
            "chunk_id": r.payload["chunk_id"],
            "endpoint_id": r.payload["endpoint_id"],
            "http_method": r.payload["http_method"],
            "api_path": r.payload["api_path"],
            "chunk_type": r.payload["chunk_type"],
            "text": r.payload["text"],
        }
        for r in results
    ]


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Run a semantic search query against the Qdrant index.")
    ap.add_argument("query", help="The search query, e.g. 'delete a repository'")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    args = ap.parse_args()

    results = vector_search(args.query, top_k=args.top_k, qdrant_url=args.qdrant_url)

    print(f"\nQuery: {args.query!r}\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. score={r['score']:.4f}  {r['http_method']} {r['api_path']}  [{r['chunk_type']}]")


if __name__ == "__main__":
    main()