"""
indexer.py

Takes chunks.json (from chunker.py) and builds the two retrieval indexes:
  1. A Qdrant collection of dense embeddings (bge-m3, local via sentence-transformers)
  2. A BM25 index over the same chunk text (rank_bm25, pure Python)

Both indexes are keyed by the same `chunk_id`, so hybrid fusion (fusion.py,
next step) can merge results from either side into one ranked list.

IMPORTANT — environment note:
`bge-m3` weights are pulled from Hugging Face on first run (~2GB, cached
locally after that). This requires outbound access to huggingface.co.
Run this script on your own machine / server, not in a network-restricted
sandbox. Everything else (Qdrant, BM25) has no external dependency once
the Python packages are installed.

Usage:
    python -m src.ingestion.indexer --chunks-file data/processed/chunks.json
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIM = 1024  # bge-m3 dense vector size
COLLECTION_NAME = "ask_my_docs_chunks"


def load_chunks(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_bm25_index(chunks: list[dict], out_path: str | Path) -> BM25Okapi:
    """
    Tokenize chunk text and build a BM25 index. We keep it dead simple —
    lowercase + whitespace/punctuation split. GitHub API text has a lot of
    meaningful punctuation (e.g. `/repos/{owner}/{repo}`, `422`), so we
    avoid aggressive stemming that would mangle path segments or codes.
    """
    tokenized_corpus = [_tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    # Persist alongside the chunk_id order, since BM25Okapi has no built-in
    # save/load — we pickle the whole object plus the id mapping it needs.
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "bm25": bm25,
                "chunk_ids": [c["chunk_id"] for c in chunks],
            },
            f,
        )
    return bm25


_STOPWORDS = {
    "a", "an", "the", "is", "are", "do", "does", "did", "for", "of", "to",
    "how", "what", "i", "in", "on", "with", "and", "or", "was", "were", "can", "you",
}


def _tokenize(text: str) -> list[str]:
    import re

    text = text.lower()
    # keep path-like tokens (e.g. /repos/{owner}/{repo}) and status codes intact-ish
    tokens = re.findall(r"[a-z0-9_./{}\-]+", text)
    return [t for t in tokens if t.strip() and t not in _STOPWORDS]


def build_vector_index(
    chunks: list[dict],
    qdrant_url: str = "http://localhost:6333",
    batch_size: int = 64,
    use_local_memory: bool = False,
    embeddings_cache_path: str | Path = "data/processed/embeddings_cache.npz",
    force_reembed: bool = False,
) -> None:
    """
    Embed every chunk with bge-m3 and upsert into Qdrant.

    Set use_local_memory=True for quick local testing without a running
    Qdrant server (uses Qdrant's in-memory mode). For real use, run Qdrant
    via `docker-compose up -d` and leave this False.

    IMPORTANT — caching: embedding 1,500+ chunks on CPU can take a long
    time (hours on modest hardware). We cache the computed vectors to disk
    keyed by chunk_id, so if you ever need to rebuild the Qdrant collection
    (e.g. after resetting corrupted storage) you do NOT need to re-run the
    model at all — this function will detect the cache, skip encoding
    entirely, and just re-upsert the cached vectors. That's seconds, not
    hours. Only pass force_reembed=True if chunks.json actually changed
    (different text) and the cache is stale.
    """
    import numpy as np

    embeddings_cache_path = Path(embeddings_cache_path)
    chunk_ids = [c["chunk_id"] for c in chunks]

    cached_vectors: dict[str, list[float]] = {}
    if embeddings_cache_path.exists() and not force_reembed:
        print(f"Found embedding cache at {embeddings_cache_path} — loading instead of re-encoding...")
        cache = np.load(embeddings_cache_path, allow_pickle=True)
        cached_ids = list(cache["chunk_ids"])
        cached_vecs = cache["vectors"]
        cached_vectors = dict(zip(cached_ids, cached_vecs))

    missing_ids = [cid for cid in chunk_ids if cid not in cached_vectors]

    if missing_ids:
        if cached_vectors:
            print(f"Cache covers {len(cached_vectors)}/{len(chunk_ids)} chunks — "
                  f"encoding the remaining {len(missing_ids)} new/changed chunks...")
        else:
            print(f"No cache found — encoding all {len(chunk_ids)} chunks with {EMBEDDING_MODEL_NAME} "
                  f"(this is the slow, one-time step)...")

        from sentence_transformers import SentenceTransformer
        import torch
        import multiprocessing

        # Fix for slow CPU encoding: PyTorch sometimes defaults to using only
        # 1-2 threads even on multi-core machines, which can make embedding
        # take hours instead of minutes. Explicitly use all available cores.
        n_cores = multiprocessing.cpu_count()
        default_threads = torch.get_num_threads()
        torch.set_num_threads(n_cores)
        print(f"Using {n_cores} CPU threads for encoding (was using {default_threads} by default)")

        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        chunks_by_id = {c["chunk_id"]: c for c in chunks}
        missing_chunks = [chunks_by_id[cid] for cid in missing_ids]

        for i in range(0, len(missing_chunks), batch_size):
            batch = missing_chunks[i : i + batch_size]
            texts = [c["text"] for c in batch]
            vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=batch_size)
            for j, c in enumerate(batch):
                cached_vectors[c["chunk_id"]] = vectors[j]
            print(f"  encoded {min(i + batch_size, len(missing_chunks))}/{len(missing_chunks)} new chunks")

        # persist the (now-complete) cache to disk immediately
        embeddings_cache_path.parent.mkdir(parents=True, exist_ok=True)
        all_ids = list(cached_vectors.keys())
        all_vecs = np.array([cached_vectors[cid] for cid in all_ids])
        np.savez(embeddings_cache_path, chunk_ids=np.array(all_ids, dtype=object), vectors=all_vecs)
        print(f"Embedding cache saved to {embeddings_cache_path} — future rebuilds will skip encoding entirely.")
    else:
        print(f"All {len(chunk_ids)} chunks found in cache — skipping the model entirely. This should take seconds.")

    # ---- upsert into Qdrant (always runs, this part is fast) ----
    client = QdrantClient(":memory:") if use_local_memory else QdrantClient(url=qdrant_url)

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    # Create payload indexes for every field we filter on (see
    # metadata_filter.py). Without these, Qdrant falls back to an
    # unindexed scan on filtered queries — slow, and known to trigger an
    # internal panic ("OffsetZero") on some Qdrant versions. Doing this
    # up front, before any points are inserted, avoids that entirely.
    from qdrant_client.models import PayloadSchemaType

    for field_name, schema_type in {
        "http_method": PayloadSchemaType.KEYWORD,
        "api_category": PayloadSchemaType.KEYWORD,
        "api_subcategory": PayloadSchemaType.KEYWORD,
        "chunk_type": PayloadSchemaType.KEYWORD,
        "auth_required": PayloadSchemaType.BOOL,
    }.items():
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field_name,
            field_schema=schema_type,
        )

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        points = [
            PointStruct(
                id=i + j,  # Qdrant needs int/uuid ids; keep chunk_id in payload
                vector=list(cached_vectors[batch[j]["chunk_id"]]),
                payload=batch[j],  # full chunk dict — enables metadata filtering
            )
            for j in range(len(batch))
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"  upserted {min(i + batch_size, len(chunks))}/{len(chunks)} chunks into Qdrant")

    print(f"Vector index built: {len(chunks)} chunks in Qdrant collection '{COLLECTION_NAME}'")


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Build BM25 + vector indexes from chunks.json")
    ap.add_argument("--chunks-file", default="data/processed/chunks.json")
    ap.add_argument("--bm25-out", default="data/processed/bm25_index.pkl")
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--skip-vector", action="store_true", help="Only build BM25 (useful without Qdrant/HF access)")
    ap.add_argument("--local-memory", action="store_true", help="Use in-memory Qdrant instead of a running server")
    ap.add_argument("--force-reembed", action="store_true",
                    help="Ignore the embedding cache and re-encode everything from scratch")
    args = ap.parse_args()

    chunks = load_chunks(args.chunks_file)
    print(f"Loaded {len(chunks)} chunks from {args.chunks_file}")

    print("Building BM25 index...")
    build_bm25_index(chunks, args.bm25_out)
    print(f"BM25 index written to {args.bm25_out}")

    if not args.skip_vector:
        print("Building vector index (requires Hugging Face access for bge-m3 weights)...")
        build_vector_index(chunks, qdrant_url=args.qdrant_url, use_local_memory=args.local_memory,
                            force_reembed=args.force_reembed)
    else:
        print("Skipped vector index (--skip-vector set)")


if __name__ == "__main__":
    main()