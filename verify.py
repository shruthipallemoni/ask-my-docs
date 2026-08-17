"""
verify_pipeline.py

A single, self-contained sanity check for everything built so far:
  1. Does data/processed/endpoints.json exist and look correct?
  2. Does data/processed/chunks.json exist and look correct?
  3. Does the BM25 index exist and actually return sensible-ish results?

Run this any time you want to confirm the pipeline still works — after
pulling new code, after re-running the parser/chunker, or just to sanity
check your local setup matches what was built.

Usage:
    python verify_pipeline.py

Exit code 0 = everything checked out. Non-zero = something's missing or broken.
"""

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).parent
CHECKS_PASSED = []
CHECKS_FAILED = []


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        CHECKS_PASSED.append(name)
        print(f"  [PASS] {name}")
    else:
        CHECKS_FAILED.append(name)
        print(f"  [FAIL] {name}  {('- ' + detail) if detail else ''}")


def section(title: str):
    print(f"\n{'='*60}\n{title}\n{'='*60}")


def main():
    # ---- 1. endpoints.json --------------------------------------------
    section("1. Checking parsed endpoints (data/processed/endpoints.json)")
    endpoints_path = ROOT / "data" / "processed" / "endpoints.json"
    if not endpoints_path.exists():
        check("endpoints.json exists", False, f"expected at {endpoints_path}")
        endpoints = []
    else:
        endpoints = json.loads(endpoints_path.read_text())
        check("endpoints.json exists", True)
        check("endpoints.json is a non-empty list", isinstance(endpoints, list) and len(endpoints) > 0)
        check("endpoints.json has 1000+ endpoints", len(endpoints) >= 1000, f"found {len(endpoints)}")

        sample = next((e for e in endpoints if e["endpoint_id"] == "POST /repos/{owner}/{repo}/issues"), None)
        check("known endpoint 'create an issue' is present", sample is not None)
        if sample:
            check("endpoint has non-empty description", len(sample.get("description", "")) > 20)
            check("endpoint has resolved parameters (no leftover $ref)", "$ref" not in json.dumps(sample))
            check("endpoint has a source_url", bool(sample.get("source_url")))

    # ---- 2. chunks.json -------------------------------------------------
    section("2. Checking chunks (data/processed/chunks.json)")
    chunks_path = ROOT / "data" / "processed" / "chunks.json"
    if not chunks_path.exists():
        check("chunks.json exists", False, f"expected at {chunks_path}")
        chunks = []
    else:
        chunks = json.loads(chunks_path.read_text())
        check("chunks.json exists", True)
        check("chunks.json is a non-empty list", isinstance(chunks, list) and len(chunks) > 0)
        check("chunk count >= endpoint count (some endpoints split into 2 chunks)",
              len(chunks) >= len(endpoints) if endpoints else False,
              f"{len(chunks)} chunks vs {len(endpoints)} endpoints")

        chunk_types = {c["chunk_type"] for c in chunks}
        check("both 'description' and 'request_schema' chunk types present",
              {"description", "request_schema"}.issubset(chunk_types),
              f"found types: {chunk_types}")

        # every chunk should be linked to a real endpoint_id
        endpoint_ids = {e["endpoint_id"] for e in endpoints} if endpoints else set()
        orphans = [c["chunk_id"] for c in chunks if endpoint_ids and c["endpoint_id"] not in endpoint_ids]
        check("no orphan chunks (all link back to a real endpoint)", len(orphans) == 0,
              f"{len(orphans)} orphan chunks, e.g. {orphans[:3]}")

        ids = [c["chunk_id"] for c in chunks]
        check("all chunk_ids are unique", len(ids) == len(set(ids)))

    # ---- 3. BM25 index -----------------------------------------------
    section("3. Checking BM25 index (data/processed/bm25_index.pkl)")
    bm25_path = ROOT / "data" / "processed" / "bm25_index.pkl"
    if not bm25_path.exists():
        check("bm25_index.pkl exists", False, f"expected at {bm25_path}")
    else:
        check("bm25_index.pkl exists", True)
        with open(bm25_path, "rb") as f:
            index_data = pickle.load(f)
        check("index has 'bm25' and 'chunk_ids' keys", {"bm25", "chunk_ids"}.issubset(index_data.keys()))

        bm25 = index_data["bm25"]
        chunk_ids = index_data["chunk_ids"]
        check("chunk_ids count matches chunks.json", len(chunk_ids) == len(chunks) if chunks else False,
              f"{len(chunk_ids)} vs {len(chunks)}")

        # Run a real query and confirm it returns *something* ranked sensibly
        try:
            sys.path.insert(0, str(ROOT))
            from src.ingestion.indexer import _tokenize

            query = "how do I star a gist"
            scores = bm25.get_scores(_tokenize(query))
            top_idx = max(range(len(scores)), key=lambda i: scores[i])
            top_chunk_id = chunk_ids[top_idx]
            chunk_by_id = {c["chunk_id"]: c for c in chunks} if chunks else {}
            top_chunk = chunk_by_id.get(top_chunk_id, {})
            print(f"  Query: {query!r}")
            print(f"  Top result: {top_chunk.get('http_method')} {top_chunk.get('api_path')} "
                  f"(score={scores[top_idx]:.2f})")
            check("BM25 query returns a result with non-zero score", scores[top_idx] > 0)
        except Exception as e:
            check("BM25 query runs without error", False, str(e))

    # ---- summary --------------------------------------------------------
    section("SUMMARY")
    print(f"  Passed: {len(CHECKS_PASSED)}")
    print(f"  Failed: {len(CHECKS_FAILED)}")
    if CHECKS_FAILED:
        print("\n  Failed checks:")
        for name in CHECKS_FAILED:
            print(f"    - {name}")
        print("\n  Something's missing or broken above — see [FAIL] lines for details.")
        sys.exit(1)
    else:
        print("\n  Everything checks out. Phase 1 ingestion pipeline (minus vector embeddings) is verified working.")
        sys.exit(0)


if __name__ == "__main__":
    main()