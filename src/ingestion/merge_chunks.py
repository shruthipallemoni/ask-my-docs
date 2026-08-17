"""
merge_chunks.py

Combines the endpoint chunks (chunker.py output) and the conceptual doc
chunks (markdown_parser.py output) into one chunks.json — the single file
every downstream script (indexer, BM25, retrieval) already reads from.

Keeps a backup of the pre-merge endpoint-only file, since chunks.json gets
overwritten and you may want to inspect the "before" state.

Usage (from project root):
    python -m src.ingestion.merge_chunks
"""

import json
import shutil
from pathlib import Path


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint-chunks", default="data/processed/chunks.json")
    ap.add_argument("--conceptual-chunks", default="data/processed/conceptual_chunks.json")
    ap.add_argument("--out", default="data/processed/chunks.json")
    ap.add_argument("--backup-suffix", default=".endpoint_only_backup")
    args = ap.parse_args()

    endpoint_path = Path(args.endpoint_chunks)
    conceptual_path = Path(args.conceptual_chunks)
    out_path = Path(args.out)

    with open(endpoint_path, "r", encoding="utf-8") as f:
        endpoint_chunks = json.load(f)
    with open(conceptual_path, "r", encoding="utf-8") as f:
        conceptual_chunks = json.load(f)

    # Back up the endpoint-only file before overwriting, since --out defaults
    # to the same path as --endpoint-chunks.
    if out_path == endpoint_path:
        backup_path = endpoint_path.with_suffix(endpoint_path.suffix + args.backup_suffix)
        shutil.copy(endpoint_path, backup_path)
        print(f"Backed up endpoint-only chunks to {backup_path}")

    merged = endpoint_chunks + conceptual_chunks

    # sanity check: no duplicate chunk_ids across the two sources
    ids = [c["chunk_id"] for c in merged]
    duplicates = {cid for cid in ids if ids.count(cid) > 1}
    if duplicates:
        print(f"WARNING: {len(duplicates)} duplicate chunk_ids found: {list(duplicates)[:5]}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    print(f"Merged {len(endpoint_chunks)} endpoint chunks + {len(conceptual_chunks)} conceptual chunks "
          f"= {len(merged)} total -> {out_path}")


if __name__ == "__main__":
    main()