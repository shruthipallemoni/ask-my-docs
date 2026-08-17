"""
check_thresholds.py

Reads eval/report/retrieval_eval_results.json (from run_retrieval_eval.py)
and exits non-zero if any metric falls below its threshold — this is the
actual gate GitHub Actions uses to block a PR.

Thresholds are set slightly below our current measured baseline (Hit
Rate@5=85.0%, MRR=0.725) — tight enough to catch a real regression, loose
enough that normal variance (e.g. golden set additions, minor chunking
changes) doesn't cause false-positive failures. Adjust as the corpus and
golden set evolve.

Usage (from project root, after running run_retrieval_eval.py):
    python -m eval.check_thresholds
Exit code 0 = pass, 1 = fail (at least one metric below threshold).
"""

import json
import sys
from pathlib import Path

THRESHOLDS = {
    "hit_rate_at_5": 0.75,   # measured baseline: 0.850
    "mrr": 0.60,             # measured baseline: 0.725
}


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="eval/report/retrieval_eval_results.json")
    args = ap.parse_args()

    path = Path(args.results)
    if not path.exists():
        print(f"FAIL: {path} not found — did run_retrieval_eval.py run successfully?")
        sys.exit(1)

    data = json.loads(path.read_text())
    summary = data["summary"]

    print("Checking retrieval eval results against thresholds...\n")

    all_passed = True
    for metric, threshold in THRESHOLDS.items():
        value = summary.get(metric)
        if value is None:
            print(f"  FAIL: metric '{metric}' not found in results")
            all_passed = False
            continue

        passed = value >= threshold
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {metric}: {value:.3f} (threshold: >= {threshold})")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All thresholds passed.")
        sys.exit(0)
    else:
        print("One or more metrics fell below threshold — blocking merge.")
        print("\nFailing questions for debugging:")
        for f in summary.get("failures", [])[:10]:
            print(f"  [{f['id']}] {f['query']!r}")
        sys.exit(1)


if __name__ == "__main__":
    main()