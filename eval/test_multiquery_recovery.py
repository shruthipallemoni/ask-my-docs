"""
test_multiquery_recovery.py

Targeted follow-up test: takes the specific questions that failed under
plain hybrid search + rerank, and re-runs them through multi_query_search
(paraphrase + fuse) to see whether generating alternate phrasings recovers
the correct endpoint. This is deliberately narrow — testing 6 known
failures is cheap and fast, versus re-running the full 40-question golden
set through the more expensive multi-query path.

Usage (from project root, Qdrant running + GROQ_API_KEY set):
    python -m eval.test_multiquery_recovery
"""

from src.retrieval.multi_query import multi_query_search

# The 6 questions that failed plain hybrid search + rerank, pulled directly
# from the latest eval/report/retrieval_eval_results.json failures list.
FAILING_CASES = [
    {
        "id": "gs_017",
        "query": "What's the endpoint to invite someone to collaborate and then check pending invitations?",
        "expected": {"GET /repos/{owner}/{repo}/invitations", "PUT /repos/{owner}/{repo}/collaborators/{username}"},
    },
    {
        "id": "gs_019",
        "query": "What are the required fields when creating an issue?",
        "expected": {"POST /repos/{owner}/{repo}/issues"},
    },
    {
        "id": "gs_024",
        "query": "Is the path parameter for owner and repo case sensitive when creating an issue?",
        "expected": {"POST /repos/{owner}/{repo}/issues"},
    },
    {
        "id": "gs_026",
        "query": "What's the difference between the repo scope and the public_repo scope?",
        "expected": {"CONCEPTUAL apps/oauth-apps/building-oauth-apps/scopes-for-oauth-apps"},
    },
    {
        "id": "gs_031",
        "query": "What does a 410 status code mean when creating an issue?",
        "expected": {"POST /repos/{owner}/{repo}/issues"},
    },
    {
        "id": "gs_032",
        "query": "What HTTP status code is returned on a successful repository creation?",
        "expected": {"POST /orgs/{org}/repos"},
    },
]


def main():
    recovered = 0
    still_failing = 0

    for case in FAILING_CASES:
        print(f"\n{'='*60}")
        print(f"[{case['id']}] {case['query']!r}")
        results, variants = multi_query_search(case["query"], n_variants=3, top_k=5)

        print("  Generated variants:")
        for v in variants[1:]:
            print(f"    - {v!r}")

        found_at = None
        for i, r in enumerate(results, 1):
            if r["endpoint_id"] in case["expected"]:
                found_at = i
                break

        if found_at:
            print(f"  RECOVERED at rank {found_at}: {results[found_at-1]['http_method']} {results[found_at-1]['api_path']}")
            recovered += 1
        else:
            print(f"  STILL MISSING. Top result instead: {results[0]['http_method']} {results[0]['api_path']}")
            still_failing += 1

    print(f"\n{'='*60}")
    print(f"SUMMARY: {recovered}/{len(FAILING_CASES)} recovered with multi-query, {still_failing} still failing")


if __name__ == "__main__":
    main()