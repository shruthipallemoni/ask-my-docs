"""
query_expansion.py

Expands common developer/GitHub shorthand BEFORE retrieval — "pr" -> also
searches "pull request", "repo" -> also "repository", etc. Unlike
multi_query.py (which uses an LLM to generate paraphrases), this is pure
dictionary lookup: free, instant, no API call, no latency cost. That's a
deliberate trade-off — this catches a known, finite set of abbreviations
cheaply; multi-query handles the open-ended "any possible phrasing"
problem, at a real cost. Use this one always; use multi-query selectively.

Design choice: we APPEND the expansion alongside the original term rather
than replacing it. "merge a pr" -> "merge a pr pull request". This way:
  - BM25 still gets an exact match if "pr" itself appears in the docs
  - BM25 and vector search ALSO get the fully-spelled-out term
  - We never lose information by guessing wrong about which form is
    "more correct" — both are searched.

Usage (from project root, no external services needed):
    python -m src.retrieval.query_expansion "how do I merge a pr"
"""

from __future__ import annotations

import re

# Deliberately conservative — only unambiguous GitHub/dev shorthand.
# Avoid expanding common English words that happen to also be abbreviations
# in some other context (e.g. we do NOT expand "issue" since it's already
# a full word, not shorthand for anything).
SHORTHAND_MAP: dict[str, str] = {
    "pr": "pull request",
    "prs": "pull requests",
    "repo": "repository",
    "repos": "repositories",
    "org": "organization",
    "orgs": "organizations",
    "gh": "github",
    "auth": "authentication",
    "perms": "permissions",
    "sha": "commit hash",
    "wf": "workflow",
    "wfs": "workflows",
    "pat": "personal access token",
    "gha": "github actions",
    "webhook": "webhook http callback",
}

# Match shorthand tokens as whole words only (word boundaries), case-insensitive.
_TOKEN_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in SHORTHAND_MAP.keys()) + r")\b",
    re.IGNORECASE,
)


def expand_query(query: str) -> str:
    """
    Returns the query with known shorthand expanded and appended alongside
    the original term. If no shorthand is found, returns the query unchanged.
    """
    matches_found = []

    def _replace(match: re.Match) -> str:
        token = match.group(0)
        expansion = SHORTHAND_MAP[token.lower()]
        matches_found.append((token, expansion))
        return f"{token} {expansion}"

    expanded = _TOKEN_PATTERN.sub(_replace, query)
    return expanded


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Expand GitHub/dev shorthand in a query.")
    ap.add_argument("query")
    args = ap.parse_args()

    expanded = expand_query(args.query)
    print(f"Original: {args.query!r}")
    print(f"Expanded: {expanded!r}")


if __name__ == "__main__":
    main()