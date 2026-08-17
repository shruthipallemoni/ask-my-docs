"""
citation_validator.py

Parses [cite: <endpoint_id>] tags out of a generated answer and checks
each one against the set of endpoint_ids that were ACTUALLY retrieved for
that query. This is the safety net that catches hallucinated citations —
an LLM confidently citing an endpoint that was never in the context at all.

Two distinct things this checks, worth keeping separate:
  1. VALIDITY — every citation that exists must point to a real, retrieved
     endpoint_id. A citation to something not retrieved is a hallucination
     and should trigger a reject/regenerate (see DESIGN.md section 7).
  2. COVERAGE — how many citations are there relative to the answer's
     length? An answer with zero citations, or just one citation covering
     several paragraphs of claims, isn't necessarily WRONG, but it's not
     verifiably grounded either — we flag it as "sparse" rather than
     "invalid", since it's a real observed behavior of smaller/faster
     models (see the gpt-oss-20b example that motivated this module) and
     shouldn't be silently treated as equivalent to a well-cited answer.

Usage (from project root):
    python -m src.generation.citation_validator
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict

CITATION_PATTERN = re.compile(r"\[cite:\s*([^\]]+)\]")

# Below this many words-per-citation, we flag the answer as sparsely cited.
# This is a heuristic, not a hard rule — tune based on what you see in eval.
SPARSE_CITATION_WORDS_PER_CITE_THRESHOLD = 150


@dataclass
class CitationCheckResult:
    citations_found: list[str]
    valid_citations: list[str]
    hallucinated_citations: list[str]  # cited but NOT in the retrieved set
    is_valid: bool                     # False if any hallucinated citation exists
    is_sparsely_cited: bool            # True if citation density is low
    word_count: int
    citation_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def extract_citations(answer_text: str) -> list[str]:
    """Pull every endpoint_id out of [cite: ...] tags, in order of appearance."""
    return [m.strip() for m in CITATION_PATTERN.findall(answer_text)]


def validate_citations(answer_text: str, retrieved_endpoint_ids: set[str]) -> CitationCheckResult:
    """
    retrieved_endpoint_ids: the set of endpoint_ids that were ACTUALLY
    retrieved and shown to the model as context for this query (e.g. from
    the expanded chunks in parent_child.py). Anything cited outside this
    set is, by definition, not grounded in what the model was given.
    """
    citations = extract_citations(answer_text)
    valid = [c for c in citations if c in retrieved_endpoint_ids]
    hallucinated = [c for c in citations if c not in retrieved_endpoint_ids]

    word_count = len(answer_text.split())
    citation_count = len(citations)
    words_per_citation = word_count / citation_count if citation_count else float("inf")

    return CitationCheckResult(
        citations_found=citations,
        valid_citations=valid,
        hallucinated_citations=hallucinated,
        is_valid=len(hallucinated) == 0,
        is_sparsely_cited=words_per_citation > SPARSE_CITATION_WORDS_PER_CITE_THRESHOLD,
        word_count=word_count,
        citation_count=citation_count,
    )


def strip_citation_tags(answer_text: str) -> str:
    """Remove [cite: ...] tags for display to an end user who doesn't need to see the raw tag syntax."""
    return CITATION_PATTERN.sub("", answer_text).replace("  ", " ").strip()


def main():
    """Quick self-test with a few realistic examples — no external services needed."""
    retrieved_ids = {"POST /repos/{owner}/{repo}/issues", "GET /repos/{owner}/{repo}/issues"}

    examples = [
        # well-cited, valid
        "Send a POST request [cite: POST /repos/{owner}/{repo}/issues]. "
        "The only required field is title [cite: POST /repos/{owner}/{repo}/issues}]".replace("}]", "]"),
        # hallucinated citation — endpoint never retrieved
        "You can also use PATCH /repos/{owner}/{repo}/issues/{number} [cite: PATCH /repos/{owner}/{repo}/issues/{number}] to edit it.",
        # zero citations at all
        "Just send a POST request with a title field and you're done.",
    ]

    for i, answer in enumerate(examples, 1):
        result = validate_citations(answer, retrieved_ids)
        print(f"--- Example {i} ---")
        print(f"  citations found: {result.citations_found}")
        print(f"  valid: {result.is_valid}  |  hallucinated: {result.hallucinated_citations}")
        print(f"  sparsely cited: {result.is_sparsely_cited}  ({result.word_count} words, {result.citation_count} citations)")
        print()


if __name__ == "__main__":
    main()