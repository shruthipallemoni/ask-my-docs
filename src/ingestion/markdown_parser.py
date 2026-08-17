"""
markdown_parser.py

Parses GitHub's conceptual documentation pages (guides like "Scopes for
OAuth apps", "Rate limits for the REST API") from the github/docs repo —
the prose pages that explain concepts spanning multiple endpoints, which
the OpenAPI spec alone doesn't cover. This directly fixes the gap we found
empirically: a question like "what scopes do I need" has no single home
in our endpoint-only corpus, so retrieval was inconsistent.

GitHub's docs markdown is NOT plain markdown — it uses Liquid templating
(`{% ifversion %}`, `{% data variables.product.x %}`, `{% data
reusables.x %}`) and a custom `[AUTOTITLE](/path)` link syntax. Naively
treating this as plain markdown would leave template tags and broken
links littered through our chunk text. This parser does best-effort
cleanup:
  - Resolves the most common `{% data variables.product.* %}` tags to
    their real product names (a small, hardcoded map — not a full Liquid
    engine, which would require fetching GitHub's entire reusables/data
    directory tree just to resolve a handful of tags).
  - Strips `{% ifversion %}`/`{% endif %}` conditionals, KEEPING the inner
    text (we don't have version context, so we default to showing content
    rather than hiding it — a deliberate, documented trade-off).
  - Drops `{% data reusables.* %}` lines entirely — these pull in shared
    text snippets from another part of the repo we're not fetching. This
    means some intro paragraphs may be sparse; noted as a known gap.
  - Converts `[AUTOTITLE](/path)` into a plain "(see: /path)" reference
    since we don't have the title-resolution data.

Chunking: splits each page on H2 (`##`) headings — the natural unit for a
conceptual guide, same principle as endpoint-level chunking for the API
spec (DESIGN.md section 3).

Output chunks reuse the EXACT SAME schema as chunker.py's endpoint chunks
(endpoint_id, http_method, api_path, etc.) with sensible placeholder
values for fields that don't apply to conceptual pages. This is
deliberate — it means conceptual chunks slot into the existing BM25 index,
vector index, parent-child grouping, and citation validator with ZERO
changes needed elsewhere in the pipeline.

Usage (from project root):
    python -m src.ingestion.markdown_parser --out data/processed/conceptual_chunks.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import requests

GITHUB_DOCS_RAW_BASE = "https://raw.githubusercontent.com/github/docs/main"
GITHUB_DOCS_SITE_BASE = "https://docs.github.com"

# Hard cap on a single chunk's size, in characters. H2-heading splitting
# alone isn't always enough — some sections (e.g. a long walkthrough with
# multiple embedded code examples) can run to 15,000+ characters in a
# single section. That caused a real problem in testing: one oversized
# chunk, batched alongside normal-sized ones, made the embedding model's
# attention mechanism allocate several GB of memory (attention scales
# quadratically with sequence length) and crashed the encoding step.
# Anything over this threshold gets further split by paragraph boundaries.
MAX_CHUNK_CHARS = 2500

# Confirmed-real conceptual pages (verified against the live repo) covering
# exactly the gap we found: authentication, scopes, rate limits, troubleshooting.
# This list is deliberately small and curated rather than "ingest everything" —
# scope creep here would pull in thousands of unrelated product pages (Actions,
# Codespaces, billing, etc.) that dilute retrieval precision for what is meant
# to be a GitHub REST API assistant, not a full GitHub docs assistant.
CONCEPTUAL_DOC_PATHS = [
    "content/rest/authentication/authenticating-to-the-rest-api.md",
    "content/apps/oauth-apps/building-oauth-apps/scopes-for-oauth-apps.md",
    "content/rest/using-the-rest-api/rate-limits-for-the-rest-api.md",
    "content/rest/using-the-rest-api/getting-started-with-the-rest-api.md",
    "content/rest/using-the-rest-api/troubleshooting-the-rest-api.md",
    "content/webhooks/using-webhooks/validating-webhook-deliveries.md",
]

# Best-effort resolution for the most common Liquid product-name variables.
# Not exhaustive — anything not in this map gets its raw tag stripped,
# which occasionally leaves a slightly awkward sentence. Documented trade-off.
LIQUID_VARIABLE_MAP = {
    "product.prodname_dotcom": "GitHub",
    "product.prodname_ghe_server": "GitHub Enterprise Server",
    "product.prodname_ghe_cloud": "GitHub Enterprise Cloud",
    "product.prodname_github_app": "GitHub App",
    "product.prodname_github_apps": "GitHub Apps",
    "product.prodname_oauth_app": "OAuth App",
    "product.prodname_oauth_apps": "OAuth Apps",
    "product.rest_url": "https://api.github.com",
    "product.prodname_actions": "GitHub Actions",
    "product.prodname_codespaces": "Codespaces",
}


def _clean_liquid_template(text: str) -> str:
    # Resolve known {% data variables.x.y %} tags to real text
    def _resolve_variable(match: re.Match) -> str:
        var_path = match.group(1).strip()
        return LIQUID_VARIABLE_MAP.get(var_path, "")

    # NOTE: path segments can contain hyphens (e.g. "rest-api.secondary-rate-
    # limit-rest-graphql"), so the character class must include '-', not just
    # \w (word chars) and '.'. An earlier version missed this and let
    # hyphenated reusable tags slip through unstripped.
    text = re.sub(r"\{%-?\s*data\s+variables\.([\w.-]+)\s*-?%\}", _resolve_variable, text)

    # Drop {% data reusables.x %} lines entirely — external snippet we don't have
    text = re.sub(r"\{%-?\s*data\s+reusables\.[\w.-]+\s*-?%\}", "", text)

    # Strip {% ifversion ... %} / {% elsif ... %} / {% else %} / {% endif %}
    # tags but KEEP the inner text (we don't have version context to pick a
    # branch, so default to showing everything rather than hiding content —
    # slightly redundant/contradictory text is a better failure mode for a
    # retrieval corpus than silently missing content).
    text = re.sub(r"\{%-?\s*ifversion[^%]*-?%\}", "", text)
    text = re.sub(r"\{%-?\s*elsif[^%]*-?%\}", "", text)
    text = re.sub(r"\{%-?\s*else\s*-?%\}", "", text)
    text = re.sub(r"\{%-?\s*endif\s*-?%\}", "", text)

    # [AUTOTITLE](/path) -> plain path reference (we don't have title resolution)
    text = re.sub(r"\[AUTOTITLE\]\(([^)]+)\)", r"(see: \1)", text)

    # collapse leftover multiple blank lines from removed tags
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_frontmatter(raw_text: str) -> tuple[dict, str]:
    """Splits YAML frontmatter (between --- markers) from the body. Minimal parsing — just title."""
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", raw_text, re.DOTALL)
    if not match:
        return {}, raw_text

    frontmatter_text, body = match.group(1), match.group(2)
    title_match = re.search(r"^title:\s*['\"]?(.+?)['\"]?\s*$", frontmatter_text, re.MULTILINE)
    title = title_match.group(1) if title_match else "Untitled"
    return {"title": title}, body


def fetch_doc(doc_path: str) -> str:
    url = f"{GITHUB_DOCS_RAW_BASE}/{doc_path}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.text


def _doc_url(doc_path: str) -> str:
    """Convert a repo path like 'content/rest/authentication/foo.md' into a real docs.github.com URL."""
    site_path = doc_path.replace("content/", "").replace(".md", "")
    return f"{GITHUB_DOCS_SITE_BASE}/en/{site_path}"


def _split_oversized_section(content: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """
    Splits content that exceeds max_chars into smaller pieces along
    paragraph boundaries (blank-line-separated), greedily packing
    paragraphs until the next one would exceed the limit. Never splits
    mid-paragraph, and never splits inside a fenced code block (checked by
    counting ``` markers so we don't cut a code example in half).
    """
    if len(content) <= max_chars:
        return [content]

    paragraphs = content.split("\n\n")
    parts: list[str] = []
    current = ""
    fence_open = False

    for para in paragraphs:
        # Track whether we're inside an open ``` fence so we never split
        # a multi-paragraph code block across two chunks.
        fence_count_in_para = para.count("```")
        would_toggle_fence = fence_count_in_para % 2 == 1

        candidate = f"{current}\n\n{para}" if current else para

        if len(candidate) > max_chars and current and not fence_open:
            parts.append(current)
            current = para
        else:
            current = candidate

        if would_toggle_fence:
            fence_open = not fence_open

    if current:
        parts.append(current)

    return parts


def parse_doc_to_chunks(doc_path: str, raw_text: str) -> list[dict]:
    """Split one cleaned conceptual doc into H2-section chunks, in the shared chunk schema."""
    metadata, body = _strip_frontmatter(raw_text)
    title = metadata.get("title", "Untitled")
    cleaned_body = _clean_liquid_template(body)

    # Split on H2 headings. Keep the intro (before the first ##) as its own
    # section under the page title, since intros often contain the core
    # definition a query is looking for.
    sections = re.split(r"\n##\s+", cleaned_body)
    doc_slug = doc_path.replace("content/", "").replace(".md", "")
    source_url = _doc_url(doc_path)

    chunks = []
    seen_slugs: dict[str, int] = {}
    for i, section in enumerate(sections):
        if not section.strip():
            continue
        if i == 0:
            heading = title
            content = section
        else:
            lines = section.split("\n", 1)
            heading = lines[0].strip()
            content = lines[1] if len(lines) > 1 else ""

        content = content.strip()
        if len(content) < 30:  # skip near-empty sections (e.g. just a stripped note block)
            continue

        content_parts = _split_oversized_section(content)

        for part_idx, part_content in enumerate(content_parts):
            base_slug = re.sub(r"[^a-z0-9]+", "_", heading.lower()).strip("_")
            if len(content_parts) > 1:
                base_slug = f"{base_slug}_part{part_idx + 1}"

            # Defense in depth: if two sections in the same doc produce the same
            # slug (e.g. identical heading text under different version
            # branches), disambiguate with a counter rather than silently
            # colliding and overwriting one chunk with another downstream.
            if base_slug in seen_slugs:
                seen_slugs[base_slug] += 1
                slug = f"{base_slug}_{seen_slugs[base_slug]}"
            else:
                seen_slugs[base_slug] = 1
                slug = base_slug

            chunk_id = f"conceptual::{doc_slug}::{slug}"

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "endpoint_id": f"CONCEPTUAL {doc_slug}",  # reuses the field so parent-child grouping still works
                    "chunk_type": "conceptual",
                    "text": f"{title} — {heading}\n\n{part_content}",
                    "http_method": "N/A",
                    "api_path": doc_slug,
                    "api_category": "conceptual",
                    "api_subcategory": heading,
                    "auth_required": False,
                    "required_params": [],
                    "optional_params": [],
                    "response_codes": [],
                    "source_url": source_url,
                    "spec_version": "docs-main",
                }
            )
    return chunks


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Fetch and parse GitHub's conceptual docs pages into chunks.")
    ap.add_argument("--out", default="data/processed/conceptual_chunks.json")
    args = ap.parse_args()

    all_chunks = []
    for doc_path in CONCEPTUAL_DOC_PATHS:
        print(f"Fetching {doc_path}...")
        raw_text = fetch_doc(doc_path)
        chunks = parse_doc_to_chunks(doc_path, raw_text)
        print(f"  -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\nTotal: {len(all_chunks)} conceptual chunks from {len(CONCEPTUAL_DOC_PATHS)} pages -> {out_path}")


if __name__ == "__main__":
    main()