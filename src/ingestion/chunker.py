"""
chunker.py

Turns parsed Endpoint objects (from openapi_parser.py) into retrieval-ready
Chunks, following the structural chunking strategy from DESIGN.md section 3:

  - One "description" chunk per endpoint (overview + params + response codes)
  - One "request_schema" chunk if the endpoint has a request body (POST/PUT/PATCH)
  - All chunks for the same endpoint share `endpoint_id` so parent-child
    retrieval (pull the full endpoint doc from any matched chunk) is a simple
    lookup, not a similarity search.

Why split description vs. request schema instead of one giant chunk:
large endpoints (e.g. "Update a pull request") have request bodies with
15+ fields. Cramming that into one chunk with the prose description pushes
well past what embedding models represent well in a single vector, and
buries the params a query is actually about. Splitting keeps each chunk
focused, and parent-child retrieval reassembles the full picture at
generation time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class Chunk:
    chunk_id: str
    endpoint_id: str
    chunk_type: str  # "description" | "request_schema" | "code_example" | "conceptual"
    text: str  # the actual content to embed
    http_method: str
    api_path: str
    api_category: str
    api_subcategory: str
    auth_required: bool
    required_params: list[str]
    optional_params: list[str]
    response_codes: list[str]
    source_url: str | None
    spec_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _format_description_chunk_text(ep: dict) -> str:
    """Human/LLM-readable text for the description+params+responses chunk."""
    lines = [
        f"{ep['http_method']} {ep['api_path']}",
        f"Summary: {ep['summary']}",
        "",
        ep["description"] or "(no additional description provided)",
        "",
    ]

    if ep["parameters"]:
        lines.append("Parameters:")
        for p in ep["parameters"]:
            req = "required" if p["required"] else "optional"
            lines.append(f"  - {p['name']} ({p['location']}, {p['type']}, {req}): {p['description']}")
        lines.append("")

    if ep["response_codes"]:
        lines.append(f"Possible response status codes: {', '.join(ep['response_codes'])}")

    return "\n".join(lines).strip()


def _format_request_schema_chunk_text(ep: dict) -> str:
    """Human/LLM-readable text for the request body schema chunk."""
    schema = ep["request_body_schema"]
    lines = [
        f"Request body schema for {ep['http_method']} {ep['api_path']}",
        f"({ep['summary']})",
        "",
    ]
    required = schema.get("required_fields", [])
    if required:
        lines.append(f"Required fields: {', '.join(required)}")
    lines.append("")
    lines.append("Fields:")
    for name, meta in schema.get("properties", {}).items():
        marker = " (required)" if name in required else " (optional)"
        lines.append(f"  - {name}{marker}, type: {meta.get('type')}: {meta.get('description', '').strip()}")
    return "\n".join(lines).strip()


def chunk_endpoint(ep: dict) -> list[Chunk]:
    """Produce all chunks for a single endpoint."""
    chunks: list[Chunk] = []

    required_params = [p["name"] for p in ep["parameters"] if p["required"]]
    optional_params = [p["name"] for p in ep["parameters"] if not p["required"]]
    if ep.get("request_body_schema"):
        required_params += [
            f"body.{f}" for f in ep["request_body_schema"].get("required_fields", [])
        ]
        optional_params += [
            f"body.{f}"
            for f in ep["request_body_schema"].get("properties", {})
            if f not in ep["request_body_schema"].get("required_fields", [])
        ]

    base_kwargs = dict(
        endpoint_id=ep["endpoint_id"],
        http_method=ep["http_method"],
        api_path=ep["api_path"],
        api_category=ep["api_category"],
        api_subcategory=ep["api_subcategory"],
        auth_required=ep["requires_auth_heuristic"],
        required_params=required_params,
        optional_params=optional_params,
        response_codes=ep["response_codes"],
        source_url=ep["source_url"],
        spec_version=ep["spec_version"],
    )

    # 1. description chunk (always present)
    desc_id = f"{ep['operation_id'] or ep['endpoint_id']}::description".replace(" ", "_")
    chunks.append(
        Chunk(
            chunk_id=desc_id,
            chunk_type="description",
            text=_format_description_chunk_text(ep),
            **base_kwargs,
        )
    )

    # 2. request schema chunk (only if endpoint accepts a body)
    if ep.get("request_body_schema") and ep["request_body_schema"].get("properties"):
        schema_id = f"{ep['operation_id'] or ep['endpoint_id']}::request_schema".replace(" ", "_")
        chunks.append(
            Chunk(
                chunk_id=schema_id,
                chunk_type="request_schema",
                text=_format_request_schema_chunk_text(ep),
                **base_kwargs,
            )
        )

    return chunks


def chunk_all(endpoints: list[dict]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for ep in endpoints:
        all_chunks.extend(chunk_endpoint(ep))
    return all_chunks


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Chunk parsed endpoints into retrieval-ready chunks.")
    ap.add_argument("--endpoints-file", default="data/processed/endpoints.json")
    ap.add_argument("--out", default="data/processed/chunks.json")
    args = ap.parse_args()

    with open(args.endpoints_file, "r", encoding="utf-8") as f:
        endpoints = json.load(f)

    chunks = chunk_all(endpoints)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in chunks], f, indent=2)

    n_desc = sum(1 for c in chunks if c.chunk_type == "description")
    n_schema = sum(1 for c in chunks if c.chunk_type == "request_schema")
    print(f"Chunked {len(endpoints)} endpoints -> {len(chunks)} chunks")
    print(f"  description chunks: {n_desc}")
    print(f"  request_schema chunks: {n_schema}")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()