"""
openapi_parser.py

Parses GitHub's official REST API OpenAPI spec (github/rest-api-description)
into structured Endpoint objects — one per (path, HTTP method) operation.

Source of truth: https://raw.githubusercontent.com/github/rest-api-description/main/descriptions/api.github.com/api.github.com.json

Design note (important, honest trade-off):
GitHub's public OpenAPI spec does NOT include machine-readable OAuth scope
requirements per endpoint — that information only exists in prose on the
docs site. So `requires_auth` here is a *heuristic* (derived from HTTP
method + path pattern), not ground truth. We flag it as such in the
metadata rather than pretending it's precise. A future enrichment pass
could scrape the prose "Fine-grained access tokens" tables per endpoint
page to get exact scopes — noted as a Phase 2 improvement.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


GITHUB_SPEC_URL = (
    "https://raw.githubusercontent.com/github/rest-api-description/main/"
    "descriptions/api.github.com/api.github.com.json"
)


@dataclass
class Parameter:
    name: str
    location: str  # "path" | "query" | "header"
    required: bool
    type: str
    description: str


@dataclass
class Endpoint:
    endpoint_id: str  # e.g. "POST /repos/{owner}/{repo}/issues"
    http_method: str
    api_path: str
    operation_id: str
    summary: str
    description: str
    api_category: str
    api_subcategory: str
    parameters: list[Parameter] = field(default_factory=list)
    request_body_schema: dict[str, Any] | None = None
    response_codes: list[str] = field(default_factory=list)
    response_schema_summary: dict[str, Any] | None = None
    source_url: str | None = None
    spec_version: str = ""
    requires_auth_heuristic: bool = True
    github_app_enabled: bool = False
    triggers_notification: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


class OpenAPIParser:
    def __init__(self, spec: dict[str, Any]):
        self.spec = spec
        self.spec_version = spec.get("info", {}).get("version", "unknown")

    @classmethod
    def from_file(cls, path: str | Path) -> "OpenAPIParser":
        with open(path, "r", encoding="utf-8") as f:
            spec = json.load(f)
        return cls(spec)

    # ---- $ref resolution -------------------------------------------------

    def _resolve_ref(self, ref: str) -> Any:
        """Resolve a local '#/components/...' JSON pointer against the full spec."""
        assert ref.startswith("#/"), f"Only local refs supported, got: {ref}"
        node: Any = self.spec
        for part in ref[2:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            node = node[part]
        return node

    def _deref(self, node: Any, _depth: int = 0) -> Any:
        """Recursively resolve $ref pointers in a nested dict/list structure."""
        if _depth > 15:  # guard against pathological recursive refs
            return node
        if isinstance(node, dict):
            if "$ref" in node and len(node) == 1:
                resolved = self._resolve_ref(node["$ref"])
                return self._deref(resolved, _depth + 1)
            return {k: self._deref(v, _depth + 1) for k, v in node.items()}
        if isinstance(node, list):
            return [self._deref(item, _depth + 1) for item in node]
        return node

    # ---- extraction helpers ------------------------------------------------

    def _extract_parameters(self, raw_params: list[dict]) -> list[Parameter]:
        params = []
        for p in raw_params:
            resolved = self._deref(p)
            schema = resolved.get("schema", {})
            params.append(
                Parameter(
                    name=resolved.get("name", ""),
                    location=resolved.get("in", ""),
                    required=bool(resolved.get("required", False)),
                    type=schema.get("type", "unknown"),
                    description=(resolved.get("description") or "").strip(),
                )
            )
        return params

    def _extract_request_body_schema(self, request_body: dict | None) -> dict | None:
        if not request_body:
            return None
        resolved = self._deref(request_body)
        content = resolved.get("content", {})
        json_content = content.get("application/json", {})
        schema = json_content.get("schema", {})
        if not schema:
            return None
        # Simplify: keep top-level property names, types, required flags, descriptions.
        # We deliberately don't keep the FULL nested schema in metadata (too heavy) —
        # the full schema lives in the chunk text itself, this is just for filtering.
        props = schema.get("properties", {})
        simplified = {
            "type": schema.get("type"),
            "required_fields": schema.get("required", []),
            "properties": {
                name: {
                    "type": pdef.get("type", pdef.get("oneOf", [{}])[0].get("type", "unknown")),
                    "description": (pdef.get("description") or "")[:200],
                }
                for name, pdef in props.items()
            },
        }
        return simplified

    def _guess_requires_auth(self, method: str, path: str, x_github: dict) -> bool:
        """
        Heuristic only (see module docstring). GET requests to non-user-scoped
        public resources are often unauthenticated; everything else (POST/PUT/
        PATCH/DELETE, or paths under /user, /orgs/*/settings, etc.) is treated
        as requiring auth. This is intentionally conservative.
        """
        if method.upper() != "GET":
            return True
        if path.startswith("/user") or "/settings" in path or "/installation" in path:
            return True
        return False

    def _category_from_tags(self, op: dict) -> tuple[str, str]:
        x_github = op.get("x-github", {})
        category = x_github.get("category") or (op.get("tags", ["uncategorized"])[0])
        subcategory = x_github.get("subcategory") or ""
        return category, subcategory

    # ---- main parse ---------------------------------------------------------

    def parse(self) -> list[Endpoint]:
        endpoints: list[Endpoint] = []
        paths = self.spec.get("paths", {})

        for api_path, path_item in paths.items():
            for method, op in path_item.items():
                if method.lower() not in {"get", "post", "put", "patch", "delete"}:
                    continue  # skip "parameters", "description" siblings etc.

                op_resolved_params = self._extract_parameters(op.get("parameters", []))
                x_github = op.get("x-github", {})
                category, subcategory = self._category_from_tags(op)
                external_docs = op.get("externalDocs", {})

                endpoint = Endpoint(
                    endpoint_id=f"{method.upper()} {api_path}",
                    http_method=method.upper(),
                    api_path=api_path,
                    operation_id=op.get("operationId", ""),
                    summary=op.get("summary", "").strip(),
                    description=(op.get("description") or "").strip(),
                    api_category=category,
                    api_subcategory=subcategory,
                    parameters=op_resolved_params,
                    request_body_schema=self._extract_request_body_schema(op.get("requestBody")),
                    response_codes=sorted(op.get("responses", {}).keys()),
                    source_url=external_docs.get("url"),
                    spec_version=self.spec_version,
                    requires_auth_heuristic=self._guess_requires_auth(method, api_path, x_github),
                    github_app_enabled=bool(x_github.get("enabledForGitHubApps", False)),
                    triggers_notification=bool(x_github.get("triggersNotification", False)),
                )
                endpoints.append(endpoint)

        return endpoints


def main():
    """CLI entry point: parse the local spec file and write structured JSON."""
    import argparse

    ap = argparse.ArgumentParser(description="Parse GitHub's OpenAPI spec into structured endpoints.")
    ap.add_argument("--spec-file", default="data/raw/api.github.com.json")
    ap.add_argument("--out", default="data/processed/endpoints.json")
    ap.add_argument("--limit", type=int, default=None, help="Only parse first N endpoints (debugging)")
    args = ap.parse_args()

    parser = OpenAPIParser.from_file(args.spec_file)
    endpoints = parser.parse()
    if args.limit:
        endpoints = endpoints[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([e.to_dict() for e in endpoints], f, indent=2)

    print(f"Parsed {len(endpoints)} endpoints -> {out_path}")
    categories = sorted({e.api_category for e in endpoints})
    print(f"Categories found ({len(categories)}): {categories[:15]}{'...' if len(categories) > 15 else ''}")


if __name__ == "__main__":
    main()