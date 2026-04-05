"""Export versioned OpenAPI schema for SDK generation.

This script produces backend/openapi_v1.json and ensures each operation has:
- tags
- operationId
- x-ninai-cognitive-domain
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.main import app

HTTP_METHODS = {"get", "post", "put", "patch", "delete", "options", "head"}


def _make_operation_id(method: str, path: str) -> str:
    cleaned = path.strip("/").replace("{", "").replace("}", "")
    cleaned = cleaned.replace("/", "_").replace("-", "_")
    if not cleaned:
        cleaned = "root"
    return f"{method.lower()}_{cleaned}"


def _make_domain(tag: str) -> str:
    return str(tag or "general").strip().lower().replace(" ", "-")


def main() -> None:
    spec = app.openapi()

    for path, item in spec.get("paths", {}).items():
        for method, operation in item.items():
            if method.lower() not in HTTP_METHODS or not isinstance(operation, dict):
                continue

            tags = operation.get("tags") or ["General"]
            operation["tags"] = tags

            operation_id = operation.get("operationId")
            if not operation_id:
                operation["operationId"] = _make_operation_id(method, path)

            if "x-ninai-cognitive-domain" not in operation:
                operation["x-ninai-cognitive-domain"] = _make_domain(tags[0])

    out = Path(__file__).resolve().parents[1] / "openapi_v1.json"
    out.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"Wrote {len(spec.get('paths', {}))} paths to {out}")


if __name__ == "__main__":
    main()
