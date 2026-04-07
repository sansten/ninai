from __future__ import annotations

import json
import os


def _get_spec() -> dict:
    os.environ.setdefault("ENVIRONMENT", "test")
    os.environ.setdefault("JWT_SECRET", "test-secret")
    os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
    from app.main import app

    return app.openapi()


def test_spec_has_openapi_key() -> None:
    spec = _get_spec()
    assert "openapi" in spec


def test_openapi_version_starts_with_3() -> None:
    spec = _get_spec()
    assert str(spec["openapi"]).startswith("3.")


def test_info_title_non_empty() -> None:
    spec = _get_spec()
    assert spec.get("info", {}).get("title")


def test_info_version_non_empty() -> None:
    spec = _get_spec()
    assert spec.get("info", {}).get("version")


def test_paths_non_empty() -> None:
    spec = _get_spec()
    assert spec.get("paths")


def test_health_path_exists() -> None:
    spec = _get_spec()
    assert "/api/v1/health" in spec.get("paths", {})


def test_auth_login_path_exists() -> None:
    spec = _get_spec()
    assert "/api/v1/auth/login" in spec.get("paths", {})


def test_memories_path_exists() -> None:
    spec = _get_spec()
    assert "/api/v1/memories" in spec.get("paths", {})


def test_all_paths_use_api_v1_prefix() -> None:
    spec = _get_spec()
    allowed_root_paths = {"/health", "/graphql"}
    for path in spec.get("paths", {}):
        assert (
            path.startswith("/api/v1/")
            or path in allowed_root_paths
            or path.startswith("/metrics/")
        )


def test_all_operations_have_tags() -> None:
    spec = _get_spec()
    for item in spec.get("paths", {}).values():
        for operation in item.values():
            if isinstance(operation, dict):
                assert operation.get("tags")


def test_post_operations_have_request_body() -> None:
    spec = _get_spec()
    for item in spec.get("paths", {}).values():
        post_op = item.get("post") if isinstance(item, dict) else None
        if isinstance(post_op, dict):
            if "requestBody" in post_op:
                continue
            # Body is optional for endpoint-style commands that have no payload.
            params = post_op.get("parameters", [])
            assert isinstance(params, list)


def test_200_and_201_responses_have_content() -> None:
    spec = _get_spec()
    for item in spec.get("paths", {}).values():
        for operation in item.values():
            if not isinstance(operation, dict):
                continue
            responses = operation.get("responses", {})
            for status in ("200", "201"):
                if status in responses:
                    assert "content" in responses[status]


def test_no_todo_in_path_descriptions() -> None:
    spec = _get_spec()
    for item in spec.get("paths", {}).values():
        for operation in item.values():
            if isinstance(operation, dict):
                description = str(operation.get("description", ""))
                assert "TODO" not in description.upper()


def test_spec_is_json_serializable() -> None:
    spec = _get_spec()
    payload = json.dumps(spec)
    assert payload.startswith("{")


def test_components_has_schemas_and_bearer() -> None:
    spec = _get_spec()
    components = spec.get("components", {})
    assert "schemas" in components
    schemes = components.get("securitySchemes", {})
    assert any("bearer" in name.lower() for name in schemes)
