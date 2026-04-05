from __future__ import annotations

from app.main import app


def _has_path(path: str) -> bool:
    return any(getattr(r, "path", None) == path for r in app.router.routes)


def test_self_model_alias_routes_registered():
    assert _has_path("/api/v1/self-model/bundle")
    assert _has_path("/api/v1/selfmodel/bundle")


def test_llm_and_tools_routes_registered():
    assert _has_path("/api/v1/llm/complete-json")
    assert _has_path("/api/v1/tools")
    assert _has_path("/api/v1/tools/invoke")


def test_goal_agent_routes_registered():
    assert _has_path("/api/v1/goals/propose")
    assert _has_path("/api/v1/goals/link-suggestions")


def test_knowledge_feature_routes_registered():
    assert _has_path("/api/v1/memories/{memory_id}/recommendations")
    assert _has_path("/api/v1/recommendations/metrics")
    assert _has_path("/api/v1/graph/relationships/populate")
    assert _has_path("/api/v1/graph/relationships")
    assert _has_path("/api/v1/topics")
    assert _has_path("/api/v1/knowledge/reports/summary")


def test_explainability_routes_registered():
    assert _has_path("/api/v1/explain/{decision_id}")
    assert _has_path("/api/v1/explain/memory/{memory_id}")
    assert _has_path("/api/v1/explain/conflict/{conflict_id}")
    assert _has_path("/api/v1/explain/anomaly/{anomaly_id}")
    assert _has_path("/api/v1/audit/trail/{session_id}")
