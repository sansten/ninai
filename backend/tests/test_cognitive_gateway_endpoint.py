from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.api.v1.endpoints.cognitive_gateway import (
    _apply_query_intelligence_filters,
    _coerce_hnms_mode,
    _expand_query_with_intelligence,
    gateway_answer,
)
from app.schemas.memory import SearchHnmsMode


def test_expand_query_with_query_intelligence_entities_and_intent_terms():
    expanded = _expand_query_with_intelligence(
        "When did the outage happen?",
        {
            "query_intent": "find_timeline",
            "extracted_entities": ["API Gateway", "Incident 42"],
        },
    )
    assert "API Gateway" in expanded
    assert "Incident 42" in expanded
    assert "timeline" in expanded


def test_apply_query_intelligence_filters_respects_min_credibility_and_temporal_bias():
    filtered = _apply_query_intelligence_filters(
        [
            {
                "id": "temporal",
                "content": "timeline event",
                "score": 0.3,
                "occurred_at": "2026-04-29T12:00:00+00:00",
                "credibility_score": 0.8,
            },
            {
                "id": "low-cred",
                "content": "weak note",
                "score": 0.9,
                "credibility_score": 0.2,
            },
        ],
        {
            "dynamic_filters": {
                "min_credibility": 0.5,
                "has_temporal_data": True,
            }
        },
    )
    assert [m["id"] for m in filtered] == ["temporal"]


def test_coerce_hnms_mode_accepts_string_values():
    assert _coerce_hnms_mode("research") == SearchHnmsMode.RESEARCH
    assert _coerce_hnms_mode("invalid") is None


@pytest.mark.asyncio
async def test_gateway_answer_endpoint_exposes_answer_source_and_llm_error():
    class _FakeGateway:
        async def answer(self, **kwargs):
            return SimpleNamespace(
                answer="Stockholm",
                model="heuristic",
                context_turns=0,
                used_llm=False,
                answer_source="heuristic",
                llm_error="HTTPError(404, 'model not found')",
            )

    result = await gateway_answer(
        payload={"question": "Where?", "memories": [{"content": "They moved to Stockholm."}]},
        tenant=SimpleNamespace(),
        gateway=_FakeGateway(),
    )
    assert result["answer_source"] == "heuristic"
    assert result["llm_error"] == "HTTPError(404, 'model not found')"
