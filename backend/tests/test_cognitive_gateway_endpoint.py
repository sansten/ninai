from __future__ import annotations

from types import SimpleNamespace

import pytest

import app.api.v1.endpoints.cognitive_gateway as gateway_endpoint_module
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
    called = {"set_tenant_context": 0}

    async def _fake_set_tenant_context(*args, **kwargs):
        called["set_tenant_context"] += 1
        return None

    gateway_endpoint_module.set_tenant_context = _fake_set_tenant_context

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
        payload={
            "question": "Where?",
            "memories": [{"content": "They moved to Stockholm."}],
            "prompt_override": "Answer with one place only.",
        },
        tenant=SimpleNamespace(
            user_id="user-1",
            org_id="org-1",
            roles_string="org_admin",
            clearance_level=5,
        ),
        requester=SimpleNamespace(),
        db=SimpleNamespace(),
        gateway=_FakeGateway(),
    )
    assert result["answer_source"] == "heuristic"
    assert result["llm_error"] == "HTTPError(404, 'model not found')"
    assert result["llm_failure_mode"] is None
    assert result["llm_endpoint"] is None
    assert called["set_tenant_context"] == 0


@pytest.mark.asyncio
async def test_gateway_answer_endpoint_routes_large_structured_prompt_through_grounded_service(monkeypatch):
    called = {"set_tenant_context": 0, "grounded": 0, "gateway": 0}

    async def _fake_set_tenant_context(*args, **kwargs):
        called["set_tenant_context"] += 1
        return None

    gateway_endpoint_module.set_tenant_context = _fake_set_tenant_context

    class _FakeGroundedService:
        def __init__(self, gateway):
            self.gateway = gateway

        async def answer(self, **kwargs):
            called["grounded"] += 1
            evidence_package = kwargs["evidence_package"]
            assert evidence_package["memory_hits"]
            return SimpleNamespace(
                answer="researching adoption agencies",
                model="test-model",
                context_turns=3,
                used_llm=True,
                answer_source="gateway_compressed",
                llm_error=None,
                llm_failure_mode=None,
                llm_endpoint="http://ollama-primary:11434",
                grounded=True,
                confidence=0.8,
                support=["memory:Adoption agencies"],
                contradictions=[],
                uncertainty_reason=None,
            )

    monkeypatch.setattr(gateway_endpoint_module, "GroundedAnswerService", _FakeGroundedService)

    class _FakeGateway:
        async def answer(self, **kwargs):
            called["gateway"] += 1
            return SimpleNamespace(answer="should not be used")

    large_prompt = "Conversation:\n" + ("Turn 1: filler line.\n" * 120) + "\nQuestion: What did Melanie plan?\nAnswer:"

    result = await gateway_answer(
        payload={
            "question": "What are Melanie's plans for the summer with respect to adoption?",
            "memories": [
                {
                    "content": "Melanie planned to spend the summer researching adoption agencies.",
                    "extra_metadata": {
                        "fact_supporting_facts": [
                            {
                                "subject": "Melanie",
                                "predicate": "summer_plan",
                                "object": "researching adoption agencies",
                                "status": "active",
                                "confidence": 0.95,
                            }
                        ]
                    },
                }
            ],
            "prompt_override": large_prompt,
        },
        tenant=SimpleNamespace(
            user_id="user-1",
            org_id="org-1",
            roles_string="org_admin",
            clearance_level=5,
        ),
        requester=SimpleNamespace(),
        db=SimpleNamespace(),
        gateway=_FakeGateway(),
    )

    assert result["answer"] == "researching adoption agencies"
    assert result["answer_source"] == "gateway_compressed"
    assert result["llm_endpoint"] == "http://ollama-primary:11434"
    assert called["grounded"] == 1
    assert called["gateway"] == 0
    assert called["set_tenant_context"] == 0
