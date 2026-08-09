"""Tests for TheoryOfMindAgent — Phase 38."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from app.agents.theory_of_mind_agent import (
    TheoryOfMindAgent,
    _KNOWLEDGE_LEVELS,
    _TONE_ADJUSTMENTS,
    _DETAIL_LEVELS,
    _EXPERT_WORD_THRESHOLD,
    _EXPERT_INTERACTION_THRESHOLD,
    _INTERMEDIATE_WORD_THRESHOLD,
    _INTERMEDIATE_INTERACTION_THRESHOLD,
    _HIGH_CONFIDENCE_THRESHOLD,
    _LOW_CONFIDENCE_THRESHOLD,
    _POSITIVE_FEEDBACK_THRESHOLD,
    _MAX_EXPERTISE_DOMAINS,
    _MAX_PEER_MODELS,
    _infer_knowledge_level,
    _tone_from_level,
    _detail_from_level,
    _build_user_model,
    _build_peer_agent_models,
    _infer_intent,
    _safe_default,
    _parse_llm_output,
    run_heuristic_async,
)
from app.agents.registry import get_agent


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _interaction(
    query: str = "what is the budget?",
    domain: str = "finance",
    word_count: int | None = None,
) -> dict:
    entry: dict = {"query": query, "domain": domain}
    if word_count is not None:
        entry["word_count"] = word_count
    return entry


def _feedback(sentiment: str = "positive", category: str = "accuracy", domain: str = "finance") -> dict:
    return {"sentiment": sentiment, "category": category, "domain": domain}


def _peer_result(status: str = "success", confidence: float = 0.8, domain: str = "finance") -> dict:
    return {"status": status, "confidence": confidence, "domain": domain}


def _make_context(
    user_interactions: list | None = None,
    feedback_signals: list | None = None,
    peer_agent_results: dict | None = None,
    attention_signals: dict | None = None,
    job_id: str | None = None,
) -> dict:
    enrichment: dict = {}
    if user_interactions is not None:
        enrichment["user_interactions"] = user_interactions
    if feedback_signals is not None:
        enrichment["feedback_signals"] = feedback_signals
    if peer_agent_results is not None:
        enrichment["peer_agent_results"] = peer_agent_results
    if attention_signals is not None:
        enrichment["attention_signals"] = attention_signals
    ctx: dict = {"memory": {"enrichment": enrichment}}
    if job_id:
        ctx["runtime"] = {"job_id": job_id}
    return ctx


# ---------------------------------------------------------------------------
# _infer_knowledge_level
# ---------------------------------------------------------------------------

def test_knowledge_level_expert_both_thresholds():
    assert _infer_knowledge_level(
        float(_EXPERT_WORD_THRESHOLD),
        _EXPERT_INTERACTION_THRESHOLD,
    ) == "expert"


def test_knowledge_level_expert_exceeds_thresholds():
    assert _infer_knowledge_level(20.0, 10) == "expert"


def test_knowledge_level_intermediate_word_count():
    assert _infer_knowledge_level(float(_INTERMEDIATE_WORD_THRESHOLD), 1) == "intermediate"


def test_knowledge_level_intermediate_interaction_count():
    assert _infer_knowledge_level(5.0, _INTERMEDIATE_INTERACTION_THRESHOLD) == "intermediate"


def test_knowledge_level_novice_low_everything():
    assert _infer_knowledge_level(3.0, 1) == "novice"


def test_knowledge_level_expert_needs_both():
    # Meets word threshold but not interaction count → intermediate
    assert _infer_knowledge_level(float(_EXPERT_WORD_THRESHOLD), 1) == "intermediate"


# ---------------------------------------------------------------------------
# _tone_from_level
# ---------------------------------------------------------------------------

def test_tone_expert_is_technical():
    assert _tone_from_level("expert") == "technical"


def test_tone_novice_is_simplified():
    assert _tone_from_level("novice") == "simplified"


def test_tone_intermediate_is_neutral():
    assert _tone_from_level("intermediate") == "neutral"


# ---------------------------------------------------------------------------
# _detail_from_level
# ---------------------------------------------------------------------------

def test_detail_expert_is_detailed():
    assert _detail_from_level("expert", 0) == "detailed"


def test_detail_novice_is_brief():
    assert _detail_from_level("novice", 0) == "brief"


def test_detail_intermediate_standard():
    assert _detail_from_level("intermediate", 0) == "standard"


def test_detail_positive_feedback_overrides_novice():
    assert _detail_from_level("novice", _POSITIVE_FEEDBACK_THRESHOLD) == "detailed"


def test_detail_positive_feedback_overrides_intermediate():
    assert _detail_from_level("intermediate", _POSITIVE_FEEDBACK_THRESHOLD) == "detailed"


# ---------------------------------------------------------------------------
# _build_user_model
# ---------------------------------------------------------------------------

def test_build_user_model_empty_interactions():
    model = _build_user_model([], [])
    assert model["knowledge_level"] == "novice"
    assert model["inferred_expertise"] == []
    assert model["interaction_count"] == 0


def test_build_user_model_expert_from_word_count():
    interactions = [
        _interaction(
            query="analyse the quarterly financial projections including variance",
            domain="finance",
            word_count=_EXPERT_WORD_THRESHOLD,
        )
    ] * _EXPERT_INTERACTION_THRESHOLD
    model = _build_user_model(interactions, [])
    assert model["knowledge_level"] == "expert"


def test_build_user_model_novice_short_queries():
    interactions = [_interaction(query="help me", domain="hr", word_count=2)]
    model = _build_user_model(interactions, [])
    assert model["knowledge_level"] == "novice"


def test_build_user_model_domain_expertise_extracted():
    interactions = [
        _interaction(domain="finance", word_count=5),
        _interaction(domain="finance", word_count=5),
        _interaction(domain="legal", word_count=5),
    ]
    model = _build_user_model(interactions, [])
    assert "finance" in model["inferred_expertise"]
    assert model["inferred_expertise"][0] == "finance"


def test_build_user_model_unknown_domain_excluded():
    interactions = [_interaction(domain="unknown", word_count=5)]
    model = _build_user_model(interactions, [])
    assert "unknown" not in model["inferred_expertise"]


def test_build_user_model_max_expertise_domains():
    interactions = [_interaction(domain=f"domain_{i}", word_count=5) for i in range(10)]
    model = _build_user_model(interactions, [])
    assert len(model["inferred_expertise"]) <= _MAX_EXPERTISE_DOMAINS


def test_build_user_model_word_count_computed_from_query():
    # No word_count field; computed from query
    interactions = [_interaction(query="short query", domain="it")]
    model = _build_user_model(interactions, [])
    assert isinstance(model["interaction_count"], int)


def test_build_user_model_positive_feedback_detail_level():
    interactions = [_interaction(domain="legal", word_count=5)]
    feedback = [_feedback(sentiment="positive")] * _POSITIVE_FEEDBACK_THRESHOLD
    model = _build_user_model(interactions, feedback)
    assert model["preferred_detail_level"] == "detailed"


def test_build_user_model_no_feedback_novice_brief():
    interactions = [_interaction(query="ok", domain="hr", word_count=1)]
    model = _build_user_model(interactions, [])
    assert model["preferred_detail_level"] == "brief"


def test_build_user_model_intermediate_standard_detail():
    interactions = [_interaction(domain="it", word_count=_INTERMEDIATE_WORD_THRESHOLD)]
    model = _build_user_model(interactions, [])
    assert model["preferred_detail_level"] == "standard"


def test_build_user_model_mixed_sentiments():
    interactions = [_interaction(domain="finance", word_count=5)]
    feedback = [
        _feedback(sentiment="positive"),
        _feedback(sentiment="negative"),
        _feedback(sentiment="neutral"),
    ]
    model = _build_user_model(interactions, feedback)
    # Only 1 positive — below threshold
    assert model["preferred_detail_level"] in _DETAIL_LEVELS


def test_build_user_model_interaction_count_correct():
    interactions = [_interaction() for _ in range(7)]
    model = _build_user_model(interactions, [])
    assert model["interaction_count"] == 7


def test_build_user_model_helpful_sentiment_counts():
    interactions = [_interaction(domain="ops", word_count=5)]
    feedback = [_feedback(sentiment="helpful")] * _POSITIVE_FEEDBACK_THRESHOLD
    model = _build_user_model(interactions, feedback)
    assert model["preferred_detail_level"] == "detailed"


# ---------------------------------------------------------------------------
# _build_peer_agent_models
# ---------------------------------------------------------------------------

def test_build_peer_empty_dict():
    assert _build_peer_agent_models({}) == []


def test_build_peer_all_success():
    results = {"AgentA": [_peer_result("success", 0.9, "finance")] * 3}
    models = _build_peer_agent_models(results)
    assert len(models) == 1
    assert models[0]["reliability_score"] == 1.0


def test_build_peer_all_failure():
    results = {"AgentB": [_peer_result("failure", 0.2, "hr")] * 3}
    models = _build_peer_agent_models(results)
    assert models[0]["reliability_score"] == 0.0


def test_build_peer_mixed_reliability():
    results = {"AgentC": [
        _peer_result("success", 0.8, "finance"),
        _peer_result("failure", 0.3, "hr"),
    ]}
    models = _build_peer_agent_models(results)
    assert models[0]["reliability_score"] == 0.5


def test_build_peer_strength_domain_detected():
    results = {"AgentD": [_peer_result("success", _HIGH_CONFIDENCE_THRESHOLD, "legal")]}
    models = _build_peer_agent_models(results)
    assert "legal" in models[0]["strength_domains"]


def test_build_peer_weakness_domain_detected():
    results = {"AgentE": [_peer_result("failure", _LOW_CONFIDENCE_THRESHOLD - 0.01, "ops")]}
    models = _build_peer_agent_models(results)
    assert "ops" in models[0]["weakness_domains"]


def test_build_peer_empty_results_list_skipped():
    results = {"AgentF": []}
    models = _build_peer_agent_models(results)
    assert models == []


def test_build_peer_multiple_agents():
    results = {
        "AgentG": [_peer_result("success", 0.8, "finance")],
        "AgentH": [_peer_result("failure", 0.2, "hr")],
    }
    models = _build_peer_agent_models(results)
    assert len(models) == 2


def test_build_peer_confidence_averaged():
    results = {"AgentI": [
        _peer_result("success", 0.6, "it"),
        _peer_result("success", 0.8, "it"),
    ]}
    models = _build_peer_agent_models(results)
    assert abs(models[0]["confidence"] - 0.7) < 0.01


def test_build_peer_max_models_capped():
    results = {f"Agent_{i}": [_peer_result()] for i in range(_MAX_PEER_MODELS + 5)}
    models = _build_peer_agent_models(results)
    assert len(models) <= _MAX_PEER_MODELS


def test_build_peer_non_dict_results_ignored():
    results = {"AgentJ": ["not_a_dict", None, _peer_result("success", 0.7, "finance")]}
    models = _build_peer_agent_models(results)
    # Only the valid dict result is counted
    assert models[0]["reliability_score"] == 1.0


# ---------------------------------------------------------------------------
# _infer_intent
# ---------------------------------------------------------------------------

def test_infer_intent_most_common_domain():
    interactions = [
        _interaction(domain="finance"),
        _interaction(domain="finance"),
        _interaction(domain="hr"),
    ]
    assert _infer_intent(interactions, {}) == "finance"


def _attention_signal(topic: str, relevance: float = 0.8) -> dict:
    """OrgAttentionAgent's real output shape, not the {focus_domain: ...}
    dict this test file previously (incorrectly) assumed."""
    return {"team": topic, "topic": topic, "entity_type": "topic", "relevance": relevance}


def test_infer_intent_no_interactions_uses_attention():
    assert _infer_intent([], [_attention_signal("legal")]) == "legal"


def test_infer_intent_no_signals_returns_general():
    assert _infer_intent([], []) == "general"


def test_infer_intent_interactions_override_attention():
    interactions = [_interaction(domain="ops")]
    assert _infer_intent(interactions, [_attention_signal("finance")]) == "ops"


def test_infer_intent_unknown_domain_skipped():
    interactions = [_interaction(domain="unknown")]
    assert _infer_intent(interactions, [_attention_signal("legal")]) == "legal"


def test_infer_intent_empty_domain_skipped():
    interactions = [{"query": "help", "domain": ""}]
    assert _infer_intent(interactions, [_attention_signal("hr")]) == "hr"


def test_infer_intent_picks_highest_relevance_signal():
    signals = [_attention_signal("hr", relevance=0.3), _attention_signal("legal", relevance=0.9)]
    assert _infer_intent([], signals) == "legal"


def test_infer_intent_single_valid_domain():
    interactions = [_interaction(domain="legal")]
    assert _infer_intent(interactions, {}) == "legal"


# ---------------------------------------------------------------------------
# _safe_default
# ---------------------------------------------------------------------------

def test_safe_default_structure():
    d = _safe_default()
    assert "user_model" in d
    assert "peer_agent_models" in d
    assert "inferred_intent" in d
    assert "tone_adjustment" in d
    assert "confidence" in d


def test_safe_default_values():
    d = _safe_default()
    assert d["user_model"]["knowledge_level"] == "novice"
    assert d["peer_agent_models"] == []
    assert d["inferred_intent"] == "general"
    assert d["tone_adjustment"] == "neutral"
    assert d["confidence"] == 0.5
    assert d["rationale"] == "heuristic"


# ---------------------------------------------------------------------------
# _parse_llm_output
# ---------------------------------------------------------------------------

def _valid_llm_resp(**overrides) -> dict:
    base = {
        "user_model": {
            "knowledge_level": "intermediate",
            "inferred_expertise": ["finance"],
            "preferred_detail_level": "standard",
            "interaction_count": 4,
        },
        "peer_agent_models": [
            {
                "agent_name": "AgentA",
                "reliability_score": 0.8,
                "strength_domains": ["finance"],
                "weakness_domains": [],
                "confidence": 0.75,
            }
        ],
        "inferred_intent": "finance",
        "tone_adjustment": "neutral",
        "confidence": 0.7,
    }
    base.update(overrides)
    return base


def test_parse_llm_valid():
    result = _parse_llm_output(_valid_llm_resp())
    assert result is not None
    assert result["rationale"] == "llm"
    assert result["tone_adjustment"] == "neutral"


def test_parse_llm_not_dict():
    assert _parse_llm_output("invalid") is None


def test_parse_llm_user_model_missing():
    resp = _valid_llm_resp()
    del resp["user_model"]
    assert _parse_llm_output(resp) is None


def test_parse_llm_knowledge_level_invalid():
    resp = _valid_llm_resp()
    resp["user_model"]["knowledge_level"] = "guru"
    assert _parse_llm_output(resp) is None


def test_parse_llm_tone_adjustment_invalid():
    resp = _valid_llm_resp(tone_adjustment="casual")
    assert _parse_llm_output(resp) is None


def test_parse_llm_confidence_out_of_range():
    assert _parse_llm_output(_valid_llm_resp(confidence=1.5)) is None


def test_parse_llm_confidence_negative():
    assert _parse_llm_output(_valid_llm_resp(confidence=-0.1)) is None


def test_parse_llm_peer_models_not_list():
    assert _parse_llm_output(_valid_llm_resp(peer_agent_models="bad")) is None


def test_parse_llm_inferred_intent_missing():
    resp = _valid_llm_resp()
    del resp["inferred_intent"]
    assert _parse_llm_output(resp) is None


def test_parse_llm_inferred_intent_empty():
    assert _parse_llm_output(_valid_llm_resp(inferred_intent="   ")) is None


def test_parse_llm_peer_missing_agent_name_skipped():
    resp = _valid_llm_resp()
    resp["peer_agent_models"] = [{"reliability_score": 0.5}]  # no agent_name
    result = _parse_llm_output(resp)
    assert result is not None
    assert result["peer_agent_models"] == []


def test_parse_llm_peer_non_dict_skipped():
    resp = _valid_llm_resp()
    resp["peer_agent_models"] = ["not_a_dict"]
    result = _parse_llm_output(resp)
    assert result is not None
    assert result["peer_agent_models"] == []


def test_parse_llm_detail_level_invalid_corrected():
    resp = _valid_llm_resp()
    resp["user_model"]["preferred_detail_level"] = "verbose"
    result = _parse_llm_output(resp)
    assert result is not None
    assert result["user_model"]["preferred_detail_level"] == "standard"


def test_parse_llm_user_model_not_dict():
    resp = _valid_llm_resp()
    resp["user_model"] = "not_a_dict"
    assert _parse_llm_output(resp) is None


# ---------------------------------------------------------------------------
# run_heuristic_async
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_heuristic_empty_inputs():
    result = await run_heuristic_async([], [], {}, {})
    assert result["user_model"]["knowledge_level"] == "novice"
    assert result["peer_agent_models"] == []
    assert result["inferred_intent"] == "general"
    assert result["tone_adjustment"] == "simplified"
    assert result["confidence"] == 0.5
    assert result["rationale"] == "heuristic"


@pytest.mark.asyncio
async def test_heuristic_expert_user():
    interactions = [
        _interaction(query=" ".join(["word"] * _EXPERT_WORD_THRESHOLD), domain="finance")
    ] * _EXPERT_INTERACTION_THRESHOLD
    result = await run_heuristic_async(interactions, [], {}, {})
    assert result["user_model"]["knowledge_level"] == "expert"
    assert result["tone_adjustment"] == "technical"


@pytest.mark.asyncio
async def test_heuristic_novice_user():
    interactions = [_interaction(query="help", domain="hr", word_count=1)]
    result = await run_heuristic_async(interactions, [], {}, {})
    assert result["user_model"]["knowledge_level"] == "novice"
    assert result["tone_adjustment"] == "simplified"


@pytest.mark.asyncio
async def test_heuristic_intent_from_interactions():
    interactions = [
        _interaction(domain="legal"),
        _interaction(domain="legal"),
        _interaction(domain="finance"),
    ]
    result = await run_heuristic_async(interactions, [], {}, {})
    assert result["inferred_intent"] == "legal"


@pytest.mark.asyncio
async def test_heuristic_intent_from_attention():
    result = await run_heuristic_async([], [], {}, [_attention_signal("ops")])
    assert result["inferred_intent"] == "ops"


@pytest.mark.asyncio
async def test_heuristic_peer_models_built():
    peers = {
        "AgentX": [_peer_result("success", 0.9, "finance")],
        "AgentY": [_peer_result("failure", 0.3, "hr")],
    }
    result = await run_heuristic_async([], [], peers, {})
    assert len(result["peer_agent_models"]) == 2


@pytest.mark.asyncio
async def test_heuristic_confidence_from_peers():
    peers = {"AgentZ": [_peer_result("success", 0.8, "it")]}
    result = await run_heuristic_async([], [], peers, {})
    assert result["confidence"] == pytest.approx(0.8, abs=0.01)


@pytest.mark.asyncio
async def test_heuristic_no_peers_confidence_half():
    result = await run_heuristic_async([], [], {}, {})
    assert result["confidence"] == 0.5


@pytest.mark.asyncio
async def test_heuristic_positive_feedback_detail_level():
    interactions = [_interaction(query="ok", domain="it", word_count=1)]
    feedback = [_feedback(sentiment="positive")] * _POSITIVE_FEEDBACK_THRESHOLD
    result = await run_heuristic_async(interactions, feedback, {}, {})
    assert result["user_model"]["preferred_detail_level"] == "detailed"


@pytest.mark.asyncio
async def test_heuristic_multiple_peers_avg_confidence():
    peers = {
        "P1": [_peer_result("success", 0.6, "hr")],
        "P2": [_peer_result("success", 0.8, "it")],
    }
    result = await run_heuristic_async([], [], peers, {})
    assert result["confidence"] == pytest.approx(0.7, abs=0.01)


@pytest.mark.asyncio
async def test_heuristic_intermediate_user():
    interactions = [
        _interaction(domain="it", word_count=_INTERMEDIATE_WORD_THRESHOLD)
    ] * (_INTERMEDIATE_INTERACTION_THRESHOLD - 1)
    result = await run_heuristic_async(interactions, [], {}, {})
    assert result["user_model"]["knowledge_level"] == "intermediate"
    assert result["tone_adjustment"] == "neutral"


# ---------------------------------------------------------------------------
# TheoryOfMindAgent.validate_outputs
# ---------------------------------------------------------------------------

def _make_agent() -> TheoryOfMindAgent:
    return TheoryOfMindAgent()


def _make_result(outputs: dict, status: str = "success"):
    from app.agents.types import AgentResult
    now = datetime.now(timezone.utc)
    return AgentResult(
        agent_name="TheoryOfMindAgent",
        agent_version="v1",
        memory_id="mem-1",
        status=status,
        confidence=0.7,
        outputs=outputs,
        warnings=[],
        errors=[],
        started_at=now,
        finished_at=now,
        trace_id=None,
    )


def _valid_outputs() -> dict:
    return {
        "user_model": {
            "knowledge_level": "intermediate",
            "inferred_expertise": ["finance"],
            "preferred_detail_level": "standard",
            "interaction_count": 3,
        },
        "peer_agent_models": [],
        "inferred_intent": "finance",
        "tone_adjustment": "neutral",
        "confidence": 0.7,
        "rationale": "heuristic",
    }


def test_validate_outputs_valid():
    agent = _make_agent()
    agent.validate_outputs(_make_result(_valid_outputs()))  # no exception


def test_validate_outputs_user_model_not_dict():
    agent = _make_agent()
    outputs = _valid_outputs()
    outputs["user_model"] = "bad"
    with pytest.raises(ValueError, match="user_model"):
        agent.validate_outputs(_make_result(outputs))


def test_validate_outputs_knowledge_level_invalid():
    agent = _make_agent()
    outputs = _valid_outputs()
    outputs["user_model"]["knowledge_level"] = "guru"
    with pytest.raises(ValueError, match="knowledge_level"):
        agent.validate_outputs(_make_result(outputs))


def test_validate_outputs_peer_models_not_list():
    agent = _make_agent()
    outputs = _valid_outputs()
    outputs["peer_agent_models"] = "bad"
    with pytest.raises(ValueError, match="peer_agent_models"):
        agent.validate_outputs(_make_result(outputs))


def test_validate_outputs_inferred_intent_not_str():
    agent = _make_agent()
    outputs = _valid_outputs()
    outputs["inferred_intent"] = 42
    with pytest.raises(ValueError, match="inferred_intent"):
        agent.validate_outputs(_make_result(outputs))


def test_validate_outputs_inferred_intent_empty():
    agent = _make_agent()
    outputs = _valid_outputs()
    outputs["inferred_intent"] = ""
    with pytest.raises(ValueError, match="inferred_intent"):
        agent.validate_outputs(_make_result(outputs))


def test_validate_outputs_tone_invalid():
    agent = _make_agent()
    outputs = _valid_outputs()
    outputs["tone_adjustment"] = "casual"
    with pytest.raises(ValueError, match="tone_adjustment"):
        agent.validate_outputs(_make_result(outputs))


def test_validate_outputs_confidence_out_of_range():
    agent = _make_agent()
    outputs = _valid_outputs()
    outputs["confidence"] = 1.5
    with pytest.raises(ValueError, match="confidence"):
        agent.validate_outputs(_make_result(outputs))


def test_validate_outputs_confidence_not_numeric():
    agent = _make_agent()
    outputs = _valid_outputs()
    outputs["confidence"] = "high"
    with pytest.raises(ValueError, match="confidence"):
        agent.validate_outputs(_make_result(outputs))


def test_validate_outputs_skipped_on_non_success():
    agent = _make_agent()
    bad_outputs = {"confidence": 99}  # would fail if validated
    agent.validate_outputs(_make_result(bad_outputs, status="failed"))  # no exception


# ---------------------------------------------------------------------------
# TheoryOfMindAgent.run
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_heuristic_strategy():
    agent = _make_agent()
    ctx = _make_context(user_interactions=[_interaction(domain="finance", word_count=5)])
    with patch("app.agents.theory_of_mind_agent.settings") as mock_settings:
        mock_settings.AGENT_STRATEGY = "heuristic"
        result = await agent.run("mem-1", ctx)
    assert result.status == "success"
    assert result.outputs["rationale"] == "heuristic"


@pytest.mark.asyncio
async def test_run_llm_strategy_valid_response():
    agent = _make_agent()
    ctx = _make_context()
    llm_resp = {
        "user_model": {
            "knowledge_level": "expert",
            "inferred_expertise": ["finance"],
            "preferred_detail_level": "detailed",
            "interaction_count": 10,
        },
        "peer_agent_models": [],
        "inferred_intent": "finance",
        "tone_adjustment": "technical",
        "confidence": 0.85,
    }
    mock_client = MagicMock()
    mock_client.complete_json = AsyncMock(return_value=llm_resp)
    with patch("app.agents.theory_of_mind_agent.settings") as mock_settings, \
         patch("app.agents.theory_of_mind_agent.create_llm_client", return_value=mock_client):
        mock_settings.AGENT_STRATEGY = "llm"
        mock_settings.VLLM_BASE_URL = "http://localhost:11434"
        mock_settings.VLLM_MODEL = "qwen2.5:7b"
        mock_settings.VLLM_TIMEOUT_SECONDS = 5.0
        mock_settings.VLLM_MAX_CONCURRENCY = 2
        result = await agent.run("mem-1", ctx)
    assert result.status == "success"
    assert result.outputs["rationale"] == "llm"
    assert result.outputs["tone_adjustment"] == "technical"


@pytest.mark.asyncio
async def test_run_llm_invalid_response_falls_back():
    agent = _make_agent()
    ctx = _make_context()
    mock_client = MagicMock()
    mock_client.complete_json = AsyncMock(return_value={"bad": "response"})
    with patch("app.agents.theory_of_mind_agent.settings") as mock_settings, \
         patch("app.agents.theory_of_mind_agent.create_llm_client", return_value=mock_client):
        mock_settings.AGENT_STRATEGY = "llm"
        mock_settings.VLLM_BASE_URL = "http://localhost:11434"
        mock_settings.VLLM_MODEL = "qwen2.5:7b"
        mock_settings.VLLM_TIMEOUT_SECONDS = 5.0
        mock_settings.VLLM_MAX_CONCURRENCY = 2
        result = await agent.run("mem-1", ctx)
    assert result.status == "success"
    assert result.outputs["rationale"] == "heuristic"


@pytest.mark.asyncio
async def test_run_llm_exception_falls_back():
    agent = _make_agent()
    ctx = _make_context()
    mock_client = MagicMock()
    mock_client.complete_json = AsyncMock(side_effect=RuntimeError("timeout"))
    with patch("app.agents.theory_of_mind_agent.settings") as mock_settings, \
         patch("app.agents.theory_of_mind_agent.create_llm_client", return_value=mock_client):
        mock_settings.AGENT_STRATEGY = "llm"
        mock_settings.VLLM_BASE_URL = "http://localhost:11434"
        mock_settings.VLLM_MODEL = "qwen2.5:7b"
        mock_settings.VLLM_TIMEOUT_SECONDS = 5.0
        mock_settings.VLLM_MAX_CONCURRENCY = 2
        result = await agent.run("mem-1", ctx)
    assert result.status == "success"
    assert result.outputs["rationale"] == "heuristic"


@pytest.mark.asyncio
async def test_run_empty_context():
    agent = _make_agent()
    with patch("app.agents.theory_of_mind_agent.settings") as mock_settings:
        mock_settings.AGENT_STRATEGY = "heuristic"
        result = await agent.run("mem-x", {})
    assert result.status == "success"
    assert result.outputs["inferred_intent"] == "general"


@pytest.mark.asyncio
async def test_run_expert_user_context():
    agent = _make_agent()
    interactions = [
        _interaction(domain="legal", word_count=_EXPERT_WORD_THRESHOLD)
    ] * _EXPERT_INTERACTION_THRESHOLD
    ctx = _make_context(user_interactions=interactions)
    with patch("app.agents.theory_of_mind_agent.settings") as mock_settings:
        mock_settings.AGENT_STRATEGY = "heuristic"
        result = await agent.run("mem-2", ctx)
    assert result.outputs["user_model"]["knowledge_level"] == "expert"
    assert result.outputs["tone_adjustment"] == "technical"


@pytest.mark.asyncio
async def test_run_novice_user_context():
    agent = _make_agent()
    ctx = _make_context(user_interactions=[_interaction(query="hi", word_count=1)])
    with patch("app.agents.theory_of_mind_agent.settings") as mock_settings:
        mock_settings.AGENT_STRATEGY = "heuristic"
        result = await agent.run("mem-3", ctx)
    assert result.outputs["tone_adjustment"] == "simplified"


@pytest.mark.asyncio
async def test_run_peer_agents_modeled():
    agent = _make_agent()
    peers = {"CredibilityAgent": [_peer_result("success", 0.9, "finance")]}
    ctx = _make_context(peer_agent_results=peers)
    with patch("app.agents.theory_of_mind_agent.settings") as mock_settings:
        mock_settings.AGENT_STRATEGY = "heuristic"
        result = await agent.run("mem-4", ctx)
    assert len(result.outputs["peer_agent_models"]) == 1
    assert result.outputs["peer_agent_models"][0]["agent_name"] == "CredibilityAgent"


@pytest.mark.asyncio
async def test_run_timestamps_set():
    agent = _make_agent()
    with patch("app.agents.theory_of_mind_agent.settings") as mock_settings:
        mock_settings.AGENT_STRATEGY = "heuristic"
        result = await agent.run("mem-5", {})
    assert result.started_at is not None
    assert result.finished_at is not None
    assert result.finished_at >= result.started_at


@pytest.mark.asyncio
async def test_run_trace_id_propagated():
    agent = _make_agent()
    ctx = _make_context(job_id="job-99")
    with patch("app.agents.theory_of_mind_agent.settings") as mock_settings:
        mock_settings.AGENT_STRATEGY = "heuristic"
        result = await agent.run("mem-6", ctx)
    assert result.trace_id == "job-99"


# ---------------------------------------------------------------------------
# Agent metadata
# ---------------------------------------------------------------------------

def test_dependencies():
    agent = _make_agent()
    deps = agent.dependencies()
    assert "CredibilityAgent" in deps
    assert "OrgAttentionAgent" in deps
    assert "FeedbackIntegrationAgent" in deps


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_theory_of_mind():
    agent = get_agent("theory_of_mind")
    assert isinstance(agent, TheoryOfMindAgent)


def test_registry_theoryofmindagent():
    agent = get_agent("theoryofmindagent")
    assert isinstance(agent, TheoryOfMindAgent)


def test_registry_theoryofmind():
    agent = get_agent("theoryofmind")
    assert isinstance(agent, TheoryOfMindAgent)
