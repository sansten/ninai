"""Tests for Phase 52 - AdaptivePersonaAgent and PersonaProfileService."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.adaptive_persona_agent import (
    AdaptivePersonaAgent,
    _expand_acronyms,
    _strip_parentheticals,
    _truncate_to_sentences,
)
from app.agents.registry import get_agent
from app.agents.types import AgentResult
from app.models.persona_profile import PersonaProfile
from app.services.persona_profile_service import (
    PersonaProfileService,
    _default_ema_for_level,
    _level_from_ema,
    _signal_to_expertise_target,
    _verbosity_from_signal,
)


def _ctx(*, content="API and SLA are degraded.", persona=None, context_type="memory_read"):
    return {
        "memory": {
            "content": content,
            "enrichment": {
                "content": content,
                "persona": persona
                or {
                    "expertise_level": "intermediate",
                    "preferred_verbosity": "normal",
                },
                "context_type": context_type,
            },
        },
        "runtime": {"job_id": "trace-1"},
    }


def _result(outputs, status="success"):
    now = datetime.now(timezone.utc)
    return AgentResult(
        agent_name="AdaptivePersonaAgent",
        agent_version="v1",
        memory_id="m1",
        status=status,
        confidence=0.7,
        outputs=outputs,
        warnings=[],
        errors=[],
        started_at=now,
        finished_at=now,
    )


# ---------------------------------------------------------------------------
# Pure helper tests
# ---------------------------------------------------------------------------

def test_strip_parentheticals_removes_segments():
    assert _strip_parentheticals("A (x) B") == "A B"


def test_truncate_to_sentences_limits_output():
    text = "One. Two. Three."
    assert _truncate_to_sentences(text, 2) == "One. Two."


def test_expand_acronyms_expands_known_terms():
    text, changed = _expand_acronyms("SLA and API")
    assert changed is True
    assert "service level agreement" in text


def test_expand_acronyms_no_change_for_unknown_terms():
    text, changed = _expand_acronyms("plain sentence")
    assert changed is False
    assert text == "plain sentence"


def test_default_ema_by_level():
    assert _default_ema_for_level("novice") == 0.2
    assert _default_ema_for_level("intermediate") == 0.5
    assert _default_ema_for_level("expert") == 0.8


def test_level_from_ema_buckets():
    assert _level_from_ema(0.1) == "novice"
    assert _level_from_ema(0.5) == "intermediate"
    assert _level_from_ema(0.9) == "expert"


def test_verbosity_from_signal_detail():
    assert _verbosity_from_signal({"requested_detail": True, "query_length": 10}) == "detailed"


def test_verbosity_from_signal_brief_short_query():
    assert _verbosity_from_signal({"requested_detail": False, "query_length": 10}) == "brief"


def test_verbosity_from_signal_normal_default():
    assert _verbosity_from_signal({"requested_detail": False, "query_length": 120}) == "normal"


def test_signal_to_expertise_target_increases_with_jargon_and_detail():
    low = _signal_to_expertise_target({"query_length": 20, "used_jargon": False, "requested_detail": False})
    high = _signal_to_expertise_target({"query_length": 200, "used_jargon": True, "requested_detail": True})
    assert high > low


# ---------------------------------------------------------------------------
# Agent heuristic tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("app.agents.adaptive_persona_agent.settings.AGENT_STRATEGY", "heuristic")
async def test_novice_detailed_expands_acronyms_and_adds_context():
    agent = AdaptivePersonaAgent()
    persona = {"expertise_level": "novice", "preferred_verbosity": "detailed"}
    result = await agent.run("m1", _ctx(content="SLA and API issue.", persona=persona))
    assert "service level agreement" in result.outputs["adapted_content"]
    assert "What this means:" in result.outputs["adapted_content"]
    assert "expanded acronyms" in result.outputs["changes_made"]


@pytest.mark.asyncio
@patch("app.agents.adaptive_persona_agent.settings.AGENT_STRATEGY", "heuristic")
async def test_expert_brief_strips_parentheticals_and_truncates():
    agent = AdaptivePersonaAgent()
    persona = {"expertise_level": "expert", "preferred_verbosity": "brief"}
    result = await agent.run("m1", _ctx(content="A (details). First sentence. Second sentence. Third sentence.", persona=persona))
    out = result.outputs["adapted_content"]
    assert "(details)" not in out
    assert out.count(".") <= 2


@pytest.mark.asyncio
@patch("app.agents.adaptive_persona_agent.settings.AGENT_STRATEGY", "heuristic")
async def test_intermediate_normal_pass_through():
    agent = AdaptivePersonaAgent()
    text = "No transformation needed."
    result = await agent.run("m1", _ctx(content=text))
    assert result.outputs["adapted_content"] == text


@pytest.mark.asyncio
@patch("app.agents.adaptive_persona_agent.settings.AGENT_STRATEGY", "heuristic")
async def test_alert_context_forces_brief():
    agent = AdaptivePersonaAgent()
    persona = {"expertise_level": "novice", "preferred_verbosity": "detailed"}
    result = await agent.run("m1", _ctx(content="One. Two. Three.", persona=persona, context_type="alert"))
    assert result.outputs["persona_applied"].endswith("_brief")


@pytest.mark.asyncio
@patch("app.agents.adaptive_persona_agent.settings.AGENT_STRATEGY", "heuristic")
async def test_persona_applied_format():
    agent = AdaptivePersonaAgent()
    persona = {"expertise_level": "expert", "preferred_verbosity": "brief"}
    result = await agent.run("m1", _ctx(persona=persona))
    assert result.outputs["persona_applied"] == "expert_brief"


@pytest.mark.asyncio
@patch("app.agents.adaptive_persona_agent.settings.AGENT_STRATEGY", "heuristic")
async def test_changes_made_is_list():
    agent = AdaptivePersonaAgent()
    persona = {"expertise_level": "expert", "preferred_verbosity": "brief"}
    result = await agent.run("m1", _ctx(persona=persona))
    assert isinstance(result.outputs["changes_made"], list)


@pytest.mark.asyncio
@patch("app.agents.adaptive_persona_agent.settings.AGENT_STRATEGY", "heuristic")
async def test_confidence_higher_when_changes_applied():
    agent = AdaptivePersonaAgent()
    novice = {"expertise_level": "novice", "preferred_verbosity": "detailed"}
    expert = {"expertise_level": "intermediate", "preferred_verbosity": "normal"}
    r_changed = await agent.run("m1", _ctx(content="API", persona=novice))
    r_passthrough = await agent.run("m1", _ctx(content="plain", persona=expert))
    assert r_changed.outputs["confidence"] >= r_passthrough.outputs["confidence"]


@pytest.mark.asyncio
@patch("app.agents.adaptive_persona_agent.settings.AGENT_STRATEGY", "heuristic")
async def test_empty_content_is_supported():
    agent = AdaptivePersonaAgent()
    result = await agent.run("m1", _ctx(content=""))
    assert result.status == "success"


@pytest.mark.asyncio
@patch("app.agents.adaptive_persona_agent.settings.AGENT_STRATEGY", "heuristic")
async def test_trace_id_propagated():
    agent = AdaptivePersonaAgent()
    result = await agent.run("m1", _ctx())
    assert result.trace_id == "trace-1"


# ---------------------------------------------------------------------------
# Agent LLM tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("app.agents.adaptive_persona_agent.settings.AGENT_STRATEGY", "llm")
async def test_llm_valid_json_used():
    agent = AdaptivePersonaAgent()
    fake = AsyncMock()
    fake.complete_json = AsyncMock(
        return_value={
            "adapted_content": "x",
            "persona_applied": "expert_brief",
            "changes_made": ["trimmed"],
            "confidence": 0.9,
        }
    )
    with patch("app.agents.adaptive_persona_agent.create_llm_client", return_value=fake):
        result = await agent.run("m1", _ctx())
    assert result.outputs["rationale"] == "llm"


@pytest.mark.asyncio
@patch("app.agents.adaptive_persona_agent.settings.AGENT_STRATEGY", "llm")
async def test_llm_invalid_json_falls_back_to_heuristic():
    agent = AdaptivePersonaAgent()
    fake = AsyncMock()
    fake.complete_json = AsyncMock(return_value={"bad": "shape"})
    with patch("app.agents.adaptive_persona_agent.create_llm_client", return_value=fake):
        result = await agent.run("m1", _ctx())
    assert result.outputs["rationale"] == "heuristic"


@pytest.mark.asyncio
@patch("app.agents.adaptive_persona_agent.settings.AGENT_STRATEGY", "llm")
async def test_llm_prompt_contains_persona_fields():
    agent = AdaptivePersonaAgent()
    fake = AsyncMock()
    fake.complete_json = AsyncMock(return_value={"adapted_content": "x", "persona_applied": "a_b", "changes_made": []})
    with patch("app.agents.adaptive_persona_agent.create_llm_client", return_value=fake):
        await agent.run("m1", _ctx())
    prompt = fake.complete_json.call_args.kwargs["prompt"]
    assert "PERSONA" in prompt
    assert "CONTEXT_TYPE" in prompt


# ---------------------------------------------------------------------------
# validate_outputs tests
# ---------------------------------------------------------------------------

def test_validate_outputs_passes_valid():
    agent = AdaptivePersonaAgent()
    agent.validate_outputs(_result({"adapted_content": "x", "persona_applied": "a", "changes_made": []}))


def test_validate_outputs_fails_on_non_string_content():
    agent = AdaptivePersonaAgent()
    with pytest.raises(ValueError, match="adapted_content"):
        agent.validate_outputs(_result({"adapted_content": 1, "persona_applied": "a", "changes_made": []}))


def test_validate_outputs_fails_on_non_string_persona_applied():
    agent = AdaptivePersonaAgent()
    with pytest.raises(ValueError, match="persona_applied"):
        agent.validate_outputs(_result({"adapted_content": "x", "persona_applied": 1, "changes_made": []}))


def test_validate_outputs_fails_on_non_list_changes():
    agent = AdaptivePersonaAgent()
    with pytest.raises(ValueError, match="changes_made"):
        agent.validate_outputs(_result({"adapted_content": "x", "persona_applied": "a", "changes_made": "bad"}))


def test_validate_outputs_skips_on_failed_status():
    agent = AdaptivePersonaAgent()
    agent.validate_outputs(_result({"bad": "payload"}, status="failed"))


# ---------------------------------------------------------------------------
# PersonaProfileService tests
# ---------------------------------------------------------------------------

class _ScalarResult:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


@pytest.mark.asyncio
async def test_service_get_or_create_returns_existing_profile():
    svc = PersonaProfileService()
    db = AsyncMock()
    db.add = MagicMock()
    existing = PersonaProfile(user_id="u1", org_id="o1")
    db.execute = AsyncMock(return_value=_ScalarResult(existing))

    out = await svc.get_or_create(db=db, user_id="u1", org_id="o1")

    assert out is existing
    db.add.assert_not_called()


@pytest.mark.asyncio
async def test_service_get_or_create_creates_when_missing():
    svc = PersonaProfileService()
    db = AsyncMock()
    db.add = MagicMock()
    db.execute = AsyncMock(return_value=_ScalarResult(None))

    out = await svc.get_or_create(db=db, user_id="u1", org_id="o1")

    assert out.user_id == "u1"
    assert out.org_id == "o1"
    assert out.expertise_level == "intermediate"
    assert out.preferred_verbosity == "normal"
    db.add.assert_called_once()
    db.flush.assert_awaited()


@pytest.mark.asyncio
async def test_service_update_from_interaction_increments_interaction_count():
    svc = PersonaProfileService()
    db = AsyncMock()
    db.add = MagicMock()
    profile = PersonaProfile(user_id="u1", org_id="o1", interaction_count=0, domain_vocabulary={})
    db.execute = AsyncMock(return_value=_ScalarResult(profile))

    out = await svc.update_from_interaction(
        db=db,
        user_id="u1",
        org_id="o1",
        signal={"query_length": 120, "used_jargon": False, "requested_detail": False},
    )

    assert out.interaction_count == 1


@pytest.mark.asyncio
async def test_service_update_expertise_shifts_toward_expert_with_repeated_signals():
    svc = PersonaProfileService()
    db = AsyncMock()
    db.add = MagicMock()
    profile = PersonaProfile(
        user_id="u1",
        org_id="o1",
        expertise_level="intermediate",
        domain_vocabulary={"_expertise_ema": 0.5},
        interaction_count=0,
    )
    db.execute = AsyncMock(return_value=_ScalarResult(profile))

    for _ in range(8):
        await svc.update_from_interaction(
            db=db,
            user_id="u1",
            org_id="o1",
            signal={"query_length": 240, "used_jargon": True, "requested_detail": True},
        )

    assert profile.expertise_level == "expert"


@pytest.mark.asyncio
async def test_service_update_sets_detailed_verbosity_on_detail_signal():
    svc = PersonaProfileService()
    db = AsyncMock()
    db.add = MagicMock()
    profile = PersonaProfile(user_id="u1", org_id="o1", domain_vocabulary={})
    db.execute = AsyncMock(return_value=_ScalarResult(profile))

    out = await svc.update_from_interaction(
        db=db,
        user_id="u1",
        org_id="o1",
        signal={"query_length": 80, "used_jargon": False, "requested_detail": True},
    )

    assert out.preferred_verbosity == "detailed"


@pytest.mark.asyncio
async def test_service_update_adds_jargon_hint_when_used():
    svc = PersonaProfileService()
    db = AsyncMock()
    db.add = MagicMock()
    profile = PersonaProfile(user_id="u1", org_id="o1", domain_vocabulary={"acronyms": []})
    db.execute = AsyncMock(return_value=_ScalarResult(profile))

    out = await svc.update_from_interaction(
        db=db,
        user_id="u1",
        org_id="o1",
        signal={"query_length": 40, "used_jargon": True, "requested_detail": False},
    )

    assert "domain-jargon" in out.domain_vocabulary.get("acronyms", [])


@pytest.mark.asyncio
async def test_service_get_style_hints_shape():
    svc = PersonaProfileService()
    db = AsyncMock()
    db.add = MagicMock()
    profile = PersonaProfile(
        user_id="u1",
        org_id="o1",
        expertise_level="intermediate",
        preferred_verbosity="normal",
        domain_vocabulary={"acronyms": ["SLA"], "preferred_terms": {"latency": "response time"}},
    )
    db.execute = AsyncMock(return_value=_ScalarResult(profile))

    hints = await svc.get_style_hints(db=db, user_id="u1", org_id="o1")

    assert set(hints.keys()) == {"tone", "verbosity", "vocabulary_hints"}
    assert isinstance(hints["vocabulary_hints"], list)


@pytest.mark.asyncio
async def test_service_get_style_hints_novice_tone_supportive():
    svc = PersonaProfileService()
    db = AsyncMock()
    db.add = MagicMock()
    profile = PersonaProfile(user_id="u1", org_id="o1", expertise_level="novice", preferred_verbosity="normal", domain_vocabulary={})
    db.execute = AsyncMock(return_value=_ScalarResult(profile))

    hints = await svc.get_style_hints(db=db, user_id="u1", org_id="o1")
    assert hints["tone"] == "supportive"


@pytest.mark.asyncio
async def test_service_get_style_hints_expert_tone_technical():
    svc = PersonaProfileService()
    db = AsyncMock()
    db.add = MagicMock()
    profile = PersonaProfile(user_id="u1", org_id="o1", expertise_level="expert", preferred_verbosity="brief", domain_vocabulary={})
    db.execute = AsyncMock(return_value=_ScalarResult(profile))

    hints = await svc.get_style_hints(db=db, user_id="u1", org_id="o1")
    assert hints["tone"] == "technical"


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

def test_registry_returns_adaptive_persona_agent():
    assert isinstance(get_agent("adaptive_persona"), AdaptivePersonaAgent)
    assert isinstance(get_agent("persona"), AdaptivePersonaAgent)


def test_agent_name_and_version():
    agent = AdaptivePersonaAgent()
    assert agent.name == "AdaptivePersonaAgent"
    assert agent.version == "v1"
