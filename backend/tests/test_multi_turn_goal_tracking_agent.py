"""Tests for MultiTurnGoalTrackingAgent — Phase 34."""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.multi_turn_goal_tracking_agent import (
    MultiTurnGoalTrackingAgent,
    _goal_description,
    _tokenize,
    classify_goal_stalled,
    compute_agent_confidence,
    compute_continuity_score,
    match_prior_goal,
    run_heuristic,
)
from app.agents.registry import get_agent
from app.agents.types import AgentResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mk_goal(
    description: str = "deploy the service",
    subtasks: list[str] | None = None,
    completion_fraction: float = 0.4,
    blocking_subtask: str | None = None,
    goal_id: str | None = None,
) -> dict[str, Any]:
    return {
        "goal_id": goal_id or str(uuid.uuid4()),
        "description": description,
        "subtasks": subtasks or [description],
        "completion_fraction": completion_fraction,
        "last_seen": "2026-01-01T00:00:00+00:00",
        "blocking_subtask": blocking_subtask,
    }


def _mk_enrichment(
    goal_detected: bool = True,
    subtasks: list[str] | None = None,
    completion_fraction: float = 0.5,
    blocking_subtask: str | None = None,
    temporal_phase: str = "execution",
) -> dict[str, Any]:
    return {
        "goal_detected": goal_detected,
        "subtasks": subtasks or ["deploy the service", "run tests"],
        "completion_fraction": completion_fraction,
        "blocking_subtask": blocking_subtask,
        "temporal_phase": temporal_phase,
    }


def _mk_context(
    enrichment: dict[str, Any] | None = None,
    session_goals: list[dict[str, Any]] | None = None,
    content: str = "We need to deploy the service and run tests.",
    strategy: str = "heuristic",
) -> dict[str, Any]:
    return {
        "memory": {
            "content": content,
            "enrichment": enrichment or _mk_enrichment(),
        },
        "session_context": {"goals": session_goals or []},
        "runtime": {"job_id": "test-job"},
        "_strategy_override": strategy,
    }


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------

def test_tokenize_basic():
    tokens = _tokenize("Deploy the service now")
    assert "deploy" in tokens
    assert "service" in tokens
    # single/two-char words excluded
    assert "a" not in tokens
    assert "to" not in tokens


def test_tokenize_empty():
    assert _tokenize("") == set()


def test_tokenize_deduplicates():
    tokens = _tokenize("run run run tests tests")
    assert tokens == {"run", "tests"}


# ---------------------------------------------------------------------------
# _goal_description
# ---------------------------------------------------------------------------

def test_goal_description_from_subtasks():
    desc = _goal_description(["deploy the service", "run tests"])
    assert desc == "deploy the service"


def test_goal_description_from_content_fallback():
    desc = _goal_description([], "We plan to migrate the database.")
    assert "migrate" in desc


def test_goal_description_empty():
    assert _goal_description([], "") == ""


def test_goal_description_truncates():
    long_subtask = "x" * 300
    desc = _goal_description([long_subtask])
    assert len(desc) == 200


# ---------------------------------------------------------------------------
# match_prior_goal
# ---------------------------------------------------------------------------

def test_match_prior_goal_empty_session():
    assert match_prior_goal(["deploy service"], []) is None


def test_match_prior_goal_no_subtasks():
    goals = [_mk_goal()]
    assert match_prior_goal([], goals) is None


def test_match_prior_goal_finds_best():
    goals = [
        _mk_goal("analyze logs", subtasks=["analyze logs pipeline"]),
        _mk_goal("deploy the service", subtasks=["deploy the service to production"]),
    ]
    current = ["deploy the service", "run integration tests"]
    result = match_prior_goal(current, goals)
    assert result is not None
    assert result["description"] == "deploy the service"


def test_match_prior_goal_no_match():
    goals = [_mk_goal("analyze logs", subtasks=["analyze logs pipeline"])]
    current = ["migrate database schema"]
    result = match_prior_goal(current, goals, threshold=5)
    assert result is None


def test_match_prior_goal_uses_description_tokens():
    goals = [_mk_goal("deploy service to kubernetes", subtasks=[])]
    current = ["deploy service to kubernetes cluster"]
    result = match_prior_goal(current, goals)
    assert result is not None


def test_match_prior_goal_threshold():
    goals = [_mk_goal("run tests", subtasks=["run tests"])]
    current = ["run tests now"]
    # threshold=2 should match "run" + "tests"
    result = match_prior_goal(current, goals, threshold=2)
    assert result is not None


# ---------------------------------------------------------------------------
# classify_goal_stalled
# ---------------------------------------------------------------------------

def test_classify_goal_stalled_due_to_stored_blocking():
    goal = _mk_goal(blocking_subtask="waiting on security review")
    assert classify_goal_stalled(goal, None, "execution") is True


def test_classify_goal_stalled_due_to_current_blocking():
    goal = _mk_goal(blocking_subtask=None, completion_fraction=0.5)
    assert classify_goal_stalled(goal, "waiting on approval", "execution") is True


def test_classify_goal_stalled_low_fraction_planning():
    goal = _mk_goal(completion_fraction=0.05, blocking_subtask=None)
    assert classify_goal_stalled(goal, None, "planning") is True


def test_classify_goal_not_stalled_normal():
    goal = _mk_goal(completion_fraction=0.5, blocking_subtask=None)
    assert classify_goal_stalled(goal, None, "execution") is False


def test_classify_goal_not_stalled_planning_above_threshold():
    goal = _mk_goal(completion_fraction=0.50, blocking_subtask=None)
    assert classify_goal_stalled(goal, None, "planning") is False


def test_classify_goal_stalled_fraction_zero_planning():
    goal = _mk_goal(completion_fraction=0.0, blocking_subtask=None)
    assert classify_goal_stalled(goal, None, "planning") is True


# ---------------------------------------------------------------------------
# compute_continuity_score
# ---------------------------------------------------------------------------

def test_continuity_score_no_goals():
    score = compute_continuity_score(
        is_returning=False,
        active_count=0,
        stalled_count=0,
        completion_fractions=[],
    )
    assert score == 0.0


def test_continuity_score_returning():
    score = compute_continuity_score(
        is_returning=True,
        active_count=1,
        stalled_count=0,
        completion_fractions=[0.5],
    )
    assert 0.7 <= score <= 1.0


def test_continuity_score_active_only():
    score = compute_continuity_score(
        is_returning=False,
        active_count=2,
        stalled_count=0,
        completion_fractions=[0.3, 0.6],
    )
    assert 0.4 < score <= 1.0


def test_continuity_score_stalled_no_active_not_returning():
    score = compute_continuity_score(
        is_returning=False,
        active_count=0,
        stalled_count=2,
        completion_fractions=[],
    )
    assert score == 0.0


def test_continuity_score_bounded():
    score = compute_continuity_score(
        is_returning=True,
        active_count=10,
        stalled_count=0,
        completion_fractions=[1.0] * 10,
    )
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# compute_agent_confidence
# ---------------------------------------------------------------------------

def test_confidence_all_signals():
    c = compute_agent_confidence(
        goal_detected=True,
        has_session_goals=True,
        is_returning=True,
        active_count=2,
    )
    assert 0.80 <= c <= 0.88


def test_confidence_no_signals():
    c = compute_agent_confidence(
        goal_detected=False,
        has_session_goals=False,
        is_returning=False,
        active_count=0,
    )
    assert c == 0.50


def test_confidence_goal_detected_only():
    c = compute_agent_confidence(
        goal_detected=True,
        has_session_goals=False,
        is_returning=False,
        active_count=0,
    )
    assert c == 0.60


# ---------------------------------------------------------------------------
# run_heuristic
# ---------------------------------------------------------------------------

def test_heuristic_no_goal_no_session():
    enrichment = _mk_enrichment(goal_detected=False, subtasks=[])
    out = run_heuristic(enrichment, [], "some content")
    assert out["active_goals"] == []
    assert out["stalled_goals"] == []
    assert out["goal_continuity_score"] == 0.0
    assert out["is_returning_goal"] is False
    assert out["rationale"] == "heuristic"


def test_heuristic_new_goal_no_session():
    enrichment = _mk_enrichment(goal_detected=True, subtasks=["deploy the service", "run tests"])
    out = run_heuristic(enrichment, [], "deploy")
    assert len(out["active_goals"]) == 1
    assert out["stalled_goals"] == []
    assert out["is_returning_goal"] is False
    assert out["goal_continuity_score"] > 0.0


def test_heuristic_returning_goal():
    prior = _mk_goal("deploy the service", subtasks=["deploy the service", "run tests"])
    enrichment = _mk_enrichment(goal_detected=True, subtasks=["deploy the service", "verify rollout"])
    out = run_heuristic(enrichment, [prior], "deploy")
    assert out["is_returning_goal"] is True
    # Active goals should include the returned goal (not stalled)
    assert any(g["goal_id"] == prior["goal_id"] for g in out["active_goals"])


def test_heuristic_stalled_new_goal():
    enrichment = _mk_enrichment(
        goal_detected=True,
        subtasks=["migrate database"],
        blocking_subtask="waiting on DBA approval",
    )
    out = run_heuristic(enrichment, [], "migrate")
    assert len(out["stalled_goals"]) == 1
    assert out["stalled_goals"][0]["stall_reason"] == "waiting on DBA approval"
    assert out["active_goals"] == []


def test_heuristic_carries_forward_prior_active():
    other_goal = _mk_goal("analyze logs", subtasks=["analyze logs"], completion_fraction=0.5)
    enrichment = _mk_enrichment(goal_detected=True, subtasks=["deploy service"])
    out = run_heuristic(enrichment, [other_goal], "deploy")
    # Both goals should appear: new one + carried-forward other
    ids = [g["goal_id"] for g in out["active_goals"]]
    assert other_goal["goal_id"] in ids


def test_heuristic_carries_forward_prior_stalled():
    stalled_prior = _mk_goal(
        "fix bug", subtasks=["fix bug"], blocking_subtask="waiting on logs"
    )
    enrichment = _mk_enrichment(goal_detected=False, subtasks=[])
    out = run_heuristic(enrichment, [stalled_prior], "")
    assert any(g["goal_id"] == stalled_prior["goal_id"] for g in out["stalled_goals"])


def test_heuristic_matched_goal_not_duplicated():
    prior = _mk_goal("deploy the service", subtasks=["deploy the service"])
    enrichment = _mk_enrichment(goal_detected=True, subtasks=["deploy the service"])
    out = run_heuristic(enrichment, [prior], "deploy")
    # The prior goal should appear exactly once (as the matched goal)
    all_ids = [g["goal_id"] for g in out["active_goals"] + out["stalled_goals"]]
    assert all_ids.count(prior["goal_id"]) == 1


def test_heuristic_multiple_prior_goals():
    goals = [
        _mk_goal("deploy service", subtasks=["deploy service"], completion_fraction=0.5),
        _mk_goal("fix monitoring", subtasks=["fix monitoring"], completion_fraction=0.3),
        _mk_goal("update docs", subtasks=["update docs"], completion_fraction=0.2),
    ]
    enrichment = _mk_enrichment(goal_detected=False, subtasks=[])
    out = run_heuristic(enrichment, goals, "")
    assert len(out["active_goals"]) == 3


def test_heuristic_max_goals_capped():
    goals = [
        _mk_goal(f"goal {i}", subtasks=[f"subtask {i}"], completion_fraction=0.5)
        for i in range(25)
    ]
    enrichment = _mk_enrichment(goal_detected=False, subtasks=[])
    out = run_heuristic(enrichment, goals, "")
    assert len(out["active_goals"]) <= 20


def test_heuristic_low_fraction_planning_stalls():
    prior = _mk_goal("plan migration", subtasks=["plan migration"], completion_fraction=0.05)
    enrichment = _mk_enrichment(goal_detected=False, subtasks=[], temporal_phase="planning")
    out = run_heuristic(enrichment, [prior], "")
    assert any(g["goal_id"] == prior["goal_id"] for g in out["stalled_goals"])


def test_heuristic_goal_id_assigned_to_new_goal():
    enrichment = _mk_enrichment(goal_detected=True, subtasks=["implement feature"])
    out = run_heuristic(enrichment, [], "implement feature")
    assert len(out["active_goals"]) == 1
    gid = out["active_goals"][0]["goal_id"]
    assert isinstance(gid, str) and len(gid) > 0


def test_heuristic_completion_fraction_updated_on_return():
    prior = _mk_goal("deploy service", subtasks=["deploy service"], completion_fraction=0.2)
    enrichment = _mk_enrichment(
        goal_detected=True,
        subtasks=["deploy service"],
        completion_fraction=0.8,
    )
    out = run_heuristic(enrichment, [prior], "deploy")
    assert out["is_returning_goal"] is True
    matched = next(g for g in out["active_goals"] if g["goal_id"] == prior["goal_id"])
    assert matched["completion_fraction"] == 0.8


# ---------------------------------------------------------------------------
# Agent — validate_outputs
# ---------------------------------------------------------------------------

def _make_result(outputs: dict[str, Any], status: str = "success") -> AgentResult:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    return AgentResult(
        agent_name="MultiTurnGoalTrackingAgent",
        agent_version="v1",
        memory_id="m1",
        status=status,
        confidence=outputs.get("confidence", 0.5),
        outputs=outputs,
        warnings=[],
        errors=[],
        started_at=now,
        finished_at=now,
        trace_id=None,
    )


def test_validate_outputs_valid():
    agent = MultiTurnGoalTrackingAgent()
    result = _make_result({
        "active_goals": [],
        "stalled_goals": [],
        "goal_continuity_score": 0.5,
        "is_returning_goal": False,
        "confidence": 0.7,
        "rationale": "heuristic",
    })
    agent.validate_outputs(result)  # no error


def test_validate_outputs_missing_active_goals():
    agent = MultiTurnGoalTrackingAgent()
    result = _make_result({
        "stalled_goals": [],
        "goal_continuity_score": 0.5,
        "is_returning_goal": False,
        "confidence": 0.7,
    })
    with pytest.raises(ValueError, match="active_goals"):
        agent.validate_outputs(result)


def test_validate_outputs_missing_stalled_goals():
    agent = MultiTurnGoalTrackingAgent()
    result = _make_result({
        "active_goals": [],
        "goal_continuity_score": 0.5,
        "is_returning_goal": False,
        "confidence": 0.7,
    })
    with pytest.raises(ValueError, match="stalled_goals"):
        agent.validate_outputs(result)


def test_validate_outputs_invalid_continuity_score():
    agent = MultiTurnGoalTrackingAgent()
    result = _make_result({
        "active_goals": [],
        "stalled_goals": [],
        "goal_continuity_score": 1.5,
        "is_returning_goal": False,
        "confidence": 0.7,
    })
    with pytest.raises(ValueError, match="goal_continuity_score"):
        agent.validate_outputs(result)


def test_validate_outputs_is_returning_not_bool():
    agent = MultiTurnGoalTrackingAgent()
    result = _make_result({
        "active_goals": [],
        "stalled_goals": [],
        "goal_continuity_score": 0.5,
        "is_returning_goal": "yes",
        "confidence": 0.7,
    })
    with pytest.raises(ValueError, match="is_returning_goal"):
        agent.validate_outputs(result)


def test_validate_outputs_invalid_confidence():
    agent = MultiTurnGoalTrackingAgent()
    # Build result with valid top-level confidence but inject invalid in outputs dict
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    result = AgentResult(
        agent_name="MultiTurnGoalTrackingAgent",
        agent_version="v1",
        memory_id="m1",
        status="success",
        confidence=0.5,  # valid at model level
        outputs={
            "active_goals": [],
            "stalled_goals": [],
            "goal_continuity_score": 0.5,
            "is_returning_goal": False,
            "confidence": -0.1,  # invalid in outputs dict
        },
        warnings=[],
        errors=[],
        started_at=now,
        finished_at=now,
        trace_id=None,
    )
    with pytest.raises(ValueError, match="confidence"):
        agent.validate_outputs(result)


def test_validate_outputs_skips_non_success():
    agent = MultiTurnGoalTrackingAgent()
    result = _make_result({}, status="failed")
    agent.validate_outputs(result)  # no error — skipped


# ---------------------------------------------------------------------------
# Agent — run (heuristic path)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_heuristic_no_goal():
    agent = MultiTurnGoalTrackingAgent()
    ctx = _mk_context(
        enrichment=_mk_enrichment(goal_detected=False, subtasks=[]),
        strategy="heuristic",
    )
    with patch.object(agent, "_get_strategy", return_value="heuristic", create=True):
        with patch("app.agents.multi_turn_goal_tracking_agent.settings") as mock_settings:
            mock_settings.MULTI_TURN_GOAL_STRATEGY = "heuristic"
            result = await agent.run("m1", ctx)

    assert result.status == "success"
    assert result.outputs["active_goals"] == []
    assert result.outputs["is_returning_goal"] is False


@pytest.mark.asyncio
async def test_run_heuristic_new_goal():
    agent = MultiTurnGoalTrackingAgent()
    ctx = _mk_context(
        enrichment=_mk_enrichment(goal_detected=True, subtasks=["deploy the service"]),
        strategy="heuristic",
    )
    with patch("app.agents.multi_turn_goal_tracking_agent.settings") as mock_settings:
        mock_settings.MULTI_TURN_GOAL_STRATEGY = "heuristic"
        result = await agent.run("m1", ctx)

    assert result.status == "success"
    assert len(result.outputs["active_goals"]) == 1
    assert result.outputs["is_returning_goal"] is False


@pytest.mark.asyncio
async def test_run_heuristic_returning_goal():
    prior_id = str(uuid.uuid4())
    prior = _mk_goal("deploy the service", subtasks=["deploy the service"], goal_id=prior_id)
    agent = MultiTurnGoalTrackingAgent()
    ctx = _mk_context(
        enrichment=_mk_enrichment(goal_detected=True, subtasks=["deploy the service"]),
        session_goals=[prior],
    )
    with patch("app.agents.multi_turn_goal_tracking_agent.settings") as mock_settings:
        mock_settings.MULTI_TURN_GOAL_STRATEGY = "heuristic"
        result = await agent.run("m1", ctx)

    assert result.outputs["is_returning_goal"] is True
    assert any(g["goal_id"] == prior_id for g in result.outputs["active_goals"])


@pytest.mark.asyncio
async def test_run_heuristic_stalled_goal():
    agent = MultiTurnGoalTrackingAgent()
    ctx = _mk_context(
        enrichment=_mk_enrichment(
            goal_detected=True,
            subtasks=["migrate db"],
            blocking_subtask="waiting on DBA",
        ),
    )
    with patch("app.agents.multi_turn_goal_tracking_agent.settings") as mock_settings:
        mock_settings.MULTI_TURN_GOAL_STRATEGY = "heuristic"
        result = await agent.run("m1", ctx)

    assert len(result.outputs["stalled_goals"]) == 1
    assert result.outputs["active_goals"] == []


@pytest.mark.asyncio
async def test_run_heuristic_with_session_context():
    prior = _mk_goal("analyze logs", subtasks=["analyze logs"])
    agent = MultiTurnGoalTrackingAgent()
    ctx = _mk_context(
        enrichment=_mk_enrichment(goal_detected=False, subtasks=[]),
        session_goals=[prior],
    )
    with patch("app.agents.multi_turn_goal_tracking_agent.settings") as mock_settings:
        mock_settings.MULTI_TURN_GOAL_STRATEGY = "heuristic"
        result = await agent.run("m1", ctx)

    all_ids = [g["goal_id"] for g in result.outputs["active_goals"] + result.outputs["stalled_goals"]]
    assert prior["goal_id"] in all_ids


@pytest.mark.asyncio
async def test_run_trace_id_propagated():
    agent = MultiTurnGoalTrackingAgent()
    ctx = _mk_context(enrichment=_mk_enrichment(goal_detected=False, subtasks=[]))
    ctx["runtime"] = {"job_id": "trace-42"}
    with patch("app.agents.multi_turn_goal_tracking_agent.settings") as mock_settings:
        mock_settings.MULTI_TURN_GOAL_STRATEGY = "heuristic"
        result = await agent.run("m1", ctx)

    assert result.trace_id == "trace-42"


@pytest.mark.asyncio
async def test_run_no_session_context():
    agent = MultiTurnGoalTrackingAgent()
    ctx = {
        "memory": {"content": "x", "enrichment": _mk_enrichment(goal_detected=False)},
        "runtime": {},
    }
    with patch("app.agents.multi_turn_goal_tracking_agent.settings") as mock_settings:
        mock_settings.MULTI_TURN_GOAL_STRATEGY = "heuristic"
        result = await agent.run("m1", ctx)

    assert result.status == "success"
    assert result.outputs["active_goals"] == []


@pytest.mark.asyncio
async def test_run_empty_context():
    agent = MultiTurnGoalTrackingAgent()
    with patch("app.agents.multi_turn_goal_tracking_agent.settings") as mock_settings:
        mock_settings.MULTI_TURN_GOAL_STRATEGY = "heuristic"
        result = await agent.run("m1", {})

    assert result.status == "success"


@pytest.mark.asyncio
async def test_run_null_enrichment():
    agent = MultiTurnGoalTrackingAgent()
    ctx = {"memory": {"content": "x", "enrichment": None}, "session_context": {"goals": []}}
    with patch("app.agents.multi_turn_goal_tracking_agent.settings") as mock_settings:
        mock_settings.MULTI_TURN_GOAL_STRATEGY = "heuristic"
        result = await agent.run("m1", ctx)

    assert result.status == "success"


@pytest.mark.asyncio
async def test_run_goal_continuity_score_range():
    agent = MultiTurnGoalTrackingAgent()
    ctx = _mk_context(
        enrichment=_mk_enrichment(goal_detected=True, subtasks=["build feature"]),
    )
    with patch("app.agents.multi_turn_goal_tracking_agent.settings") as mock_settings:
        mock_settings.MULTI_TURN_GOAL_STRATEGY = "heuristic"
        result = await agent.run("m1", ctx)

    gcs = result.outputs["goal_continuity_score"]
    assert 0.0 <= gcs <= 1.0


@pytest.mark.asyncio
async def test_run_confidence_range():
    agent = MultiTurnGoalTrackingAgent()
    ctx = _mk_context(enrichment=_mk_enrichment())
    with patch("app.agents.multi_turn_goal_tracking_agent.settings") as mock_settings:
        mock_settings.MULTI_TURN_GOAL_STRATEGY = "heuristic"
        result = await agent.run("m1", ctx)

    assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_llm_path_success():
    agent = MultiTurnGoalTrackingAgent()
    llm_resp = {
        "active_goals": [{"goal_id": "g1", "description": "deploy", "completion_fraction": 0.6, "last_seen": "now"}],
        "stalled_goals": [],
        "goal_continuity_score": 0.75,
        "is_returning_goal": True,
        "confidence": 0.80,
        "rationale": "llm analysis",
    }
    mock_client = MagicMock()
    mock_client.complete_json = AsyncMock(return_value=llm_resp)

    ctx = _mk_context(enrichment=_mk_enrichment(), strategy="llm")
    with patch("app.agents.multi_turn_goal_tracking_agent.settings") as mock_settings:
        mock_settings.MULTI_TURN_GOAL_STRATEGY = "llm"
        mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
        mock_settings.OLLAMA_MODEL = "llama3.1:8b"
        mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
        mock_settings.OLLAMA_MAX_CONCURRENCY = 2
        with patch(
            "app.agents.multi_turn_goal_tracking_agent.create_ollama_client",
            return_value=mock_client,
        ):
            result = await agent.run("m1", ctx)

    assert result.outputs["is_returning_goal"] is True
    assert result.outputs["rationale"] == "llm analysis"


@pytest.mark.asyncio
async def test_run_llm_fallback_on_bad_response():
    agent = MultiTurnGoalTrackingAgent()
    mock_client = MagicMock()
    mock_client.complete_json = AsyncMock(return_value={"bad": "data"})

    ctx = _mk_context(
        enrichment=_mk_enrichment(goal_detected=True, subtasks=["deploy service"]),
        strategy="llm",
    )
    with patch("app.agents.multi_turn_goal_tracking_agent.settings") as mock_settings:
        mock_settings.MULTI_TURN_GOAL_STRATEGY = "llm"
        mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
        mock_settings.OLLAMA_MODEL = "llama3.1:8b"
        mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
        mock_settings.OLLAMA_MAX_CONCURRENCY = 2
        with patch(
            "app.agents.multi_turn_goal_tracking_agent.create_ollama_client",
            return_value=mock_client,
        ):
            result = await agent.run("m1", ctx)

    assert result.status == "success"
    assert result.outputs["rationale"] == "heuristic"


@pytest.mark.asyncio
async def test_run_llm_fallback_on_none_response():
    agent = MultiTurnGoalTrackingAgent()
    mock_client = MagicMock()
    mock_client.complete_json = AsyncMock(return_value=None)

    ctx = _mk_context(enrichment=_mk_enrichment(), strategy="llm")
    with patch("app.agents.multi_turn_goal_tracking_agent.settings") as mock_settings:
        mock_settings.MULTI_TURN_GOAL_STRATEGY = "llm"
        mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
        mock_settings.OLLAMA_MODEL = "llama3.1:8b"
        mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
        mock_settings.OLLAMA_MAX_CONCURRENCY = 2
        with patch(
            "app.agents.multi_turn_goal_tracking_agent.create_ollama_client",
            return_value=mock_client,
        ):
            result = await agent.run("m1", ctx)

    assert result.outputs["rationale"] == "heuristic"


@pytest.mark.asyncio
async def test_run_agent_strategy_fallback():
    """When MULTI_TURN_GOAL_STRATEGY is not set, falls back to AGENT_STRATEGY."""
    agent = MultiTurnGoalTrackingAgent()
    ctx = _mk_context(enrichment=_mk_enrichment(goal_detected=False, subtasks=[]))
    with patch("app.agents.multi_turn_goal_tracking_agent.settings") as mock_settings:
        mock_settings.MULTI_TURN_GOAL_STRATEGY = None
        mock_settings.AGENT_STRATEGY = "heuristic"
        result = await agent.run("m1", ctx)

    assert result.status == "success"


# ---------------------------------------------------------------------------
# Agent metadata
# ---------------------------------------------------------------------------

def test_agent_name():
    assert MultiTurnGoalTrackingAgent().name == "MultiTurnGoalTrackingAgent"


def test_agent_version():
    assert MultiTurnGoalTrackingAgent().version == "v1"


def test_agent_dependencies():
    deps = MultiTurnGoalTrackingAgent().dependencies()
    assert "GoalDecompositionAgent" in deps
    assert "TemporalReasoningAgent" in deps
    assert "ProactiveMemoryPushAgent" in deps


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_lookup_snake():
    assert isinstance(get_agent("multi_turn_goal_tracking"), MultiTurnGoalTrackingAgent)


def test_registry_lookup_camel():
    assert isinstance(get_agent("MultiTurnGoalTrackingAgent"), MultiTurnGoalTrackingAgent)


def test_registry_lookup_short():
    assert isinstance(get_agent("multi_turn_goal"), MultiTurnGoalTrackingAgent)


def test_registry_unknown_unchanged():
    assert get_agent("not_a_real_agent") is None
