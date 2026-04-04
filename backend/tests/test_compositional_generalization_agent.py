"""Tests for CompositionalGeneralizationAgent — Phase 41."""

from __future__ import annotations

import pytest

from app.agents.compositional_generalization_agent import (
    CompositionalGeneralizationAgent,
    _tokenize,
    _step_score,
    find_best_source_step,
    compose_procedure,
    compute_agent_confidence,
    _extract_subtasks_from_content,
)
from app.agents.types import AgentContext


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PLAYBOOK_AWS = {
    "id": "pb-aws-deploy",
    "name": "AWS Deployment Playbook",
    "domain": "infrastructure",
    "steps": [
        "provision EC2 instance with required AMI",
        "configure security groups and IAM roles",
        "deploy application artifact to instance",
        "run smoke tests against deployed endpoint",
        "update load balancer target group",
    ],
    "success_rate": 0.90,
    "occurrence_count": 15,
}

PLAYBOOK_ONPREM = {
    "id": "pb-onprem-setup",
    "name": "On-Prem Server Setup",
    "domain": "infrastructure",
    "steps": [
        "provision bare-metal server with OS image",
        "configure firewall rules and network access",
        "install application dependencies",
        "run integration tests on server",
    ],
    "success_rate": 0.75,
    "occurrence_count": 8,
}

PLAYBOOK_DB = {
    "id": "pb-db-migration",
    "name": "Database Migration Playbook",
    "domain": "database",
    "steps": [
        "take full database backup before migration",
        "apply schema migrations in staging environment",
        "validate data integrity post migration",
        "promote migration to production",
    ],
    "success_rate": 0.85,
    "occurrence_count": 20,
}

CANDIDATES = [PLAYBOOK_AWS, PLAYBOOK_ONPREM, PLAYBOOK_DB]


def _ctx(content: str, subtasks: list[str] | None = None,
         candidates: list | None = None) -> AgentContext:
    return {
        "memory": {
            "content": content,
            "enrichment": {
                "domain": "infrastructure",
                "subtasks": subtasks or [],
                "playbook_candidates": candidates if candidates is not None else CANDIDATES,
            },
        },
        "runtime": {"job_id": "trace-41"},
    }


# ---------------------------------------------------------------------------
# Unit: _tokenize
# ---------------------------------------------------------------------------

def test_tokenize_basic():
    tokens = _tokenize("deploy application to server")
    assert "deploy" in tokens
    assert "application" in tokens
    assert "server" in tokens


def test_tokenize_removes_stop_words():
    tokens = _tokenize("the and for the")
    assert tokens == set()


def test_tokenize_min_length():
    tokens = _tokenize("do it now")
    # "now" is a stop word; "do" and "it" are too short
    assert tokens == set()


def test_tokenize_empty():
    assert _tokenize("") == set()


def test_tokenize_mixed_case():
    tokens = _tokenize("Deploy APPLICATION")
    assert "deploy" in tokens
    assert "application" in tokens


# ---------------------------------------------------------------------------
# Unit: _step_score
# ---------------------------------------------------------------------------

def test_step_score_full_overlap():
    tokens = _tokenize("deploy application artifact")
    score = _step_score("deploy application artifact to instance", tokens)
    assert score > 0.5


def test_step_score_no_overlap():
    tokens = _tokenize("database backup migration")
    score = _step_score("provision EC2 instance", tokens)
    assert score == 0.0


def test_step_score_empty_step():
    assert _step_score("", {"deploy", "server"}) == 0.0


def test_step_score_empty_tokens():
    assert _step_score("deploy to server", set()) == 0.0


def test_step_score_partial():
    tokens = _tokenize("deploy application")
    score = _step_score("deploy application artifact to instance", tokens)
    assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# Unit: find_best_source_step
# ---------------------------------------------------------------------------

def test_find_best_source_step_match():
    pb_id, step, score = find_best_source_step(
        "deploy application artifact", [PLAYBOOK_AWS]
    )
    assert pb_id == "pb-aws-deploy"
    assert step is not None
    assert score > 0.0


def test_find_best_source_step_no_candidates():
    pb_id, step, score = find_best_source_step("deploy something", [])
    assert pb_id is None
    assert step is None
    assert score == 0.0


def test_find_best_source_step_no_match():
    pb_id, step, score = find_best_source_step(
        "completely unrelated xyzzy foobar", [PLAYBOOK_AWS]
    )
    assert pb_id is None
    assert step is None
    assert score == 0.0


def test_find_best_source_step_prefers_high_success_rate():
    low_rate_pb = {
        "id": "pb-low",
        "steps": ["deploy application artifact to server"],
        "success_rate": 0.10,
        "occurrence_count": 2,
    }
    high_rate_pb = {
        "id": "pb-high",
        "steps": ["deploy application artifact to server"],
        "success_rate": 0.95,
        "occurrence_count": 20,
    }
    pb_id, _, score_low = find_best_source_step(
        "deploy application artifact", [low_rate_pb]
    )
    pb_id_h, _, score_high = find_best_source_step(
        "deploy application artifact", [high_rate_pb]
    )
    # High success rate should yield equal or higher weighted score.
    assert score_high >= score_low


def test_find_best_source_step_empty_subtask():
    pb_id, step, score = find_best_source_step("", [PLAYBOOK_AWS])
    assert pb_id is None


def test_find_best_source_step_multi_candidate():
    subtask = "run tests after deployment"
    pb_id, step, score = find_best_source_step(subtask, CANDIDATES)
    assert pb_id is not None
    # Should find a "run tests" step in one of the playbooks.
    assert score > 0.0


# ---------------------------------------------------------------------------
# Unit: compose_procedure
# ---------------------------------------------------------------------------

def test_compose_procedure_all_matched():
    subtasks = [
        "deploy application artifact to instance",
        "run smoke tests against endpoint",
    ]
    composed, sources, novel, conf = compose_procedure(
        subtasks=subtasks,
        playbook_candidates=[PLAYBOOK_AWS],
        content_tokens=_tokenize("deploy application and run tests"),
    )
    assert len(composed) == 2
    assert len(novel) == 0
    assert conf == 1.0
    assert "pb-aws-deploy" in sources


def test_compose_procedure_novel_steps():
    subtasks = [
        "completely novel unrecognized xyzzy procedure",
        "another unknown foobar task",
    ]
    composed, sources, novel, conf = compose_procedure(
        subtasks=subtasks,
        playbook_candidates=CANDIDATES,
        content_tokens=_tokenize("novel task"),
    )
    assert len(novel) == 2
    assert conf == 0.0
    assert len(sources) == 0


def test_compose_procedure_mixed():
    subtasks = [
        "deploy application artifact to instance",
        "completely novel unrecognized xyzzy procedure",
    ]
    composed, sources, novel, conf = compose_procedure(
        subtasks=subtasks,
        playbook_candidates=[PLAYBOOK_AWS],
        content_tokens=_tokenize("deploy application"),
    )
    assert len(composed) == 2
    assert len(novel) == 1
    assert 0.0 < conf < 1.0


def test_compose_procedure_empty_subtasks():
    composed, sources, novel, conf = compose_procedure(
        subtasks=[],
        playbook_candidates=CANDIDATES,
        content_tokens=set(),
    )
    assert composed == []
    assert sources == []
    assert novel == []
    assert conf == 0.0


def test_compose_procedure_preserves_order():
    subtasks = [
        "take full database backup before migration",
        "apply schema migrations in staging",
        "validate data integrity post migration",
        "promote migration to production",
    ]
    composed, sources, novel, conf = compose_procedure(
        subtasks=subtasks,
        playbook_candidates=[PLAYBOOK_DB],
        content_tokens=_tokenize("database migration"),
    )
    assert len(composed) == 4
    assert "pb-db-migration" in sources


def test_compose_procedure_deduplicates_source_playbooks():
    subtasks = [
        "deploy application artifact to instance",
        "run smoke tests against deployed endpoint",
    ]
    composed, sources, novel, conf = compose_procedure(
        subtasks=subtasks,
        playbook_candidates=[PLAYBOOK_AWS],
        content_tokens=set(),
    )
    # Should list pb-aws-deploy only once even though two steps came from it.
    assert sources.count("pb-aws-deploy") == 1


# ---------------------------------------------------------------------------
# Unit: compute_agent_confidence
# ---------------------------------------------------------------------------

def test_confidence_high_adaptation():
    c = compute_agent_confidence(
        adaptation_confidence=0.90, novel_step_count=0, total_steps=5, candidate_count=3
    )
    assert c == 0.85


def test_confidence_medium_adaptation():
    c = compute_agent_confidence(
        adaptation_confidence=0.65, novel_step_count=2, total_steps=5, candidate_count=3
    )
    assert c == 0.75


def test_confidence_low_adaptation():
    c = compute_agent_confidence(
        adaptation_confidence=0.30, novel_step_count=4, total_steps=5, candidate_count=2
    )
    assert c == 0.55


def test_confidence_zero_candidates():
    c = compute_agent_confidence(
        adaptation_confidence=0.0, novel_step_count=5, total_steps=5, candidate_count=0
    )
    assert c == 0.40


def test_confidence_no_steps():
    c = compute_agent_confidence(
        adaptation_confidence=0.0, novel_step_count=0, total_steps=0, candidate_count=3
    )
    assert c == 0.40


# ---------------------------------------------------------------------------
# Unit: _extract_subtasks_from_content
# ---------------------------------------------------------------------------

def test_extract_subtasks_from_content_basic():
    content = "First provision the server. Then deploy the application. Finally run tests."
    subtasks = _extract_subtasks_from_content(content)
    assert len(subtasks) >= 2


def test_extract_subtasks_empty():
    assert _extract_subtasks_from_content("") == []


def test_extract_subtasks_max_ten():
    content = ". ".join(["Deploy step number " + str(i) for i in range(20)])
    subtasks = _extract_subtasks_from_content(content)
    assert len(subtasks) <= 10


def test_extract_subtasks_max_200_chars():
    long = "Deploy " + "x" * 300
    subtasks = _extract_subtasks_from_content(long)
    for s in subtasks:
        assert len(s) <= 200


# ---------------------------------------------------------------------------
# Integration: heuristic path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_heuristic_basic_run(monkeypatch):
    import app.agents.compositional_generalization_agent as cga_mod
    monkeypatch.setattr(cga_mod, "settings",
                        type("S", (), {"AGENT_STRATEGY": "heuristic",
                                       "COMPOSITIONAL_STRATEGY": None})())
    agent = CompositionalGeneralizationAgent()
    ctx = _ctx(
        content="deploy application to on-premise server",
        subtasks=[
            "provision server with OS image",
            "deploy application artifact",
            "run smoke tests",
        ],
    )
    result = await agent.run("mem-001", ctx)
    assert result.status == "success"
    assert len(result.outputs["composed_procedure"]) == 3
    assert isinstance(result.outputs["source_playbooks"], list)
    assert 0.0 <= result.outputs["adaptation_confidence"] <= 1.0
    assert isinstance(result.outputs["novel_steps"], list)


@pytest.mark.asyncio
async def test_heuristic_no_subtasks_falls_back_to_content():
    agent = CompositionalGeneralizationAgent()
    ctx = _ctx(
        content="deploy application artifact to instance. run smoke tests.",
        subtasks=[],
    )
    result = await agent.run("mem-002", ctx)
    assert result.status == "success"
    assert len(result.outputs["composed_procedure"]) >= 1


@pytest.mark.asyncio
async def test_heuristic_all_novel(monkeypatch):
    import app.agents.compositional_generalization_agent as cga_mod
    monkeypatch.setattr(cga_mod, "settings",
                        type("S", (), {"AGENT_STRATEGY": "heuristic",
                                       "COMPOSITIONAL_STRATEGY": None})())
    agent = CompositionalGeneralizationAgent()
    ctx = _ctx(
        content="xyzzy foobar quux",
        subtasks=["completely novel xyzzy task", "unrecognized foobar procedure"],
    )
    result = await agent.run("mem-003", ctx)
    assert result.status == "success"
    assert result.outputs["adaptation_confidence"] == 0.0
    assert len(result.outputs["novel_steps"]) == 2


@pytest.mark.asyncio
async def test_heuristic_empty_content():
    agent = CompositionalGeneralizationAgent()
    ctx = _ctx(content="", subtasks=[])
    result = await agent.run("mem-004", ctx)
    assert result.status == "success"
    assert result.outputs["composed_procedure"] == []


@pytest.mark.asyncio
async def test_heuristic_no_candidates(monkeypatch):
    import app.agents.compositional_generalization_agent as cga_mod
    monkeypatch.setattr(cga_mod, "settings",
                        type("S", (), {"AGENT_STRATEGY": "heuristic",
                                       "COMPOSITIONAL_STRATEGY": None})())
    agent = CompositionalGeneralizationAgent()
    ctx = _ctx(
        content="deploy application",
        subtasks=["deploy application artifact"],
        candidates=[],
    )
    result = await agent.run("mem-005", ctx)
    assert result.status == "success"
    assert result.outputs["source_playbooks"] == []
    assert result.outputs["adaptation_confidence"] == 0.0


@pytest.mark.asyncio
async def test_heuristic_confidence_range():
    agent = CompositionalGeneralizationAgent()
    ctx = _ctx(
        content="deploy application to infrastructure",
        subtasks=[
            "provision EC2 instance with required AMI",
            "deploy application artifact to instance",
            "run smoke tests against deployed endpoint",
        ],
    )
    result = await agent.run("mem-006", ctx)
    assert 0.0 <= result.outputs["confidence"] <= 1.0


@pytest.mark.asyncio
async def test_heuristic_source_playbooks_populated():
    agent = CompositionalGeneralizationAgent()
    ctx = _ctx(
        content="deploy application",
        subtasks=["deploy application artifact to instance"],
    )
    result = await agent.run("mem-007", ctx)
    assert result.status == "success"
    assert len(result.outputs["source_playbooks"]) >= 1


@pytest.mark.asyncio
async def test_heuristic_trace_id_propagated():
    agent = CompositionalGeneralizationAgent()
    ctx = _ctx(content="deploy application", subtasks=[])
    ctx["runtime"] = {"job_id": "trace-xyz"}
    result = await agent.run("mem-008", ctx)
    assert result.trace_id == "trace-xyz"


@pytest.mark.asyncio
async def test_heuristic_adapt_aws_to_onprem():
    """Core use case: adapt AWS playbook steps to on-prem scenario."""
    agent = CompositionalGeneralizationAgent()
    ctx = _ctx(
        content="deploy application to on-premise server",
        subtasks=[
            "provision server with OS image",
            "configure firewall rules and network access",
            "deploy application artifact",
            "run integration tests",
            "update load balancer configuration",
        ],
        candidates=CANDIDATES,
    )
    result = await agent.run("mem-009", ctx)
    assert result.status == "success"
    # Should source at least some steps from playbooks.
    assert result.outputs["adaptation_confidence"] > 0.0
    # Should return a complete composed procedure.
    assert len(result.outputs["composed_procedure"]) == 5


@pytest.mark.asyncio
async def test_heuristic_multi_playbook_sourcing():
    """Steps should be sourced from multiple playbooks when appropriate."""
    agent = CompositionalGeneralizationAgent()
    ctx = _ctx(
        content="deploy app and migrate database",
        subtasks=[
            "deploy application artifact to instance",
            "take full database backup before migration",
            "validate data integrity post migration",
        ],
        candidates=CANDIDATES,
    )
    result = await agent.run("mem-010", ctx)
    assert result.status == "success"
    # Should reference at least two different playbooks.
    assert len(result.outputs["source_playbooks"]) >= 2


# ---------------------------------------------------------------------------
# Integration: LLM path fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_path_falls_back_to_heuristic_on_bad_response(monkeypatch):
    import app.agents.compositional_generalization_agent as cga_mod

    class FakeClient:
        async def complete_json(self, prompt, schema_hint, tool_event_sink=None):
            return {"bad_key": "wrong"}

    monkeypatch.setattr(cga_mod, "create_ollama_client", lambda **kw: FakeClient())
    monkeypatch.setattr(cga_mod, "settings",
                        type("S", (), {"AGENT_STRATEGY": "llm",
                                       "OLLAMA_BASE_URL": "http://localhost:11434",
                                       "OLLAMA_MODEL": "llama3.1:8b",
                                       "OLLAMA_TIMEOUT_SECONDS": 5.0,
                                       "OLLAMA_MAX_CONCURRENCY": 2,
                                       "COMPOSITIONAL_STRATEGY": None,
                                       "get_ollama_model": lambda self, p=None: "llama3.1:8b"})())

    agent = CompositionalGeneralizationAgent()
    ctx = _ctx(
        content="deploy application artifact",
        subtasks=["deploy application artifact to instance"],
    )
    result = await agent.run("mem-011", ctx)
    assert result.status == "success"
    assert result.outputs["rationale"] == "heuristic"


@pytest.mark.asyncio
async def test_llm_path_accepts_valid_response(monkeypatch):
    import app.agents.compositional_generalization_agent as cga_mod

    class FakeClient:
        async def complete_json(self, prompt, schema_hint, tool_event_sink=None):
            return {
                "composed_procedure": ["step one", "step two"],
                "source_playbooks": ["pb-aws-deploy"],
                "adaptation_confidence": 0.75,
                "novel_steps": [],
                "confidence": 0.80,
                "rationale": "llm",
            }

    monkeypatch.setattr(cga_mod, "create_ollama_client", lambda **kw: FakeClient())
    monkeypatch.setattr(cga_mod, "settings",
                        type("S", (), {"AGENT_STRATEGY": "llm",
                                       "OLLAMA_BASE_URL": "http://localhost:11434",
                                       "OLLAMA_MODEL": "llama3.1:8b",
                                       "OLLAMA_TIMEOUT_SECONDS": 5.0,
                                       "OLLAMA_MAX_CONCURRENCY": 2,
                                       "COMPOSITIONAL_STRATEGY": None,
                                       "get_ollama_model": lambda self, p=None: "llama3.1:8b"})())

    agent = CompositionalGeneralizationAgent()
    ctx = _ctx(content="deploy application", subtasks=[])
    result = await agent.run("mem-012", ctx)
    assert result.status == "success"
    assert result.outputs["composed_procedure"] == ["step one", "step two"]
    assert result.outputs["rationale"] == "llm"


# ---------------------------------------------------------------------------
# validate_outputs
# ---------------------------------------------------------------------------

def test_validate_outputs_passes_valid():
    agent = CompositionalGeneralizationAgent()
    from app.agents.types import AgentResult
    from datetime import datetime, timezone
    result = AgentResult(
        agent_name=agent.name,
        agent_version=agent.version,
        memory_id="m1",
        status="success",
        confidence=0.75,
        outputs={
            "composed_procedure": ["step 1", "step 2"],
            "source_playbooks": ["pb-1"],
            "adaptation_confidence": 0.75,
            "novel_steps": [],
            "confidence": 0.75,
            "rationale": "heuristic",
        },
        warnings=[],
        errors=[],
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        trace_id=None,
    )
    agent.validate_outputs(result)  # should not raise


def test_validate_outputs_missing_composed_procedure():
    agent = CompositionalGeneralizationAgent()
    from app.agents.types import AgentResult
    from datetime import datetime, timezone
    result = AgentResult(
        agent_name=agent.name,
        agent_version=agent.version,
        memory_id="m1",
        status="success",
        confidence=0.75,
        outputs={
            "source_playbooks": [],
            "adaptation_confidence": 0.5,
            "novel_steps": [],
            "confidence": 0.75,
        },
        warnings=[], errors=[],
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        trace_id=None,
    )
    with pytest.raises(ValueError, match="composed_procedure"):
        agent.validate_outputs(result)


def test_validate_outputs_bad_adaptation_confidence():
    agent = CompositionalGeneralizationAgent()
    from app.agents.types import AgentResult
    from datetime import datetime, timezone
    result = AgentResult(
        agent_name=agent.name,
        agent_version=agent.version,
        memory_id="m1",
        status="success",
        confidence=0.75,
        outputs={
            "composed_procedure": [],
            "source_playbooks": [],
            "adaptation_confidence": 1.5,  # out of range
            "novel_steps": [],
            "confidence": 0.75,
        },
        warnings=[], errors=[],
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        trace_id=None,
    )
    with pytest.raises(ValueError, match="adaptation_confidence"):
        agent.validate_outputs(result)


def test_validate_outputs_skipped_on_error_status():
    agent = CompositionalGeneralizationAgent()
    from app.agents.types import AgentResult
    from datetime import datetime, timezone
    result = AgentResult(
        agent_name=agent.name,
        agent_version=agent.version,
        memory_id="m1",
        status="failed",
        confidence=0.0,
        outputs={},
        warnings=[], errors=["something went wrong"],
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        trace_id=None,
    )
    agent.validate_outputs(result)  # should not raise


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_lookup():
    from app.agents.registry import get_agent
    agent = get_agent("compositional_generalization")
    assert isinstance(agent, CompositionalGeneralizationAgent)


def test_registry_lookup_camelcase():
    from app.agents.registry import get_agent
    agent = get_agent("compositionalgeneralizationagent")
    assert isinstance(agent, CompositionalGeneralizationAgent)


def test_registry_lookup_unknown():
    from app.agents.registry import get_agent
    assert get_agent("nonexistent_agent_xyz") is None


# ---------------------------------------------------------------------------
# Agent metadata
# ---------------------------------------------------------------------------

def test_agent_name():
    assert CompositionalGeneralizationAgent.name == "CompositionalGeneralizationAgent"


def test_agent_version():
    assert CompositionalGeneralizationAgent.version == "v1"


def test_agent_dependencies():
    deps = CompositionalGeneralizationAgent().dependencies()
    assert "PlaybookAgent" in deps
    assert "GoalDecompositionAgent" in deps
    assert "PlaybookExecutionTrackerAgent" in deps
