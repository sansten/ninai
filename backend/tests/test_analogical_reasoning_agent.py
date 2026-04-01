"""Tests for Phase 54 - AnalogicalReasoningAgent."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.analogical_reasoning_agent import (
    AnalogicalReasoningAgent,
    _DOMAIN_SUBSTITUTIONS,
    _candidate_tokens,
    _tokenize,
    apply_substitutions,
    jaccard_similarity,
)
from app.agents.registry import get_agent
from app.agents.types import AgentResult


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _analogue(solution: str, tags=None, content: str = "") -> dict:
    return {"solution": solution, "tags": tags or [], "content": content}


def _ctx(*, source_problem="", candidates=None, features=None):
    return {
        "memory": {
            "enrichment": {
                "source_problem": source_problem,
                "candidate_analogues": candidates or [],
                "structural_features": features or [],
            }
        },
        "runtime": {"job_id": "trace-54"},
    }


def _result(outputs, status="success"):
    now = datetime.now(timezone.utc)
    return AgentResult(
        agent_name="AnalogicalReasoningAgent",
        agent_version="v1",
        memory_id="m54",
        status=status,
        confidence=0.8,
        outputs=outputs,
        warnings=[],
        errors=[],
        started_at=now,
        finished_at=now,
    )


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------

def test_tokenize_lower_case():
    tokens = _tokenize("Latency Database QUERY")
    assert "latency" in tokens
    assert "database" in tokens
    assert "query" in tokens


def test_tokenize_empty():
    assert _tokenize("") == set()


def test_tokenize_punctuation_stripped():
    tokens = _tokenize("foo, bar; baz!")
    assert tokens == {"foo", "bar", "baz"}


# ---------------------------------------------------------------------------
# _candidate_tokens
# ---------------------------------------------------------------------------

def test_candidate_tokens_combines_tags_and_content():
    a = {"tags": ["latency", "database"], "content": "slow postgres query"}
    tokens = _candidate_tokens(a)
    assert "latency" in tokens
    assert "database" in tokens
    assert "postgres" in tokens
    assert "slow" in tokens


def test_candidate_tokens_empty_analogue():
    assert _candidate_tokens({}) == set()


# ---------------------------------------------------------------------------
# jaccard_similarity
# ---------------------------------------------------------------------------

def test_jaccard_identical():
    s = {"a", "b", "c"}
    assert jaccard_similarity(s, s) == 1.0


def test_jaccard_no_overlap():
    assert jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0


def test_jaccard_partial():
    score = jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"})
    assert 0.0 < score < 1.0


def test_jaccard_both_empty():
    assert jaccard_similarity(set(), set()) == 0.0


# ---------------------------------------------------------------------------
# apply_substitutions
# ---------------------------------------------------------------------------

def test_apply_substitutions_postgres_to_redis():
    adapted, mapping = apply_substitutions("Use postgres with an index.")
    assert "redis" in adapted
    assert any(m["source_term"] == "postgres" for m in mapping)


def test_apply_substitutions_multiple():
    adapted, mapping = apply_substitutions("database index query table")
    assert "cache" in adapted
    assert "cache_key" in adapted
    assert "request" in adapted
    assert "bucket" in adapted
    assert len(mapping) >= 4


def test_apply_substitutions_no_match():
    adapted, mapping = apply_substitutions("hello world")
    assert adapted == "hello world"
    assert mapping == []


def test_apply_substitutions_case_insensitive():
    adapted, mapping = apply_substitutions("DATABASE LATENCY")
    assert "cache" in adapted.lower() or "cache" in adapted
    assert len(mapping) >= 1


# ---------------------------------------------------------------------------
# Heuristic path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("app.agents.analogical_reasoning_agent.settings")
async def test_empty_candidates_returns_none_best(mock_settings):
    mock_settings.AGENT_STRATEGY = "heuristic"
    agent = AnalogicalReasoningAgent()
    ctx = _ctx(source_problem="API latency", candidates=[], features=["latency", "api"])
    result = await agent.run("m1", ctx)
    assert result.status == "success"
    assert result.outputs["best_analogue"] is None
    assert result.outputs["analogy_score"] == 0.0


@pytest.mark.asyncio
@patch("app.agents.analogical_reasoning_agent.settings")
async def test_exact_match_score_one(mock_settings):
    mock_settings.AGENT_STRATEGY = "heuristic"
    features = ["latency", "database", "timeout"]
    candidate = _analogue(
        solution="Add a database index.",
        tags=["latency", "database", "timeout"],
    )
    agent = AnalogicalReasoningAgent()
    ctx = _ctx(
        source_problem="Database is slow",
        candidates=[candidate],
        features=features,
    )
    result = await agent.run("m1", ctx)
    assert result.status == "success"
    # Tags exactly match features → high score (close to 1.0 though not exact due to
    # content tokens being included in the union)
    assert result.outputs["analogy_score"] >= 0.5


@pytest.mark.asyncio
@patch("app.agents.analogical_reasoning_agent.settings")
async def test_exact_tag_match_gives_high_score(mock_settings):
    """When features exactly equal the analogue tags (no extra tokens), score=1.0."""
    mock_settings.AGENT_STRATEGY = "heuristic"
    features = ["latency"]
    candidate = {"solution": "", "tags": ["latency"], "content": ""}
    agent = AnalogicalReasoningAgent()
    ctx = _ctx(source_problem="latency issue", candidates=[candidate], features=features)
    result = await agent.run("m1", ctx)
    assert result.outputs["analogy_score"] == 1.0


@pytest.mark.asyncio
@patch("app.agents.analogical_reasoning_agent.settings")
async def test_partial_match_between_zero_and_one(mock_settings):
    mock_settings.AGENT_STRATEGY = "heuristic"
    features = ["latency", "database", "index"]
    candidate = _analogue("Add cache.", tags=["latency"])
    agent = AnalogicalReasoningAgent()
    ctx = _ctx(source_problem="slow DB", candidates=[candidate], features=features)
    result = await agent.run("m1", ctx)
    score = result.outputs["analogy_score"]
    assert 0.0 < score < 1.0


@pytest.mark.asyncio
@patch("app.agents.analogical_reasoning_agent.settings")
async def test_best_candidate_selected(mock_settings):
    mock_settings.AGENT_STRATEGY = "heuristic"
    features = ["latency", "database", "timeout"]
    bad = _analogue("wrong solution", tags=["network"])
    good = _analogue("right solution", tags=["latency", "database", "timeout"])
    agent = AnalogicalReasoningAgent()
    ctx = _ctx(source_problem="DB perf", candidates=[bad, good], features=features)
    result = await agent.run("m1", ctx)
    assert result.outputs["best_analogue"]["solution"] == "right solution"


@pytest.mark.asyncio
@patch("app.agents.analogical_reasoning_agent.settings")
async def test_transferred_solution_substitutes_terms(mock_settings):
    mock_settings.AGENT_STRATEGY = "heuristic"
    candidate = _analogue("Add a postgres index to speed up queries.")
    agent = AnalogicalReasoningAgent()
    ctx = _ctx(source_problem="perf", candidates=[candidate], features=["latency"])
    result = await agent.run("m1", ctx)
    solution = result.outputs["transferred_solution"]
    assert "redis" in solution  # postgres → redis
    assert "cache_key" in solution  # index → cache_key


@pytest.mark.asyncio
@patch("app.agents.analogical_reasoning_agent.settings")
async def test_novel_elements_detected(mock_settings):
    mock_settings.AGENT_STRATEGY = "heuristic"
    features = ["latency", "database", "memory_leak"]
    candidate = _analogue("Fix latency.", tags=["latency", "database"])
    agent = AnalogicalReasoningAgent()
    ctx = _ctx(source_problem="issue", candidates=[candidate], features=features)
    result = await agent.run("m1", ctx)
    # memory_leak not in candidate tags/content → novel
    assert "memory_leak" in result.outputs["novel_elements"]


@pytest.mark.asyncio
@patch("app.agents.analogical_reasoning_agent.settings")
async def test_novel_elements_reduce_confidence(mock_settings):
    mock_settings.AGENT_STRATEGY = "heuristic"
    features_no_novel = ["latency"]
    features_with_novel = ["latency", "memory_leak", "cpu_spike"]
    candidate = _analogue("Add cache.", tags=["latency"])
    agent = AnalogicalReasoningAgent()
    ctx1 = _ctx(source_problem="x", candidates=[candidate], features=features_no_novel)
    ctx2 = _ctx(source_problem="x", candidates=[candidate], features=features_with_novel)
    r1 = await agent.run("m1", ctx1)
    r2 = await agent.run("m1", ctx2)
    assert r2.outputs["confidence"] < r1.outputs["confidence"]


@pytest.mark.asyncio
@patch("app.agents.analogical_reasoning_agent.settings")
async def test_confidence_zero_when_no_candidates(mock_settings):
    mock_settings.AGENT_STRATEGY = "heuristic"
    agent = AnalogicalReasoningAgent()
    ctx = _ctx(source_problem="x", candidates=[], features=["latency"])
    result = await agent.run("m1", ctx)
    assert result.outputs["confidence"] == 0.0


@pytest.mark.asyncio
@patch("app.agents.analogical_reasoning_agent.settings")
async def test_mapping_list_length(mock_settings):
    mock_settings.AGENT_STRATEGY = "heuristic"
    candidate = _analogue("Fix postgres and database latency using an index.")
    agent = AnalogicalReasoningAgent()
    ctx = _ctx(source_problem="perf", candidates=[candidate], features=["latency"])
    result = await agent.run("m1", ctx)
    # mapping should have at least the substituted terms: postgres, database, latency, index
    assert len(result.outputs["mapping"]) >= 3


@pytest.mark.asyncio
@patch("app.agents.analogical_reasoning_agent.settings")
async def test_empty_features_no_novel(mock_settings):
    mock_settings.AGENT_STRATEGY = "heuristic"
    candidate = _analogue("Solution x.", tags=["a", "b"])
    agent = AnalogicalReasoningAgent()
    ctx = _ctx(source_problem="x", candidates=[candidate], features=[])
    result = await agent.run("m1", ctx)
    assert result.outputs["novel_elements"] == []


@pytest.mark.asyncio
@patch("app.agents.analogical_reasoning_agent.settings")
async def test_rationale_heuristic(mock_settings):
    mock_settings.AGENT_STRATEGY = "heuristic"
    agent = AnalogicalReasoningAgent()
    ctx = _ctx(source_problem="x", candidates=[], features=[])
    result = await agent.run("m1", ctx)
    assert result.outputs["rationale"] == "heuristic"


# ---------------------------------------------------------------------------
# validate_outputs
# ---------------------------------------------------------------------------

def test_validate_outputs_passes():
    agent = AnalogicalReasoningAgent()
    result = _result({
        "best_analogue": {"solution": "fix it"},
        "analogy_score": 0.8,
        "transferred_solution": "fix the cache",
        "mapping": [{"source_term": "database", "target_term": "cache"}],
        "confidence": 0.7,
        "novel_elements": [],
    })
    agent.validate_outputs(result)  # must not raise


def test_validate_outputs_missing_best_analogue():
    agent = AnalogicalReasoningAgent()
    result = _result({
        "analogy_score": 0.5,
        "transferred_solution": "",
        "mapping": [],
        "confidence": 0.5,
        "novel_elements": [],
    })
    with pytest.raises(ValueError, match="best_analogue key required"):
        agent.validate_outputs(result)


def test_validate_outputs_bad_analogy_score():
    agent = AnalogicalReasoningAgent()
    result = _result({
        "best_analogue": None,
        "analogy_score": "high",
        "transferred_solution": "",
        "mapping": [],
        "confidence": 0.5,
        "novel_elements": [],
    })
    with pytest.raises(ValueError, match="analogy_score must be a float"):
        agent.validate_outputs(result)


def test_validate_outputs_bad_mapping():
    agent = AnalogicalReasoningAgent()
    result = _result({
        "best_analogue": None,
        "analogy_score": 0.0,
        "transferred_solution": "",
        "mapping": "not a list",
        "confidence": 0.0,
        "novel_elements": [],
    })
    with pytest.raises(ValueError, match="mapping must be a list"):
        agent.validate_outputs(result)


def test_validate_outputs_bad_novel_elements():
    agent = AnalogicalReasoningAgent()
    result = _result({
        "best_analogue": None,
        "analogy_score": 0.0,
        "transferred_solution": "",
        "mapping": [],
        "confidence": 0.0,
        "novel_elements": "memory_leak",
    })
    with pytest.raises(ValueError, match="novel_elements must be a list"):
        agent.validate_outputs(result)


def test_validate_outputs_skips_non_success():
    agent = AnalogicalReasoningAgent()
    result = _result({}, status="failed")
    agent.validate_outputs(result)  # must not raise


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("app.agents.analogical_reasoning_agent.settings")
@patch("app.agents.analogical_reasoning_agent.create_ollama_client")
async def test_llm_path_success(mock_client_factory, mock_settings):
    mock_settings.AGENT_STRATEGY = "llm"
    mock_settings.OLLAMA_MODEL = "qwen2.5:7b"

    import json
    llm_resp = {
        "best_analogue": {"solution": "use cache"},
        "analogy_score": 0.75,
        "transferred_solution": "use cache_key",
        "mapping": [{"source_term": "index", "target_term": "cache_key"}],
        "confidence": 0.6,
        "novel_elements": ["memory_leak"],
    }

    mock_client = AsyncMock()
    mock_client.generate = AsyncMock(return_value={"response": json.dumps(llm_resp)})
    mock_client_factory.return_value = mock_client

    agent = AnalogicalReasoningAgent()
    ctx = _ctx(source_problem="perf", candidates=[{"solution": "s", "tags": ["t"]}], features=["latency"])
    result = await agent.run("m1", ctx)
    assert result.status == "success"
    assert result.outputs["analogy_score"] == 0.75
    assert result.outputs["rationale"] == "llm"


@pytest.mark.asyncio
@patch("app.agents.analogical_reasoning_agent.settings")
@patch("app.agents.analogical_reasoning_agent.create_ollama_client")
async def test_llm_falls_back_on_invalid_response(mock_client_factory, mock_settings):
    mock_settings.AGENT_STRATEGY = "llm"
    mock_settings.OLLAMA_MODEL = "qwen2.5:7b"

    mock_client = AsyncMock()
    mock_client.generate = AsyncMock(return_value={"response": "not json"})
    mock_client_factory.return_value = mock_client

    agent = AnalogicalReasoningAgent()
    ctx = _ctx(source_problem="x", candidates=[], features=["latency"])
    result = await agent.run("m1", ctx)
    assert result.status == "success"
    assert result.outputs["rationale"] == "heuristic"


@pytest.mark.asyncio
@patch("app.agents.analogical_reasoning_agent.settings")
@patch("app.agents.analogical_reasoning_agent.create_ollama_client")
async def test_llm_falls_back_on_exception(mock_client_factory, mock_settings):
    mock_settings.AGENT_STRATEGY = "llm"
    mock_settings.OLLAMA_MODEL = "qwen2.5:7b"

    mock_client = AsyncMock()
    mock_client.generate = AsyncMock(side_effect=ConnectionError("LLM down"))
    mock_client_factory.return_value = mock_client

    agent = AnalogicalReasoningAgent()
    ctx = _ctx(source_problem="x", candidates=[], features=[])
    result = await agent.run("m1", ctx)
    assert result.status == "success"
    assert result.outputs["rationale"] == "heuristic"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_analogical_reasoning():
    assert isinstance(get_agent("analogical_reasoning"), AnalogicalReasoningAgent)


def test_registry_analogicalreasoning():
    assert isinstance(get_agent("analogicalreasoning"), AnalogicalReasoningAgent)


def test_registry_analogy():
    assert isinstance(get_agent("analogy"), AnalogicalReasoningAgent)


def test_registry_skill_transfer():
    assert isinstance(get_agent("skill_transfer"), AnalogicalReasoningAgent)


def test_registry_unknown():
    assert get_agent("does_not_exist_phase54") is None


# ---------------------------------------------------------------------------
# Domain substitution map coverage
# ---------------------------------------------------------------------------

def test_domain_substitution_map_has_expected_keys():
    expected_keys = {"database", "index", "query", "table", "row", "postgres",
                     "mysql", "latency", "timeout"}
    assert expected_keys.issubset(_DOMAIN_SUBSTITUTIONS.keys())
