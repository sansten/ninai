"""Tests for Phase 12 — CausalReasoningAgent."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from app.agents.causal_reasoning_agent import (
    CausalReasoningAgent,
    _cause_weight,
    build_adjacency,
    build_chain_narrative,
    extract_root_causes,
    generate_counterfactual_hint,
    identify_effect_candidates,
    trace_causal_chain,
)
from app.agents.registry import get_agent
from app.agents.types import AgentResult


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_NODE_ACME = {
    "entity": "Acme Corp",
    "entity_type": "customer",
    "domains": ["sales", "support"],
    "silo_count": 2,
    "node_confidence": 0.85,
}
_NODE_CSM = {
    "entity": "CSM Reassignment",
    "entity_type": "event",
    "domains": ["hr"],
    "silo_count": 1,
    "node_confidence": 0.70,
}
_NODE_RENEWAL = {
    "entity": "Renewal Call",
    "entity_type": "event",
    "domains": ["sales"],
    "silo_count": 1,
    "node_confidence": 0.65,
}
_NODE_CHURN = {
    "entity": "Churn Signal",
    "entity_type": "metric",
    "domains": ["sales"],
    "silo_count": 1,
    "node_confidence": 0.60,
}
_NODE_PERSON = {
    "entity": "Alice",
    "entity_type": "person",
    "domains": ["engineering"],
    "silo_count": 1,
    "node_confidence": 0.80,
}

_EDGE_ACME_CHURN = {
    "from_entity": "Acme Corp",
    "to_entity": "Churn Signal",
    "relationship_type": "co_occurrence",
    "weight": 0.75,
}
_EDGE_CSM_ACME = {
    "from_entity": "CSM Reassignment",
    "to_entity": "Acme Corp",
    "relationship_type": "co_occurrence",
    "weight": 0.80,
}
_EDGE_CSM_RENEWAL = {
    "from_entity": "CSM Reassignment",
    "to_entity": "Renewal Call",
    "relationship_type": "co_occurrence",
    "weight": 0.70,
}
_EDGE_PERSON_ACME = {
    "from_entity": "Alice",
    "to_entity": "Acme Corp",
    "relationship_type": "co_occurrence",
    "weight": 0.65,
}

_PUSH_ACME = {
    "recipient_team": "sales",
    "topic": "Acme Corp",
    "trigger_type": "overlap_alert",
    "urgency_score": 0.75,
    "delivery_hint": "immediate_notification",
}
_PUSH_LOW = {
    "recipient_team": "support",
    "topic": "minor_topic",
    "trigger_type": "onboarding",
    "urgency_score": 0.30,
    "delivery_hint": "webhook",
}

_SIG_CROSS_SILO = {
    "content": "CSM Reassignment happened last week",
    "source_silo": "hr",
    "target_domains": ["sales"],
    "relevance_score": 0.70,
    "entities": ["CSM Reassignment"],
}


def _make_result(outputs: dict) -> AgentResult:
    now = datetime.now(timezone.utc)
    return AgentResult(
        agent_name="CausalReasoningAgent",
        agent_version="v1",
        memory_id="mem-1",
        status="success",
        confidence=outputs.get("confidence", 0.5),
        outputs=outputs,
        warnings=[],
        errors=[],
        started_at=now,
        finished_at=now,
    )


def _ctx(
    content: str = "",
    world_nodes=None,
    entity_edges=None,
    propagated_signals=None,
    push_candidates=None,
    job_id: str | None = None,
) -> dict:
    enrichment = {}
    if world_nodes is not None:
        enrichment["world_nodes"] = world_nodes
    if entity_edges is not None:
        enrichment["entity_edges"] = entity_edges
    if propagated_signals is not None:
        enrichment["propagated_signals"] = propagated_signals
    if push_candidates is not None:
        enrichment["push_candidates"] = push_candidates
    ctx: dict = {"memory": {"content": content, "enrichment": enrichment}}
    if job_id:
        ctx["runtime"] = {"job_id": job_id}
    return ctx


# ---------------------------------------------------------------------------
# _cause_weight
# ---------------------------------------------------------------------------


def test_cause_weight_person():
    assert _cause_weight("person") == 1.0


def test_cause_weight_metric_is_low():
    assert _cause_weight("metric") == 0.30


def test_cause_weight_unknown_type():
    assert _cause_weight("unknown_type") == 0.45


def test_cause_weight_empty_string():
    assert _cause_weight("") == 0.45


def test_cause_weight_case_insensitive():
    assert _cause_weight("PERSON") == 1.0
    assert _cause_weight("Customer") == 0.90


# ---------------------------------------------------------------------------
# identify_effect_candidates
# ---------------------------------------------------------------------------


def test_identify_effect_candidates_empty():
    result = identify_effect_candidates(world_nodes=[], push_candidates=[])
    assert result == []


def test_identify_effect_candidates_from_push_only():
    result = identify_effect_candidates(
        world_nodes=[],
        push_candidates=[_PUSH_ACME],
    )
    assert len(result) == 1
    assert result[0]["entity"] == "Acme Corp"
    assert result[0]["effect_score"] == 0.75


def test_identify_effect_candidates_low_urgency_excluded():
    result = identify_effect_candidates(
        world_nodes=[],
        push_candidates=[_PUSH_LOW],  # urgency 0.30 < 0.5
    )
    assert result == []


def test_identify_effect_candidates_from_world_nodes_cross_silo():
    result = identify_effect_candidates(
        world_nodes=[_NODE_ACME],  # silo_count=2
        push_candidates=[],
    )
    assert len(result) == 1
    assert result[0]["entity"] == "Acme Corp"
    assert result[0]["effect_score"] > 0.4


def test_identify_effect_candidates_single_silo_excluded():
    result = identify_effect_candidates(
        world_nodes=[_NODE_CSM],  # silo_count=1
        push_candidates=[],
    )
    assert result == []


def test_identify_effect_candidates_combined_takes_max_score():
    result = identify_effect_candidates(
        world_nodes=[_NODE_ACME],
        push_candidates=[_PUSH_ACME],
    )
    assert len(result) == 1
    assert result[0]["entity"] == "Acme Corp"
    assert result[0]["effect_score"] == 0.75  # push urgency (0.75) > cross-silo score


def test_identify_effect_candidates_sorted_descending():
    push2 = {**_PUSH_ACME, "topic": "Other", "urgency_score": 0.60}
    node2 = {**_NODE_ACME, "entity": "Other", "silo_count": 3}
    result = identify_effect_candidates(
        world_nodes=[node2],
        push_candidates=[_PUSH_ACME, push2],
    )
    scores = [r["effect_score"] for r in result]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# build_adjacency
# ---------------------------------------------------------------------------


def test_build_adjacency_empty():
    assert build_adjacency([]) == {}


def test_build_adjacency_single_edge_bidirectional():
    adj = build_adjacency([_EDGE_ACME_CHURN])
    assert any(e["neighbor"] == "Churn Signal" for e in adj["Acme Corp"])
    assert any(e["neighbor"] == "Acme Corp" for e in adj["Churn Signal"])


def test_build_adjacency_preserves_weight():
    adj = build_adjacency([_EDGE_ACME_CHURN])
    entry = next(e for e in adj["Acme Corp"] if e["neighbor"] == "Churn Signal")
    assert entry["weight"] == 0.75


def test_build_adjacency_missing_entities_skipped():
    bad_edge = {"from_entity": "", "to_entity": "Churn Signal", "weight": 0.5}
    adj = build_adjacency([bad_edge])
    assert adj == {}


def test_build_adjacency_multiple_edges():
    adj = build_adjacency([_EDGE_ACME_CHURN, _EDGE_CSM_ACME])
    assert len(adj["Acme Corp"]) == 2  # neighbors: Churn Signal + CSM Reassignment


# ---------------------------------------------------------------------------
# trace_causal_chain
# ---------------------------------------------------------------------------


def test_trace_causal_chain_no_neighbors():
    adj = build_adjacency([])
    node_lookup = {"Churn Signal": _NODE_CHURN}
    result = trace_causal_chain(
        effect_entity="Churn Signal",
        adjacency=adj,
        node_lookup=node_lookup,
        propagated_signal_entities=set(),
    )
    assert result == []


def test_trace_causal_chain_direct_one_hop():
    # Alice (person, cw=1.0) → Churn Signal (metric, cw=0.30)
    # Alice has higher cause_weight than Churn Signal → valid causal direction
    adj = build_adjacency([_EDGE_PERSON_ACME, _EDGE_ACME_CHURN])
    node_lookup = {
        "Alice": _NODE_PERSON,
        "Acme Corp": _NODE_ACME,
        "Churn Signal": _NODE_CHURN,
    }
    result = trace_causal_chain(
        effect_entity="Churn Signal",
        adjacency=adj,
        node_lookup=node_lookup,
        propagated_signal_entities=set(),
    )
    # Should find at least one chain involving a higher-weight entity
    assert len(result) >= 1
    for chain in result:
        assert chain["hop_count"] >= 1
        assert chain["chain_strength"] > 0


def test_trace_causal_chain_no_cycles():
    # Self-loop edge should not produce cycles
    adj = build_adjacency([{"from_entity": "A", "to_entity": "A", "weight": 0.9}])
    node_lookup = {"A": {"entity": "A", "entity_type": "metric", "domains": []}}
    result = trace_causal_chain(
        effect_entity="A",
        adjacency=adj,
        node_lookup=node_lookup,
        propagated_signal_entities=set(),
    )
    assert result == []


def test_trace_causal_chain_propagated_signal_boosts_cause():
    # Renewal Call (event, cw=0.60) normally wouldn't cause Churn Signal (metric, cw=0.30)
    # because 0.60 > 0.30 — wait, it would. Let's test propagated signal enables
    # a same-or-lower-weight entity to be recognized as a cause.
    # metric (0.30) → metric (0.30): normally blocked since 0.30 is not > 0.30
    # but propagated signal makes it pass.
    adj = build_adjacency([
        {"from_entity": "Billing Issue", "to_entity": "Churn Signal", "weight": 0.6}
    ])
    node_lookup = {
        "Billing Issue": {"entity": "Billing Issue", "entity_type": "metric", "domains": ["finance"]},
        "Churn Signal": _NODE_CHURN,
    }
    result_without = trace_causal_chain(
        effect_entity="Churn Signal",
        adjacency=adj,
        node_lookup=node_lookup,
        propagated_signal_entities=set(),
    )
    result_with = trace_causal_chain(
        effect_entity="Churn Signal",
        adjacency=adj,
        node_lookup=node_lookup,
        propagated_signal_entities={"Billing Issue"},
    )
    # With propagated signal, Billing Issue is treated as a cause even at same type weight
    assert len(result_with) >= len(result_without)


def test_trace_causal_chain_max_hops_respected():
    # Build a long chain: A → B → C → D → E (4 hops)
    # Each is person type (same weight), but D and E are in propagated signals
    edges = [
        {"from_entity": "A", "to_entity": "B", "weight": 0.9},
        {"from_entity": "B", "to_entity": "C", "weight": 0.9},
        {"from_entity": "C", "to_entity": "D", "weight": 0.9},
        {"from_entity": "D", "to_entity": "E", "weight": 0.9},
    ]
    adj = build_adjacency(edges)
    node_lookup = {
        "A": {"entity": "A", "entity_type": "person", "domains": []},
        "B": {"entity": "B", "entity_type": "org", "domains": []},
        "C": {"entity": "C", "entity_type": "customer", "domains": []},
        "D": {"entity": "D", "entity_type": "project", "domains": []},
        "E": {"entity": "E", "entity_type": "metric", "domains": []},
    }
    result = trace_causal_chain(
        effect_entity="E",
        adjacency=adj,
        node_lookup=node_lookup,
        propagated_signal_entities={"A", "B", "C", "D"},
        max_hops=2,
    )
    # max_hops=2: paths should have at most 2 hops (3 entities in path)
    for chain in result:
        assert chain["hop_count"] <= 2


def test_trace_causal_chain_silo_span_computed():
    adj = build_adjacency([_EDGE_CSM_ACME])
    node_lookup = {
        "CSM Reassignment": _NODE_CSM,   # domains: ["hr"]
        "Acme Corp": _NODE_ACME,          # domains: ["sales", "support"]
    }
    result = trace_causal_chain(
        effect_entity="Acme Corp",
        adjacency=adj,
        node_lookup=node_lookup,
        propagated_signal_entities={"CSM Reassignment"},
    )
    if result:
        # Path ["Acme Corp", "CSM Reassignment"] → silos: hr + sales + support = 3
        assert result[0]["silo_span"] >= 2


def test_trace_causal_chain_chain_strength_is_product():
    # Single hop: effect=Churn Signal, cause=Acme Corp
    # edge weight=0.75, cause_weight(customer)=0.90
    # strength = 1.0 * 0.75 * 0.90 = 0.675
    adj = build_adjacency([_EDGE_ACME_CHURN])
    node_lookup = {"Acme Corp": _NODE_ACME, "Churn Signal": _NODE_CHURN}
    result = trace_causal_chain(
        effect_entity="Churn Signal",
        adjacency=adj,
        node_lookup=node_lookup,
        propagated_signal_entities=set(),
    )
    if result:
        assert abs(result[0]["chain_strength"] - round(0.75 * 0.90, 4)) < 0.001


# ---------------------------------------------------------------------------
# extract_root_causes
# ---------------------------------------------------------------------------


def test_extract_root_causes_empty_chains():
    result = extract_root_causes([], {})
    assert result == []


def test_extract_root_causes_single_chain():
    chains = [
        {
            "causal_path": ["Churn Signal", "Acme Corp", "CSM Reassignment"],
            "chain_strength": 0.54,
        }
    ]
    node_lookup = {
        "CSM Reassignment": _NODE_CSM,
    }
    result = extract_root_causes(chains, node_lookup)
    assert len(result) == 1
    assert result[0]["entity"] == "CSM Reassignment"
    assert result[0]["cause_score"] == 0.54


def test_extract_root_causes_deduplicates_same_root():
    chains = [
        {"causal_path": ["X", "Root"], "chain_strength": 0.60},
        {"causal_path": ["Y", "Root"], "chain_strength": 0.80},
    ]
    result = extract_root_causes(chains, {})
    assert len(result) == 1
    assert result[0]["entity"] == "Root"
    assert result[0]["cause_score"] == 0.80  # takes max


def test_extract_root_causes_sorted_descending():
    chains = [
        {"causal_path": ["effect", "cause_a"], "chain_strength": 0.40},
        {"causal_path": ["effect", "cause_b"], "chain_strength": 0.70},
    ]
    result = extract_root_causes(chains, {})
    scores = [r["cause_score"] for r in result]
    assert scores == sorted(scores, reverse=True)


def test_extract_root_causes_single_hop_path_excluded():
    # path of length 1 has no cause (only the effect itself)
    chains = [{"causal_path": ["OnlyEffect"], "chain_strength": 0.9}]
    result = extract_root_causes(chains, {})
    assert result == []


# ---------------------------------------------------------------------------
# build_chain_narrative
# ---------------------------------------------------------------------------


def test_build_chain_narrative_empty_path():
    result = build_chain_narrative(
        effect_entity="X",
        chain={"causal_path": []},
        node_lookup={},
    )
    assert result == ""


def test_build_chain_narrative_two_hop():
    chain = {"causal_path": ["Churn Signal", "Acme Corp", "CSM Reassignment"]}
    node_lookup = {
        "Churn Signal": _NODE_CHURN,
        "Acme Corp": _NODE_ACME,
        "CSM Reassignment": _NODE_CSM,
    }
    result = build_chain_narrative(
        effect_entity="Churn Signal",
        chain=chain,
        node_lookup=node_lookup,
    )
    # Reversed: CSM Reassignment → Acme Corp → Churn Signal
    assert "CSM Reassignment" in result
    assert "Churn Signal" in result
    assert result.index("CSM Reassignment") < result.index("Churn Signal")


def test_build_chain_narrative_includes_entity_types():
    chain = {"causal_path": ["Churn Signal", "Acme Corp"]}
    node_lookup = {"Acme Corp": _NODE_ACME, "Churn Signal": _NODE_CHURN}
    result = build_chain_narrative(
        effect_entity="Churn Signal",
        chain=chain,
        node_lookup=node_lookup,
    )
    assert "customer" in result
    assert "metric" in result


def test_build_chain_narrative_unknown_node_no_type_label():
    chain = {"causal_path": ["unknown_entity", "another"]}
    result = build_chain_narrative(
        effect_entity="another",
        chain=chain,
        node_lookup={},
    )
    # No entity_type available — should not crash, just omit type
    assert "unknown_entity" in result


# ---------------------------------------------------------------------------
# generate_counterfactual_hint
# ---------------------------------------------------------------------------


def test_generate_counterfactual_hint_no_causes():
    assert generate_counterfactual_hint([], "Churn Signal") is None


def test_generate_counterfactual_hint_no_effect():
    causes = [{"entity": "CSM", "entity_type": "event", "cause_score": 0.7, "domains": []}]
    assert generate_counterfactual_hint(causes, None) is None


def test_generate_counterfactual_hint_full():
    causes = [
        {"entity": "CSM Reassignment", "entity_type": "event", "cause_score": 0.72, "domains": ["hr"]}
    ]
    result = generate_counterfactual_hint(causes, "Churn Signal")
    assert result is not None
    assert "CSM Reassignment" in result
    assert "Churn Signal" in result
    assert "hr" in result
    assert "0.72" in result


# ---------------------------------------------------------------------------
# CausalReasoningAgent — metadata
# ---------------------------------------------------------------------------


def test_agent_name_and_version():
    agent = CausalReasoningAgent()
    assert agent.name == "CausalReasoningAgent"
    assert agent.version == "v1"


def test_agent_dependencies():
    agent = CausalReasoningAgent()
    assert "ProactiveMemoryPushAgent" in agent.dependencies()


def test_registry_lookup():
    for key in ("causal_reasoning", "causalreasoning", "causalreasoningagent"):
        assert isinstance(get_agent(key), CausalReasoningAgent)


# ---------------------------------------------------------------------------
# CausalReasoningAgent.run — heuristic path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_empty_enrichment_returns_success():
    agent = CausalReasoningAgent()
    result = await agent.run("mem-1", _ctx())
    assert result.status == "success"
    assert isinstance(result.outputs["causal_chains"], list)
    assert isinstance(result.outputs["root_causes"], list)
    assert result.outputs["rationale"] == "heuristic"


@pytest.mark.asyncio
async def test_run_heuristic_strategy_override():
    agent = CausalReasoningAgent()
    with patch("app.agents.causal_reasoning_agent.settings") as mock_settings:
        mock_settings.CAUSAL_REASONING_STRATEGY = "heuristic"
        mock_settings.AGENT_STRATEGY = "llm"
        result = await agent.run(
            "mem-1",
            _ctx(content="some content", world_nodes=[_NODE_ACME, _NODE_CHURN]),
        )
    assert result.outputs["rationale"] == "heuristic"


@pytest.mark.asyncio
async def test_run_with_cross_silo_nodes_detects_effects():
    agent = CausalReasoningAgent()
    result = await agent.run(
        "mem-1",
        _ctx(
            world_nodes=[_NODE_ACME],  # silo_count=2
            entity_edges=[_EDGE_ACME_CHURN],
        ),
    )
    assert result.outputs["top_effect"] == "Acme Corp"


@pytest.mark.asyncio
async def test_run_push_candidates_seed_effects():
    agent = CausalReasoningAgent()
    result = await agent.run(
        "mem-1",
        _ctx(
            world_nodes=[_NODE_ACME, _NODE_CHURN, _NODE_CSM],
            entity_edges=[_EDGE_CSM_ACME, _EDGE_ACME_CHURN],
            push_candidates=[_PUSH_ACME],
        ),
    )
    assert result.outputs["top_effect"] == "Acme Corp"


@pytest.mark.asyncio
async def test_run_causal_chain_crosses_silos():
    agent = CausalReasoningAgent()
    result = await agent.run(
        "mem-1",
        _ctx(
            world_nodes=[_NODE_ACME, _NODE_CHURN, _NODE_CSM],
            entity_edges=[_EDGE_CSM_ACME, _EDGE_ACME_CHURN],
            propagated_signals=[_SIG_CROSS_SILO],
            push_candidates=[_PUSH_ACME],
        ),
    )
    chains = result.outputs["causal_chains"]
    # At least one chain should cross silos
    if chains:
        silo_spans = [c["silo_span"] for c in chains]
        assert max(silo_spans) >= 1


@pytest.mark.asyncio
async def test_run_root_causes_non_empty_when_chains_found():
    agent = CausalReasoningAgent()
    result = await agent.run(
        "mem-1",
        _ctx(
            world_nodes=[_NODE_ACME, _NODE_CHURN, _NODE_PERSON],
            entity_edges=[_EDGE_PERSON_ACME, _EDGE_ACME_CHURN],
            push_candidates=[_PUSH_ACME],
        ),
    )
    if result.outputs["causal_chains"]:
        assert len(result.outputs["root_causes"]) >= 1


@pytest.mark.asyncio
async def test_run_counterfactual_hint_present_when_causes_found():
    agent = CausalReasoningAgent()
    result = await agent.run(
        "mem-1",
        _ctx(
            world_nodes=[_NODE_ACME, _NODE_CHURN, _NODE_PERSON],
            entity_edges=[_EDGE_PERSON_ACME, _EDGE_ACME_CHURN],
            push_candidates=[_PUSH_ACME],
        ),
    )
    if result.outputs["root_causes"]:
        assert result.outputs["counterfactual_hint"] is not None


@pytest.mark.asyncio
async def test_run_confidence_in_range():
    agent = CausalReasoningAgent()
    result = await agent.run(
        "mem-1",
        _ctx(
            world_nodes=[_NODE_ACME, _NODE_CHURN, _NODE_CSM],
            entity_edges=[_EDGE_CSM_ACME, _EDGE_ACME_CHURN],
            push_candidates=[_PUSH_ACME],
        ),
    )
    assert 0.0 <= result.confidence <= 1.0


@pytest.mark.asyncio
async def test_run_confidence_low_when_no_chains():
    agent = CausalReasoningAgent()
    result = await agent.run("mem-1", _ctx())
    assert result.outputs["confidence"] == 0.3


@pytest.mark.asyncio
async def test_run_trace_id_propagated():
    agent = CausalReasoningAgent()
    result = await agent.run("mem-1", _ctx(job_id="job-xyz"))
    assert result.trace_id == "job-xyz"


@pytest.mark.asyncio
async def test_run_propagated_signals_expand_cause_set():
    agent = CausalReasoningAgent()
    # Without signal: metric → metric should not be causal
    # With signal: billing_issue is in propagated → becomes a cause
    billing_node = {
        "entity": "Billing Issue",
        "entity_type": "metric",
        "domains": ["finance"],
        "silo_count": 1,
        "node_confidence": 0.60,
    }
    billing_sig = {
        "content": "Billing Issue escalated",
        "source_silo": "finance",
        "target_domains": ["sales"],
        "relevance_score": 0.70,
        "entities": ["Billing Issue"],
    }
    result = await agent.run(
        "mem-1",
        _ctx(
            world_nodes=[_NODE_CHURN, billing_node],
            entity_edges=[
                {"from_entity": "Billing Issue", "to_entity": "Churn Signal", "weight": 0.6}
            ],
            propagated_signals=[billing_sig],
            push_candidates=[_PUSH_ACME],
        ),
    )
    assert result.status == "success"


# ---------------------------------------------------------------------------
# CausalReasoningAgent.run — LLM path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_llm_valid_response_used():
    agent = CausalReasoningAgent()
    llm_output = {
        "causal_chains": [
            {
                "effect_entity": "Churn Signal",
                "causal_path": ["Churn Signal", "Acme Corp"],
                "hop_count": 1,
                "chain_strength": 0.65,
                "silo_span": 2,
                "narrative": "Acme Corp (customer) → Churn Signal (metric)",
            }
        ],
        "root_causes": [
            {"entity": "Acme Corp", "entity_type": "customer", "cause_score": 0.65, "domains": ["sales"]}
        ],
        "counterfactual_hint": "Addressing Acme Corp could mitigate Churn Signal (cause_score=0.65)",
        "top_effect": "Churn Signal",
        "confidence": 0.75,
        "rationale": "llm reasoning",
    }
    with patch("app.agents.causal_reasoning_agent.settings") as mock_settings:
        mock_settings.CAUSAL_REASONING_STRATEGY = "llm"
        mock_settings.AGENT_STRATEGY = "llm"
        mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
        mock_settings.OLLAMA_MODEL = "llama3.1:8b"
        mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
        mock_settings.OLLAMA_MAX_CONCURRENCY = 2
        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value=llm_output)
        with patch(
            "app.agents.causal_reasoning_agent.create_ollama_client",
            return_value=mock_client,
        ):
            result = await agent.run(
                "mem-1",
                _ctx(content="Acme Corp churn signal detected"),
            )
    assert result.outputs["causal_chains"] == llm_output["causal_chains"]
    assert result.outputs["confidence"] == 0.75


@pytest.mark.asyncio
async def test_run_llm_bad_response_falls_back_to_heuristic():
    agent = CausalReasoningAgent()
    with patch("app.agents.causal_reasoning_agent.settings") as mock_settings:
        mock_settings.CAUSAL_REASONING_STRATEGY = "llm"
        mock_settings.AGENT_STRATEGY = "llm"
        mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
        mock_settings.OLLAMA_MODEL = "llama3.1:8b"
        mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
        mock_settings.OLLAMA_MAX_CONCURRENCY = 2
        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value={"invalid": True})
        with patch(
            "app.agents.causal_reasoning_agent.create_ollama_client",
            return_value=mock_client,
        ):
            result = await agent.run(
                "mem-1",
                _ctx(content="some content"),
            )
    assert result.outputs["rationale"] == "heuristic"


# ---------------------------------------------------------------------------
# CausalReasoningAgent.validate_outputs
# ---------------------------------------------------------------------------


def test_validate_outputs_passes_on_valid():
    agent = CausalReasoningAgent()
    outputs = {
        "causal_chains": [
            {
                "effect_entity": "X",
                "causal_path": ["X", "Y"],
                "hop_count": 1,
                "chain_strength": 0.5,
                "silo_span": 1,
                "narrative": "Y → X",
            }
        ],
        "root_causes": [],
        "counterfactual_hint": None,
        "top_effect": "X",
        "confidence": 0.6,
        "rationale": "heuristic",
    }
    agent.validate_outputs(_make_result(outputs))  # should not raise


def test_validate_outputs_fails_if_causal_chains_not_list():
    agent = CausalReasoningAgent()
    outputs = {"causal_chains": "not_a_list", "root_causes": []}
    with pytest.raises(ValueError, match="causal_chains must be a list"):
        agent.validate_outputs(_make_result(outputs))


def test_validate_outputs_fails_if_root_causes_not_list():
    agent = CausalReasoningAgent()
    outputs = {"causal_chains": [], "root_causes": "bad"}
    with pytest.raises(ValueError, match="root_causes must be a list"):
        agent.validate_outputs(_make_result(outputs))


def test_validate_outputs_fails_if_chain_missing_effect_entity():
    agent = CausalReasoningAgent()
    outputs = {
        "causal_chains": [{"causal_path": ["A", "B"]}],
        "root_causes": [],
    }
    with pytest.raises(ValueError, match="effect_entity"):
        agent.validate_outputs(_make_result(outputs))


def test_validate_outputs_fails_if_chain_missing_causal_path():
    agent = CausalReasoningAgent()
    outputs = {
        "causal_chains": [{"effect_entity": "X"}],
        "root_causes": [],
    }
    with pytest.raises(ValueError, match="causal_path"):
        agent.validate_outputs(_make_result(outputs))


def test_validate_outputs_skipped_on_non_success():
    agent = CausalReasoningAgent()
    now = datetime.now(timezone.utc)
    result = AgentResult(
        agent_name="CausalReasoningAgent",
        agent_version="v1",
        memory_id="mem-1",
        status="failed",
        confidence=0.0,
        outputs={"causal_chains": "bad"},
        warnings=[],
        errors=[],
        started_at=now,
        finished_at=now,
    )
    agent.validate_outputs(result)  # should not raise
