"""Real-world validation: memory decay and credibility scoring.

Dataset: IT helpdesk and enterprise event records (Kaggle-style).
Validates that decay half-lives and credibility tiers produce semantically
correct results — not just structurally valid outputs.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

os.environ.setdefault("AGENT_STRATEGY", "heuristic")

from app.agents.memory_decay_agent import MemoryDecayAgent
from app.agents.credibility_agent import CredibilityAgent
from tests.e2e.data import (
    HELPDESK_TICKETS,
    get_ticket,
    make_credibility_context,
    make_memory_context,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decay_ctx(ticket: dict, *, last_accessed_days_ago: int | None = None) -> dict:
    ctx = make_memory_context(ticket)
    if last_accessed_days_ago is not None:
        ts = (
            datetime.now(timezone.utc) - timedelta(days=last_accessed_days_ago)
        ).isoformat()
        ctx["memory"]["enrichment"]["last_accessed_at"] = ts
    return ctx


def _decay_ctx_with_ref(ticket: dict, *, reference_count: int) -> dict:
    ctx = make_memory_context(ticket)
    ctx["memory"]["enrichment"]["reference_count"] = reference_count
    return ctx


# ---------------------------------------------------------------------------
# Decay tests
# ---------------------------------------------------------------------------


class TestDecayWithRealData:
    """Validate exponential decay semantics against realistic helpdesk data."""

    async def test_fresh_incident_high_freshness(self) -> None:
        ticket = get_ticket("INC-001")  # 2 days old, half-life=7
        agent = MemoryDecayAgent()
        result = await agent.run(ticket["ticket_id"], _decay_ctx(ticket))

        assert result.status == "success"
        fs = result.outputs["freshness_score"]
        # e^(-ln2/7 * 2) ≈ 0.82 — should be comfortably above 0.75
        assert fs > 0.75, f"Expected freshness > 0.75 for 2-day-old incident, got {fs}"
        assert result.outputs["is_stale"] is False

    async def test_stale_incident_low_freshness(self) -> None:
        ticket = get_ticket("INC-010")  # 45 days old, half-life=7
        agent = MemoryDecayAgent()
        result = await agent.run(ticket["ticket_id"], _decay_ctx(ticket))

        assert result.status == "success"
        fs = result.outputs["freshness_score"]
        # e^(-ln2/7 * 45) ≈ 0.011 — plus reference boost (0.06) still well below 0.30
        assert fs < 0.30, f"Expected freshness < 0.30 for 45-day-old incident, got {fs}"
        assert result.outputs["is_stale"] is True

    async def test_incident_decays_faster_than_document(self) -> None:
        # Build synthetic contexts both 30 days old
        base_ts = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

        inc_ctx = {
            "memory": {
                "content": "Production auth service down.",
                "created_at": base_ts,
                "enrichment": {"domain": "incident", "reference_count": 0},
            },
            "runtime": {"job_id": "e2e-decay-compare"},
        }
        doc_ctx = {
            "memory": {
                "content": "Architecture decision record for event sourcing.",
                "created_at": base_ts,
                "enrichment": {"domain": "document", "reference_count": 0},
            },
            "runtime": {"job_id": "e2e-decay-compare-2"},
        }

        agent = MemoryDecayAgent()
        inc_result = await agent.run("mem-inc-30", inc_ctx)
        doc_result = await agent.run("mem-doc-30", doc_ctx)

        inc_fs = inc_result.outputs["freshness_score"]
        doc_fs = doc_result.outputs["freshness_score"]

        # incident half-life=7 vs document half-life=365; same age means incident is much staler
        assert inc_fs < doc_fs, (
            f"Incident freshness ({inc_fs}) should be lower than document freshness ({doc_fs}) "
            f"at 30 days old"
        )

    async def test_strategy_doc_still_fresh_at_90_days(self) -> None:
        ticket = get_ticket("STR-001")  # 120 days old — let's use a custom 90-day context
        base_ts = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        ctx = {
            "memory": {
                "content": ticket["description"],
                "created_at": base_ts,
                "enrichment": {"domain": "strategy", "reference_count": 0},
            },
            "runtime": {"job_id": "e2e-strategy-90"},
        }
        agent = MemoryDecayAgent()
        result = await agent.run("mem-str-90", ctx)

        assert result.status == "success"
        fs = result.outputs["freshness_score"]
        # strategy half-life=180; at 90 days base freshness ≈ 0.50, no reference boost
        assert fs > 0.35, f"Expected strategy doc freshness > 0.35 at 90 days, got {fs}"
        assert result.outputs["half_life"] == 180.0

    async def test_recently_accessed_memory_gets_recency_boost(self) -> None:
        ticket = get_ticket("INC-003")  # 3 days old
        ctx_no_access = _decay_ctx(ticket)
        ctx_accessed = _decay_ctx(ticket, last_accessed_days_ago=1)

        agent = MemoryDecayAgent()
        r_no_access = await agent.run(ticket["ticket_id"] + "-a", ctx_no_access)
        r_accessed = await agent.run(ticket["ticket_id"] + "-b", ctx_accessed)

        fs_no = r_no_access.outputs["freshness_score"]
        fs_yes = r_accessed.outputs["freshness_score"]

        assert fs_yes > fs_no, (
            f"Memory accessed 1 day ago should have higher freshness ({fs_yes}) "
            f"than unaccessed ({fs_no})"
        )
        assert r_accessed.outputs["recency_boost"] > 0.0

    async def test_highly_referenced_memory_boost(self) -> None:
        ticket = get_ticket("INC-001")  # 2 days old
        ctx_zero = _decay_ctx_with_ref(ticket, reference_count=0)
        ctx_high = _decay_ctx_with_ref(ticket, reference_count=10)

        agent = MemoryDecayAgent()
        r_zero = await agent.run("mem-ref-0", ctx_zero)
        r_high = await agent.run("mem-ref-10", ctx_high)

        fs_zero = r_zero.outputs["freshness_score"]
        fs_high = r_high.outputs["freshness_score"]

        assert fs_high > fs_zero, (
            f"10 references should yield higher freshness ({fs_high}) than 0 ({fs_zero})"
        )
        # Max reference boost is 0.20; at 10 refs (0.02 each) = 0.20
        assert r_high.outputs["reference_boost"] == 0.20

    async def test_deployment_event_half_life(self) -> None:
        # At exactly the deployment half-life (21 days), base freshness should be ≈ 0.50
        base_ts = (datetime.now(timezone.utc) - timedelta(days=21)).isoformat()
        ctx = {
            "memory": {
                "content": "v2.4.0 deployed to production.",
                "created_at": base_ts,
                "enrichment": {"domain": "deployment", "reference_count": 0},
            },
            "runtime": {"job_id": "e2e-deploy-half-life"},
        }
        agent = MemoryDecayAgent()
        result = await agent.run("mem-dep-21", ctx)

        assert result.status == "success"
        fs = result.outputs["freshness_score"]
        # Should be close to 0.50 (half-life point); allow ±0.10 for floating point
        assert 0.40 <= fs <= 0.60, (
            f"At 21 days (half-life of deployment), expected freshness ≈ 0.50, got {fs}"
        )
        assert result.outputs["half_life"] == 21.0

    async def test_all_domains_produce_valid_range(self) -> None:
        agent = MemoryDecayAgent()
        for ticket in HELPDESK_TICKETS:
            ctx = make_memory_context(ticket)
            result = await agent.run(ticket["ticket_id"], ctx)
            fs = result.outputs["freshness_score"]
            assert 0.0 <= fs <= 1.0, (
                f"Ticket {ticket['ticket_id']}: freshness_score {fs} out of [0, 1]"
            )
            assert isinstance(result.outputs["is_stale"], bool)
            assert result.outputs["age_days"] >= 0.0


# ---------------------------------------------------------------------------
# Credibility tests
# ---------------------------------------------------------------------------


class TestCredibilityWithRealData:
    """Validate credibility scoring semantics against realistic helpdesk data."""

    async def test_engineer_api_source_high_credibility(self) -> None:
        ticket = get_ticket("INC-001")  # engineer, slack, entities present
        agent = CredibilityAgent()
        result = await agent.run(ticket["ticket_id"], make_credibility_context(ticket))

        assert result.status == "success"
        score = result.outputs["credibility_score"]
        # engineer in _PRIMARY_AUTHOR_ROLES, slack is not in known types → falls to author_role
        # expect primary tier → base 0.80, entities present
        assert score > 0.70, (
            f"Expected credibility > 0.70 for engineer/slack source, got {score}"
        )

    async def test_vague_user_chat_low_credibility(self) -> None:
        ticket = get_ticket("INC-004")  # user, chat, no entities
        agent = CredibilityAgent()
        result = await agent.run(ticket["ticket_id"], make_credibility_context(ticket))

        assert result.status == "success"
        score = result.outputs["credibility_score"]
        # user not in any role set → unknown tier, no citation, no entities
        assert score < 0.55, (
            f"Expected credibility < 0.55 for vague user/chat report, got {score}"
        )

    async def test_system_alert_credibility(self) -> None:
        ticket = get_ticket("INC-002")  # system source, system role
        agent = CredibilityAgent()
        result = await agent.run(ticket["ticket_id"], make_credibility_context(ticket))

        assert result.status == "success"
        score = result.outputs["credibility_score"]
        tier = result.outputs["source_tier"]
        # system is in _PRIMARY_SOURCE_TYPES → primary tier → base 0.80
        assert score > 0.65, f"Expected system alert credibility > 0.65, got {score}"
        assert tier == "primary"

    async def test_corroborated_memory_higher_credibility(self) -> None:
        ticket = get_ticket("INC-003")
        corroborating = [
            {"id": "mem-a", "content": "Payment gateway confirmed timing out.", "entities": {}},
            {"id": "mem-b", "content": "Stripe latency spike confirmed in APAC region.", "entities": {}},
            {"id": "mem-c", "content": "Checkout failure rate at 61% confirmed by monitoring.", "entities": {}},
        ]
        agent = CredibilityAgent()
        r_bare = await agent.run("inc-003-bare", make_credibility_context(ticket))
        r_corr = await agent.run(
            "inc-003-corr",
            make_credibility_context(ticket, corroborating=corroborating),
        )

        score_bare = r_bare.outputs["credibility_score"]
        score_corr = r_corr.outputs["credibility_score"]

        assert score_corr > score_bare, (
            f"Corroborated memory ({score_corr}) should score higher than uncorroborated ({score_bare})"
        )
        assert r_corr.outputs["corroboration_count"] == 3

    async def test_conflict_exposure_reduces_credibility(self) -> None:
        ticket = get_ticket("CON-001")  # claims auth is healthy
        conflicts_with_auth = [
            {"entity": "authservice", "severity": "high", "conflict_type": "state_mismatch"},
            {"entity": "authservice", "severity": "high", "conflict_type": "state_mismatch"},
        ]
        agent = CredibilityAgent()
        r_clean = await agent.run("con-001-clean", make_credibility_context(ticket))
        r_conflicted = await agent.run(
            "con-001-conflict",
            make_credibility_context(ticket, conflicts=conflicts_with_auth),
        )

        score_clean = r_clean.outputs["credibility_score"]
        score_conflicted = r_conflicted.outputs["credibility_score"]

        assert score_conflicted < score_clean, (
            f"Conflicted memory ({score_conflicted}) should score lower than clean ({score_clean})"
        )

    async def test_credibility_ordering_by_source_tier(self) -> None:
        # Same content and author role — vary only source_type
        content = (
            "Auth service JWT validation failed. "
            "All tokens rejected; 401 errors on all endpoints."
        )
        entities = {"AuthService": "system"}

        def _ctx(source_type: str) -> dict:
            return {
                "memory": {
                    "content": content,
                    "source_type": source_type,
                    "author_role": "",
                    "enrichment": {
                        "entities": entities,
                        "conflicts": [],
                        "corroborating_memories": [],
                    },
                },
                "runtime": {"job_id": f"e2e-tier-{source_type}"},
            }

        agent = CredibilityAgent()
        sources = ["system", "api", "document", "file", "scrape"]
        scores = {}
        for src in sources:
            result = await agent.run(f"tier-{src}", _ctx(src))
            scores[src] = result.outputs["credibility_score"]

        # system and api are primary, document and file are secondary, scrape is tertiary
        assert scores["system"] >= scores["document"], (
            f"system ({scores['system']}) should be >= document ({scores['document']})"
        )
        assert scores["api"] >= scores["scrape"], (
            f"api ({scores['api']}) should be >= scrape ({scores['scrape']})"
        )
        assert scores["document"] > scores["scrape"], (
            f"document ({scores['document']}) should be > scrape ({scores['scrape']})"
        )

    async def test_all_tickets_produce_valid_credibility(self) -> None:
        agent = CredibilityAgent()
        for ticket in HELPDESK_TICKETS:
            ctx = make_credibility_context(ticket)
            result = await agent.run(ticket["ticket_id"], ctx)
            score = result.outputs["credibility_score"]
            assert 0.0 <= score <= 1.0, (
                f"Ticket {ticket['ticket_id']}: credibility_score {score} out of [0, 1]"
            )
            assert result.outputs["source_tier"] in {"primary", "secondary", "tertiary", "unknown"}
            assert isinstance(result.outputs["credibility_flags"], list)
