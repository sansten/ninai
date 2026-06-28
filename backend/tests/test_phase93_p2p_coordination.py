"""Phase 93 — P2PCoordinationService tests."""
from __future__ import annotations

import time
import uuid

import pytest

from app.services.p2p_coordination_service import (
    AgentCapabilityManifest,
    BidResult,
    ClaimResult,
    P2PCoordinationService,
    Task,
)


def _svc() -> P2PCoordinationService:
    return P2PCoordinationService()


# ---------------------------------------------------------------------------
# Agent registration
# ---------------------------------------------------------------------------

class TestAgentRegistration:
    def test_register_returns_manifest(self):
        svc = _svc()
        m = svc.register_agent("agent-a", capabilities=["research", "synthesis"])
        assert isinstance(m, AgentCapabilityManifest)
        assert m.agent_id == "agent-a"

    def test_capabilities_lowercased(self):
        svc = _svc()
        m = svc.register_agent("agent-a", capabilities=["RESEARCH", "EMAIL"])
        assert "research" in m.capabilities
        assert "email" in m.capabilities

    def test_discover_returns_registered_agent(self):
        svc = _svc()
        svc.register_agent("agent-a", capabilities=["research"])
        agents = svc.discover_agents()
        ids = [a.agent_id for a in agents]
        assert "agent-a" in ids

    def test_deregister_removes_agent(self):
        svc = _svc()
        svc.register_agent("agent-a", capabilities=["research"])
        svc.deregister_agent("agent-a")
        agents = svc.discover_agents()
        assert all(a.agent_id != "agent-a" for a in agents)

    def test_discover_filtered_by_capability(self):
        svc = _svc()
        svc.register_agent("researcher", capabilities=["research", "synthesis"])
        svc.register_agent("mailer", capabilities=["email", "slack"])
        research_agents = svc.discover_agents(capability="research")
        ids = [a.agent_id for a in research_agents]
        assert "researcher" in ids
        assert "mailer" not in ids

    def test_expired_agent_not_returned(self):
        svc = _svc()
        m = svc.register_agent("dying-agent", capabilities=["x"], ttl_seconds=1)
        # Manually backdate
        m.registered_at = time.time() - 10
        agents = svc.discover_agents()
        ids = [a.agent_id for a in agents]
        assert "dying-agent" not in ids

    def test_can_handle_partial_match(self):
        m = AgentCapabilityManifest(agent_id="a", capabilities=["research"])
        assert m.can_handle("research_synthesis")
        assert m.can_handle("research")
        assert not m.can_handle("email")

    def test_multiple_agents_registered(self):
        svc = _svc()
        for i in range(5):
            svc.register_agent(f"agent-{i}", capabilities=["task"])
        assert len(svc.discover_agents()) == 5


# ---------------------------------------------------------------------------
# Task posting
# ---------------------------------------------------------------------------

class TestTaskPosting:
    def test_post_task_returns_id(self):
        svc = _svc()
        tid = svc.post_task(task_type="research", payload={"q": "test"})
        assert isinstance(tid, str)
        assert len(tid) > 0

    def test_posted_task_is_pending(self):
        svc = _svc()
        tid = svc.post_task(task_type="research", payload={})
        task = svc.get_task(tid)
        assert task is not None
        assert task.status == "pending"

    def test_task_type_and_payload_preserved(self):
        svc = _svc()
        tid = svc.post_task(task_type="email", payload={"to": "user@x.com"})
        task = svc.get_task(tid)
        assert task.task_type == "email"
        assert task.payload["to"] == "user@x.com"

    def test_priority_clamped(self):
        svc = _svc()
        tid = svc.post_task(task_type="x", payload={}, priority=5.0)
        task = svc.get_task(tid)
        assert task.priority <= 1.0

    def test_get_pending_tasks_sorted_by_priority(self):
        svc = _svc()
        svc.post_task(task_type="t", payload={}, priority=0.2)
        svc.post_task(task_type="t", payload={}, priority=0.9)
        svc.post_task(task_type="t", payload={}, priority=0.5)
        pending = svc.get_pending_tasks()
        priorities = [t.priority for t in pending]
        assert priorities == sorted(priorities, reverse=True)

    def test_get_pending_filtered_by_capability(self):
        svc = _svc()
        svc.post_task(task_type="research", payload={})
        svc.post_task(task_type="email", payload={})
        pending = svc.get_pending_tasks(capability="research")
        assert all("research" in t.task_type for t in pending)

    def test_get_task_unknown_returns_none(self):
        svc = _svc()
        assert svc.get_task("nonexistent") is None


# ---------------------------------------------------------------------------
# Bidding
# ---------------------------------------------------------------------------

class TestBidding:
    def test_bid_accepted_for_pending_task(self):
        svc = _svc()
        tid = svc.post_task(task_type="research", payload={})
        result = svc.bid_for_task(tid, agent_id="agent-a", confidence=0.85)
        assert result.accepted is True
        assert result.confidence == 0.85

    def test_bid_rejected_for_unknown_task(self):
        svc = _svc()
        result = svc.bid_for_task("nonexistent", agent_id="agent-a", confidence=0.9)
        assert result.accepted is False

    def test_higher_bid_is_leading(self):
        svc = _svc()
        tid = svc.post_task(task_type="x", payload={})
        svc.bid_for_task(tid, agent_id="agent-a", confidence=0.70)
        result = svc.bid_for_task(tid, agent_id="agent-b", confidence=0.95)
        assert result.is_leading_bid is True

    def test_lower_bid_not_leading(self):
        svc = _svc()
        tid = svc.post_task(task_type="x", payload={})
        svc.bid_for_task(tid, agent_id="agent-a", confidence=0.95)
        result = svc.bid_for_task(tid, agent_id="agent-b", confidence=0.60)
        assert result.is_leading_bid is False

    def test_leading_bidder_returns_winner(self):
        svc = _svc()
        tid = svc.post_task(task_type="x", payload={})
        svc.bid_for_task(tid, agent_id="agent-a", confidence=0.70)
        svc.bid_for_task(tid, agent_id="agent-b", confidence=0.95)
        leader = svc.leading_bidder(tid)
        assert leader is not None
        assert leader[0] == "agent-b"
        assert leader[1] == 0.95

    def test_leading_bidder_none_for_no_bids(self):
        svc = _svc()
        tid = svc.post_task(task_type="x", payload={})
        assert svc.leading_bidder(tid) is None

    def test_bid_on_claimed_task_rejected(self):
        svc = _svc()
        tid = svc.post_task(task_type="x", payload={})
        svc.claim_task(tid, agent_id="agent-a")
        result = svc.bid_for_task(tid, agent_id="agent-b", confidence=0.9)
        assert result.accepted is False

    def test_confidence_clamped_to_zero_one(self):
        svc = _svc()
        tid = svc.post_task(task_type="x", payload={})
        result = svc.bid_for_task(tid, agent_id="agent-a", confidence=5.0)
        assert result.confidence <= 1.0


# ---------------------------------------------------------------------------
# Claiming
# ---------------------------------------------------------------------------

class TestClaiming:
    def test_claim_succeeds_for_pending_task(self):
        svc = _svc()
        tid = svc.post_task(task_type="x", payload={})
        result = svc.claim_task(tid, agent_id="agent-a")
        assert result.claimed is True

    def test_claim_sets_owner(self):
        svc = _svc()
        tid = svc.post_task(task_type="x", payload={})
        svc.claim_task(tid, agent_id="agent-a")
        task = svc.get_task(tid)
        assert task.owner == "agent-a"
        assert task.status == "claimed"

    def test_double_claim_fails(self):
        svc = _svc()
        tid = svc.post_task(task_type="x", payload={})
        svc.claim_task(tid, agent_id="agent-a")
        result2 = svc.claim_task(tid, agent_id="agent-b")
        assert result2.claimed is False

    def test_claim_unknown_task_returns_not_claimed(self):
        svc = _svc()
        result = svc.claim_task("no-such-task", agent_id="agent-a")
        assert result.claimed is False

    def test_claim_already_completed_fails(self):
        svc = _svc()
        tid = svc.post_task(task_type="x", payload={})
        svc.claim_task(tid, agent_id="agent-a")
        svc.resolve_task(tid, agent_id="agent-a", outcome="success")
        result = svc.claim_task(tid, agent_id="agent-b")
        assert result.claimed is False


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

class TestResolution:
    def test_resolve_marks_completed(self):
        svc = _svc()
        tid = svc.post_task(task_type="x", payload={})
        svc.claim_task(tid, agent_id="agent-a")
        task = svc.resolve_task(tid, agent_id="agent-a", outcome="success")
        assert task.status == "completed"

    def test_resolve_failure_marks_failed(self):
        svc = _svc()
        tid = svc.post_task(task_type="x", payload={})
        svc.claim_task(tid, agent_id="agent-a")
        task = svc.resolve_task(tid, agent_id="agent-a", outcome="error")
        assert task.status == "failed"

    def test_resolve_by_non_owner_raises(self):
        svc = _svc()
        tid = svc.post_task(task_type="x", payload={})
        svc.claim_task(tid, agent_id="agent-a")
        with pytest.raises(ValueError):
            svc.resolve_task(tid, agent_id="agent-b", outcome="success")

    def test_resolve_unknown_task_raises(self):
        svc = _svc()
        with pytest.raises(KeyError):
            svc.resolve_task("no-task", agent_id="a", outcome="success")

    def test_result_data_preserved(self):
        svc = _svc()
        tid = svc.post_task(task_type="x", payload={})
        svc.claim_task(tid, agent_id="agent-a")
        task = svc.resolve_task(tid, agent_id="agent-a", outcome="success",
                                result_data={"summary": "done"})
        assert task.completion["result_data"]["summary"] == "done"


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_initial(self):
        svc = _svc()
        s = svc.stats()
        assert s["pending_tasks"] == 0
        assert s["total_tasks"] == 0

    def test_stats_after_posting(self):
        svc = _svc()
        svc.post_task(task_type="x", payload={})
        s = svc.stats()
        assert s["pending_tasks"] == 1
        assert s["total_tasks"] == 1

    def test_stats_after_full_lifecycle(self):
        svc = _svc()
        tid = svc.post_task(task_type="x", payload={})
        svc.claim_task(tid, agent_id="a")
        svc.resolve_task(tid, agent_id="a", outcome="success")
        s = svc.stats()
        assert s["completed_tasks"] == 1
        assert s["pending_tasks"] == 0

    def test_live_agents_counted(self):
        svc = _svc()
        svc.register_agent("a1", capabilities=["x"])
        svc.register_agent("a2", capabilities=["y"])
        s = svc.stats()
        assert s["live_agents"] == 2
