"""Phase 87 — Redis write-through persistence for MemoryConsensusService.

Tests verify that state survives a service restart when a Redis URL is provided,
and that the in-memory-only mode (no redis_url) is fully backward compatible.

Uses fakeredis for test isolation — no real Redis required.
When fakeredis is not installed, all tests are skipped automatically.
"""
from __future__ import annotations

import uuid
import pytest

try:
    import fakeredis
    _FAKEREDIS_AVAILABLE = True
except ImportError:
    _FAKEREDIS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _FAKEREDIS_AVAILABLE,
    reason="fakeredis not installed; skipping Redis persistence tests",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _redis_server():
    """Return a shared fakeredis server instance for the test session."""
    return fakeredis.FakeServer()


def _make_svc(server, *, quorum_k=2, quorum_n=3):
    """Build a MemoryConsensusService that persists to a fakeredis server."""
    from app.services.memory_consensus_service import MemoryConsensusService
    import redis as _redis_sync

    svc = MemoryConsensusService(quorum_k=quorum_k, quorum_n=quorum_n)
    # Inject fakeredis client directly instead of going through redis_url
    svc._r = fakeredis.FakeRedis(server=server, decode_responses=True)
    return svc


def _make_svc_loaded(server, *, quorum_k=2, quorum_n=3):
    """Simulate a pod restart by creating a new service and loading from fakeredis."""
    from app.services.memory_consensus_service import MemoryConsensusService
    svc = MemoryConsensusService(quorum_k=quorum_k, quorum_n=quorum_n)
    svc._r = fakeredis.FakeRedis(server=server, decode_responses=True)
    svc._load_from_redis()
    return svc


# ---------------------------------------------------------------------------
# Backward compatibility — no redis_url
# ---------------------------------------------------------------------------

class TestInMemoryModeBackwardCompat:
    def test_no_redis_url_works_as_before(self):
        from app.services.memory_consensus_service import MemoryConsensusService
        svc = MemoryConsensusService()
        claim = svc.submit_claim(content="fact", submitter="a", org_id="org1")
        assert claim.status == "pending"

    def test_no_redis_means_r_is_none(self):
        from app.services.memory_consensus_service import MemoryConsensusService
        svc = MemoryConsensusService()
        assert svc._r is None

    def test_full_lifecycle_without_redis(self):
        from app.services.memory_consensus_service import MemoryConsensusService
        svc = MemoryConsensusService(quorum_k=1, quorum_n=1)
        claim = svc.submit_claim(content="f", submitter="a", org_id="org1")
        result = svc.evaluate_claim(claim_id=claim.claim_id, evaluator="b", verdict="confirm")
        assert result["status"] == "promoted"


# ---------------------------------------------------------------------------
# Claim persistence
# ---------------------------------------------------------------------------

class TestClaimPersistence:
    def test_claim_survives_restart(self):
        server = _redis_server()
        svc1 = _make_svc(server)
        claim = svc1.submit_claim(content="important fact", submitter="agent-a", org_id="org1")

        svc2 = _make_svc_loaded(server)
        assert claim.claim_id in svc2._claims
        restored = svc2._claims[claim.claim_id]
        assert restored.content == "important fact"
        assert restored.org_id == "org1"
        assert restored.status == "pending"

    def test_claim_status_updated_on_quarantine(self):
        server = _redis_server()
        svc1 = _make_svc(server, quorum_k=2, quorum_n=2)
        claim = svc1.submit_claim(content="bad fact", submitter="agent-a", org_id="org1")
        svc1.evaluate_claim(claim_id=claim.claim_id, evaluator="agent-b", verdict="contradict")

        svc2 = _make_svc_loaded(server)
        restored = svc2._claims[claim.claim_id]
        assert restored.status == "quarantined"

    def test_claim_status_updated_on_promote(self):
        server = _redis_server()
        svc1 = _make_svc(server, quorum_k=1, quorum_n=2)
        claim = svc1.submit_claim(content="good fact", submitter="agent-a", org_id="org1")
        svc1.evaluate_claim(claim_id=claim.claim_id, evaluator="agent-b", verdict="confirm")

        svc2 = _make_svc_loaded(server)
        restored = svc2._claims[claim.claim_id]
        assert restored.status == "promoted"

    def test_multiple_claims_all_persisted(self):
        server = _redis_server()
        svc1 = _make_svc(server)
        ids = set()
        for i in range(5):
            c = svc1.submit_claim(content=f"fact {i}", submitter="a", org_id="org1")
            ids.add(c.claim_id)

        svc2 = _make_svc_loaded(server)
        for cid in ids:
            assert cid in svc2._claims


# ---------------------------------------------------------------------------
# Vote persistence
# ---------------------------------------------------------------------------

class TestVotePersistence:
    def test_votes_survive_restart(self):
        server = _redis_server()
        svc1 = _make_svc(server, quorum_k=2, quorum_n=3)
        claim = svc1.submit_claim(content="f", submitter="a", org_id="org1")
        svc1.evaluate_claim(claim_id=claim.claim_id, evaluator="b", verdict="confirm")

        svc2 = _make_svc_loaded(server)
        assert claim.claim_id in svc2._votes
        votes = svc2._votes[claim.claim_id]
        assert len(votes) == 1
        assert votes[0].verdict == "confirm"
        assert votes[0].evaluator == "b"

    def test_multiple_votes_preserved(self):
        server = _redis_server()
        svc1 = _make_svc(server, quorum_k=3, quorum_n=3)
        claim = svc1.submit_claim(content="f", submitter="a", org_id="org1")
        svc1.evaluate_claim(claim_id=claim.claim_id, evaluator="b", verdict="confirm")
        svc1.evaluate_claim(claim_id=claim.claim_id, evaluator="c", verdict="contradict")

        svc2 = _make_svc_loaded(server)
        votes = svc2._votes[claim.claim_id]
        verdicts = {v.verdict for v in votes}
        assert "confirm" in verdicts
        assert "contradict" in verdicts


# ---------------------------------------------------------------------------
# Shared fact persistence
# ---------------------------------------------------------------------------

class TestSharedFactPersistence:
    def test_promoted_fact_survives_restart(self):
        server = _redis_server()
        svc1 = _make_svc(server, quorum_k=1, quorum_n=2)
        claim = svc1.submit_claim(content="great fact", submitter="a", org_id="org1", confidence=0.9)
        svc1.evaluate_claim(claim_id=claim.claim_id, evaluator="b", verdict="confirm")

        svc2 = _make_svc_loaded(server)
        assert claim.claim_id in svc2._shared
        fact = svc2._shared[claim.claim_id]
        assert fact.content == "great fact"
        assert not fact.tombstoned

    def test_tombstoned_fact_persisted(self):
        server = _redis_server()
        svc1 = _make_svc(server, quorum_k=1, quorum_n=2)
        claim = svc1.submit_claim(content="wrong fact", submitter="a", org_id="org1")
        svc1.evaluate_claim(claim_id=claim.claim_id, evaluator="b", verdict="confirm")
        svc1.tombstone(claim.claim_id)

        svc2 = _make_svc_loaded(server)
        assert claim.claim_id in svc2._shared
        fact = svc2._shared[claim.claim_id]
        assert fact.tombstoned
        assert fact.tombstoned_at is not None

    def test_get_shared_facts_after_restart(self):
        server = _redis_server()
        svc1 = _make_svc(server, quorum_k=1, quorum_n=2)
        for i in range(3):
            c = svc1.submit_claim(content=f"fact {i}", submitter="a", org_id="org1")
            svc1.evaluate_claim(claim_id=c.claim_id, evaluator="b", verdict="confirm")

        svc2 = _make_svc_loaded(server)
        facts = svc2.get_shared_facts(org_id="org1")
        assert len(facts) == 3


# ---------------------------------------------------------------------------
# Receipt persistence
# ---------------------------------------------------------------------------

class TestReceiptPersistence:
    def test_receipts_survive_restart(self):
        server = _redis_server()
        svc1 = _make_svc(server, quorum_k=1, quorum_n=2)
        claim = svc1.submit_claim(content="f", submitter="a", org_id="org1")
        svc1.evaluate_claim(claim_id=claim.claim_id, evaluator="b", verdict="confirm")
        svc1.acknowledge_receipt(claim_id=claim.claim_id, agent_id="consumer-1")
        svc1.acknowledge_receipt(claim_id=claim.claim_id, agent_id="consumer-2")

        svc2 = _make_svc_loaded(server)
        radius = svc2.contamination_radius(claim.claim_id)
        assert radius["recipient_count"] == 2
        assert "consumer-1" in radius["recipients"]
        assert "consumer-2" in radius["recipients"]

    def test_tombstone_clears_receipts_in_redis(self):
        server = _redis_server()
        svc1 = _make_svc(server, quorum_k=1, quorum_n=2)
        claim = svc1.submit_claim(content="f", submitter="a", org_id="org1")
        svc1.evaluate_claim(claim_id=claim.claim_id, evaluator="b", verdict="confirm")
        svc1.acknowledge_receipt(claim_id=claim.claim_id, agent_id="consumer-1")
        tb = svc1.tombstone(claim.claim_id)
        assert "consumer-1" in tb["affected_agents"]

        svc2 = _make_svc_loaded(server)
        # After tombstone, receipts for this claim should be cleared
        radius = svc2.contamination_radius(claim.claim_id)
        # claim is tombstoned — no recipients tracked
        assert radius["recipient_count"] == 0


# ---------------------------------------------------------------------------
# Cross-pod correctness simulation
# ---------------------------------------------------------------------------

class TestCrossPodCorrectness:
    def test_pod_a_submits_pod_b_votes(self):
        """Simulate two pod instances sharing the same Redis."""
        server = _redis_server()
        pod_a = _make_svc(server, quorum_k=1, quorum_n=2)
        pod_b = _make_svc(server)

        # Pod A submits claim
        claim = pod_a.submit_claim(content="shared fact", submitter="agent-a", org_id="org1")

        # Pod B loads state from Redis and votes
        pod_b._load_from_redis()
        assert claim.claim_id in pod_b._claims

        result = pod_b.evaluate_claim(claim_id=claim.claim_id, evaluator="agent-b", verdict="confirm")
        assert result["status"] == "promoted"

        # Pod A loads updated state and sees the promotion
        pod_a._load_from_redis()
        assert pod_a._claims[claim.claim_id].status == "promoted"
