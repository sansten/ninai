"""Tests for MemoryConsensusService (Phase 85)."""

from __future__ import annotations

import pytest

from app.services.memory_consensus_service import (
    MemoryConsensusService,
    MemoryClaim,
    SharedFact,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _svc(k: int = 2, n: int = 3) -> MemoryConsensusService:
    return MemoryConsensusService(quorum_k=k, quorum_n=n)


def _claim(svc: MemoryConsensusService, **kwargs) -> MemoryClaim:
    defaults = dict(content="The sky is blue", submitter="agent-A", org_id="org-1", confidence=0.8)
    defaults.update(kwargs)
    return svc.submit_claim(**defaults)


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------

class TestSubmitClaim:
    def test_returns_memory_claim_instance(self):
        svc = _svc()
        claim = _claim(svc)
        assert isinstance(claim, MemoryClaim)

    def test_initial_status_is_pending(self):
        svc = _svc()
        claim = _claim(svc)
        assert claim.status == "pending"

    def test_submitted_at_is_set(self):
        svc = _svc()
        claim = _claim(svc)
        assert claim.submitted_at is not None

    def test_provenance_stored(self):
        svc = _svc()
        claim = _claim(svc, provenance={"source_memory_id": "mem-42"})
        assert claim.provenance == {"source_memory_id": "mem-42"}

    def test_confidence_clamped_high(self):
        svc = _svc()
        claim = _claim(svc, confidence=9.0)
        assert claim.confidence == 1.0

    def test_confidence_clamped_low(self):
        svc = _svc()
        claim = _claim(svc, confidence=-1.0)
        assert claim.confidence == 0.0

    def test_duplicate_claim_id_raises(self):
        svc = _svc()
        _claim(svc, claim_id="fixed-id")
        with pytest.raises(ValueError, match="already exists"):
            _claim(svc, claim_id="fixed-id")

    def test_custom_claim_id_honoured(self):
        svc = _svc()
        claim = _claim(svc, claim_id="my-custom-id")
        assert claim.claim_id == "my-custom-id"

    def test_empty_content_accepted(self):
        svc = _svc()
        claim = _claim(svc, content="")
        assert claim.content == ""

    def test_multiple_claims_tracked_independently(self):
        svc = _svc()
        c1 = _claim(svc, content="fact A", claim_id="c1")
        c2 = _claim(svc, content="fact B", claim_id="c2")
        assert c1.claim_id != c2.claim_id
        assert len(svc.get_pending_claims()) == 2


# ---------------------------------------------------------------------------
# Evaluate — vote recording
# ---------------------------------------------------------------------------

class TestEvaluateClaim:
    def test_confirm_recorded(self):
        svc = _svc()
        c = _claim(svc)
        result = svc.evaluate_claim(claim_id=c.claim_id, evaluator="agent-B", verdict="confirm", confidence=0.9)
        assert result["confirms"] == 1

    def test_contradict_recorded(self):
        svc = _svc()
        c = _claim(svc)
        result = svc.evaluate_claim(claim_id=c.claim_id, evaluator="agent-B", verdict="contradict", confidence=0.9)
        assert result["contradicts"] == 1

    def test_insufficient_recorded(self):
        svc = _svc()
        c = _claim(svc)
        result = svc.evaluate_claim(claim_id=c.claim_id, evaluator="agent-B", verdict="insufficient", confidence=0.5)
        assert result["action"] == "vote_recorded"

    def test_unknown_claim_id_raises_key_error(self):
        svc = _svc()
        with pytest.raises(KeyError):
            svc.evaluate_claim(claim_id="no-such-id", evaluator="agent-B", verdict="confirm")

    def test_invalid_verdict_raises_value_error(self):
        svc = _svc()
        c = _claim(svc)
        with pytest.raises(ValueError, match="verdict must be"):
            svc.evaluate_claim(claim_id=c.claim_id, evaluator="agent-B", verdict="maybe")

    def test_submitter_cannot_self_vote(self):
        svc = _svc()
        c = _claim(svc, submitter="agent-A")
        with pytest.raises(ValueError, match="Submitter"):
            svc.evaluate_claim(claim_id=c.claim_id, evaluator="agent-A", verdict="confirm")

    def test_duplicate_evaluator_raises(self):
        svc = _svc()
        c = _claim(svc)
        svc.evaluate_claim(claim_id=c.claim_id, evaluator="agent-B", verdict="confirm")
        with pytest.raises(ValueError, match="already voted"):
            svc.evaluate_claim(claim_id=c.claim_id, evaluator="agent-B", verdict="confirm")

    def test_vote_on_non_pending_claim_raises(self):
        svc = _svc(k=1, n=3)
        c = _claim(svc)
        svc.evaluate_claim(claim_id=c.claim_id, evaluator="agent-B", verdict="confirm")
        # Now promoted
        with pytest.raises(ValueError, match="not pending"):
            svc.evaluate_claim(claim_id=c.claim_id, evaluator="agent-C", verdict="confirm")

    def test_action_vote_recorded_before_quorum(self):
        svc = _svc(k=2, n=3)
        c = _claim(svc)
        result = svc.evaluate_claim(claim_id=c.claim_id, evaluator="agent-B", verdict="confirm")
        assert result["action"] == "vote_recorded"
        assert result["status"] == "pending"


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------

class TestPromotion:
    def test_k_confirms_promotes(self):
        svc = _svc(k=2, n=3)
        c = _claim(svc)
        svc.evaluate_claim(claim_id=c.claim_id, evaluator="agent-B", verdict="confirm")
        result = svc.evaluate_claim(claim_id=c.claim_id, evaluator="agent-C", verdict="confirm")
        assert result["action"] == "promoted"
        assert result["status"] == "promoted"

    def test_promoted_fact_in_shared_pool(self):
        svc = _svc(k=2, n=3)
        c = _claim(svc)
        svc.evaluate_claim(claim_id=c.claim_id, evaluator="agent-B", verdict="confirm")
        svc.evaluate_claim(claim_id=c.claim_id, evaluator="agent-C", verdict="confirm")
        facts = svc.get_shared_facts()
        assert len(facts) == 1
        assert facts[0].claim_id == c.claim_id

    def test_promoted_claim_status_updated(self):
        svc = _svc(k=1, n=3)
        c = _claim(svc)
        svc.evaluate_claim(claim_id=c.claim_id, evaluator="agent-B", verdict="confirm")
        assert c.status == "promoted"

    def test_agent_contributors_includes_submitter_and_confirmers(self):
        svc = _svc(k=2, n=3)
        c = _claim(svc, submitter="agent-A")
        svc.evaluate_claim(claim_id=c.claim_id, evaluator="agent-B", verdict="confirm")
        svc.evaluate_claim(claim_id=c.claim_id, evaluator="agent-C", verdict="confirm")
        fact = svc.get_shared_facts()[0]
        assert "agent-A" in fact.agent_contributors
        assert "agent-B" in fact.agent_contributors
        assert "agent-C" in fact.agent_contributors

    def test_consensus_count_reflects_confirmers(self):
        svc = _svc(k=2, n=3)
        c = _claim(svc)
        svc.evaluate_claim(claim_id=c.claim_id, evaluator="agent-B", verdict="confirm")
        svc.evaluate_claim(claim_id=c.claim_id, evaluator="agent-C", verdict="confirm")
        assert svc.get_shared_facts()[0].consensus_count == 2

    def test_result_shared_fact_dict_present_on_promote(self):
        svc = _svc(k=1, n=3)
        c = _claim(svc)
        result = svc.evaluate_claim(claim_id=c.claim_id, evaluator="agent-B", verdict="confirm")
        sf = result["shared_fact"]
        assert sf is not None
        assert sf["content"] == c.content
        assert sf["tombstoned"] is False

    def test_k3_requires_three_confirms(self):
        svc = _svc(k=3, n=5)
        c = _claim(svc)
        r1 = svc.evaluate_claim(claim_id=c.claim_id, evaluator="b", verdict="confirm")
        r2 = svc.evaluate_claim(claim_id=c.claim_id, evaluator="c", verdict="confirm")
        assert r1["action"] == "vote_recorded"
        assert r2["action"] == "vote_recorded"
        r3 = svc.evaluate_claim(claim_id=c.claim_id, evaluator="d", verdict="confirm")
        assert r3["action"] == "promoted"

    def test_promote_despite_some_contradicts_if_confirms_reach_k(self):
        svc = _svc(k=2, n=4)
        c = _claim(svc)
        svc.evaluate_claim(claim_id=c.claim_id, evaluator="b", verdict="contradict")
        svc.evaluate_claim(claim_id=c.claim_id, evaluator="c", verdict="confirm")
        result = svc.evaluate_claim(claim_id=c.claim_id, evaluator="d", verdict="confirm")
        assert result["action"] == "promoted"


# ---------------------------------------------------------------------------
# Quarantine
# ---------------------------------------------------------------------------

class TestQuarantine:
    def test_single_contradict_no_confirms_quarantines(self):
        # k=2, n=2: after 1 contradict, remaining=1, confirms=0; 0+1=1 < k=2 → quarantine
        svc = _svc(k=2, n=2)
        c = _claim(svc)
        result = svc.evaluate_claim(claim_id=c.claim_id, evaluator="agent-B", verdict="contradict")
        assert result["action"] == "quarantined"
        assert c.status == "quarantined"

    def test_more_contradicts_than_confirms_quarantines(self):
        # k=3, n=3: after confirm=1, contradict=1; remaining=1, 1+1=2 < k=3 → quarantine
        svc = _svc(k=3, n=3)
        c = _claim(svc)
        svc.evaluate_claim(claim_id=c.claim_id, evaluator="b", verdict="confirm")
        result = svc.evaluate_claim(claim_id=c.claim_id, evaluator="c", verdict="contradict")
        assert result["action"] == "quarantined"

    def test_quarantined_claim_not_in_shared_pool(self):
        svc = _svc(k=2, n=3)
        c = _claim(svc)
        svc.evaluate_claim(claim_id=c.claim_id, evaluator="b", verdict="contradict")
        assert len(svc.get_shared_facts()) == 0


# ---------------------------------------------------------------------------
# Expiry
# ---------------------------------------------------------------------------

class TestExpiry:
    def test_all_insufficient_expires_after_n(self):
        svc = _svc(k=2, n=2)
        c = _claim(svc)
        svc.evaluate_claim(claim_id=c.claim_id, evaluator="b", verdict="insufficient")
        result = svc.evaluate_claim(claim_id=c.claim_id, evaluator="c", verdict="insufficient")
        assert result["action"] == "expired"
        assert c.status == "expired"

    def test_expired_claim_not_in_shared_pool(self):
        svc = _svc(k=2, n=2)
        c = _claim(svc)
        svc.evaluate_claim(claim_id=c.claim_id, evaluator="b", verdict="insufficient")
        svc.evaluate_claim(claim_id=c.claim_id, evaluator="c", verdict="insufficient")
        assert len(svc.get_shared_facts()) == 0


# ---------------------------------------------------------------------------
# Quorum status
# ---------------------------------------------------------------------------

class TestQuorumStatus:
    def test_quorum_status_reflects_votes(self):
        svc = _svc(k=2, n=3)
        c = _claim(svc)
        svc.evaluate_claim(claim_id=c.claim_id, evaluator="b", verdict="confirm")
        qs = svc.quorum_status(c.claim_id)
        assert qs["confirms"] == 1
        assert qs["contradicts"] == 0
        assert qs["insufficient"] == 0
        assert qs["remaining_slots"] == 2

    def test_quorum_status_unknown_claim_raises(self):
        svc = _svc()
        with pytest.raises(KeyError):
            svc.quorum_status("no-such-id")

    def test_quorum_status_fields_present(self):
        svc = _svc()
        c = _claim(svc)
        qs = svc.quorum_status(c.claim_id)
        for key in ("claim_id", "status", "confirms", "contradicts", "insufficient", "total_votes", "remaining_slots", "quorum_k", "quorum_n"):
            assert key in qs


# ---------------------------------------------------------------------------
# Receipt tracking
# ---------------------------------------------------------------------------

class TestReceiptTracking:
    def test_acknowledge_receipt_tracked(self):
        svc = _svc(k=1, n=3)
        c = _claim(svc)
        svc.evaluate_claim(claim_id=c.claim_id, evaluator="b", verdict="confirm")
        svc.acknowledge_receipt(claim_id=c.claim_id, agent_id="consumer-X")
        radius = svc.contamination_radius(c.claim_id)
        assert "consumer-X" in radius["recipients"]
        assert radius["recipient_count"] == 1

    def test_multiple_agents_acknowledged(self):
        svc = _svc(k=1, n=3)
        c = _claim(svc)
        svc.evaluate_claim(claim_id=c.claim_id, evaluator="b", verdict="confirm")
        svc.acknowledge_receipt(claim_id=c.claim_id, agent_id="agent-X")
        svc.acknowledge_receipt(claim_id=c.claim_id, agent_id="agent-Y")
        radius = svc.contamination_radius(c.claim_id)
        assert radius["recipient_count"] == 2

    def test_acknowledge_non_promoted_raises(self):
        svc = _svc(k=2, n=3)
        c = _claim(svc)
        with pytest.raises(ValueError, match="No promoted fact"):
            svc.acknowledge_receipt(claim_id=c.claim_id, agent_id="agent-X")

    def test_contamination_radius_zero_initially(self):
        svc = _svc(k=1, n=3)
        c = _claim(svc)
        svc.evaluate_claim(claim_id=c.claim_id, evaluator="b", verdict="confirm")
        radius = svc.contamination_radius(c.claim_id)
        assert radius["recipient_count"] == 0


# ---------------------------------------------------------------------------
# Tombstone
# ---------------------------------------------------------------------------

class TestTombstone:
    def test_tombstone_promoted_fact(self):
        svc = _svc(k=1, n=3)
        c = _claim(svc)
        svc.evaluate_claim(claim_id=c.claim_id, evaluator="b", verdict="confirm")
        result = svc.tombstone(c.claim_id)
        assert result["claim_id"] == c.claim_id
        assert "tombstoned_at" in result

    def test_tombstoned_fact_excluded_from_shared_pool(self):
        svc = _svc(k=1, n=3)
        c = _claim(svc)
        svc.evaluate_claim(claim_id=c.claim_id, evaluator="b", verdict="confirm")
        svc.tombstone(c.claim_id)
        assert len(svc.get_shared_facts()) == 0

    def test_tombstone_returns_affected_agents(self):
        svc = _svc(k=1, n=3)
        c = _claim(svc)
        svc.evaluate_claim(claim_id=c.claim_id, evaluator="b", verdict="confirm")
        svc.acknowledge_receipt(claim_id=c.claim_id, agent_id="consumer-1")
        svc.acknowledge_receipt(claim_id=c.claim_id, agent_id="consumer-2")
        result = svc.tombstone(c.claim_id)
        assert set(result["affected_agents"]) == {"consumer-1", "consumer-2"}

    def test_tombstone_clears_receipts(self):
        svc = _svc(k=1, n=3)
        c = _claim(svc)
        svc.evaluate_claim(claim_id=c.claim_id, evaluator="b", verdict="confirm")
        svc.acknowledge_receipt(claim_id=c.claim_id, agent_id="consumer-1")
        svc.tombstone(c.claim_id)
        radius = svc.contamination_radius(c.claim_id)
        assert radius["recipient_count"] == 0

    def test_tombstone_pending_raises(self):
        svc = _svc(k=2, n=3)
        c = _claim(svc)
        with pytest.raises(ValueError, match="No promoted fact"):
            svc.tombstone(c.claim_id)

    def test_double_tombstone_raises(self):
        svc = _svc(k=1, n=3)
        c = _claim(svc)
        svc.evaluate_claim(claim_id=c.claim_id, evaluator="b", verdict="confirm")
        svc.tombstone(c.claim_id)
        with pytest.raises(ValueError, match="already tombstoned"):
            svc.tombstone(c.claim_id)

    def test_claim_status_set_to_tombstoned(self):
        svc = _svc(k=1, n=3)
        c = _claim(svc)
        svc.evaluate_claim(claim_id=c.claim_id, evaluator="b", verdict="confirm")
        svc.tombstone(c.claim_id)
        assert c.status == "tombstoned"

    def test_acknowledge_tombstoned_fact_raises(self):
        svc = _svc(k=1, n=3)
        c = _claim(svc)
        svc.evaluate_claim(claim_id=c.claim_id, evaluator="b", verdict="confirm")
        svc.tombstone(c.claim_id)
        with pytest.raises(ValueError, match="tombstoned"):
            svc.acknowledge_receipt(claim_id=c.claim_id, agent_id="late-consumer")


# ---------------------------------------------------------------------------
# List methods
# ---------------------------------------------------------------------------

class TestListMethods:
    def test_get_pending_returns_only_pending(self):
        svc = _svc(k=1, n=3)
        c1 = _claim(svc, claim_id="c1")
        c2 = _claim(svc, claim_id="c2")
        svc.evaluate_claim(claim_id=c1.claim_id, evaluator="b", verdict="confirm")
        pending = svc.get_pending_claims()
        assert len(pending) == 1
        assert pending[0].claim_id == "c2"

    def test_get_shared_returns_only_promoted(self):
        svc = _svc(k=1, n=3)
        c1 = _claim(svc, claim_id="c1")
        _claim(svc, claim_id="c2")
        svc.evaluate_claim(claim_id=c1.claim_id, evaluator="b", verdict="confirm")
        shared = svc.get_shared_facts()
        assert len(shared) == 1
        assert shared[0].claim_id == "c1"

    def test_org_id_filters_pending(self):
        svc = _svc()
        _claim(svc, org_id="org-1", claim_id="c1")
        _claim(svc, org_id="org-2", claim_id="c2")
        assert len(svc.get_pending_claims(org_id="org-1")) == 1
        assert len(svc.get_pending_claims(org_id="org-2")) == 1

    def test_org_id_filters_shared(self):
        svc = _svc(k=1, n=3)
        c1 = _claim(svc, org_id="org-1", claim_id="c1")
        c2 = _claim(svc, org_id="org-2", claim_id="c2")
        svc.evaluate_claim(claim_id=c1.claim_id, evaluator="b", verdict="confirm")
        svc.evaluate_claim(claim_id=c2.claim_id, evaluator="b", verdict="confirm")
        assert len(svc.get_shared_facts(org_id="org-1")) == 1


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestConstructorValidation:
    def test_quorum_k_zero_raises(self):
        with pytest.raises(ValueError, match="quorum_k"):
            MemoryConsensusService(quorum_k=0, quorum_n=3)

    def test_quorum_n_less_than_k_raises(self):
        with pytest.raises(ValueError, match="quorum_n"):
            MemoryConsensusService(quorum_k=3, quorum_n=2)

    def test_k_equals_n_valid(self):
        svc = MemoryConsensusService(quorum_k=2, quorum_n=2)
        assert svc._quorum_k == 2
