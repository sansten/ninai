"""Consensus-gated memory promotion service (Phase 85).

Multi-agent context sharing is unsolved because broadcast memory propagates
bad beliefs — if one agent is wrong, every downstream agent inherits the error.

This service applies a distributed-consensus model to memory promotion:
a fact cannot enter the shared pool until k independent agents confirm it.
Contradicted facts are quarantined for human review rather than broadcast.
Post-promotion revocation (tombstone) tracks which agents received the fact
so belief-correction can be targeted rather than requiring a full reset.

Flow:
  1. Agent A submits a MemoryClaim (content + provenance + confidence).
  2. OrchestrationBus fans the claim to N evaluator agents.
  3. Each evaluator returns confirm / contradict / insufficient.
  4. If confirms >= quorum_k  →  SharedFact promoted, recipients tracked.
     If contradicts > confirms  →  claim quarantined (human review queue).
     If all N evaluators voted and no quorum  →  claim expires.
  5. When a promoted fact is later challenged, tombstone() revokes it and
     returns the list of agents that acknowledged receipt for targeted correction.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MemoryClaim:
    claim_id: str
    content: str
    submitter: str
    org_id: str
    confidence: float
    submitted_at: datetime
    provenance: dict[str, Any]
    status: str  # pending | promoted | quarantined | expired | tombstoned


@dataclass
class ClaimVote:
    claim_id: str
    evaluator: str
    verdict: str  # confirm | contradict | insufficient
    confidence: float
    rationale: str
    voted_at: datetime


@dataclass
class SharedFact:
    claim_id: str
    content: str
    submitter: str
    org_id: str
    initial_confidence: float
    consensus_count: int
    agent_contributors: list[str]
    promoted_at: datetime
    tombstoned: bool = False
    tombstoned_at: datetime | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class MemoryConsensusService:
    """Consensus-gated shared memory pool.

    Parameters
    ----------
    quorum_k:
        Minimum number of independent *confirm* votes required to promote a
        claim to the shared pool. Default 2.
    quorum_n:
        Maximum number of evaluators per claim. Once N agents have voted the
        decision is final regardless of whether quorum was reached. Default 3.
    """

    VALID_VERDICTS = frozenset({"confirm", "contradict", "insufficient"})

    def __init__(self, *, quorum_k: int = 2, quorum_n: int = 3) -> None:
        if quorum_k < 1:
            raise ValueError("quorum_k must be >= 1")
        if quorum_n < quorum_k:
            raise ValueError("quorum_n must be >= quorum_k")

        self._quorum_k = quorum_k
        self._quorum_n = quorum_n

        self._claims: dict[str, MemoryClaim] = {}
        self._votes: dict[str, list[ClaimVote]] = {}
        self._shared: dict[str, SharedFact] = {}
        # agent_id → set of claim_ids the agent acknowledged receiving
        self._receipts: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit_claim(
        self,
        *,
        content: str,
        submitter: str,
        org_id: str,
        confidence: float = 0.7,
        provenance: dict[str, Any] | None = None,
        claim_id: str | None = None,
    ) -> MemoryClaim:
        """Register a new memory fact for consensus evaluation.

        Parameters
        ----------
        content:    The fact to be shared — a short, assertable statement.
        submitter:  Identifier of the agent submitting the claim.
        org_id:     Tenant scope.
        confidence: Submitter's own confidence in the claim (0–1).
        provenance: Arbitrary metadata (source memory_id, session, etc.).
        claim_id:   Optional deterministic ID; auto-generated if omitted.
        """
        cid = str(claim_id or uuid.uuid4())
        if cid in self._claims:
            raise ValueError(f"claim_id already exists: {cid}")

        claim = MemoryClaim(
            claim_id=cid,
            content=str(content or ""),
            submitter=str(submitter or ""),
            org_id=str(org_id or ""),
            confidence=max(0.0, min(1.0, float(confidence or 0.0))),
            submitted_at=datetime.now(timezone.utc),
            provenance=dict(provenance or {}),
            status="pending",
        )
        self._claims[cid] = claim
        self._votes[cid] = []
        return claim

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_claim(
        self,
        *,
        claim_id: str,
        evaluator: str,
        verdict: str,
        confidence: float = 0.7,
        rationale: str = "",
    ) -> dict[str, Any]:
        """Cast a vote on a pending claim.

        Returns a dict describing the outcome after this vote:
          - status:       current claim status after processing
          - confirms:     running confirm count
          - contradicts:  running contradict count
          - action:       "vote_recorded" | "promoted" | "quarantined" | "expired"
          - shared_fact:  SharedFact dict if promoted, else None
        """
        cid = str(claim_id or "")
        if cid not in self._claims:
            raise KeyError(f"Unknown claim_id: {cid}")

        claim = self._claims[cid]
        if claim.status != "pending":
            raise ValueError(
                f"Claim {cid} is not pending (status={claim.status!r}); cannot accept more votes"
            )

        if verdict not in self.VALID_VERDICTS:
            raise ValueError(f"verdict must be one of {sorted(self.VALID_VERDICTS)}, got {verdict!r}")

        evaluator_s = str(evaluator or "")
        if evaluator_s == claim.submitter:
            raise ValueError("Submitter cannot evaluate their own claim")

        existing_voters = {v.evaluator for v in self._votes[cid]}
        if evaluator_s in existing_voters:
            raise ValueError(f"Agent {evaluator_s!r} has already voted on claim {cid}")

        vote = ClaimVote(
            claim_id=cid,
            evaluator=evaluator_s,
            verdict=verdict,
            confidence=max(0.0, min(1.0, float(confidence or 0.0))),
            rationale=str(rationale or ""),
            voted_at=datetime.now(timezone.utc),
        )
        self._votes[cid].append(vote)

        return self._decide(cid)

    def _decide(self, claim_id: str) -> dict[str, Any]:
        claim = self._claims[claim_id]
        votes = self._votes[claim_id]

        confirms = sum(1 for v in votes if v.verdict == "confirm")
        contradicts = sum(1 for v in votes if v.verdict == "contradict")
        total_decisive = len(votes)

        # Quorum reached — promote
        if confirms >= self._quorum_k:
            shared = self._promote(claim_id)
            return {
                "status": "promoted",
                "confirms": confirms,
                "contradicts": contradicts,
                "action": "promoted",
                "shared_fact": self._shared_fact_dict(shared),
            }

        # Quarantine — quorum is mathematically unreachable given remaining slots
        remaining = max(0, self._quorum_n - total_decisive)
        if contradicts > 0 and (confirms + remaining) < self._quorum_k:
            claim.status = "quarantined"
            return {
                "status": "quarantined",
                "confirms": confirms,
                "contradicts": contradicts,
                "action": "quarantined",
                "shared_fact": None,
            }

        # All N evaluators voted and no quorum — expire
        if total_decisive >= self._quorum_n:
            claim.status = "expired"
            return {
                "status": "expired",
                "confirms": confirms,
                "contradicts": contradicts,
                "action": "expired",
                "shared_fact": None,
            }

        # Vote recorded; decision pending
        return {
            "status": "pending",
            "confirms": confirms,
            "contradicts": contradicts,
            "action": "vote_recorded",
            "shared_fact": None,
        }

    def _promote(self, claim_id: str) -> SharedFact:
        claim = self._claims[claim_id]
        votes = self._votes[claim_id]
        confirming_agents = [v.evaluator for v in votes if v.verdict == "confirm"]
        shared = SharedFact(
            claim_id=claim_id,
            content=claim.content,
            submitter=claim.submitter,
            org_id=claim.org_id,
            initial_confidence=claim.confidence,
            consensus_count=len(confirming_agents),
            agent_contributors=[claim.submitter] + confirming_agents,
            promoted_at=datetime.now(timezone.utc),
            provenance=claim.provenance,
        )
        self._shared[claim_id] = shared
        claim.status = "promoted"
        return shared

    # ------------------------------------------------------------------
    # Shared-pool read
    # ------------------------------------------------------------------

    def get_pending_claims(self, org_id: str | None = None) -> list[MemoryClaim]:
        """All claims currently awaiting evaluation."""
        results = [c for c in self._claims.values() if c.status == "pending"]
        if org_id is not None:
            results = [c for c in results if c.org_id == str(org_id)]
        return results

    def get_shared_facts(self, org_id: str | None = None) -> list[SharedFact]:
        """All promoted (non-tombstoned) shared facts."""
        results = [f for f in self._shared.values() if not f.tombstoned]
        if org_id is not None:
            results = [f for f in results if f.org_id == str(org_id)]
        return results

    def quorum_status(self, claim_id: str) -> dict[str, Any]:
        """Current vote counts and remaining evaluator slots for a claim."""
        cid = str(claim_id or "")
        if cid not in self._claims:
            raise KeyError(f"Unknown claim_id: {cid}")
        votes = self._votes[cid]
        confirms = sum(1 for v in votes if v.verdict == "confirm")
        contradicts = sum(1 for v in votes if v.verdict == "contradict")
        insufficient = sum(1 for v in votes if v.verdict == "insufficient")
        total = len(votes)
        return {
            "claim_id": cid,
            "status": self._claims[cid].status,
            "confirms": confirms,
            "contradicts": contradicts,
            "insufficient": insufficient,
            "total_votes": total,
            "remaining_slots": max(0, self._quorum_n - total),
            "quorum_k": self._quorum_k,
            "quorum_n": self._quorum_n,
        }

    # ------------------------------------------------------------------
    # Receipt tracking
    # ------------------------------------------------------------------

    def acknowledge_receipt(self, *, claim_id: str, agent_id: str) -> None:
        """Record that an agent received and is acting on a promoted fact.

        Raises ValueError if the fact is not promoted (or is tombstoned).
        """
        cid = str(claim_id or "")
        if cid not in self._shared:
            raise ValueError(f"No promoted fact with claim_id: {cid}")
        if self._shared[cid].tombstoned:
            raise ValueError(f"Fact {cid} is tombstoned; cannot acknowledge")
        aid = str(agent_id or "")
        self._receipts.setdefault(aid, set()).add(cid)

    def contamination_radius(self, claim_id: str) -> dict[str, Any]:
        """Number of agents that acknowledged receipt of this promoted fact."""
        cid = str(claim_id or "")
        affected = [aid for aid, cids in self._receipts.items() if cid in cids]
        return {
            "claim_id": cid,
            "recipient_count": len(affected),
            "recipients": affected,
        }

    # ------------------------------------------------------------------
    # Tombstone / revocation
    # ------------------------------------------------------------------

    def tombstone(self, claim_id: str) -> dict[str, Any]:
        """Revoke a promoted fact and identify affected agents for correction.

        Only promoted facts can be tombstoned. Returns a dict with:
          - claim_id
          - affected_agents: list of agent_ids that acknowledged receipt
          - tombstoned_at
        """
        cid = str(claim_id or "")
        if cid not in self._shared:
            raise ValueError(f"No promoted fact with claim_id: {cid}; only promoted facts can be tombstoned")
        fact = self._shared[cid]
        if fact.tombstoned:
            raise ValueError(f"Fact {cid} is already tombstoned")

        now = datetime.now(timezone.utc)
        fact.tombstoned = True
        fact.tombstoned_at = now
        self._claims[cid].status = "tombstoned"

        affected = [aid for aid, cids in self._receipts.items() if cid in cids]
        # Clear receipts so contamination_radius returns 0 post-tombstone
        for aid in affected:
            self._receipts[aid].discard(cid)

        return {
            "claim_id": cid,
            "affected_agents": affected,
            "tombstoned_at": now.isoformat(),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _shared_fact_dict(fact: SharedFact) -> dict[str, Any]:
        return {
            "claim_id": fact.claim_id,
            "content": fact.content,
            "submitter": fact.submitter,
            "org_id": fact.org_id,
            "initial_confidence": fact.initial_confidence,
            "consensus_count": fact.consensus_count,
            "agent_contributors": fact.agent_contributors,
            "promoted_at": fact.promoted_at.isoformat(),
            "tombstoned": fact.tombstoned,
            "provenance": fact.provenance,
        }
