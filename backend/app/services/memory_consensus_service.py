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

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


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

    def __init__(self, *, quorum_k: int = 2, quorum_n: int = 3, redis_url: str | None = None) -> None:
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

        # Optional Redis write-through persistence — survives pod restarts and
        # enables cross-pod consistency. When None, in-memory dicts are the only store.
        self._r = None
        self._redis_prefix = "mcs"
        if redis_url:
            try:
                import redis as _redis_sync
                self._r = _redis_sync.Redis.from_url(redis_url, decode_responses=True)
                self._load_from_redis()
            except Exception as exc:
                logger.warning("MemoryConsensusService: Redis init failed (%s); using in-memory only", exc)
                self._r = None

    # ------------------------------------------------------------------
    # Redis persistence helpers
    # ------------------------------------------------------------------

    def _rkey(self, *parts: str) -> str:
        return ":".join([self._redis_prefix] + list(parts))

    def _persist_claim(self, claim: MemoryClaim) -> None:
        if self._r is None:
            return
        try:
            data = {
                "claim_id": claim.claim_id, "content": claim.content,
                "submitter": claim.submitter, "org_id": claim.org_id,
                "confidence": claim.confidence,
                "submitted_at": claim.submitted_at.isoformat(),
                "provenance": claim.provenance, "status": claim.status,
            }
            self._r.set(self._rkey("claim", claim.claim_id), json.dumps(data))
            self._r.sadd(self._rkey("claim_ids"), claim.claim_id)
        except Exception as exc:
            logger.warning("Redis persist_claim failed: %s", exc)

    def _persist_votes(self, claim_id: str, votes: list[ClaimVote]) -> None:
        if self._r is None:
            return
        try:
            data = [
                {"claim_id": v.claim_id, "evaluator": v.evaluator, "verdict": v.verdict,
                 "confidence": v.confidence, "rationale": v.rationale,
                 "voted_at": v.voted_at.isoformat()}
                for v in votes
            ]
            self._r.set(self._rkey("votes", claim_id), json.dumps(data))
        except Exception as exc:
            logger.warning("Redis persist_votes failed: %s", exc)

    def _persist_shared(self, fact: SharedFact) -> None:
        if self._r is None:
            return
        try:
            data = {
                "claim_id": fact.claim_id, "content": fact.content,
                "submitter": fact.submitter, "org_id": fact.org_id,
                "initial_confidence": fact.initial_confidence,
                "consensus_count": fact.consensus_count,
                "agent_contributors": fact.agent_contributors,
                "promoted_at": fact.promoted_at.isoformat(),
                "tombstoned": fact.tombstoned,
                "tombstoned_at": fact.tombstoned_at.isoformat() if fact.tombstoned_at else None,
                "provenance": fact.provenance,
            }
            self._r.set(self._rkey("shared", fact.claim_id), json.dumps(data))
            if not fact.tombstoned:
                self._r.sadd(self._rkey("shared_ids"), fact.claim_id)
            else:
                self._r.srem(self._rkey("shared_ids"), fact.claim_id)
        except Exception as exc:
            logger.warning("Redis persist_shared failed: %s", exc)

    def _persist_receipts(self, agent_id: str, claim_ids: set[str]) -> None:
        if self._r is None:
            return
        try:
            self._r.set(self._rkey("receipts", agent_id), json.dumps(list(claim_ids)))
        except Exception as exc:
            logger.warning("Redis persist_receipts failed: %s", exc)

    def _load_from_redis(self) -> None:
        """Restore in-memory state from Redis on startup."""
        if self._r is None:
            return
        try:
            claim_ids = self._r.smembers(self._rkey("claim_ids")) or set()
            for cid in claim_ids:
                raw = self._r.get(self._rkey("claim", cid))
                if not raw:
                    continue
                d = json.loads(raw)
                claim = MemoryClaim(
                    claim_id=d["claim_id"], content=d["content"],
                    submitter=d["submitter"], org_id=d["org_id"],
                    confidence=d["confidence"],
                    submitted_at=datetime.fromisoformat(d["submitted_at"]),
                    provenance=d.get("provenance") or {}, status=d["status"],
                )
                self._claims[cid] = claim
                votes_raw = self._r.get(self._rkey("votes", cid))
                if votes_raw:
                    self._votes[cid] = [
                        ClaimVote(
                            claim_id=v["claim_id"], evaluator=v["evaluator"],
                            verdict=v["verdict"], confidence=v["confidence"],
                            rationale=v.get("rationale", ""),
                            voted_at=datetime.fromisoformat(v["voted_at"]),
                        )
                        for v in json.loads(votes_raw)
                    ]
                else:
                    self._votes[cid] = []

            shared_ids = self._r.smembers(self._rkey("shared_ids")) or set()
            for sid in shared_ids:
                raw = self._r.get(self._rkey("shared", sid))
                if not raw:
                    continue
                d = json.loads(raw)
                self._shared[sid] = SharedFact(
                    claim_id=d["claim_id"], content=d["content"],
                    submitter=d["submitter"], org_id=d["org_id"],
                    initial_confidence=d["initial_confidence"],
                    consensus_count=d["consensus_count"],
                    agent_contributors=d["agent_contributors"],
                    promoted_at=datetime.fromisoformat(d["promoted_at"]),
                    tombstoned=d.get("tombstoned", False),
                    tombstoned_at=(datetime.fromisoformat(d["tombstoned_at"]) if d.get("tombstoned_at") else None),
                    provenance=d.get("provenance") or {},
                )

            # Receipts — scan for all receipt keys
            for key in self._r.scan_iter(self._rkey("receipts", "*")):
                agent_id = key.split(":")[-1]
                raw = self._r.get(key)
                if raw:
                    self._receipts[agent_id] = set(json.loads(raw))
        except Exception as exc:
            logger.warning("MemoryConsensusService: Redis load failed (%s); starting empty", exc)

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
        self._persist_claim(claim)
        self._persist_votes(cid, [])
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
        self._persist_votes(cid, self._votes[cid])

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
            self._persist_claim(claim)
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
            self._persist_claim(claim)
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
        self._persist_claim(claim)
        self._persist_shared(shared)
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
        self._persist_receipts(aid, self._receipts[aid])

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
            self._persist_receipts(aid, self._receipts[aid])

        self._persist_claim(self._claims[cid])
        self._persist_shared(fact)

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
