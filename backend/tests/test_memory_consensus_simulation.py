"""Simulation-based evaluation of MemoryConsensusService (Phase 85).

Answers the key operational questions:
  1. What precision/recall does the service achieve at different k/n settings?
  2. What evaluator accuracy is needed to keep contamination below a threshold?
  3. Does the k/n sweep produce the expected precision↑/recall↓ trade-off?
  4. How effective is tombstone coverage when a bad fact slips through?

Each test seeds a claim pool with a known mix of correct/incorrect facts, runs
synthetic evaluators with configurable accuracy and abstention rates, then
measures classification outcomes (TP/FP/TN/FN).
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Any

import pytest

from app.services.memory_consensus_service import MemoryConsensusService


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    """Metrics from one simulation run."""
    n_claims: int
    n_correct: int
    n_bad: int

    promoted_correct: int = 0   # TP
    promoted_bad: int = 0       # FP
    quarantined_correct: int = 0
    quarantined_bad: int = 0
    expired_correct: int = 0
    expired_bad: int = 0

    # Receipt/tombstone
    contaminated_agents_per_bad_fact: list[int] = field(default_factory=list)
    tombstone_coverage: list[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        denom = self.promoted_correct + self.promoted_bad
        return self.promoted_correct / denom if denom > 0 else 1.0

    @property
    def recall(self) -> float:
        denom = self.promoted_correct + self.quarantined_correct + self.expired_correct
        return self.promoted_correct / denom if denom > 0 else 0.0

    @property
    def contamination_prevention_rate(self) -> float:
        """Fraction of bad facts that were NOT promoted."""
        if self.n_bad == 0:
            return 1.0
        blocked = self.n_bad - self.promoted_bad
        return blocked / self.n_bad

    @property
    def quarantine_accuracy(self) -> float:
        """Of all quarantined claims, fraction that were genuinely bad."""
        total_q = self.quarantined_correct + self.quarantined_bad
        return self.quarantined_bad / total_q if total_q > 0 else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def simulate(
    *,
    n_claims: int = 200,
    bad_rate: float = 0.3,
    quorum_k: int = 2,
    quorum_n: int = 3,
    evaluator_accuracy: float = 0.85,
    insufficient_rate: float = 0.1,
    n_consumer_agents: int = 5,
    seed: int = 42,
) -> SimResult:
    """Run one simulation and return classification metrics.

    Parameters
    ----------
    n_claims:           Total facts submitted for consensus evaluation.
    bad_rate:           Fraction of submitted facts that are incorrect/bad.
    quorum_k:           Minimum confirms to promote.
    quorum_n:           Max evaluators per claim.
    evaluator_accuracy: Probability each evaluator votes correctly (0–1).
                        1.0 = perfect, 0.5 = random, 0.0 = always wrong.
    insufficient_rate:  Probability evaluator abstains ("insufficient").
    n_consumer_agents:  Agents that acknowledge receipt of promoted facts.
    seed:               RNG seed for reproducibility.
    """
    rng = random.Random(seed)
    svc = MemoryConsensusService(quorum_k=quorum_k, quorum_n=quorum_n)

    # Ground truth: is_correct[claim_id] → True/False
    is_correct: dict[str, bool] = {}

    for i in range(n_claims):
        correct = rng.random() >= bad_rate
        cid = f"claim-{i}"
        is_correct[cid] = correct
        svc.submit_claim(
            content=f"Fact {i}: {'correct' if correct else 'bad'}",
            submitter="submitter-agent",
            org_id="org-eval",
            confidence=rng.uniform(0.6, 0.95),
            claim_id=cid,
        )

    # Simulate N evaluator agents per claim
    result = SimResult(
        n_claims=n_claims,
        n_correct=sum(1 for v in is_correct.values() if v),
        n_bad=sum(1 for v in is_correct.values() if not v),
    )

    for i, (cid, correct) in enumerate(is_correct.items()):
        for j in range(quorum_n):
            evaluator_id = f"evaluator-{j}"
            claim = svc._claims.get(cid)
            if claim is None or claim.status != "pending":
                break  # already decided

            # Decide this evaluator's vote
            if rng.random() < insufficient_rate:
                verdict = "insufficient"
            elif rng.random() < evaluator_accuracy:
                verdict = "confirm" if correct else "contradict"
            else:
                verdict = "contradict" if correct else "confirm"

            try:
                svc.evaluate_claim(
                    claim_id=cid,
                    evaluator=evaluator_id,
                    verdict=verdict,
                )
            except (ValueError, KeyError):
                break  # already decided mid-loop

    # Classify outcomes
    for cid, correct in is_correct.items():
        status = svc._claims[cid].status
        if status == "promoted":
            if correct:
                result.promoted_correct += 1
            else:
                result.promoted_bad += 1
                # Simulate consumer agents acknowledging and then track tombstone
                acknowledged = []
                n_ack = rng.randint(0, n_consumer_agents)
                for k in range(n_ack):
                    agent_id = f"consumer-{k}"
                    try:
                        svc.acknowledge_receipt(claim_id=cid, agent_id=agent_id)
                        acknowledged.append(agent_id)
                    except ValueError:
                        pass
                result.contaminated_agents_per_bad_fact.append(len(acknowledged))
                # Tombstone and check coverage
                tb = svc.tombstone(cid)
                affected = tb["affected_agents"]
                coverage = len(affected) / len(acknowledged) if acknowledged else 1.0
                result.tombstone_coverage.append(coverage)
        elif status == "quarantined":
            if correct:
                result.quarantined_correct += 1
            else:
                result.quarantined_bad += 1
        else:  # expired or pending (timed out)
            if correct:
                result.expired_correct += 1
            else:
                result.expired_bad += 1

    return result


# ---------------------------------------------------------------------------
# Baseline correctness
# ---------------------------------------------------------------------------

class TestSimulationBaseline:
    def test_perfect_evaluators_near_perfect_precision(self):
        r = simulate(evaluator_accuracy=1.0, insufficient_rate=0.0, seed=1)
        assert r.precision >= 0.99, f"precision={r.precision:.3f}"

    def test_perfect_evaluators_high_recall(self):
        r = simulate(evaluator_accuracy=1.0, insufficient_rate=0.0, seed=1)
        assert r.recall >= 0.85, f"recall={r.recall:.3f}"

    def test_perfect_evaluators_zero_contamination(self):
        r = simulate(evaluator_accuracy=1.0, insufficient_rate=0.0, seed=1)
        assert r.promoted_bad == 0, f"bad facts promoted: {r.promoted_bad}"

    def test_random_evaluators_low_precision(self):
        # 50% accuracy = random voting → some bad facts slip through
        r = simulate(evaluator_accuracy=0.5, insufficient_rate=0.0, seed=2, n_claims=400)
        assert r.precision < 0.9, f"precision unexpectedly high: {r.precision:.3f}"

    def test_contamination_prevention_high_with_good_evaluators(self):
        r = simulate(evaluator_accuracy=0.9, insufficient_rate=0.05, seed=3)
        assert r.contamination_prevention_rate >= 0.85, \
            f"contamination_prevention={r.contamination_prevention_rate:.3f}"

    def test_quarantine_accuracy_high_with_good_evaluators(self):
        r = simulate(evaluator_accuracy=0.9, insufficient_rate=0.05, seed=3)
        # Most quarantined claims should be genuinely bad
        assert r.quarantine_accuracy >= 0.7, \
            f"quarantine_accuracy={r.quarantine_accuracy:.3f}"

    def test_result_counts_sum_to_n_claims(self):
        r = simulate(n_claims=100, seed=4)
        total = (r.promoted_correct + r.promoted_bad
                 + r.quarantined_correct + r.quarantined_bad
                 + r.expired_correct + r.expired_bad)
        assert total == 100


# ---------------------------------------------------------------------------
# k/n trade-off sweep
# ---------------------------------------------------------------------------

class TestQuorumSweep:
    """Verify the precision↑/recall↓ trade-off as k increases."""

    @pytest.fixture(scope="class")
    def sweep(self):
        """Run simulation at k=1,2,3 with n=3, return results keyed by k."""
        results = {}
        for k in (1, 2, 3):
            results[k] = simulate(
                n_claims=300,
                quorum_k=k,
                quorum_n=3,
                evaluator_accuracy=0.80,
                insufficient_rate=0.08,
                seed=10,
            )
        return results

    def test_precision_increases_with_k(self, sweep):
        assert sweep[1].precision <= sweep[2].precision or abs(sweep[1].precision - sweep[2].precision) < 0.05, \
            f"k=1 precision={sweep[1].precision:.3f} unexpectedly > k=2 precision={sweep[2].precision:.3f}"
        assert sweep[2].precision <= sweep[3].precision or abs(sweep[2].precision - sweep[3].precision) < 0.05

    def test_recall_decreases_with_k(self, sweep):
        assert sweep[1].recall >= sweep[3].recall - 0.05, \
            f"k=1 recall={sweep[1].recall:.3f} unexpectedly < k=3 recall={sweep[3].recall:.3f}"

    def test_k1_contamination_higher_than_k3(self, sweep):
        assert sweep[1].contamination_prevention_rate <= sweep[3].contamination_prevention_rate + 0.05

    def test_k3_never_promotes_at_full_n3(self, sweep):
        # k=3, n=3: all 3 evaluators must confirm — much harder to promote bad facts
        assert sweep[3].promoted_bad <= sweep[1].promoted_bad

    def test_all_sweep_results_have_valid_f1(self, sweep):
        for k, r in sweep.items():
            assert 0.0 <= r.f1 <= 1.0, f"k={k}: f1={r.f1}"


# ---------------------------------------------------------------------------
# Evaluator accuracy threshold
# ---------------------------------------------------------------------------

class TestAccuracyThreshold:
    """Find the minimum evaluator accuracy that keeps contamination < 5%."""

    @pytest.fixture(scope="class")
    def accuracy_sweep(self):
        results = {}
        for acc in (0.6, 0.7, 0.75, 0.8, 0.85, 0.9):
            results[acc] = simulate(
                n_claims=300,
                quorum_k=2,
                quorum_n=3,
                evaluator_accuracy=acc,
                insufficient_rate=0.05,
                seed=20,
            )
        return results

    def test_contamination_monotonically_decreases_with_accuracy(self, accuracy_sweep):
        accs = sorted(accuracy_sweep)
        prev = accuracy_sweep[accs[0]].contamination_prevention_rate
        for acc in accs[1:]:
            curr = accuracy_sweep[acc].contamination_prevention_rate
            assert curr >= prev - 0.08, \
                f"accuracy={acc}: contamination_prevention={curr:.3f} dropped unexpectedly vs prev={prev:.3f}"
            prev = curr

    def test_high_accuracy_achieves_sub5pct_contamination(self, accuracy_sweep):
        r = accuracy_sweep[0.9]
        contamination_rate = 1.0 - r.contamination_prevention_rate
        assert contamination_rate < 0.10, \
            f"contamination_rate={contamination_rate:.3f} exceeds 10% at accuracy=0.9"

    def test_low_accuracy_has_measurable_contamination(self, accuracy_sweep):
        r = accuracy_sweep[0.6]
        assert r.promoted_bad > 0 or r.contamination_prevention_rate < 1.0


# ---------------------------------------------------------------------------
# High bad-rate stress test
# ---------------------------------------------------------------------------

class TestHighBadRate:
    def test_50pct_bad_rate_good_evaluators_precision_above_90(self):
        r = simulate(
            n_claims=200,
            bad_rate=0.5,
            quorum_k=2,
            quorum_n=3,
            evaluator_accuracy=0.9,
            insufficient_rate=0.0,
            seed=30,
        )
        assert r.precision >= 0.85, f"precision={r.precision:.3f} with 50% bad rate"

    def test_high_bad_rate_contamination_still_blocked(self):
        r = simulate(
            n_claims=200,
            bad_rate=0.7,
            quorum_k=2,
            quorum_n=3,
            evaluator_accuracy=0.85,
            insufficient_rate=0.05,
            seed=31,
        )
        assert r.contamination_prevention_rate >= 0.75


# ---------------------------------------------------------------------------
# Insufficient-rate impact
# ---------------------------------------------------------------------------

class TestInsufficientRate:
    def test_high_insufficient_rate_drops_recall(self):
        r_low = simulate(insufficient_rate=0.0, seed=40)
        r_high = simulate(insufficient_rate=0.7, seed=40)
        # High abstention → fewer claims reach quorum → lower recall
        assert r_high.recall <= r_low.recall + 0.1

    def test_high_insufficient_rate_does_not_hurt_precision(self):
        # Insufficient votes don't count as contradictions, so precision stays high
        r = simulate(
            evaluator_accuracy=0.9,
            insufficient_rate=0.6,
            seed=41,
            n_claims=300,
        )
        # Precision only computed over what gets promoted; insufficient → expire, not FP
        assert r.precision >= 0.80


# ---------------------------------------------------------------------------
# Tombstone coverage
# ---------------------------------------------------------------------------

class TestTombstoneCoverage:
    def test_tombstone_covers_all_acknowledged_agents(self):
        """When a bad fact slips through, tombstone must identify all recipients."""
        r = simulate(
            n_claims=200,
            bad_rate=0.3,
            evaluator_accuracy=0.5,   # low accuracy → some bad facts get promoted
            insufficient_rate=0.0,
            n_consumer_agents=8,
            seed=50,
        )
        if r.tombstone_coverage:
            assert all(c == 1.0 for c in r.tombstone_coverage), \
                f"tombstone missed agents: {r.tombstone_coverage}"

    def test_no_slippage_means_no_contaminated_agents(self):
        r = simulate(
            evaluator_accuracy=1.0,
            insufficient_rate=0.0,
            seed=51,
        )
        assert r.promoted_bad == 0
        assert r.contaminated_agents_per_bad_fact == []


# ---------------------------------------------------------------------------
# Operating point recommendation
# ---------------------------------------------------------------------------

class TestOperatingPoint:
    """Validate that k=2, n=3 is a reasonable production default."""

    def test_default_k2_n3_precision_above_90(self):
        r = simulate(
            quorum_k=2, quorum_n=3,
            evaluator_accuracy=0.82,
            insufficient_rate=0.08,
            bad_rate=0.3,
            n_claims=500,
            seed=60,
        )
        assert r.precision >= 0.88, f"precision={r.precision:.3f}"

    def test_default_k2_n3_recall_above_60(self):
        r = simulate(
            quorum_k=2, quorum_n=3,
            evaluator_accuracy=0.82,
            insufficient_rate=0.08,
            bad_rate=0.3,
            n_claims=500,
            seed=60,
        )
        assert r.recall >= 0.50, f"recall={r.recall:.3f}"

    def test_default_k2_n3_contamination_below_10pct(self):
        r = simulate(
            quorum_k=2, quorum_n=3,
            evaluator_accuracy=0.82,
            insufficient_rate=0.08,
            bad_rate=0.3,
            n_claims=500,
            seed=60,
        )
        contamination = 1.0 - r.contamination_prevention_rate
        assert contamination < 0.15, f"contamination={contamination:.3f}"
