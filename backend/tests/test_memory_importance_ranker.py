from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.services.memory_importance_ranker import MemoryImportanceRanker


_NOW = datetime(2026, 4, 2, 12, 0, 0, tzinfo=timezone.utc)


def _memory(
    *,
    created_at: datetime | str,
    credibility_score: float = 0.7,
    activation: float = 0.5,
    reference_count: int = 0,
    goal_link_count: int = 0,
    memory_id: str = "m1",
) -> dict:
    return {
        "id": memory_id,
        "created_at": created_at.isoformat() if isinstance(created_at, datetime) else created_at,
        "credibility_score": credibility_score,
        "activation": activation,
        "reference_count": reference_count,
        "goal_link_count": goal_link_count,
    }


class TestScore:
    def test_reference_count_ten_saturates(self):
        svc = MemoryImportanceRanker()
        m = _memory(created_at=_NOW, reference_count=10)
        score_10 = svc.score(memory=m, now=_NOW)
        m2 = _memory(created_at=_NOW, reference_count=100)
        score_100 = svc.score(memory=m2, now=_NOW)
        assert score_10 == score_100

    def test_reference_count_five_half_ref_component(self):
        svc = MemoryImportanceRanker()
        base = _memory(created_at=_NOW, reference_count=0)
        mid = _memory(created_at=_NOW, reference_count=5)
        assert svc.score(memory=mid, now=_NOW) > svc.score(memory=base, now=_NOW)

    def test_days_old_zero_recency_one(self):
        svc = MemoryImportanceRanker()
        m = _memory(created_at=_NOW)
        score_now = svc.score(memory=m, now=_NOW)
        old = _memory(created_at=_NOW - timedelta(days=30))
        score_old = svc.score(memory=old, now=_NOW)
        assert score_now > score_old

    def test_days_old_thirty_recency_approx_point_three_seven(self):
        svc = MemoryImportanceRanker()
        m = _memory(created_at=_NOW - timedelta(days=30), credibility_score=0, activation=0, reference_count=0, goal_link_count=0)
        score = svc.score(memory=m, now=_NOW)
        expected = round(0.20 * 0.36787944117, 4)
        assert abs(score - expected) <= 0.0001

    def test_goal_link_count_five_saturates(self):
        svc = MemoryImportanceRanker()
        s1 = svc.score(memory=_memory(created_at=_NOW, goal_link_count=5), now=_NOW)
        s2 = svc.score(memory=_memory(created_at=_NOW, goal_link_count=10), now=_NOW)
        assert s1 == s2

    def test_all_components_max_near_one(self):
        svc = MemoryImportanceRanker()
        m = _memory(created_at=_NOW, credibility_score=1.0, activation=1.0, reference_count=10, goal_link_count=5)
        assert svc.score(memory=m, now=_NOW) >= 0.99

    def test_all_components_zero_near_zero(self):
        svc = MemoryImportanceRanker()
        m = _memory(
            created_at=_NOW - timedelta(days=3650),
            credibility_score=0.0,
            activation=0.0,
            reference_count=0,
            goal_link_count=0,
        )
        assert svc.score(memory=m, now=_NOW) <= 0.01

    def test_score_in_bounds(self):
        svc = MemoryImportanceRanker()
        m = _memory(created_at=_NOW, credibility_score=2.0, activation=-2.0, reference_count=50, goal_link_count=50)
        s = svc.score(memory=m, now=_NOW)
        assert 0.0 <= s <= 1.0

    def test_defaults_for_missing_fields(self):
        svc = MemoryImportanceRanker()
        m = {"created_at": _NOW.isoformat()}
        s = svc.score(memory=m, now=_NOW)
        assert 0.0 <= s <= 1.0

    def test_invalid_created_at_fallback_now(self):
        svc = MemoryImportanceRanker()
        m = _memory(created_at="bad-date")
        s = svc.score(memory=m, now=_NOW)
        assert s > 0.0

    def test_created_at_z_suffix_supported(self):
        svc = MemoryImportanceRanker()
        m = _memory(created_at="2026-04-02T12:00:00Z")
        s = svc.score(memory=m, now=_NOW)
        assert s > 0.0

    def test_negative_counts_treated_zero(self):
        svc = MemoryImportanceRanker()
        m = _memory(created_at=_NOW, reference_count=-3, goal_link_count=-2)
        s = svc.score(memory=m, now=_NOW)
        assert s >= 0.0


class TestRank:
    def test_rank_respects_limit(self):
        svc = MemoryImportanceRanker()
        memories = [_memory(created_at=_NOW, memory_id=f"m{i}") for i in range(20)]
        ranked = svc.rank(memories=memories, now=_NOW, limit=10)
        assert len(ranked) == 10

    def test_rank_attaches_importance_score(self):
        svc = MemoryImportanceRanker()
        ranked = svc.rank(memories=[_memory(created_at=_NOW)], now=_NOW, limit=10)
        assert "_importance_score" in ranked[0]

    def test_rank_sorted_desc(self):
        svc = MemoryImportanceRanker()
        high = _memory(created_at=_NOW, reference_count=10, goal_link_count=5, credibility_score=1.0, activation=1.0, memory_id="high")
        low = _memory(created_at=_NOW - timedelta(days=3650), credibility_score=0.0, activation=0.0, reference_count=0, goal_link_count=0, memory_id="low")
        ranked = svc.rank(memories=[low, high], now=_NOW, limit=10)
        assert ranked[0]["id"] == "high"

    def test_rank_limit_zero_returns_empty(self):
        svc = MemoryImportanceRanker()
        ranked = svc.rank(memories=[_memory(created_at=_NOW)], now=_NOW, limit=0)
        assert ranked == []

    def test_rank_empty_input(self):
        svc = MemoryImportanceRanker()
        assert svc.rank(memories=[], now=_NOW, limit=10) == []

    def test_rank_negative_limit_returns_empty(self):
        svc = MemoryImportanceRanker()
        ranked = svc.rank(memories=[_memory(created_at=_NOW)], now=_NOW, limit=-1)
        assert ranked == []


class TestTier:
    def test_tier_critical(self):
        svc = MemoryImportanceRanker()
        assert svc.importance_tier(0.85) == "critical"

    def test_tier_important(self):
        svc = MemoryImportanceRanker()
        assert svc.importance_tier(0.65) == "important"

    def test_tier_normal(self):
        svc = MemoryImportanceRanker()
        assert svc.importance_tier(0.45) == "normal"

    def test_tier_archivable(self):
        svc = MemoryImportanceRanker()
        assert svc.importance_tier(0.35) == "archivable"

    def test_tier_boundaries(self):
        svc = MemoryImportanceRanker()
        assert svc.importance_tier(0.8) == "critical"
        assert svc.importance_tier(0.6) == "important"
        assert svc.importance_tier(0.4) == "normal"

    def test_tier_clamps_out_of_range(self):
        svc = MemoryImportanceRanker()
        assert svc.importance_tier(2.0) == "critical"
        assert svc.importance_tier(-1.0) == "archivable"


class TestWeightSanity:
    def test_weights_sum_one(self):
        total = (
            MemoryImportanceRanker.REFERENCE_WEIGHT
            + MemoryImportanceRanker.GOAL_LINK_WEIGHT
            + MemoryImportanceRanker.RECENCY_WEIGHT
            + MemoryImportanceRanker.CREDIBILITY_WEIGHT
            + MemoryImportanceRanker.ACTIVATION_WEIGHT
        )
        assert round(total, 6) == 1.0
