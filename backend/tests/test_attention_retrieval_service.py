from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.services.attention_retrieval_service import AttentionRetrievalService


class TestAttentionRetrievalService:
    def _svc(self) -> AttentionRetrievalService:
        return AttentionRetrievalService()

    def _memory(self, *, content: str, tags: list[str], created_at: datetime) -> dict:
        return {
            "id": content,
            "content": content,
            "tags": tags,
            "created_at": created_at,
        }

    def test_goal_match_scores_higher(self):
        svc = self._svc()
        now = datetime.now(timezone.utc)
        goal = {"goal_id": "g1", "title": "DB goal", "tags": ["database", "latency"]}
        m1 = self._memory(content="db slow", tags=["database", "latency"], created_at=now)
        m2 = self._memory(content="hr policy", tags=["hr"], created_at=now)
        s1 = svc.score(memory=m1, active_goals=[goal], active_incidents=[], query_tokens=frozenset(), now=now)
        s2 = svc.score(memory=m2, active_goals=[goal], active_incidents=[], query_tokens=frozenset(), now=now)
        assert s1 > s2

    def test_incident_match_scores_higher(self):
        svc = self._svc()
        now = datetime.now(timezone.utc)
        inc = {"memory_id": "i1", "content": "api timeout incident", "tags": ["timeout"]}
        m1 = self._memory(content="api timeout incident", tags=["ops"], created_at=now)
        m2 = self._memory(content="unrelated meeting notes", tags=["meeting"], created_at=now)
        s1 = svc.score(memory=m1, active_goals=[], active_incidents=[inc], query_tokens=frozenset(), now=now)
        s2 = svc.score(memory=m2, active_goals=[], active_incidents=[inc], query_tokens=frozenset(), now=now)
        assert s1 > s2

    def test_very_old_memory_recency_near_zero(self):
        svc = self._svc()
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=365)
        score = svc.score(
            memory=self._memory(content="x", tags=[], created_at=old),
            active_goals=[],
            active_incidents=[],
            query_tokens=frozenset(),
            now=now,
        )
        # Only recency contributes here.
        assert score < 0.01

    def test_fresh_memory_recency_full_component(self):
        svc = self._svc()
        now = datetime.now(timezone.utc)
        score = svc.score(
            memory=self._memory(content="x", tags=[], created_at=now),
            active_goals=[],
            active_incidents=[],
            query_tokens=frozenset(),
            now=now,
        )
        assert score == 0.2

    def test_rank_returns_at_most_limit(self):
        svc = self._svc()
        now = datetime.now(timezone.utc)
        memories = [self._memory(content=f"m{i}", tags=["x"], created_at=now) for i in range(20)]
        ranked = svc.rank(
            memories=memories,
            active_goals=[],
            active_incidents=[],
            query_tokens=frozenset(),
            now=now,
            limit=5,
        )
        assert len(ranked) == 5

    def test_rank_attaches_attention_score(self):
        svc = self._svc()
        now = datetime.now(timezone.utc)
        ranked = svc.rank(
            memories=[self._memory(content="m", tags=["x"], created_at=now)],
            active_goals=[],
            active_incidents=[],
            query_tokens=frozenset(),
            now=now,
            limit=1,
        )
        assert "_attention_score" in ranked[0]

    def test_empty_active_goals_goal_score_zero(self):
        svc = self._svc()
        now = datetime.now(timezone.utc)
        m = self._memory(content="a", tags=["database"], created_at=now)
        s = svc.score(memory=m, active_goals=[], active_incidents=[], query_tokens=frozenset(), now=now)
        assert s == 0.2

    def test_empty_active_incidents_incident_score_zero(self):
        svc = self._svc()
        now = datetime.now(timezone.utc)
        m = self._memory(content="a", tags=["database"], created_at=now)
        s = svc.score(memory=m, active_goals=[], active_incidents=[], query_tokens=frozenset({"a"}), now=now)
        assert s == 0.3

    def test_score_in_range(self):
        svc = self._svc()
        now = datetime.now(timezone.utc)
        m = self._memory(content="a b", tags=["a"], created_at=now)
        s = svc.score(
            memory=m,
            active_goals=[{"goal_id": "g", "title": "t", "tags": ["a"]}],
            active_incidents=[{"memory_id": "i", "content": "a b", "tags": ["a"]}],
            query_tokens=frozenset({"a", "b"}),
            now=now,
        )
        assert 0.0 <= s <= 1.0

    def test_rank_deterministic(self):
        svc = self._svc()
        now = datetime.now(timezone.utc)
        memories = [
            self._memory(content="same", tags=["x"], created_at=now),
            self._memory(content="same", tags=["x"], created_at=now),
        ]
        r1 = svc.rank(memories=memories, active_goals=[], active_incidents=[], query_tokens=frozenset(), now=now, limit=2)
        r2 = svc.rank(memories=memories, active_goals=[], active_incidents=[], query_tokens=frozenset(), now=now, limit=2)
        assert [m["id"] for m in r1] == [m["id"] for m in r2]

    def test_query_relevance_changes_score(self):
        svc = self._svc()
        now = datetime.now(timezone.utc)
        m = self._memory(content="database timeout", tags=["db"], created_at=now)
        s1 = svc.score(memory=m, active_goals=[], active_incidents=[], query_tokens=frozenset({"database"}), now=now)
        s2 = svc.score(memory=m, active_goals=[], active_incidents=[], query_tokens=frozenset({"finance"}), now=now)
        assert s1 > s2

    def test_limit_zero_returns_empty(self):
        svc = self._svc()
        now = datetime.now(timezone.utc)
        ranked = svc.rank(
            memories=[self._memory(content="a", tags=[], created_at=now)],
            active_goals=[],
            active_incidents=[],
            query_tokens=frozenset(),
            now=now,
            limit=0,
        )
        assert ranked == []

    def test_incident_uses_tags_when_content_missing(self):
        svc = self._svc()
        now = datetime.now(timezone.utc)
        incident = {"memory_id": "i", "content": "", "tags": ["latency"]}
        m1 = self._memory(content="latency observed", tags=[], created_at=now)
        m2 = self._memory(content="unrelated", tags=[], created_at=now)
        s1 = svc.score(memory=m1, active_goals=[], active_incidents=[incident], query_tokens=frozenset(), now=now)
        s2 = svc.score(memory=m2, active_goals=[], active_incidents=[incident], query_tokens=frozenset(), now=now)
        assert s1 > s2

    def test_created_at_iso_string_supported(self):
        svc = self._svc()
        now = datetime.now(timezone.utc)
        m = {
            "id": "m",
            "content": "text",
            "tags": [],
            "created_at": now.isoformat(),
        }
        s = svc.score(memory=m, active_goals=[], active_incidents=[], query_tokens=frozenset(), now=now)
        assert s == 0.2

    def test_missing_created_at_defaults_now(self):
        svc = self._svc()
        now = datetime.now(timezone.utc)
        m = {"id": "m", "content": "text", "tags": []}
        s = svc.score(memory=m, active_goals=[], active_incidents=[], query_tokens=frozenset(), now=now)
        assert s == 0.2

    def test_goal_tags_string_entries_are_normalized(self):
        svc = self._svc()
        now = datetime.now(timezone.utc)
        goal = {"goal_id": "g", "title": "x", "tags": [" DATABASE "]}
        m = self._memory(content="x", tags=["database"], created_at=now)
        s = svc.score(memory=m, active_goals=[goal], active_incidents=[], query_tokens=frozenset(), now=now)
        assert s >= 0.6

    def test_rank_sorted_descending_by_attention_score(self):
        svc = self._svc()
        now = datetime.now(timezone.utc)
        goal = {"goal_id": "g", "title": "DB", "tags": ["database"]}
        m1 = self._memory(content="database note", tags=["database"], created_at=now)
        m2 = self._memory(content="finance note", tags=["finance"], created_at=now)
        ranked = svc.rank(
            memories=[m2, m1],
            active_goals=[goal],
            active_incidents=[],
            query_tokens=frozenset(),
            now=now,
            limit=2,
        )
        assert ranked[0]["_attention_score"] >= ranked[1]["_attention_score"]

    def test_no_memories_returns_empty(self):
        svc = self._svc()
        now = datetime.now(timezone.utc)
        assert svc.rank(memories=[], active_goals=[], active_incidents=[], query_tokens=frozenset(), now=now) == []

    def test_weights_sum_to_one(self):
        svc = self._svc()
        assert (
            svc.GOAL_WEIGHT
            + svc.INCIDENT_WEIGHT
            + svc.RECENCY_WEIGHT
            + svc.BASE_RELEVANCE_WEIGHT
        ) == pytest.approx(1.0)

    def test_jaccard_empty_sets(self):
        svc = self._svc()
        assert svc._jaccard(set(), set()) == 1.0

    def test_days_old_not_negative(self):
        svc = self._svc()
        now = datetime.now(timezone.utc)
        future = now + timedelta(days=1)
        assert svc._days_old(future, now) == 0

    def test_score_handles_content_preview_fallback(self):
        svc = self._svc()
        now = datetime.now(timezone.utc)
        m = {
            "id": "m",
            "content_preview": "database timeout",
            "tags": ["ops"],
            "created_at": now,
        }
        s = svc.score(memory=m, active_goals=[], active_incidents=[], query_tokens=frozenset({"database"}), now=now)
        assert s > 0.2

    def test_rank_does_not_mutate_input(self):
        svc = self._svc()
        now = datetime.now(timezone.utc)
        memories = [self._memory(content="m", tags=[], created_at=now)]
        _ = svc.rank(memories=memories, active_goals=[], active_incidents=[], query_tokens=frozenset(), now=now, limit=1)
        assert "_attention_score" not in memories[0]

    def test_equal_scores_keep_input_order(self):
        svc = self._svc()
        now = datetime.now(timezone.utc)
        a = self._memory(content="same-a", tags=[], created_at=now)
        b = self._memory(content="same-b", tags=[], created_at=now)
        ranked = svc.rank(memories=[a, b], active_goals=[], active_incidents=[], query_tokens=frozenset(), now=now, limit=2)
        assert ranked[0]["id"] == "same-a"
        assert ranked[1]["id"] == "same-b"

    def test_query_tokens_empty_relevance_zero_component(self):
        svc = self._svc()
        now = datetime.now(timezone.utc)
        m = self._memory(content="database", tags=[], created_at=now)
        s = svc.score(memory=m, active_goals=[], active_incidents=[], query_tokens=frozenset(), now=now)
        assert s == 0.2

    def test_existing_retrieval_score_boosts_attention(self):
        svc = self._svc()
        now = datetime.now(timezone.utc)
        high = {
            "id": "high",
            "content": "generic note",
            "tags": [],
            "created_at": now,
            "score": 0.95,
        }
        low = {
            "id": "low",
            "content": "generic note",
            "tags": [],
            "created_at": now,
            "score": 0.10,
        }
        s_high = svc.score(memory=high, active_goals=[], active_incidents=[], query_tokens=frozenset(), now=now)
        s_low = svc.score(memory=low, active_goals=[], active_incidents=[], query_tokens=frozenset(), now=now)
        assert s_high > s_low
