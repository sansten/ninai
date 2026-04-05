from __future__ import annotations

from datetime import datetime, timezone, timedelta
from types import SimpleNamespace

import pytest

from app.services.cognitive_forget_service import CognitiveForgetService


class _ScalarResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return list(self._rows)


class _FakeDB:
    def __init__(self, responses):
        self._responses = list(responses)
        self.flush_count = 0
        self.commit_count = 0

    async def execute(self, _stmt):
        rows = self._responses.pop(0) if self._responses else []
        return _ScalarResult(rows)

    async def flush(self):
        self.flush_count += 1

    async def commit(self):
        self.commit_count += 1


class _FakeWebhookService:
    def __init__(self, _db):
        self.events = []

    async def emit_event(self, *, organization_id: str, event_type: str, payload: dict):
        self.events.append((organization_id, event_type, payload))


class TestCognitiveForgetService:
    def test_normalize_domains(self):
        svc = CognitiveForgetService(_FakeDB([]))
        out = svc._normalize_domains(["HR", " hr ", "Finance", "", "finance"])
        assert out == ["hr", "finance"]

    def test_memory_matches_domains(self):
        svc = CognitiveForgetService(_FakeDB([]))
        memory = SimpleNamespace(
            business_domain="hr",
            tags=["policy"],
            extra_metadata={"domain": "people"},
        )
        assert svc._memory_matches_domains(memory, ["hr"])
        assert svc._memory_matches_domains(memory, ["policy"])
        assert not svc._memory_matches_domains(memory, ["security"])

    @pytest.mark.asyncio
    async def test_mark_memories_for_cascade_deletion(self):
        db = _FakeDB([])
        svc = CognitiveForgetService(db)
        mem = SimpleNamespace(extra_metadata={}, is_active=True)
        count = await svc._mark_memories_for_cascade_deletion(
            memories=[mem],
            subject="user@company.com",
            reason="gdpr_erasure",
        )
        assert count == 1
        assert mem.is_active is False
        assert mem.extra_metadata["forget_marker"]["cascade_delete"] is True

    @pytest.mark.asyncio
    async def test_invalidate_linked_causal_edges(self):
        edge_linked = SimpleNamespace(
            cause_entity_id="m1",
            effect_entity_id="x",
            evidence_memory_ids=[],
            invalidation_count=0,
            strength=0.9,
            last_validated_at=None,
        )
        edge_other = SimpleNamespace(
            cause_entity_id="z",
            effect_entity_id="y",
            evidence_memory_ids=["other"],
            invalidation_count=0,
            strength=0.6,
            last_validated_at=None,
        )
        db = _FakeDB([[edge_linked, edge_other]])
        svc = CognitiveForgetService(db)
        count = await svc._invalidate_linked_causal_edges(
            organization_id="org1",
            memory_ids=["m1"],
        )
        assert count == 1
        assert edge_linked.invalidation_count == 1
        assert edge_linked.strength == 0.4
        assert edge_other.invalidation_count == 0

    @pytest.mark.asyncio
    async def test_recompute_affected_scores(self):
        created = datetime.now(timezone.utc) - timedelta(days=10)
        memory = SimpleNamespace(
            content_preview="Test memory",
            source_type="document",
            business_domain="hr",
            created_at=created,
            extra_metadata={},
        )
        db = _FakeDB([])
        svc = CognitiveForgetService(db)
        count = await svc._recompute_affected_scores(memories=[memory])
        assert count == 1
        recomputed = memory.extra_metadata.get("recomputed_after_forget")
        assert recomputed is not None
        assert 0.0 <= recomputed["credibility_score"] <= 1.0
        assert 0.0 <= recomputed["freshness_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_forget_full_flow(self, monkeypatch):
        db = _FakeDB([])
        svc = CognitiveForgetService(db)

        mem = SimpleNamespace(
            id="m1",
            extra_metadata={},
            is_active=True,
            content_preview="user@company.com hr policy",
            source_type="document",
            business_domain="hr",
            created_at=datetime.now(timezone.utc),
        )

        async def _resolve_subject_user_ids(**_kwargs):
            return ["u1"]

        async def _find_associated_memories(**_kwargs):
            return [mem]

        async def _emit(**_kwargs):
            return True

        monkeypatch.setattr(svc, "_resolve_subject_user_ids", _resolve_subject_user_ids)
        monkeypatch.setattr(svc, "_find_associated_memories", _find_associated_memories)
        monkeypatch.setattr(svc, "_emit_knowledge_erased_event", _emit)

        cert = await svc.forget(
            organization_id="org1",
            subject="user@company.com",
            domains=["hr"],
            reason="gdpr_erasure",
            requested_by_user_id="admin1",
        )

        assert cert.organization_id == "org1"
        assert cert.subject == "user@company.com"
        assert cert.erased_memory_count == 1
        assert cert.recomputed_memory_count == 1
        assert cert.knowledge_erased_event_emitted is True
        assert db.commit_count == 1
