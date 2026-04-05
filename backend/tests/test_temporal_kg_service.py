from __future__ import annotations

from datetime import datetime, timezone, timedelta
from types import SimpleNamespace

import pytest

from app.services.temporal_kg_service import TemporalKnowledgeGraphService


def _rel(metadata: dict, rel_type: str = "RELATES_TO"):
    return SimpleNamespace(
        metadata_=metadata,
        relationship_type=rel_type,
        organization_id="org-1",
        from_memory_id="m-a",
        to_memory_id="m-b",
    )


class TestTemporalKnowledgeGraphService:
    def test_edge_valid_as_of_with_open_window(self):
        svc = TemporalKnowledgeGraphService(db=None)  # type: ignore[arg-type]
        now = datetime.now(timezone.utc)
        rel = _rel({"valid_from": (now - timedelta(days=1)).isoformat(), "valid_until": None})
        assert svc.edge_valid_as_of(rel, now) is True

    def test_edge_invalid_before_valid_from(self):
        svc = TemporalKnowledgeGraphService(db=None)  # type: ignore[arg-type]
        now = datetime.now(timezone.utc)
        rel = _rel({"valid_from": (now + timedelta(days=1)).isoformat(), "valid_until": None})
        assert svc.edge_valid_as_of(rel, now) is False

    def test_filter_edges_as_of(self):
        svc = TemporalKnowledgeGraphService(db=None)  # type: ignore[arg-type]
        now = datetime.now(timezone.utc)
        rels = [
            _rel({"valid_from": (now - timedelta(days=2)).isoformat(), "valid_until": None}),
            _rel({"valid_from": (now + timedelta(days=2)).isoformat(), "valid_until": None}),
        ]
        filtered = svc.filter_edges_as_of(rels, now)
        assert len(filtered) == 1

    def test_detect_temporal_conflict_for_contradiction_overlap(self):
        svc = TemporalKnowledgeGraphService(db=None)  # type: ignore[arg-type]
        now = datetime.now(timezone.utc)
        conflict = svc.detect_temporal_conflict(
            candidate_type="CONTRADICTS",
            candidate_valid_from=now,
            candidate_valid_until=None,
            existing_type="RELATES_TO",
            existing_valid_from=now - timedelta(days=1),
            existing_valid_until=None,
        )
        assert conflict is True

    @pytest.mark.asyncio
    async def test_invalidate_contradicted_edges_sets_valid_until(self):
        contradiction_at = datetime.now(timezone.utc)
        rel_active = _rel({"valid_from": (contradiction_at - timedelta(days=3)).isoformat(), "valid_until": None})
        rel_already_closed = _rel({"valid_from": (contradiction_at - timedelta(days=5)).isoformat(), "valid_until": (contradiction_at - timedelta(days=1)).isoformat()})

        fake_scalars = SimpleNamespace(all=lambda: [rel_active, rel_already_closed])
        fake_result = SimpleNamespace(scalars=lambda: fake_scalars)

        class _FakeDB:
            def __init__(self):
                self.committed = False

            async def execute(self, stmt):
                return fake_result

            async def commit(self):
                self.committed = True

        db = _FakeDB()
        svc = TemporalKnowledgeGraphService(db=db)  # type: ignore[arg-type]

        count = await svc.invalidate_contradicted_edges(
            organization_id="org-1",
            from_memory_id="m-a",
            to_memory_id="m-b",
            contradiction_at=contradiction_at,
        )

        assert count == 1
        assert rel_active.metadata_["valid_until"] == contradiction_at.isoformat()
        assert db.committed is True
