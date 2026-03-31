"""Tests for Phase 52 Slice 1 - EnvironmentSyncService."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.services.environment_sync_service import EnvironmentSyncService


class _Clock:
    def __init__(self, start: datetime):
        self.current = start

    def now(self) -> datetime:
        return self.current

    def advance(self, seconds: int) -> None:
        self.current = self.current + timedelta(seconds=seconds)


class TestNormalizeInbound:
    def test_prefers_header_event_id(self):
        svc = EnvironmentSyncService()
        event = svc.normalize_inbound(
            org_id="org-1",
            connector_type="jira",
            payload={"id": "payload-id", "issue": {"key": "PROJ-12", "fields": {"summary": "x"}}},
            headers={"X-Ninai-Event-Id": "hdr-id"},
        )

        assert event.external_event_id == "hdr-id"
        assert event.external_object_id == "PROJ-12"

    def test_uses_payload_id_when_header_missing(self):
        svc = EnvironmentSyncService()
        event = svc.normalize_inbound(
            org_id="org-1",
            connector_type="webhook",
            payload={"id": "evt-123", "title": "hello"},
        )

        assert event.external_event_id == "evt-123"
        assert event.external_object_id == "evt-123"

    def test_parses_iso_timestamp(self):
        svc = EnvironmentSyncService()
        ts = "2026-03-31T10:15:20Z"
        event = svc.normalize_inbound(
            org_id="org-1",
            connector_type="webhook",
            payload={"event_id": "evt", "timestamp": ts, "title": "hello"},
        )

        assert event.external_updated_at == datetime(2026, 3, 31, 10, 15, 20, tzinfo=timezone.utc)


class TestApplyInbound:
    def test_apply_creates_state(self):
        svc = EnvironmentSyncService()
        event = svc.normalize_inbound(
            org_id="org-1",
            connector_type="slack",
            payload={"event_id": "e-1", "event": {"type": "message", "text": "hello", "ts": "10"}},
        )

        result = svc.apply_inbound(event)

        assert result.status == "applied"
        assert result.state is not None
        assert result.state.external_object_id == "10"

    def test_duplicate_replay_key_is_rejected(self):
        svc = EnvironmentSyncService()
        payload = {"event_id": "e-1", "title": "hello"}
        event1 = svc.normalize_inbound(org_id="org-1", connector_type="webhook", payload=payload)
        event2 = svc.normalize_inbound(org_id="org-1", connector_type="webhook", payload=payload)

        first = svc.apply_inbound(event1)
        second = svc.apply_inbound(event2)

        assert first.status == "applied"
        assert second.status == "duplicate"

    def test_out_of_order_event_is_ignored(self):
        svc = EnvironmentSyncService()
        newer = svc.normalize_inbound(
            org_id="org-1",
            connector_type="webhook",
            payload={"event_id": "new", "id": "obj-1", "timestamp": "2026-03-31T11:00:00Z", "title": "new"},
        )
        older = svc.normalize_inbound(
            org_id="org-1",
            connector_type="webhook",
            payload={"event_id": "old", "id": "obj-1", "timestamp": "2026-03-31T10:00:00Z", "title": "old"},
        )

        first = svc.apply_inbound(newer)
        second = svc.apply_inbound(older)

        assert first.status == "applied"
        assert second.status == "out_of_order"
        assert second.state is not None
        assert second.state.title == "new"

    def test_newer_event_overwrites_state(self):
        svc = EnvironmentSyncService()
        old_event = svc.normalize_inbound(
            org_id="org-1",
            connector_type="webhook",
            payload={"event_id": "old", "id": "obj-1", "timestamp": "2026-03-31T10:00:00Z", "title": "old"},
        )
        new_event = svc.normalize_inbound(
            org_id="org-1",
            connector_type="webhook",
            payload={"event_id": "new", "id": "obj-1", "timestamp": "2026-03-31T11:00:00Z", "title": "new"},
        )

        svc.apply_inbound(old_event)
        result = svc.apply_inbound(new_event)

        assert result.status == "applied"
        assert result.state is not None
        assert result.state.title == "new"


class TestDivergenceAndReconcile:
    def test_mark_internal_projection_sets_diverged_false_when_hash_matches(self):
        svc = EnvironmentSyncService()
        event = svc.normalize_inbound(
            org_id="org-1",
            connector_type="webhook",
            payload={"event_id": "e-1", "id": "obj-1", "title": "hello"},
        )
        result = svc.apply_inbound(event)
        assert result.state is not None

        state = svc.mark_internal_projection(
            org_id="org-1",
            connector_type="webhook",
            external_object_id="obj-1",
            internal_hash=result.state.state_hash,
        )

        assert state is not None
        assert state.diverged is False

    def test_mark_internal_projection_sets_diverged_true_when_hash_differs(self):
        svc = EnvironmentSyncService()
        event = svc.normalize_inbound(
            org_id="org-1",
            connector_type="webhook",
            payload={"event_id": "e-1", "id": "obj-1", "title": "hello"},
        )
        svc.apply_inbound(event)

        state = svc.mark_internal_projection(
            org_id="org-1",
            connector_type="webhook",
            external_object_id="obj-1",
            internal_hash="different-hash",
        )

        assert state is not None
        assert state.diverged is True

    def test_reconcile_candidates_include_diverged(self):
        svc = EnvironmentSyncService()
        event = svc.normalize_inbound(
            org_id="org-1",
            connector_type="webhook",
            payload={"event_id": "e-1", "id": "obj-1", "title": "hello"},
        )
        svc.apply_inbound(event)
        svc.mark_internal_projection(
            org_id="org-1",
            connector_type="webhook",
            external_object_id="obj-1",
            internal_hash="different-hash",
        )

        candidates = svc.reconcile_candidates(org_id="org-1", max_lag_seconds=3600)
        assert len(candidates) == 1

    def test_reconcile_candidates_include_lagging(self):
        clock = _Clock(datetime(2026, 3, 31, 10, 0, 0, tzinfo=timezone.utc))
        svc = EnvironmentSyncService(now_fn=clock.now)
        event = svc.normalize_inbound(
            org_id="org-1",
            connector_type="webhook",
            payload={"event_id": "e-1", "id": "obj-1", "title": "hello"},
        )
        svc.apply_inbound(event)

        clock.advance(45)
        candidates = svc.reconcile_candidates(org_id="org-1", max_lag_seconds=30)
        assert len(candidates) == 1


class TestSummaryAndReplayTtl:
    def test_summary_counts_applied_duplicate_and_out_of_order(self):
        svc = EnvironmentSyncService()

        first = svc.normalize_inbound(
            org_id="org-1",
            connector_type="webhook",
            payload={"event_id": "a", "id": "obj-1", "timestamp": "2026-03-31T11:00:00Z", "title": "a"},
        )
        duplicate = svc.normalize_inbound(
            org_id="org-1",
            connector_type="webhook",
            payload={"event_id": "a", "id": "obj-1", "timestamp": "2026-03-31T11:00:00Z", "title": "a"},
        )
        old = svc.normalize_inbound(
            org_id="org-1",
            connector_type="webhook",
            payload={"event_id": "b", "id": "obj-1", "timestamp": "2026-03-31T10:00:00Z", "title": "b"},
        )

        svc.apply_inbound(first)
        svc.apply_inbound(duplicate)
        svc.apply_inbound(old)

        summary = svc.summary("org-1")
        assert summary.applied_events == 1
        assert summary.duplicate_events == 1
        assert summary.out_of_order_events == 1

    def test_replay_ttl_allows_old_replay_key_after_expiry(self):
        clock = _Clock(datetime(2026, 3, 31, 10, 0, 0, tzinfo=timezone.utc))
        svc = EnvironmentSyncService(replay_ttl_seconds=5, now_fn=clock.now)

        payload = {"event_id": "e-1", "id": "obj-1", "title": "hello"}
        first = svc.normalize_inbound(org_id="org-1", connector_type="webhook", payload=payload)
        second = svc.normalize_inbound(org_id="org-1", connector_type="webhook", payload=payload)

        assert svc.apply_inbound(first).status == "applied"
        assert svc.apply_inbound(second).status == "duplicate"

        clock.advance(6)
        third = svc.normalize_inbound(org_id="org-1", connector_type="webhook", payload=payload)
        assert svc.apply_inbound(third).status == "applied"
