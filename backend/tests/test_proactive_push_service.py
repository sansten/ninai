from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.proactive_push_service import ProactivePushService


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return list(self._rows)


class _FakeDB:
    def __init__(self, batches):
        self._batches = list(batches)

    async def execute(self, _stmt):
        rows = self._batches.pop(0) if self._batches else []
        return _FakeResult(rows)


class _FakeWebhookService:
    emitted: list[dict] = []

    def __init__(self, _db):
        pass

    async def emit_event(self, *, organization_id: str, event_type: str, payload: dict):
        self.emitted.append(
            {
                "organization_id": organization_id,
                "event_type": event_type,
                "payload": payload,
            }
        )


class TestProactivePushService:
    def test_threshold_from_org_settings(self):
        assert ProactivePushService.get_push_threshold({"push_threshold": 0.8}) == 0.8
        assert ProactivePushService.get_push_threshold({"push_threshold": 2}) == 1.0
        assert ProactivePushService.get_push_threshold({"push_threshold": -1}) == 0.0

    def test_compute_relevance_higher_for_matching_goal(self):
        event = {
            "event_type": "incident.critical",
            "payload": {
                "summary": "payment outage in checkout service",
                "description": "critical outage with high latency",
            },
        }
        high = ProactivePushService.compute_relevance(
            event=event,
            goal="reduce checkout outage and latency incidents",
            context_snapshot={"focus": "payment service reliability"},
        )
        low = ProactivePushService.compute_relevance(
            event=event,
            goal="improve hiring process for backend team",
            context_snapshot={"focus": "candidate interviews"},
        )
        assert high > low

    @pytest.mark.asyncio
    async def test_build_candidates_filters_by_threshold(self):
        svc = ProactivePushService(_FakeDB([]))
        sessions = [
            SimpleNamespace(id="s1", goal="reduce incident rate in payments", context_snapshot={}),
            SimpleNamespace(id="s2", goal="improve sales forecasting", context_snapshot={}),
        ]
        events = [
            SimpleNamespace(
                id="e1",
                event_type="incident.critical",
                payload={"summary": "payments incident and outage"},
            ),
            SimpleNamespace(
                id="e2",
                event_type="knowledge.updated",
                payload={"summary": "quarterly hiring handbook refreshed"},
            ),
        ]
        candidates = await svc.build_candidates(sessions=sessions, events=events, threshold=0.45)
        ids = [c.event_id for c in candidates]
        assert "e1" in ids

    @pytest.mark.asyncio
    async def test_run_cycle_no_sessions_or_webhooks(self):
        db = _FakeDB([
            [],  # sessions
            [],  # webhooks
        ])
        svc = ProactivePushService(db)
        result = await svc.run_cycle(organization_id="org1", push_threshold=0.5)
        assert result["pushed"] == 0
        assert result["active_sessions"] == 0
        assert result["active_webhooks"] == 0

    @pytest.mark.asyncio
    async def test_run_cycle_emits_push_events(self, monkeypatch):
        monkeypatch.setattr("app.services.proactive_push_service.WebhookService", _FakeWebhookService)
        _FakeWebhookService.emitted = []

        sessions = [
            SimpleNamespace(
                id="s1",
                goal="prevent outage in payment API",
                context_snapshot={"focus": "payment API"},
            )
        ]
        webhooks = [SimpleNamespace(id="w1", is_active=True)]
        events = [
            SimpleNamespace(
                id="e1",
                event_type="incident.critical",
                payload={"summary": "payment API outage detected"},
            )
        ]

        db = _FakeDB([])
        svc = ProactivePushService(db)

        async def _sessions(**_kwargs):
            return sessions

        async def _webhooks(**_kwargs):
            return webhooks

        async def _events(**_kwargs):
            return events

        monkeypatch.setattr(svc, "scan_active_sessions", _sessions)
        monkeypatch.setattr(svc, "scan_subscribed_webhooks", _webhooks)
        monkeypatch.setattr(svc, "scan_recent_events", _events)

        result = await svc.run_cycle(
            organization_id="org-abc",
            push_threshold=0.2,
            max_pushes=5,
        )

        assert result["pushed"] == 1
        assert len(_FakeWebhookService.emitted) == 1
        emitted = _FakeWebhookService.emitted[0]
        assert emitted["organization_id"] == "org-abc"
        assert emitted["event_type"] == "cognitive.proactive_push"
