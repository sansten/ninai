"""Tests for environment reconciliation worker service (Phase 52 Slice 2)."""

from __future__ import annotations

import pytest

from app.services.environment_reconciliation_service import EnvironmentReconciliationService
from app.services.environment_sync_service import EnvironmentSyncService
from app.services.external_connector_service import DispatchResult


class _StubConnector:
    def __init__(self, statuses: list[str]):
        self._statuses = list(statuses)
        self.calls = []

    async def dispatch(self, *, action_type, target_url, payload, headers):
        self.calls.append(
            {
                "action_type": action_type,
                "target_url": target_url,
                "payload": payload,
                "headers": headers,
            }
        )
        status = self._statuses.pop(0) if self._statuses else "success"
        if status == "success":
            return DispatchResult(
                status="success",
                http_status_code=200,
                attempt_count=1,
            )
        return DispatchResult(
            status="failed",
            http_status_code=500,
            attempt_count=1,
            error="boom",
            retry_class="transient",
        )


@pytest.mark.asyncio
async def test_reconcile_org_success_marks_projection_clean():
    sync = EnvironmentSyncService()
    event = sync.normalize_inbound(
        org_id="org-1",
        connector_type="webhook",
        payload={"event_id": "e-1", "id": "obj-1", "title": "hello", "url": "https://example.com"},
    )
    apply = sync.apply_inbound(event)
    assert apply.state is not None

    # Mark as diverged first so it appears in candidate list regardless of lag.
    sync.mark_internal_projection(
        org_id="org-1",
        connector_type="webhook",
        external_object_id="obj-1",
        internal_hash="different",
    )

    connector = _StubConnector(["success"])
    worker = EnvironmentReconciliationService(sync_service=sync, connector=connector)

    result = await worker.reconcile_org(org_id="org-1", max_lag_seconds=3600, limit=100)
    state = sync.get_state(org_id="org-1", connector_type="webhook", external_object_id="obj-1")

    assert result.attempted == 1
    assert result.succeeded == 1
    assert result.failed == 0
    assert state is not None
    assert state.diverged is False


@pytest.mark.asyncio
async def test_reconcile_org_failure_calls_handoff_handler():
    sync = EnvironmentSyncService()
    event = sync.normalize_inbound(
        org_id="org-1",
        connector_type="webhook",
        payload={"event_id": "e-1", "id": "obj-1", "title": "hello", "url": "https://example.com"},
    )
    sync.apply_inbound(event)
    sync.mark_internal_projection(
        org_id="org-1",
        connector_type="webhook",
        external_object_id="obj-1",
        internal_hash="different",
    )

    calls = []

    async def _on_failure(state, error):
        calls.append((state.external_object_id, error))

    connector = _StubConnector(["failed"])
    worker = EnvironmentReconciliationService(
        sync_service=sync,
        connector=connector,
        failure_handler=_on_failure,
    )

    result = await worker.reconcile_org(org_id="org-1", max_lag_seconds=3600, limit=100)

    assert result.attempted == 1
    assert result.failed == 1
    assert result.dlq_handoffs == 1
    assert len(calls) == 1
    assert calls[0][0] == "obj-1"


@pytest.mark.asyncio
async def test_reconcile_respects_limit():
    sync = EnvironmentSyncService()
    for i in range(3):
        event = sync.normalize_inbound(
            org_id="org-1",
            connector_type="webhook",
            payload={"event_id": f"e-{i}", "id": f"obj-{i}", "title": f"title-{i}"},
        )
        sync.apply_inbound(event)
        sync.mark_internal_projection(
            org_id="org-1",
            connector_type="webhook",
            external_object_id=f"obj-{i}",
            internal_hash="different",
        )

    connector = _StubConnector(["success", "success", "success"])
    worker = EnvironmentReconciliationService(sync_service=sync, connector=connector)

    result = await worker.reconcile_org(org_id="org-1", max_lag_seconds=3600, limit=2)

    assert result.scanned == 3
    assert result.attempted == 2
    assert len(connector.calls) == 2
