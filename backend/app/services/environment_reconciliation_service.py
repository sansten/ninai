"""Environment reconciliation service - Phase 52 Slice 2.

Runs outbound reconciliation for lagging/diverged canonical sync states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from app.services.environment_sync_service import (
    CanonicalConnectorState,
    EnvironmentSyncService,
)
from app.services.external_connector_service import ExternalConnectorService

_DEFAULT_RECONCILE_ENDPOINT = "https://api.example.internal/ninai/sync/reconcile"


@dataclass
class ReconciliationRunResult:
    org_id: str
    scanned: int
    attempted: int
    succeeded: int
    failed: int
    dlq_handoffs: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "org_id": self.org_id,
            "scanned": self.scanned,
            "attempted": self.attempted,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "dlq_handoffs": self.dlq_handoffs,
        }


FailureHandler = Callable[[CanonicalConnectorState, str], Awaitable[None]]


class EnvironmentReconciliationService:
    """Outbound reconciliation orchestration for canonical sync state."""

    def __init__(
        self,
        *,
        sync_service: EnvironmentSyncService,
        connector: ExternalConnectorService | None = None,
        failure_handler: FailureHandler | None = None,
    ) -> None:
        self._sync_service = sync_service
        self._connector = connector or ExternalConnectorService()
        self._failure_handler = failure_handler

    async def reconcile_org(
        self,
        *,
        org_id: str,
        max_lag_seconds: int = 30,
        limit: int = 100,
    ) -> ReconciliationRunResult:
        candidates = self._sync_service.reconcile_candidates(
            org_id=org_id,
            max_lag_seconds=max_lag_seconds,
        )

        scanned = len(candidates)
        attempted = 0
        succeeded = 0
        failed = 0
        dlq_handoffs = 0

        for state in candidates[: max(1, int(limit))]:
            attempted += 1
            target = state.source_url or _DEFAULT_RECONCILE_ENDPOINT
            payload = {
                "_ninai_reconcile": True,
                "org_id": state.org_id,
                "connector_type": state.connector_type,
                "external_object_id": state.external_object_id,
                "source_event_id": state.source_event_id,
                "state_hash": state.state_hash,
                "diverged": state.diverged,
            }
            headers = {
                "X-Ninai-Sync-Reconcile": "1",
                "X-Ninai-Connector-Type": state.connector_type,
            }

            try:
                result = await self._connector.dispatch(
                    action_type="generic_rest",
                    target_url=target,
                    payload=payload,
                    headers=headers,
                )
                if result.status == "success":
                    self._sync_service.mark_internal_projection(
                        org_id=state.org_id,
                        connector_type=state.connector_type,
                        external_object_id=state.external_object_id,
                        internal_hash=state.state_hash,
                    )
                    succeeded += 1
                else:
                    failed += 1
                    error = result.error or f"dispatch failed http={result.http_status_code}"
                    if self._failure_handler is not None:
                        await self._failure_handler(state, error)
                        dlq_handoffs += 1
            except Exception as exc:
                failed += 1
                if self._failure_handler is not None:
                    await self._failure_handler(state, f"{type(exc).__name__}: {exc}")
                    dlq_handoffs += 1

        return ReconciliationRunResult(
            org_id=org_id,
            scanned=scanned,
            attempted=attempted,
            succeeded=succeeded,
            failed=failed,
            dlq_handoffs=dlq_handoffs,
        )
