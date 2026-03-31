"""Connector Observability Service — Phase 51 Slice 4.

Records per-org, per-connector-type dispatch outcome events and exposes
aggregated metrics (counters, last error, last dispatch time).

Designed as an in-process accumulator that is passed to
AutonomousActionAgent via runtime['action_observability'].  In production
this can be backed by a Redis hash, Prometheus counters, or OpenTelemetry
metrics; the interface is the same.

Thread-safety: all mutations use plain dict operations which are GIL-protected
in CPython.  For multi-process deployments, swap the backing store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Event + Metrics types
# ---------------------------------------------------------------------------

@dataclass
class DispatchEvent:
    """Single dispatch outcome recorded after each connector call."""
    org_id: str
    connector_type: str
    status: str                   # "success" | "failed"
    retry_class: str | None       # RetryClass.value or None on success
    rollback_policy: str | None   # "none" | "notify" | "compensate" | None
    rollback_triggered: bool
    attempt_count: int
    latency_ms: float | None
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ConnectorMetrics:
    """Aggregated counters for a (org_id, connector_type) pair."""
    org_id: str
    connector_type: str
    total: int = 0
    success: int = 0
    failed: int = 0
    throttled: int = 0           # failed with retry_class=throttled
    permanent_error: int = 0     # failed with retry_class=permanent
    rollback_triggered: int = 0
    last_error: str | None = None
    last_dispatch_at: datetime | None = None
    total_latency_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        """Float 0..1, or 0.0 if no calls recorded."""
        return self.success / self.total if self.total > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float | None:
        """Average latency in ms across all recorded events, or None if none."""
        return round(self.total_latency_ms / self.total, 2) if self.total > 0 else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "org_id": self.org_id,
            "connector_type": self.connector_type,
            "total": self.total,
            "success": self.success,
            "failed": self.failed,
            "throttled": self.throttled,
            "permanent_error": self.permanent_error,
            "rollback_triggered": self.rollback_triggered,
            "success_rate": round(self.success_rate, 4),
            "avg_latency_ms": self.avg_latency_ms,
            "last_error": self.last_error,
            "last_dispatch_at": self.last_dispatch_at.isoformat() if self.last_dispatch_at else None,
        }


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class ConnectorObservabilityService:
    """In-process accumulator for connector dispatch metrics.

    Keyed by (org_id, connector_type).  Designed to be injected at runtime:

        observability = ConnectorObservabilityService()
        runtime = {"action_observability": observability}
    """

    def __init__(self) -> None:
        # (org_id, connector_type) -> ConnectorMetrics
        self._metrics: dict[tuple[str, str], ConnectorMetrics] = {}

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def record(self, event: DispatchEvent) -> None:
        """Accumulate a dispatch event into the running metrics."""
        key = (event.org_id, event.connector_type)
        m = self._metrics.setdefault(
            key,
            ConnectorMetrics(org_id=event.org_id, connector_type=event.connector_type),
        )
        m.total += 1
        if event.status == "success":
            m.success += 1
        else:
            m.failed += 1
            if event.retry_class == "throttled":
                m.throttled += 1
            elif event.retry_class == "permanent":
                m.permanent_error += 1

        if event.rollback_triggered:
            m.rollback_triggered += 1

        if event.latency_ms is not None:
            m.total_latency_ms += event.latency_ms

        m.last_dispatch_at = event.ts

        if event.status == "failed":
            m.last_error = (
                f"retry_class={event.retry_class}, rollback={event.rollback_policy}"
            )

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def metrics_for(self, org_id: str, connector_type: str) -> ConnectorMetrics | None:
        """Return accumulated metrics for a specific (org, connector_type) pair."""
        return self._metrics.get((org_id, connector_type))

    def org_summary(self, org_id: str) -> list[ConnectorMetrics]:
        """Return all metric records for an org across all connector types."""
        return [m for (oid, _), m in self._metrics.items() if oid == org_id]

    def reset(self, org_id: str | None = None, connector_type: str | None = None) -> None:
        """Clear metrics.  If org_id given, clears only that org's records."""
        if org_id is None:
            self._metrics.clear()
        elif connector_type is not None:
            self._metrics.pop((org_id, connector_type), None)
        else:
            to_remove = [k for k in self._metrics if k[0] == org_id]
            for k in to_remove:
                del self._metrics[k]
