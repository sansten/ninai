"""Environment Sync Service - Phase 52 Slice 1.

Provides a canonical in-memory sync state model for inbound connector events,
plus normalization, replay protection, and out-of-order safeguards.

Scope for this slice:
- Canonical connector state model
- Inbound normalization
- Replay/out-of-order protection
- Sync lag and divergence summaries
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import json
from typing import Any, Callable

from app.services.inbound_event_service import NormalizedEvent, parse_inbound_event


@dataclass
class NormalizedInboundEvent:
    """Normalized inbound envelope used by sync state processing."""

    org_id: str
    connector_type: str
    external_event_id: str
    external_object_id: str
    external_updated_at: datetime
    payload_hash: str
    replay_key: str
    normalized: NormalizedEvent
    received_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CanonicalConnectorState:
    """Canonical state row for one external object in one connector."""

    org_id: str
    connector_type: str
    external_object_id: str
    event_type: str
    title: str
    summary: str
    severity: str | None
    actor: str | None
    source_url: str | None
    source_event_id: str
    external_updated_at: datetime
    last_seen_at: datetime
    payload_hash: str
    state_hash: str
    last_internal_hash: str | None = None
    diverged: bool = False


@dataclass
class ApplyInboundResult:
    """Result of applying a normalized inbound event to canonical state."""

    status: str  # applied | duplicate | out_of_order | invalid
    state: CanonicalConnectorState | None = None
    reason: str | None = None


@dataclass
class SyncSummary:
    """Per-org summary used by monitoring and dashboards."""

    org_id: str
    total_objects: int
    diverged_objects: int
    divergence_rate: float
    applied_events: int
    duplicate_events: int
    out_of_order_events: int


class EnvironmentSyncService:
    """In-memory canonical sync state manager for connector inbound events."""

    def __init__(
        self,
        *,
        replay_ttl_seconds: int = 24 * 3600,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self._replay_ttl = int(replay_ttl_seconds)
        self._now = now_fn or (lambda: datetime.now(timezone.utc))

        # (org_id, connector_type, external_object_id) -> CanonicalConnectorState
        self._states: dict[tuple[str, str, str], CanonicalConnectorState] = {}

        # replay_key -> first seen timestamp
        self._seen_replay: dict[str, datetime] = {}

        # org_id -> counters
        self._applied: dict[str, int] = {}
        self._duplicates: dict[str, int] = {}
        self._out_of_order: dict[str, int] = {}

    def normalize_inbound(
        self,
        *,
        org_id: str,
        connector_type: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
        received_at: datetime | None = None,
    ) -> NormalizedInboundEvent:
        """Normalize raw inbound connector data into canonical envelope."""
        received = received_at or self._now()
        normalized = parse_inbound_event(connector_type, payload)

        external_event_id = self._pick_external_event_id(payload, headers, normalized)
        external_object_id = normalized.external_id or external_event_id
        external_updated_at = self._pick_external_updated_at(payload, received)

        payload_hash = self._hash_json(payload)
        replay_key = self._hash_parts(
            org_id,
            connector_type,
            external_event_id,
            payload_hash,
        )

        return NormalizedInboundEvent(
            org_id=org_id,
            connector_type=str(connector_type),
            external_event_id=external_event_id,
            external_object_id=external_object_id,
            external_updated_at=external_updated_at,
            payload_hash=payload_hash,
            replay_key=replay_key,
            normalized=normalized,
            received_at=received,
        )

    def apply_inbound(self, event: NormalizedInboundEvent) -> ApplyInboundResult:
        """Apply normalized inbound event to canonical state with replay guards."""
        self._expire_replay_cache()

        if not event.org_id or not event.connector_type or not event.external_object_id:
            self._inc(self._duplicates, event.org_id)
            return ApplyInboundResult(status="invalid", reason="missing required identity fields")

        if event.replay_key in self._seen_replay:
            self._inc(self._duplicates, event.org_id)
            return ApplyInboundResult(status="duplicate", reason="replay key already seen")

        key = (event.org_id, event.connector_type, event.external_object_id)
        existing = self._states.get(key)
        if existing and event.external_updated_at < existing.external_updated_at:
            self._inc(self._out_of_order, event.org_id)
            self._seen_replay[event.replay_key] = self._now()
            return ApplyInboundResult(status="out_of_order", state=existing)

        state_hash = self._hash_parts(
            event.normalized.event_type,
            event.normalized.title,
            event.normalized.summary,
            str(event.normalized.severity or ""),
            str(event.normalized.actor or ""),
            str(event.normalized.url or ""),
        )

        prior_internal_hash = existing.last_internal_hash if existing else None
        state = CanonicalConnectorState(
            org_id=event.org_id,
            connector_type=event.connector_type,
            external_object_id=event.external_object_id,
            event_type=event.normalized.event_type,
            title=event.normalized.title,
            summary=event.normalized.summary,
            severity=event.normalized.severity,
            actor=event.normalized.actor,
            source_url=event.normalized.url,
            source_event_id=event.external_event_id,
            external_updated_at=event.external_updated_at,
            last_seen_at=self._now(),
            payload_hash=event.payload_hash,
            state_hash=state_hash,
            last_internal_hash=prior_internal_hash,
            diverged=(prior_internal_hash is not None and prior_internal_hash != state_hash),
        )

        self._states[key] = state
        self._seen_replay[event.replay_key] = self._now()
        self._inc(self._applied, event.org_id)
        return ApplyInboundResult(status="applied", state=state)

    def mark_internal_projection(
        self,
        *,
        org_id: str,
        connector_type: str,
        external_object_id: str,
        internal_hash: str,
    ) -> CanonicalConnectorState | None:
        """Record current internal projection hash for divergence checks."""
        key = (org_id, connector_type, external_object_id)
        state = self._states.get(key)
        if state is None:
            return None

        state.last_internal_hash = internal_hash
        state.diverged = state.state_hash != internal_hash
        return state

    def reconcile_candidates(
        self,
        *,
        org_id: str,
        max_lag_seconds: int = 30,
    ) -> list[CanonicalConnectorState]:
        """Return candidate objects that are diverged or stale by lag threshold."""
        now = self._now()
        threshold = timedelta(seconds=max(1, int(max_lag_seconds)))
        candidates: list[CanonicalConnectorState] = []

        for state in self._states.values():
            if state.org_id != org_id:
                continue
            lag = now - state.last_seen_at
            if state.diverged or lag > threshold:
                candidates.append(state)

        return candidates

    def summary(self, org_id: str) -> SyncSummary:
        """Return per-org aggregate metrics for sync dashboards."""
        states = [s for s in self._states.values() if s.org_id == org_id]
        total = len(states)
        diverged = sum(1 for s in states if s.diverged)

        return SyncSummary(
            org_id=org_id,
            total_objects=total,
            diverged_objects=diverged,
            divergence_rate=(diverged / total) if total else 0.0,
            applied_events=self._applied.get(org_id, 0),
            duplicate_events=self._duplicates.get(org_id, 0),
            out_of_order_events=self._out_of_order.get(org_id, 0),
        )

    def get_state(
        self,
        *,
        org_id: str,
        connector_type: str,
        external_object_id: str,
    ) -> CanonicalConnectorState | None:
        return self._states.get((org_id, connector_type, external_object_id))

    def reset(self, org_id: str | None = None) -> None:
        """Clear all state or only one org's state and counters."""
        if org_id is None:
            self._states.clear()
            self._seen_replay.clear()
            self._applied.clear()
            self._duplicates.clear()
            self._out_of_order.clear()
            return

        drop_keys = [k for k in self._states if k[0] == org_id]
        for k in drop_keys:
            del self._states[k]

        self._applied.pop(org_id, None)
        self._duplicates.pop(org_id, None)
        self._out_of_order.pop(org_id, None)

    def _expire_replay_cache(self) -> None:
        cutoff = self._now() - timedelta(seconds=self._replay_ttl)
        stale = [k for k, seen_at in self._seen_replay.items() if seen_at < cutoff]
        for k in stale:
            del self._seen_replay[k]

    @staticmethod
    def _hash_parts(*parts: str) -> str:
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]

    @classmethod
    def _hash_json(cls, payload: dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, default=str)
        return cls._hash_parts(raw)

    @staticmethod
    def _pick_external_event_id(
        payload: dict[str, Any],
        headers: dict[str, str] | None,
        normalized: NormalizedEvent,
    ) -> str:
        hdr = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
        candidates = [
            hdr.get("x-ninai-event-id"),
            payload.get("event_id"),
            payload.get("id"),
            payload.get("request_id"),
            normalized.external_id,
        ]
        for c in candidates:
            if c:
                return str(c)

        # Fallback remains deterministic for same payload and connector.
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:24]

    @classmethod
    def _pick_external_updated_at(
        cls,
        payload: dict[str, Any],
        received_at: datetime,
    ) -> datetime:
        for key in ("updated_at", "timestamp", "ts", "occurred_at"):
            parsed = cls._parse_ts(payload.get(key))
            if parsed is not None:
                return parsed

        event_obj = payload.get("event")
        if isinstance(event_obj, dict):
            for key in ("updated_at", "timestamp", "ts", "occurred_at"):
                parsed = cls._parse_ts(event_obj.get(key))
                if parsed is not None:
                    return parsed

        return received_at

    @staticmethod
    def _parse_ts(value: Any) -> datetime | None:
        if value is None:
            return None

        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)

        if isinstance(value, str):
            v = value.strip()
            if not v:
                return None
            if v.isdigit():
                return datetime.fromtimestamp(float(v), tz=timezone.utc)

            iso = v.replace("Z", "+00:00")
            try:
                parsed = datetime.fromisoformat(iso)
                return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                return None

        return None

    @staticmethod
    def _inc(counter: dict[str, int], org_id: str) -> None:
        counter[org_id] = counter.get(org_id, 0) + 1
