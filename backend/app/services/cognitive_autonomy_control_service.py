"""Cognitive autonomy control service.

Provides a process-local kill switch for autonomous Cognitive OS behavior.
This controls whether background/autonomous cognitive sessions are allowed
for each org, with an optional global override.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class CognitiveAutonomyState:
    org_id: str
    enabled: bool = True
    reason: str | None = None
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "org_id": self.org_id,
            "enabled": self.enabled,
            "reason": self.reason,
            "updated_at": self.updated_at.isoformat(),
        }


class CognitiveAutonomyControlService:
    """In-process org/global autonomy control plane.

    Default behavior is permissive (enabled=True) unless explicitly disabled.
    """

    def __init__(self) -> None:
        self._global_enabled: bool = True
        self._global_reason: str | None = None
        self._org_states: dict[str, CognitiveAutonomyState] = {}

    def set_global(self, *, enabled: bool, reason: str | None = None) -> None:
        self._global_enabled = bool(enabled)
        self._global_reason = reason

    def set_org(self, org_id: str, *, enabled: bool, reason: str | None = None) -> CognitiveAutonomyState:
        state = self._org_states.get(org_id)
        if state is None:
            state = CognitiveAutonomyState(org_id=org_id)
            self._org_states[org_id] = state
        state.enabled = bool(enabled)
        state.reason = reason
        state.updated_at = datetime.now(timezone.utc)
        return state

    def get_org(self, org_id: str) -> CognitiveAutonomyState | None:
        return self._org_states.get(org_id)

    def is_enabled(self, *, org_id: str) -> tuple[bool, str | None]:
        if not self._global_enabled:
            return False, self._global_reason or "global_autonomy_disabled"
        state = self._org_states.get(org_id)
        if state and not state.enabled:
            return False, state.reason or "org_autonomy_disabled"
        return True, None

    def reset(self) -> None:
        self._global_enabled = True
        self._global_reason = None
        self._org_states.clear()

    def snapshot(self, *, org_id: str) -> dict[str, Any]:
        enabled, blocked_reason = self.is_enabled(org_id=org_id)
        org_state = self.get_org(org_id)
        return {
            "global": {
                "enabled": self._global_enabled,
                "reason": self._global_reason,
            },
            "org": org_state.to_dict() if org_state else None,
            "effective": {
                "enabled": enabled,
                "reason": blocked_reason,
            },
        }


_GLOBAL_COGNITIVE_AUTONOMY_CONTROL: CognitiveAutonomyControlService | None = None


def get_cognitive_autonomy_control_service() -> CognitiveAutonomyControlService:
    global _GLOBAL_COGNITIVE_AUTONOMY_CONTROL
    if _GLOBAL_COGNITIVE_AUTONOMY_CONTROL is None:
        _GLOBAL_COGNITIVE_AUTONOMY_CONTROL = CognitiveAutonomyControlService()
    return _GLOBAL_COGNITIVE_AUTONOMY_CONTROL
