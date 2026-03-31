"""Action Kill-Switch Service — Phase 51 Slice 4.

Programmatic, per-org and per-connector-type kill-switch store.

Operators can disable all actions for an org, disable specific action types,
or enable dry-run mode — without touching the agent call site.  The agent
reads: runtime['action_kill_switch'].build_runtime_control(org_id) and
merges the result with any explicit action_runtime_control dict already
present in runtime.

Design: in-process dict store.  Production deployments can swap this for a
Redis hash (same interface) to make changes visible across workers instantly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# State type
# ---------------------------------------------------------------------------

@dataclass
class KillSwitchState:
    """Kill-switch configuration for one org."""
    org_id: str
    enabled: bool = True           # False == all actions blocked
    dry_run: bool = False          # True == actions simulated but not dispatched
    disabled_action_types: set[str] = field(default_factory=set)
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "org_id": self.org_id,
            "enabled": self.enabled,
            "dry_run": self.dry_run,
            "disabled_action_types": sorted(self.disabled_action_types),
            "updated_at": self.updated_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

_VALID_ACTION_TYPES = frozenset({
    "webhook", "pagerduty", "jira", "slack", "generic_rest",
})


class ActionKillSwitchService:
    """In-process kill-switch store.

    Example usage (wired into the agent at runtime):

        ks = ActionKillSwitchService()
        ks.disable_all(org_id="org-1")

        runtime = {
            "action_kill_switch": ks,
            # action_runtime_control NOT required — ks.build_runtime_control() fills it
        }
    """

    def __init__(self) -> None:
        self._store: dict[str, KillSwitchState] = {}

    # ------------------------------------------------------------------
    # Org-level controls
    # ------------------------------------------------------------------

    def _get_or_create(self, org_id: str) -> KillSwitchState:
        if org_id not in self._store:
            self._store[org_id] = KillSwitchState(org_id=org_id)
        return self._store[org_id]

    def set_org_state(
        self,
        org_id: str,
        *,
        enabled: bool | None = None,
        dry_run: bool | None = None,
        disabled_action_types: set[str] | list[str] | None = None,
    ) -> KillSwitchState:
        """Create or update kill-switch state for an org.

        Only the supplied keyword arguments are changed; others are preserved.
        """
        state = self._get_or_create(org_id)
        if enabled is not None:
            state.enabled = bool(enabled)
        if dry_run is not None:
            state.dry_run = bool(dry_run)
        if disabled_action_types is not None:
            valid = {str(t).strip().lower() for t in disabled_action_types}
            state.disabled_action_types = valid & _VALID_ACTION_TYPES
        state.updated_at = datetime.now(timezone.utc)
        return state

    def get_org_state(self, org_id: str) -> KillSwitchState | None:
        """Return current state for an org, or None if never configured."""
        return self._store.get(org_id)

    def disable_all(self, org_id: str) -> KillSwitchState:
        """Block all action dispatches for an org."""
        return self.set_org_state(org_id, enabled=False)

    def enable_all(self, org_id: str) -> KillSwitchState:
        """Re-enable all action dispatches for an org (preserves disabled_action_types)."""
        return self.set_org_state(org_id, enabled=True)

    def set_dry_run(self, org_id: str, *, dry_run: bool) -> KillSwitchState:
        """Toggle dry-run mode for an org."""
        return self.set_org_state(org_id, dry_run=dry_run)

    # ------------------------------------------------------------------
    # Connector-type controls
    # ------------------------------------------------------------------

    def disable_connector_type(self, org_id: str, connector_type: str) -> bool:
        """Add a connector_type to the disabled set.

        Returns True if the type was valid and added, False if unknown type.
        """
        t = str(connector_type).strip().lower()
        if t not in _VALID_ACTION_TYPES:
            return False
        state = self._get_or_create(org_id)
        state.disabled_action_types.add(t)
        state.updated_at = datetime.now(timezone.utc)
        return True

    def enable_connector_type(self, org_id: str, connector_type: str) -> bool:
        """Remove a connector_type from the disabled set.

        Returns True if the type was found and removed, False if not present.
        """
        t = str(connector_type).strip().lower()
        state = self._store.get(org_id)
        if state is None or t not in state.disabled_action_types:
            return False
        state.disabled_action_types.discard(t)
        state.updated_at = datetime.now(timezone.utc)
        return True

    # ------------------------------------------------------------------
    # Runtime control builder
    # ------------------------------------------------------------------

    def build_runtime_control(self, org_id: str) -> dict[str, Any]:
        """Return an action_runtime_control-compatible dict for the given org.

        If the org has no configured state, returns a default permissive dict.
        The returned dict is safe to pass directly as runtime['action_runtime_control'].
        """
        state = self._store.get(org_id)
        if state is None:
            return {"enabled": True, "dry_run": False, "disabled_action_types": []}
        return {
            "enabled": state.enabled,
            "dry_run": state.dry_run,
            "disabled_action_types": sorted(state.disabled_action_types),
        }

    def reset(self, org_id: str | None = None) -> None:
        """Clear kill-switch state.  If org_id given, clears only that org."""
        if org_id is None:
            self._store.clear()
        else:
            self._store.pop(org_id, None)
