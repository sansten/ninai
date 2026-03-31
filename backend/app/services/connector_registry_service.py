"""Connector Registry Service — Phase 48.

Manages registered external-system connectors per tenant.  This is the
in-process registry used at dispatch time; the persistent store is
ConnectorRegistration (DB model).

Responsibilities:
- CRUD for connector registrations (in-memory dict for test/demo; DB-backed at runtime)
- Lookup: find active connector for (org_id, connector_type)
- Ping/test a connector's target URL
- Resolve credential_ref → actual credential string (env / secret manager)

The service is designed to run without a DB session in unit tests by using
an in-memory store seeded by tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.services.external_connector_service import ExternalConnectorService, DispatchResult


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ConnectorSpec:
    """Runtime representation of a registered connector."""
    id: str
    organization_id: str
    name: str
    connector_type: str          # webhook | pagerduty | jira | slack | generic_rest
    target_url: str
    credential_ref: str | None
    headers_template: dict
    event_types: list[str]
    is_active: bool = True
    test_status: str = "untested"
    last_tested_at: datetime | None = None


@dataclass
class ConnectorTestResult:
    connector_id: str
    status: str          # "ok" | "failed"
    http_status_code: int | None
    error: str | None
    tested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Valid constants
# ---------------------------------------------------------------------------

_VALID_TYPES = frozenset({
    "webhook", "pagerduty", "jira", "slack", "generic_rest"
})
_VALID_TEST_STATUSES = frozenset({"ok", "failed", "untested"})


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ConnectorRegistryService:
    """In-memory connector registry with optional persistence hooks.

    In production, callers seed the registry from DB rows on startup and
    register/deregister as rows change.  In tests, connectors are registered
    directly without a DB.
    """

    def __init__(
        self,
        *,
        dispatcher: ExternalConnectorService | None = None,
    ) -> None:
        # org_id -> connector_id -> ConnectorSpec
        self._store: dict[str, dict[str, ConnectorSpec]] = {}
        self._dispatcher = dispatcher or ExternalConnectorService()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, spec: ConnectorSpec) -> ConnectorSpec:
        """Register or replace a connector."""
        if spec.connector_type not in _VALID_TYPES:
            raise ValueError(f"Unknown connector_type: {spec.connector_type!r}")
        if not spec.target_url:
            raise ValueError("target_url is required")
        if not spec.id:
            raise ValueError("id is required")

        org_store = self._store.setdefault(spec.organization_id, {})
        org_store[spec.id] = spec
        return spec

    def deregister(self, *, org_id: str, connector_id: str) -> bool:
        """Remove a connector.  Returns True if it existed."""
        org_store = self._store.get(org_id, {})
        if connector_id in org_store:
            del org_store[connector_id]
            return True
        return False

    def set_active(self, *, org_id: str, connector_id: str, active: bool) -> bool:
        """Toggle is_active flag.  Returns True if connector found."""
        spec = self._get(org_id, connector_id)
        if spec is None:
            return False
        # dataclasses are mutable here
        spec.is_active = active
        return True

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def _get(self, org_id: str, connector_id: str) -> ConnectorSpec | None:
        return self._store.get(org_id, {}).get(connector_id)

    def get_by_id(self, *, org_id: str, connector_id: str) -> ConnectorSpec | None:
        return self._get(org_id, connector_id)

    def get_for_action(
        self, *, org_id: str, connector_type: str
    ) -> ConnectorSpec | None:
        """Return the first active connector matching org_id + connector_type."""
        for spec in self._store.get(org_id, {}).values():
            if (
                spec.is_active
                and spec.connector_type == connector_type
            ):
                return spec
        return None

    def list_for_org(self, *, org_id: str) -> list[ConnectorSpec]:
        return list(self._store.get(org_id, {}).values())

    def list_active_for_org(self, *, org_id: str) -> list[ConnectorSpec]:
        return [s for s in self.list_for_org(org_id=org_id) if s.is_active]

    # ------------------------------------------------------------------
    # Credential resolution
    # ------------------------------------------------------------------

    def resolve_credential(
        self,
        spec: ConnectorSpec,
        *,
        env_prefix: str = "NINAI_CONNECTOR_",
    ) -> str | None:
        """Resolve credential_ref to actual value.

        Tries (in order):
        1. Environment variable: env_prefix + credential_ref.upper()
        2. credential_ref directly (if it looks like a literal token)
        3. Returns None (connector works without auth — e.g. public webhooks)
        """
        if not spec.credential_ref:
            return None
        env_key = env_prefix + spec.credential_ref.upper().replace("-", "_")
        val = os.environ.get(env_key)
        if val:
            return val
        # If credential_ref itself is short enough to be a literal (test tokens etc.)
        if len(spec.credential_ref) <= 128:
            return spec.credential_ref
        return None

    def build_dispatch_headers(self, spec: ConnectorSpec) -> dict[str, str]:
        """Merge spec.headers_template with auth header if credential present."""
        headers: dict[str, str] = {}
        if spec.headers_template:
            headers.update({str(k): str(v) for k, v in spec.headers_template.items()})

        credential = self.resolve_credential(spec)
        if credential:
            if spec.connector_type == "pagerduty":
                headers["Authorization"] = f"Token token={credential}"
            elif spec.connector_type == "jira":
                headers["Authorization"] = f"Bearer {credential}"
            else:
                headers.setdefault("Authorization", f"Bearer {credential}")

        return headers

    # ------------------------------------------------------------------
    # Test / ping
    # ------------------------------------------------------------------

    async def test_connector(
        self, *, org_id: str, connector_id: str
    ) -> ConnectorTestResult:
        """Send a lightweight test POST to the connector's target_url."""
        spec = self._get(org_id, connector_id)
        if spec is None:
            return ConnectorTestResult(
                connector_id=connector_id,
                status="failed",
                http_status_code=None,
                error="connector not found",
            )

        headers = self.build_dispatch_headers(spec)
        result: DispatchResult = await self._dispatcher.dispatch(
            action_type=spec.connector_type,
            target_url=spec.target_url,
            payload={"ninai": "ping", "connector_id": connector_id},
            headers=headers,
        )

        test_result = ConnectorTestResult(
            connector_id=connector_id,
            status="ok" if result.status == "success" else "failed",
            http_status_code=result.http_status_code,
            error=result.error,
        )
        spec.test_status = test_result.status
        spec.last_tested_at = test_result.tested_at
        return test_result
