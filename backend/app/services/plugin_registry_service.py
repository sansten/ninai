"""Plugin registry service (Feature 19)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class PluginLogEntry:
    at: datetime
    level: str
    message: str


@dataclass
class PluginSpec:
    name: str
    version: str
    description: str
    capabilities: list[str]
    entrypoint: str
    config_schema: dict[str, Any] = field(default_factory=dict)
    events_emitted: list[str] = field(default_factory=list)
    events_consumed: list[str] = field(default_factory=list)
    manifest_url: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    installed_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    status: str = "installed"


class PluginRegistryService:
    """Process-local plugin registry keyed by org_id and plugin name."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, PluginSpec]] = {}
        self._logs: dict[str, dict[str, list[PluginLogEntry]]] = {}

    def list_for_org(self, *, org_id: str) -> list[PluginSpec]:
        return list(self._store.get(org_id, {}).values())

    def get_for_org(self, *, org_id: str, name: str) -> PluginSpec | None:
        return self._store.get(org_id, {}).get(name)

    def install(self, *, org_id: str, spec: PluginSpec) -> PluginSpec:
        if not spec.name.strip():
            raise ValueError("name is required")
        if not spec.version.strip():
            raise ValueError("version is required")
        if not spec.entrypoint.strip():
            raise ValueError("entrypoint is required")

        org_store = self._store.setdefault(org_id, {})
        now = _utcnow()
        existing = org_store.get(spec.name)

        spec.updated_at = now
        if existing is None:
            spec.installed_at = now
        else:
            spec.installed_at = existing.installed_at

        org_store[spec.name] = spec
        self._append_log(org_id=org_id, name=spec.name, level="info", message="plugin installed")
        return spec

    def uninstall(self, *, org_id: str, name: str) -> bool:
        org_store = self._store.get(org_id, {})
        if name not in org_store:
            return False
        del org_store[name]
        self._append_log(org_id=org_id, name=name, level="info", message="plugin uninstalled")
        return True

    def update_config(self, *, org_id: str, name: str, config_patch: dict[str, Any]) -> PluginSpec | None:
        spec = self.get_for_org(org_id=org_id, name=name)
        if spec is None:
            return None
        merged = dict(spec.config or {})
        merged.update(config_patch or {})
        spec.config = merged
        spec.updated_at = _utcnow()
        self._append_log(org_id=org_id, name=name, level="info", message="plugin config updated")
        return spec

    def get_logs(self, *, org_id: str, name: str, limit: int = 100) -> list[PluginLogEntry]:
        rows = list(self._logs.get(org_id, {}).get(name, []))
        if limit <= 0:
            return []
        return rows[-limit:]

    def _append_log(self, *, org_id: str, name: str, level: str, message: str) -> None:
        org_logs = self._logs.setdefault(org_id, {})
        plugin_logs = org_logs.setdefault(name, [])
        plugin_logs.append(PluginLogEntry(at=_utcnow(), level=level, message=message))
