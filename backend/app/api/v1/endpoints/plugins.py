"""Plugin / Extension registry endpoints (Feature 19)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query, status

from app.middleware.tenant_context import TenantContext, require_org_admin
from app.services.plugin_registry_service import PluginRegistryService, PluginSpec

router = APIRouter()

_registry = PluginRegistryService()


def _get_registry() -> PluginRegistryService:
    return _registry


def _spec_to_dict(spec: PluginSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "version": spec.version,
        "description": spec.description,
        "capabilities": list(spec.capabilities or []),
        "entrypoint": spec.entrypoint,
        "config_schema": dict(spec.config_schema or {}),
        "events_emitted": list(spec.events_emitted or []),
        "events_consumed": list(spec.events_consumed or []),
        "manifest_url": spec.manifest_url,
        "config": dict(spec.config or {}),
        "installed_at": spec.installed_at,
        "updated_at": spec.updated_at,
        "status": spec.status,
    }


@router.get("")
async def list_plugins(
    tenant: TenantContext = Depends(require_org_admin()),
    registry: PluginRegistryService = Depends(_get_registry),
) -> dict[str, Any]:
    plugins = [_spec_to_dict(spec) for spec in registry.list_for_org(org_id=tenant.org_id)]
    return {"plugins": plugins, "total": len(plugins)}


@router.post("/install", status_code=status.HTTP_201_CREATED)
async def install_plugin(
    payload: dict[str, Any] = Body(...),
    tenant: TenantContext = Depends(require_org_admin()),
    registry: PluginRegistryService = Depends(_get_registry),
) -> dict[str, Any]:
    name = str(payload.get("name") or "").strip()
    version = str(payload.get("version") or "").strip()
    entrypoint = str(payload.get("entrypoint") or "").strip()

    if not name or not version or not entrypoint:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="name, version, and entrypoint are required",
        )

    spec = PluginSpec(
        name=name,
        version=version,
        description=str(payload.get("description") or ""),
        capabilities=[str(c) for c in (payload.get("capabilities") or [])],
        entrypoint=entrypoint,
        config_schema=dict(payload.get("config_schema") or {}),
        events_emitted=[str(v) for v in (payload.get("events_emitted") or [])],
        events_consumed=[str(v) for v in (payload.get("events_consumed") or [])],
        manifest_url=str(payload.get("manifest_url") or "") or None,
        config=dict(payload.get("config") or {}),
        status=str(payload.get("status") or "installed"),
    )

    try:
        installed = registry.install(org_id=tenant.org_id, spec=spec)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    return {"installed": True, "plugin": _spec_to_dict(installed)}


@router.delete("/{name}")
async def uninstall_plugin(
    name: str = Path(..., min_length=1),
    tenant: TenantContext = Depends(require_org_admin()),
    registry: PluginRegistryService = Depends(_get_registry),
) -> dict[str, Any]:
    deleted = registry.uninstall(org_id=tenant.org_id, name=name)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plugin not found")
    return {"deleted": True, "name": name}


@router.get("/{name}/logs")
async def get_plugin_logs(
    name: str = Path(..., min_length=1),
    limit: int = Query(default=100, ge=1, le=1000),
    tenant: TenantContext = Depends(require_org_admin()),
    registry: PluginRegistryService = Depends(_get_registry),
) -> dict[str, Any]:
    spec = registry.get_for_org(org_id=tenant.org_id, name=name)
    if spec is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plugin not found")

    logs = registry.get_logs(org_id=tenant.org_id, name=name, limit=limit)
    return {
        "name": name,
        "logs": [{"at": row.at, "level": row.level, "message": row.message} for row in logs],
        "total": len(logs),
    }


@router.patch("/{name}/config")
async def update_plugin_config(
    name: str = Path(..., min_length=1),
    payload: dict[str, Any] = Body(...),
    tenant: TenantContext = Depends(require_org_admin()),
    registry: PluginRegistryService = Depends(_get_registry),
) -> dict[str, Any]:
    config_patch = payload.get("config")
    if not isinstance(config_patch, dict):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="config object is required")

    updated = registry.update_config(org_id=tenant.org_id, name=name, config_patch=config_patch)
    if updated is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plugin not found")

    return {"updated": True, "plugin": _spec_to_dict(updated)}
