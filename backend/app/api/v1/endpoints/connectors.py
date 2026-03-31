"""Connector Hub endpoints — Phase 48.

Exposes CRUD for registered external connectors and an inbound webhook
receiver that converts external system events into Ninai memory records.

Endpoints:
  POST   /connectors                  — register a connector
  GET    /connectors                  — list all connectors for org
  GET    /connectors/{id}             — get one connector
  PATCH  /connectors/{id}/activate    — enable/disable connector
  DELETE /connectors/{id}             — remove connector
  POST   /connectors/{id}/test        — ping the connector's target URL
  POST   /connectors/inbound/{type}   — receive an inbound event from external system
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Path, status

from app.middleware.tenant_context import TenantContext, require_org_admin
from app.services.connector_registry_service import (
    ConnectorRegistryService,
    ConnectorSpec,
    ConnectorTestResult,
)
from app.services.inbound_event_service import parse_inbound_event, event_to_memory_fields

router = APIRouter()

# Module-level registry (singleton per process; seeded from DB at startup in production)
_registry = ConnectorRegistryService()


def _get_registry() -> ConnectorRegistryService:
    return _registry


# ---------------------------------------------------------------------------
# Register a connector
# ---------------------------------------------------------------------------

@router.post("/connectors", status_code=status.HTTP_201_CREATED)
async def register_connector(
    payload: dict[str, Any] = Body(...),
    tenant: TenantContext = Depends(require_org_admin()),
    registry: ConnectorRegistryService = Depends(_get_registry),
) -> dict[str, Any]:
    connector_type = str(payload.get("connector_type") or "")
    target_url = str(payload.get("target_url") or "")
    name = str(payload.get("name") or connector_type)

    if not connector_type or not target_url:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="connector_type and target_url are required",
        )

    spec = ConnectorSpec(
        id=str(uuid.uuid4()),
        organization_id=tenant.org_id,
        name=name,
        connector_type=connector_type,
        target_url=target_url,
        credential_ref=payload.get("credential_ref"),
        headers_template=payload.get("headers_template") or {},
        event_types=payload.get("event_types") or [connector_type],
        is_active=bool(payload.get("is_active", True)),
    )

    try:
        registered = registry.register(spec)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    return _spec_to_dict(registered)


# ---------------------------------------------------------------------------
# List connectors
# ---------------------------------------------------------------------------

@router.get("/connectors")
async def list_connectors(
    tenant: TenantContext = Depends(require_org_admin()),
    registry: ConnectorRegistryService = Depends(_get_registry),
) -> list[dict[str, Any]]:
    return [_spec_to_dict(s) for s in registry.list_for_org(org_id=tenant.org_id)]


# ---------------------------------------------------------------------------
# Get one connector
# ---------------------------------------------------------------------------

@router.get("/connectors/{connector_id}")
async def get_connector(
    connector_id: str = Path(...),
    tenant: TenantContext = Depends(require_org_admin()),
    registry: ConnectorRegistryService = Depends(_get_registry),
) -> dict[str, Any]:
    spec = registry.get_by_id(org_id=tenant.org_id, connector_id=connector_id)
    if spec is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="connector not found")
    return _spec_to_dict(spec)


# ---------------------------------------------------------------------------
# Activate / deactivate connector
# ---------------------------------------------------------------------------

@router.patch("/connectors/{connector_id}/activate")
async def set_connector_active(
    connector_id: str = Path(...),
    payload: dict[str, Any] = Body(...),
    tenant: TenantContext = Depends(require_org_admin()),
    registry: ConnectorRegistryService = Depends(_get_registry),
) -> dict[str, Any]:
    active = bool(payload.get("is_active", True))
    found = registry.set_active(org_id=tenant.org_id, connector_id=connector_id, active=active)
    if not found:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="connector not found")
    spec = registry.get_by_id(org_id=tenant.org_id, connector_id=connector_id)
    return _spec_to_dict(spec)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Delete connector
# ---------------------------------------------------------------------------

@router.delete("/connectors/{connector_id}")
async def delete_connector(
    connector_id: str = Path(...),
    tenant: TenantContext = Depends(require_org_admin()),
    registry: ConnectorRegistryService = Depends(_get_registry),
) -> dict[str, Any]:
    found = registry.deregister(org_id=tenant.org_id, connector_id=connector_id)
    if not found:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="connector not found")
    return {"deleted": True, "connector_id": connector_id}


# ---------------------------------------------------------------------------
# Test / ping a connector
# ---------------------------------------------------------------------------

@router.post("/connectors/{connector_id}/test")
async def test_connector(
    connector_id: str = Path(...),
    tenant: TenantContext = Depends(require_org_admin()),
    registry: ConnectorRegistryService = Depends(_get_registry),
) -> dict[str, Any]:
    result: ConnectorTestResult = await registry.test_connector(
        org_id=tenant.org_id, connector_id=connector_id
    )
    return {
        "connector_id": result.connector_id,
        "status": result.status,
        "http_status_code": result.http_status_code,
        "error": result.error,
        "tested_at": result.tested_at.isoformat(),
    }


# ---------------------------------------------------------------------------
# Inbound event receiver
# ---------------------------------------------------------------------------

@router.post("/connectors/inbound/{connector_type}", status_code=status.HTTP_202_ACCEPTED)
async def receive_inbound_event(
    connector_type: str = Path(...),
    payload: dict[str, Any] = Body(...),
    tenant: TenantContext = Depends(require_org_admin()),
) -> dict[str, Any]:
    """Convert an inbound webhook from an external system to a Ninai memory record.

    The response includes the normalised memory fields that would be used to
    create the memory.  Actual memory creation is performed by the caller in
    the full runtime wiring (connector → inbound → memory service).
    """
    event = parse_inbound_event(connector_type=connector_type, payload=payload)
    fields = event_to_memory_fields(event)

    return {
        "accepted": True,
        "connector_type": event.connector_type,
        "event_type": event.event_type,
        "external_id": event.external_id,
        "memory_fields": fields,
    }


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _spec_to_dict(spec: ConnectorSpec) -> dict[str, Any]:
    return {
        "id": spec.id,
        "organization_id": spec.organization_id,
        "name": spec.name,
        "connector_type": spec.connector_type,
        "target_url": spec.target_url,
        "credential_ref": spec.credential_ref,
        "headers_template": spec.headers_template,
        "event_types": spec.event_types,
        "is_active": spec.is_active,
        "test_status": spec.test_status,
        "last_tested_at": spec.last_tested_at.isoformat() if spec.last_tested_at else None,
    }
