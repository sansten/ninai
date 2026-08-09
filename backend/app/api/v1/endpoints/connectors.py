"""Connector Hub endpoints — Phase 48 (revised).

Exposes CRUD for registered external connectors and an inbound webhook
receiver that converts external system events into Ninai memory records.

Inbound flow (fully closed loop):
  parse event → normalize → write memory →
  if severity high/critical:
    create Goal (investigate_<connector>_<event_type>) →
    fire cognitive_loop_task to run the planner on that goal →
    fire meta_review_memory_task for audit trail

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

import hashlib
import hmac
import json
import os
import uuid
from typing import Any

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Path, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, set_tenant_context
from app.core.config import settings
from app.middleware.tenant_context import TenantContext, require_org_admin
from app.schemas.memory import MemoryCreate, MemoryScope, MemoryType
from app.services.connector_registry_service import (
    ConnectorRegistryService,
    ConnectorSpec,
    ConnectorTestResult,
)
from app.services.inbound_event_service import parse_inbound_event, event_to_memory_fields
from app.services.memory_service import MemoryService
from app.services.goal_service import GoalService

router = APIRouter()

# Module-level registry (singleton per process; seeded from DB at startup in production)
_registry = ConnectorRegistryService()

# Severity levels that warrant autonomous cognitive review + goal creation
_HIGH_SEVERITY_VALUES = frozenset({"high", "critical"})

# Goal type for inbound-event-driven investigation goals
_INBOUND_GOAL_TYPE = "analysis"


def _get_registry() -> ConnectorRegistryService:
    return _registry


def _broker_available() -> bool:
    from app.core.celery_app import celery_app
    broker = celery_app.conf.broker_url
    return bool(broker) and not str(broker).startswith("memory://")


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
# Inbound event receiver — writes memory + triggers cognitive review
# ---------------------------------------------------------------------------

def _verify_inbound_signature(connector_type: str, raw_body: bytes, signature: str | None) -> None:
    """Verify X-Ninai-Signature against NINAI_INBOUND_WEBHOOK_SECRET(_<TYPE>), if configured.

    Off by default (no-op) so existing internally-proxied callers that rely
    on require_org_admin() JWT auth alone keep working unchanged. When an
    operator configures a secret for a connector type, a caller MUST also
    present a valid signature — this is the same require-if-configured
    pattern used by the Stripe webhook handler in billing.py, generalized so
    any connector_type can opt in to payload-level authenticity checking on
    top of (not instead of) the JWT/API-key auth already enforced here.
    """
    secret = (
        os.getenv(f"NINAI_INBOUND_WEBHOOK_SECRET_{connector_type.upper()}")
        or os.getenv("NINAI_INBOUND_WEBHOOK_SECRET")
        or ""
    )
    if not secret:
        return
    if not signature:
        raise HTTPException(status_code=400, detail="Missing X-Ninai-Signature header")
    expected = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    received = signature.split("sha256=")[-1].strip()
    if not hmac.compare_digest(expected, received):
        raise HTTPException(status_code=400, detail="Invalid inbound webhook signature")


@router.post("/connectors/inbound/{connector_type}", status_code=status.HTTP_202_ACCEPTED)
async def receive_inbound_event(
    request: Request,
    connector_type: str = Path(...),
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
    x_ninai_signature: str | None = Header(default=None, alias="X-Ninai-Signature"),
) -> dict[str, Any]:
    """Convert an inbound webhook from an external system into a Ninai memory record.

    Steps:
    0. If NINAI_INBOUND_WEBHOOK_SECRET(_<TYPE>) is configured, verify the
       payload's HMAC signature (in addition to the JWT/API-key auth on the
       route) so a leaked/shared org-admin credential alone isn't sufficient
       to inject fabricated external events.
    1. Parse and normalize the external payload into a NormalizedEvent.
    2. Build MemoryCreate from the normalized fields.
    3. Write the memory (force_long_term=True so it persists in Postgres/Qdrant
       with a zero-vector embedding; a background pipeline re-embeds it).
    4. If severity is high or critical, fire meta_review_memory_task so the
       autonomous cognitive loop investigates the event.

    Returns 202 Accepted with memory_id and whether a cognitive review was triggered.
    """
    raw_body = await request.body()
    _verify_inbound_signature(connector_type, raw_body, x_ninai_signature)
    payload: dict[str, Any] = json.loads(raw_body) if raw_body else {}

    event = parse_inbound_event(connector_type=connector_type, payload=payload)
    fields = event_to_memory_fields(event)

    # --- Write the memory ---
    memory_id: str | None = None
    write_error: str | None = None
    try:
        async with db.begin():
            await set_tenant_context(
                db,
                tenant.user_id,
                tenant.org_id,
                roles=getattr(tenant, "roles", ""),
                clearance_level=getattr(tenant, "clearance_level", 0),
                justification="inbound_connector_event",
            )

            mem_svc = MemoryService(
                session=db,
                user_id=tenant.user_id,
                org_id=tenant.org_id,
                clearance_level=getattr(tenant, "clearance_level", 0),
            )

            data = MemoryCreate(
                content=str(fields.get("content") or fields.get("title") or ""),
                title=str(fields.get("title") or "")[:500] or None,
                scope=MemoryScope.ORGANIZATION,
                scope_id=tenant.org_id,
                memory_type=MemoryType.LONG_TERM,
                tags=list(fields.get("tags") or [])[:20],
                source_type=str(fields.get("source_type") or "integration"),
                source_id=event.external_id,
                extra_metadata=dict(fields.get("metadata") or {}),
            )

            zero_embedding = [0.0] * settings.EMBEDDING_DIMENSIONS
            result = await mem_svc.create_memory(data, zero_embedding)
            memory_id = str(result.id) if hasattr(result, "id") else str(result)
    except Exception as exc:
        write_error = f"{type(exc).__name__}: {exc}"

    # --- Create Goal + fire cognitive loop for high-severity events ---
    goal_id: str | None = None
    session_id: str | None = None
    cognitive_loop_triggered = False

    if memory_id and event.severity in _HIGH_SEVERITY_VALUES:
        # 1. Create an investigation goal so the cognitive planner has a target
        goal_title = (
            f"Investigate {connector_type} {event.event_type}: "
            f"{str(event.title or event.external_id or 'unknown')[:120]}"
        )
        try:
            async with db.begin():
                await set_tenant_context(
                    db,
                    tenant.user_id,
                    tenant.org_id,
                    roles=getattr(tenant, "roles", ""),
                    clearance_level=getattr(tenant, "clearance_level", 0),
                    justification="inbound_goal_creation",
                )
                goal_svc = GoalService(db)
                goal = await goal_svc.create_goal(
                    org_id=tenant.org_id,
                    actor_user_id=tenant.user_id,
                    title=goal_title,
                    description=(
                        f"Auto-created from {connector_type} inbound event. "
                        f"Severity: {event.severity}. Memory: {memory_id}."
                    ),
                    goal_type=_INBOUND_GOAL_TYPE,
                    visibility_scope="organization",
                    scope_id=tenant.org_id,
                    priority=2 if event.severity == "critical" else 1,
                    tags=[connector_type, event.event_type or "event", f"severity:{event.severity}"],
                    extra_metadata={
                        "memory_id": memory_id,
                        "external_id": event.external_id,
                        "connector_type": connector_type,
                        "severity": event.severity,
                    },
                )
                goal_id = str(goal.id)
        except Exception:
            pass  # goal creation is best-effort

        # 2. Fire cognitive loop task so the planner acts on the goal
        if goal_id and _broker_available():
            try:
                import uuid as _uuid
                from app.tasks.cognitive_loop import cognitive_loop_task
                _session_id = str(_uuid.uuid4())
                cognitive_loop_task.apply_async(
                    kwargs={
                        "org_id": tenant.org_id,
                        "session_id": _session_id,
                        "initiator_user_id": tenant.user_id,
                        "justification": f"inbound_{connector_type}_{event.severity}",
                    },
                    queue="q.cognitive_loop",
                )
                session_id = _session_id
                cognitive_loop_triggered = True
            except Exception:
                pass

        # 3. Meta-supervisor audit pass on the written memory
        if _broker_available():
            try:
                from app.tasks.meta_agent import meta_review_memory_task
                meta_review_memory_task.apply_async(
                    kwargs={
                        "org_id": tenant.org_id,
                        "memory_id": memory_id,
                        "initiator_user_id": tenant.user_id,
                        "justification": f"inbound_{connector_type}_high_severity",
                    },
                    queue="q.meta_agent",
                )
            except Exception:
                pass

    return {
        "accepted": True,
        "connector_type": event.connector_type,
        "event_type": event.event_type,
        "external_id": event.external_id,
        "memory_id": memory_id,
        "memory_write_error": write_error,
        "goal_id": goal_id,
        "cognitive_session_id": session_id,
        "cognitive_loop_triggered": cognitive_loop_triggered,
        "severity": event.severity,
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
