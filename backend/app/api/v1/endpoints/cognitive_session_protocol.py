"""Feature 12 Cognitive Session Protocol endpoints.

Provides session conversation lifecycle helpers on top of existing
CognitiveSession records under /api/v1/cognitive/sessions.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, set_tenant_context
from app.core.redis import RedisClient
from app.middleware.tenant_context import TenantContext, get_tenant_context
from app.models.cognitive_session import CognitiveSession
from app.services.cognitive_gateway_service import (
    CognitiveGatewayCapabilities,
    CognitiveGatewayService,
    detect_goal,
)

router = APIRouter()


def _require_session_access(*, tenant: TenantContext, sess: CognitiveSession) -> None:
    if getattr(sess, "organization_id", None) != tenant.org_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    if getattr(sess, "user_id", None) == tenant.user_id:
        return
    if tenant.is_org_admin:
        return
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")


async def _load_session(*, db: AsyncSession, tenant: TenantContext, session_id: str) -> CognitiveSession:
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
    res = await db.execute(select(CognitiveSession).where(CognitiveSession.id == session_id))
    sess = res.scalar_one_or_none()
    if sess is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    _require_session_access(tenant=tenant, sess=sess)
    return sess


def _snapshot(sess: CognitiveSession) -> dict[str, Any]:
    current = getattr(sess, "context_snapshot", None)
    return dict(current) if isinstance(current, dict) else {}


def _sse_frame(event: str, payload: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n".encode("utf-8")


async def _emit_session_event(*, org_id: str, event_type: str, payload: dict[str, Any]) -> None:
    try:
        fields = {
            "event_type": event_type,
            "payload": json.dumps(payload, default=str),
            "event_id": payload.get("event_id") or payload.get("session_id") or "",
        }
        await RedisClient.xadd(f"ninai:events:{org_id}:all", fields, maxlen=1000)
        await RedisClient.xadd(f"ninai:events:{org_id}:{event_type}", fields, maxlen=1000)
        # Backward compatibility for existing consumers/tests.
        await RedisClient.xadd(f"events:{org_id}", fields, maxlen=1000)
    except Exception:
        pass


@router.post("/sessions/{session_id}/message")
async def session_message(
    session_id: str,
    payload: dict[str, Any] = Body(...),
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    message = str(payload.get("message") or "")
    if not message.strip():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="message is required")

    sess = await _load_session(db=db, tenant=tenant, session_id=session_id)
    snap = _snapshot(sess)

    gateway = CognitiveGatewayService(capabilities=CognitiveGatewayCapabilities.full())
    prior_memories = list(snap.get("surfaced_memories") or [])

    read_result = await gateway.read(
        query=message,
        memories=prior_memories,
        limit=int(payload.get("limit") or 5),
        context_id=session_id,
        org_id=tenant.org_id,
    )
    decision = await gateway.decide(
        content=message,
        enrichment=dict(payload.get("context") or {}),
        context_id=session_id,
        org_id=tenant.org_id,
    )

    now_iso = datetime.now(timezone.utc).isoformat()

    turns = list(snap.get("turns") or [])
    turns.append({"role": "user", "content": message, "timestamp": now_iso})

    decisions = list(snap.get("decisions") or [])
    decisions.append(
        {
            "decision": decision.decision,
            "confidence": decision.confidence,
            "tone": decision.tone,
            "action_recommended": decision.action_recommended,
            "timestamp": now_iso,
        }
    )

    identified_goals = list(snap.get("identified_goals") or [])
    inferred_goal = detect_goal(message)
    if inferred_goal:
        identified_goals.append({"goal": inferred_goal, "source": "message", "timestamp": now_iso})

    snap["turns"] = turns
    snap["decisions"] = decisions
    snap["surfaced_memories"] = list(read_result.memories or [])
    snap["identified_goals"] = identified_goals
    snap["last_message"] = message
    snap["last_updated_at"] = now_iso

    sess.context_snapshot = snap
    await db.commit()

    await _emit_session_event(
        org_id=tenant.org_id,
        event_type="session.message",
        payload={
            "session_id": session_id,
            "message": message,
            "decision": decision.decision,
            "timestamp": now_iso,
        },
    )

    async def _generate():
        yield _sse_frame("thinking", {"step": "retrieving context", "memory_count": read_result.total})
        yield _sse_frame(
            "result",
            {
                "session_id": session_id,
                "decision": decision.decision,
                "confidence": decision.confidence,
                "tone": decision.tone,
                "action_recommended": decision.action_recommended,
                "memories": read_result.memories,
            },
        )
        yield _sse_frame("done", {"session_id": session_id})

    return StreamingResponse(_generate(), media_type="text/event-stream")


@router.get("/sessions/{session_id}/summary")
async def session_summary(
    session_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    sess = await _load_session(db=db, tenant=tenant, session_id=session_id)
    snap = _snapshot(sess)
    return {
        "session_id": str(sess.id),
        "status": str(sess.status),
        "goal": str(sess.goal),
        "goal_id": getattr(sess, "goal_id", None),
        "turn_count": len(list(snap.get("turns") or [])),
        "decision_count": len(list(snap.get("decisions") or [])),
        "memory_count": len(list(snap.get("surfaced_memories") or [])),
        "context_snapshot": snap,
    }


@router.get("/sessions/{session_id}/goals")
async def session_goals(
    session_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    sess = await _load_session(db=db, tenant=tenant, session_id=session_id)
    snap = _snapshot(sess)
    goals = list(snap.get("identified_goals") or [])
    if getattr(sess, "goal", None):
        goals.insert(0, {"goal": sess.goal, "goal_id": getattr(sess, "goal_id", None), "source": "session"})
    return {"session_id": str(sess.id), "goals": goals}


@router.get("/sessions/{session_id}/memories")
async def session_memories(
    session_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    sess = await _load_session(db=db, tenant=tenant, session_id=session_id)
    snap = _snapshot(sess)
    return {
        "session_id": str(sess.id),
        "memories": list(snap.get("surfaced_memories") or []),
    }


@router.get("/sessions/{session_id}/decisions")
async def session_decisions(
    session_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    sess = await _load_session(db=db, tenant=tenant, session_id=session_id)
    snap = _snapshot(sess)
    return {
        "session_id": str(sess.id),
        "decisions": list(snap.get("decisions") or []),
    }


@router.patch("/sessions/{session_id}")
async def patch_session(
    session_id: str,
    payload: dict[str, Any] = Body(...),
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    sess = await _load_session(db=db, tenant=tenant, session_id=session_id)
    snap = _snapshot(sess)

    updated = False

    context_update = payload.get("context_snapshot")
    if isinstance(context_update, dict):
        snap.update(context_update)
        updated = True

    goal_update = payload.get("goal")
    if isinstance(goal_update, str) and goal_update.strip():
        sess.goal = goal_update.strip()
        updated = True

    status_update = payload.get("status")
    allowed_status = {"running", "succeeded", "failed", "aborted"}
    if isinstance(status_update, str) and status_update in allowed_status:
        sess.status = status_update
        updated = True

    if not updated:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No valid updates supplied")

    snap["last_updated_at"] = datetime.now(timezone.utc).isoformat()
    sess.context_snapshot = snap
    await db.commit()

    await _emit_session_event(
        org_id=tenant.org_id,
        event_type="session.updated",
        payload={
            "session_id": session_id,
            "status": sess.status,
        },
    )

    return {
        "session_id": str(sess.id),
        "status": str(sess.status),
        "goal": str(sess.goal),
        "context_snapshot": sess.context_snapshot or {},
    }


@router.delete("/sessions/{session_id}")
async def close_session(
    session_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    sess = await _load_session(db=db, tenant=tenant, session_id=session_id)
    snap = _snapshot(sess)

    if str(sess.status) == "running":
        sess.status = "succeeded"
    snap["closed_at"] = datetime.now(timezone.utc).isoformat()
    sess.context_snapshot = snap

    await db.commit()

    await _emit_session_event(
        org_id=tenant.org_id,
        event_type="session.closed",
        payload={
            "session_id": session_id,
            "status": sess.status,
        },
    )

    return {"closed": True, "session_id": str(sess.id), "status": str(sess.status)}
