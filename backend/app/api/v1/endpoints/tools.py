"""Tool governance/debug endpoints.

These endpoints expose:
- A list of registered built-in CognitiveLoop tools
- A guarded invocation path that uses PolicyGuard + ToolInvoker

They are intended for debugging and SDK examples; they require org admin.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import TenantContext, get_tenant_context
from app.models.cognitive_iteration import CognitiveIteration
from app.models.cognitive_session import CognitiveSession
from app.schemas.tools import ToolInvokeRequest, ToolInvokeResponse, ToolSpecOut
from app.services.cognitive_loop.tools import register_builtin_tools
from app.services.cognitive_tooling.policy_guard import PolicyGuard, ToolContext
from app.services.cognitive_tooling.tool_call_log_service import ToolCallLogService
from app.services.cognitive_tooling.tool_invoker import ToolInvoker
from app.services.cognitive_tooling.tool_registry import ToolRegistry
from app.services.memory_service import MemoryService
from app.services.permission_checker import PermissionChecker
from app.services.self_model_service import SelfModelService


router = APIRouter()


def _build_registry(*, db: AsyncSession, tenant: TenantContext) -> ToolRegistry:
    registry = ToolRegistry()
    memory_service = MemoryService(
        db,
        user_id=tenant.user_id,
        org_id=tenant.org_id,
        clearance_level=int(tenant.clearance_level or 0),
    )
    register_builtin_tools(registry=registry, memory_service=memory_service)
    return registry


@router.get("", response_model=list[ToolSpecOut])
async def list_tools(
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    if not tenant.is_org_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    async with db.begin():
        await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
        registry = _build_registry(db=db, tenant=tenant)

    out: list[ToolSpecOut] = []
    for name in sorted(list(getattr(registry, "_tools", {}).keys())):
        spec = registry.get_spec(name)
        out.append(ToolSpecOut.from_spec(spec))
    return out


@router.post("/invoke", response_model=ToolInvokeResponse)
async def invoke_tool(
    body: ToolInvokeRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
    x_trace_id: str | None = Header(default=None, alias="X-Trace-ID"),
):
    async with db.begin():
        await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

        session_id = body.session_id
        iteration_id = body.iteration_id

        if iteration_id and not session_id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="iteration_id requires session_id",
            )

        now = datetime.now(timezone.utc)

        # For notebook/SDK convenience, allow omitting session_id/iteration_id and create
        # a minimal ad-hoc CognitiveSession + CognitiveIteration for ToolCallLog FK integrity.
        if not session_id and not iteration_id:
            sess = CognitiveSession(
                organization_id=tenant.org_id,
                user_id=tenant.user_id,
                agent_id=None,
                status="running",
                goal=f"ad_hoc_tool_invoke:{body.tool_name}",
                context_snapshot={
                    "scope": body.scope,
                    "scope_id": body.scope_id,
                    "classification": body.classification,
                    "justification": body.justification,
                },
                trace_id=x_trace_id,
            )
            db.add(sess)
            await db.flush()

            it = CognitiveIteration(
                session_id=str(sess.id),
                iteration_num=1,
                plan_json={},
                execution_json={},
                critique_json={},
                evaluation="needs_evidence",
                started_at=now,
                finished_at=now,
                metrics={},
            )
            db.add(it)
            await db.flush()
            session_id = str(sess.id)
            iteration_id = str(it.id)
        elif session_id and not iteration_id:
            # If a session is provided, but no iteration, append a new iteration.
            sres = await db.execute(select(CognitiveSession).where(CognitiveSession.id == session_id))
            sess = sres.scalar_one_or_none()
            if sess is None or getattr(sess, "organization_id", None) != tenant.org_id:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

            nres = await db.execute(
                select(func.max(CognitiveIteration.iteration_num)).where(CognitiveIteration.session_id == session_id)
            )
            next_num = int(nres.scalar_one_or_none() or 0) + 1
            it = CognitiveIteration(
                session_id=session_id,
                iteration_num=next_num,
                plan_json={},
                execution_json={},
                critique_json={},
                evaluation="needs_evidence",
                started_at=now,
                finished_at=now,
                metrics={},
            )
            db.add(it)
            await db.flush()
            iteration_id = str(it.id)

        registry = _build_registry(db=db, tenant=tenant)

        permission_checker = PermissionChecker(db)
        guard = PolicyGuard(permission_checker)
        log_service = ToolCallLogService(db)
        invoker = ToolInvoker(registry=registry, guard=guard, log_service=log_service)

        # Pull SelfModel profile for reliability warnings (best-effort).
        self_model: dict | None = None
        try:
            prof = await SelfModelService(db).get_profile(org_id=tenant.org_id)
            self_model = {
                "tool_reliability": prof.tool_reliability or {},
                "domain_confidence": prof.domain_confidence or {},
                "agent_accuracy": prof.agent_accuracy or {},
            }
        except Exception:
            self_model = None

        ctx = ToolContext(
            user_id=tenant.user_id,
            org_id=tenant.org_id,
            scope=body.scope,
            scope_id=body.scope_id,
            classification=body.classification,
            clearance_level=int(tenant.clearance_level or 0),
            justification=body.justification,
            self_model=self_model,
        )

        result = await invoker.invoke(
            session_id=session_id,
            iteration_id=iteration_id,
            tool_name=body.tool_name,
            tool_input=body.tool_input,
            ctx=ctx,
            swallow_exceptions=True,
        )

        # echo trace_id to help correlation in logs
        return ToolInvokeResponse(trace_id=x_trace_id, result=result)
