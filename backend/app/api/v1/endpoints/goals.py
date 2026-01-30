"""GoalGraph API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import TenantContext, get_tenant_context
from app.schemas.goal import (
    GoalActivityLogResponse,
    GoalBlockersResponse,
    GoalCreateRequest,
    GoalDetailResponse,
    GoalEdgeCreateRequest,
    GoalEdgeResponse,
    GoalMemoryLinkCreateRequest,
    GoalMemoryLinkResponse,
    GoalNodeCreateRequest,
    GoalNodeUpdateRequest,
    GoalNodeResponse,
    GoalProgressSummary,
    GoalResponse,
    GoalTreeResponse,
    GoalUpdateRequest,
)
from app.schemas.goal_agents_api import GoalLinkSuggestionsRequest, GoalProposeRequest
from app.schemas.goal_agents import GoalLinkingAgentOutput, GoalPlannerAgentOutput
from app.schemas.base import PaginatedResponse
from app.services.goal_service import GoalService
from app.services.goal_navigator import GoalNavigator
from app.services.goal_planner_agent import GoalPlannerAgent
from app.services.goal_linking_agent import GoalLinkingAgent
from app.services.permission_checker import PermissionChecker


router = APIRouter()


def _create_permission_for_scope(scope: str) -> str:
    if scope == "personal":
        return "goal:create:personal"
    if scope == "team":
        return "goal:create:team"
    # department/division/organization
    return "goal:create:department"


async def _require_permission(
    *,
    db: AsyncSession,
    tenant: TenantContext,
    permission: str,
) -> None:
    checker = PermissionChecker(db)
    decision = await checker.check_permission(tenant.user_id, tenant.org_id, permission)
    if not decision.allowed:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=decision.reason)


async def _require_any_permission(
    *,
    db: AsyncSession,
    tenant: TenantContext,
    permissions: list[str],
) -> None:
    checker = PermissionChecker(db)
    last_reason = ""
    for perm in permissions:
        decision = await checker.check_permission(tenant.user_id, tenant.org_id, perm)
        if decision.allowed:
            return
        last_reason = decision.reason
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=last_reason or "Forbidden")


@router.post("", response_model=GoalResponse)
async def create_goal(
    body: GoalCreateRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    async with db.begin():
        await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

        await _require_permission(db=db, tenant=tenant, permission=_create_permission_for_scope(body.visibility_scope))

        service = GoalService(db)
        goal = await service.create_goal(
            org_id=tenant.org_id,
            created_by_user_id=tenant.user_id,
            payload=body.model_dump(),
        )
        return GoalResponse.model_validate(goal)


@router.post("/propose", response_model=GoalPlannerAgentOutput)
async def propose_goal_from_request(
    body: GoalProposeRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """Suggest a goal graph (proposal only).

    This does not create any DB rows; it returns a deterministic JSON proposal.
    """
    async with db.begin():
        await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

        # Require that user can create at least one goal scope.
        await _require_any_permission(
            db=db,
            tenant=tenant,
            permissions=["goal:create:personal", "goal:create:team", "goal:create:department"],
        )

        agent = GoalPlannerAgent()
        out = await agent.propose_goal(
            user_request=body.user_request,
            session_context=body.session_context,
            existing_goals=body.existing_goals,
            tool_event_sink=None,
        )
        return out


@router.post("/link-suggestions", response_model=GoalLinkingAgentOutput)
async def suggest_goal_links(
    body: GoalLinkSuggestionsRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """Suggest goal-memory links (proposal only)."""
    async with db.begin():
        await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

        await _require_any_permission(
            db=db,
            tenant=tenant,
            permissions=["goal:read:personal", "goal:read:team", "goal:read:department"],
        )

        agent = GoalLinkingAgent()
        out = await agent.suggest_links(
            memory=body.memory,
            active_goals=body.active_goals,
            tool_event_sink=None,
        )
        return out


@router.get("/{goal_id}", response_model=GoalDetailResponse)
async def get_goal(
    goal_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    async with db.begin():
        await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

        await _require_any_permission(
            db=db,
            tenant=tenant,
            permissions=["goal:read:personal", "goal:read:team", "goal:read:department"],
        )

        service = GoalService(db)
        goal, nodes, edges, links, progress = await service.get_goal_detail_bundle(goal_id=goal_id, org_id=tenant.org_id)

        return GoalDetailResponse(
            **GoalResponse.model_validate(goal).model_dump(),
            nodes=[GoalNodeResponse.model_validate(n) for n in nodes],
            edges=[GoalEdgeResponse.model_validate(e) for e in edges],
            memory_links=[GoalMemoryLinkResponse.model_validate(l) for l in links],
            progress=GoalProgressSummary(
                percent_complete=progress.percent_complete,
                completed_nodes=progress.completed_nodes,
                total_nodes=progress.total_nodes,
                confidence=progress.confidence,
            ),
        )


@router.get("", response_model=PaginatedResponse[GoalResponse])
async def list_goals(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    status_filter: str | None = Query(default=None, alias="status"),
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    async with db.begin():
        await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

        await _require_any_permission(
            db=db,
            tenant=tenant,
            permissions=["goal:read:personal", "goal:read:team", "goal:read:department"],
        )

        service = GoalService(db)
        page_data = await service.list_goals(org_id=tenant.org_id, page=page, page_size=page_size, status_filter=status_filter)

        return PaginatedResponse(
            items=[GoalResponse.model_validate(g) for g in page_data.items],
            total=page_data.total,
            page=page_data.page,
            page_size=page_data.page_size,
            pages=page_data.pages,
        )


@router.patch("/{goal_id}", response_model=GoalResponse)
async def update_goal(
    goal_id: str,
    body: GoalUpdateRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    async with db.begin():
        await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

        await _require_any_permission(
            db=db,
            tenant=tenant,
            permissions=["goal:update:own", "goal:update:team", "goal:update:department"],
        )

        service = GoalService(db)
        goal = await service.update_goal(goal_id=goal_id, org_id=tenant.org_id, actor_user_id=tenant.user_id, updates=body.model_dump())
        return GoalResponse.model_validate(goal)


@router.post("/{goal_id}/nodes", response_model=GoalNodeResponse)
async def add_node(
    goal_id: str,
    body: GoalNodeCreateRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    async with db.begin():
        await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

        await _require_any_permission(
            db=db,
            tenant=tenant,
            permissions=["goal:update:own", "goal:update:team", "goal:update:department"],
        )

        service = GoalService(db)
        node = await service.add_node(org_id=tenant.org_id, goal_id=goal_id, actor_user_id=tenant.user_id, payload=body.model_dump())
        return GoalNodeResponse.model_validate(node)


@router.post("/{goal_id}/dependencies", response_model=GoalEdgeResponse)
async def add_dependency(
    goal_id: str,
    body: GoalEdgeCreateRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    async with db.begin():
        await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

        await _require_any_permission(
            db=db,
            tenant=tenant,
            permissions=["goal:update:own", "goal:update:team", "goal:update:department"],
        )

        service = GoalService(db)
        edge = await service.add_dependency(
            org_id=tenant.org_id,
            actor_user_id=tenant.user_id,
            from_node_id=body.from_node_id,
            to_node_id=body.to_node_id,
            edge_type=body.edge_type,
        )
        return GoalEdgeResponse.model_validate(edge)


@router.post("/{goal_id}/link-memory", response_model=GoalMemoryLinkResponse)
async def link_memory(
    goal_id: str,
    body: GoalMemoryLinkCreateRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    async with db.begin():
        await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

        await _require_any_permission(
            db=db,
            tenant=tenant,
            permissions=["goal:update:own", "goal:update:team", "goal:update:department"],
        )

        service = GoalService(db)
        link = await service.link_memory(
            org_id=tenant.org_id,
            actor_user_id=tenant.user_id,
            goal_id=goal_id,
            memory_id=body.memory_id,
            link_type=body.link_type,
            confidence=body.confidence,
            node_id=body.node_id,
            linked_by="user",
        )
        return GoalMemoryLinkResponse.model_validate(link)


@router.get("/{goal_id}/activity", response_model=list[GoalActivityLogResponse])
async def get_goal_activity(
    goal_id: str,
    limit: int = Query(100, ge=1, le=500),
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    async with db.begin():
        await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

        await _require_any_permission(
            db=db,
            tenant=tenant,
            permissions=["goal:read:personal", "goal:read:team", "goal:read:department"],
        )

        service = GoalService(db)
        events = await service.list_activity(goal_id=goal_id, org_id=tenant.org_id, limit=limit)
        return [GoalActivityLogResponse.model_validate(e) for e in events]


@router.patch("/{goal_id}/nodes/{node_id}", response_model=GoalNodeResponse)
async def update_node(
    goal_id: str,
    node_id: str,
    body: GoalNodeUpdateRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    async with db.begin():
        await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

        await _require_any_permission(
            db=db,
            tenant=tenant,
            permissions=["goal:update:own", "goal:update:team", "goal:update:department"],
        )

        service = GoalService(db)
        # Ensure goal exists / RLS applies.
        await service.get_goal(goal_id=goal_id, org_id=tenant.org_id)

        node = await service.update_node(node_id=node_id, org_id=tenant.org_id, actor_user_id=tenant.user_id, updates=body.model_dump())
        return GoalNodeResponse.model_validate(node)


@router.get("/{goal_id}/tree", response_model=GoalTreeResponse)
async def get_goal_tree(
    goal_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    async with db.begin():
        await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

        await _require_any_permission(
            db=db,
            tenant=tenant,
            permissions=["goal:read:personal", "goal:read:team", "goal:read:department"],
        )

        nav = GoalNavigator(db)
        nodes = await nav.get_goal_tree(org_id=tenant.org_id, goal_id=goal_id)
        edges = await nav.list_edges(org_id=tenant.org_id, goal_id=goal_id)
        return GoalTreeResponse(
            goal_id=goal_id,
            nodes=[GoalNodeResponse.model_validate(n) for n in nodes],
            edges=[GoalEdgeResponse.model_validate(e) for e in edges],
        )


@router.get("/{goal_id}/blockers", response_model=GoalBlockersResponse)
async def get_goal_blockers(
    goal_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    async with db.begin():
        await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

        await _require_any_permission(
            db=db,
            tenant=tenant,
            permissions=["goal:read:personal", "goal:read:team", "goal:read:department"],
        )

        nav = GoalNavigator(db)
        blockers = await nav.compute_blockers(org_id=tenant.org_id, goal_id=goal_id)
        return GoalBlockersResponse(goal_id=goal_id, blockers=[GoalNodeResponse.model_validate(n) for n in blockers])
