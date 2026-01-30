"""GoalGraph Celery tasks.

Rules:
- Tasks are synchronous Celery entrypoints; we bridge into async.
- Tenant context (RLS) MUST be set for every DB session.
- Heavy/agentic work must be async-only; these tasks are safe to retry.

This module implements the GoalGraph tasks listed in the AGI-primitives
requirements pack.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from celery.utils.log import get_task_logger
from sqlalchemy import and_, select

from app.core.celery_app import celery_app
from app.core.database import async_session_factory, set_tenant_context
from app.core.redis import RedisClient
from app.models.cognitive_session import CognitiveSession
from app.models.goal import Goal, GoalNode, GoalEdge
from app.models.memory import MemoryMetadata
from app.schemas.goal import GoalNodeCreateRequest
from app.services.goal_navigator import GoalNavigator
from app.services.goal_planner_agent import GoalPlannerAgent
from app.services.goal_linking_agent import GoalLinkingAgent
from app.services.goal_service import GoalService
from app.services.permission_checker import PermissionChecker


logger = get_task_logger(__name__)


def _run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)


async def _acquire_idempotency_lock(key: str, ttl_seconds: int = 600) -> bool:
    """Best-effort idempotency lock using Redis SET NX.

    If Redis is unavailable, we fail open (still safe due to DB-level upserts).
    """

    try:
        client = await RedisClient.get_client()
        # redis-py asyncio supports set(..., nx=True, ex=...)
        res = await client.set(key, "1", nx=True, ex=int(ttl_seconds))
        return bool(res)
    except Exception:
        return True


@celery_app.task(name="app.tasks.goals.goal_plan_from_session_task")
def goal_plan_from_session_task(
    *,
    org_id: str,
    session_id: str,
    initiator_user_id: str,
    roles: str = "",
    clearance_level: int = 0,
    justification: str = "goal_plan_from_session",
) -> str:
    """Create a conservative Goal proposal from an existing cognitive session.

    This is a deterministic, safe default until GoalPlannerAgent is wired.
    Idempotent per (org_id, session_id).

    Returns:
        goal_id or "skipped".
    """

    async def _run() -> str:
        lock_key = f"idem:goal_plan:{org_id}:{session_id}"
        if not await _acquire_idempotency_lock(lock_key, ttl_seconds=3600):
            return "skipped"

        async with async_session_factory() as db:
            async with db.begin():
                await set_tenant_context(
                    db,
                    initiator_user_id,
                    org_id,
                    roles=roles,
                    clearance_level=int(clearance_level or 0),
                    justification=justification,
                )

                session = await db.get(CognitiveSession, session_id)
                if not session or session.organization_id != org_id:
                    return "skipped"

                # Deterministic proposal: store a "proposed" goal with provenance.
                svc = GoalService(db)
                goal = await svc.create_goal(
                    org_id=org_id,
                    created_by_user_id=initiator_user_id,
                    payload={
                        "owner_type": "user",
                        "owner_id": initiator_user_id,
                        "title": (session.goal or "Untitled goal").strip()[:500],
                        "description": "Created from cognitive session (proposal).",
                        "goal_type": "objective",
                        "status": "proposed",
                        "priority": 2,
                        "due_at": None,
                        "confidence": 0.35,
                        "visibility_scope": "personal",
                        "scope_id": None,
                        "tags": ["proposal", "cognitive_session"],
                        "metadata": {
                            "source": "cognitive_session",
                            "session_id": session_id,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        },
                    },
                )

                # Seed a first actionable node to keep the graph useful.
                await svc.add_node(
                    org_id=org_id,
                    goal_id=goal.id,
                    actor_user_id=initiator_user_id,
                    payload={
                        "parent_node_id": None,
                        "node_type": "task",
                        "title": "Gather initial evidence",
                        "description": "Collect 2-3 supporting memories before execution.",
                        "status": "todo",
                        "priority": 2,
                        "assigned_to_user_id": initiator_user_id,
                        "assigned_to_team_id": None,
                        "ordering": 0,
                        "expected_outputs": {"evidence_memory_ids": []},
                        "success_criteria": ["At least 2 relevant evidence memories linked"],
                        "blockers": None,
                        "confidence": 0.4,
                    },
                )

                return goal.id

    return _run_async(_run())


@celery_app.task(name="app.tasks.goals.goal_plan_from_session_task")
def goal_plan_from_session_task(
    *,
    org_id: str,
    session_id: str,
    initiator_user_id: str,
    roles: str = "",
    clearance_level: int = 0,
    justification: str = "goal_plan_from_session",
) -> str:
    """Propose creating a goal from a cognitive session outcome.
    
    Uses GoalPlannerAgent to analyze the session and suggest goal creation.
    Requires manual approval before goal is actually created.
    
    Returns:
        "proposed" if goal was suggested, "skipped" otherwise.
    """

    async def _run() -> str:
        async with async_session_factory() as db:
            async with db.begin():
                await set_tenant_context(
                    db,
                    initiator_user_id,
                    org_id,
                    roles=roles,
                    clearance_level=int(clearance_level or 0),
                    justification=justification,
                )

                # Validate session exists
                res = await db.execute(
                    select(CognitiveSession).where(
                        CognitiveSession.id == session_id,
                        CognitiveSession.organization_id == org_id,
                    )
                )
                session = res.scalar_one_or_none()
                if not session:
                    return "skipped"

                # Build context from session
                session_context = {
                    "goal": getattr(session, "goal", None) or "",
                    "status": getattr(session, "status", None) or "",
                    "context_snapshot": getattr(session, "context_snapshot", None) or {},
                }

                # Get existing active goals to avoid duplication
                goals_res = await db.execute(
                    select(Goal)
                    .where(
                        Goal.organization_id == org_id,
                        Goal.status.in_(["proposed", "active"]),
                    )
                    .order_by(Goal.updated_at.desc())
                    .limit(10)
                )
                existing_goals = [
                    {
                        "id": g.id,
                        "title": g.title,
                        "goal_type": g.goal_type,
                    }
                    for g in goals_res.scalars().all()
                ]

                # Run agent
                agent = GoalPlannerAgent()
                user_request = f"Session {session_id} completed with status: {session_context['status']}. Goal: {session_context['goal']}"
                
                output = await agent.propose_goal(
                    user_request=user_request,
                    session_context=session_context,
                    existing_goals=existing_goals,
                )

                if not output.create_goal or output.confidence < 0.50:
                    return "skipped"

                # Create goal in "proposed" state (requires manual approval)
                svc = GoalService(db)
                goal = await svc.create_goal(
                    org_id=org_id,
                    actor_user_id=initiator_user_id,
                    title=output.goal.title,
                    description=output.goal.description,
                    goal_type=output.goal.goal_type,
                    status="proposed",  # Always starts as proposed
                    visibility_scope=output.goal.visibility_scope,
                    scope_id=output.goal.scope_id,
                    owner_type="user",
                    owner_id=initiator_user_id,
                    priority=output.goal.priority,
                    confidence=output.confidence,
                    due_at=output.goal.due_at,
                    tags=["auto_proposed", f"session:{session_id}"],
                    extra_metadata={
                        "source": "goal_planner_agent",
                        "source_session_id": session_id,
                        "agent_confidence": output.confidence,
                    },
                )

                # Create nodes if provided
                temp_id_map = {}
                for node in output.nodes:
                    created_node = await svc.add_subgoal(
                        org_id=org_id,
                        actor_user_id=initiator_user_id,
                        goal_id=goal.id,
                        node_type=node.node_type,
                        title=node.title,
                        description=node.description,
                        success_criteria=node.success_criteria,
                        expected_outputs=node.expected_outputs,
                    )
                    temp_id_map[node.temp_id] = created_node.id

                # Create edges
                for edge in output.edges:
                    from_id = temp_id_map.get(edge.from_temp_id)
                    to_id = temp_id_map.get(edge.to_temp_id)
                    if from_id and to_id:
                        db.add(
                            GoalEdge(
                                organization_id=org_id,
                                from_node_id=from_id,
                                to_node_id=to_id,
                                edge_type=edge.edge_type,
                            )
                        )

                await db.flush()
                
                logger.info(f"Goal planner proposed goal {goal.id} from session {session_id}")
                return "proposed"

    return str(_run_async(_run()) or "skipped")


@celery_app.task(name="app.tasks.goals.goal_link_memory_task")
def goal_link_memory_task(
    *,
    org_id: str,
    memory_id: str,
    initiator_user_id: str,
    roles: str = "",
    clearance_level: int = 0,
    justification: str = "goal_link_memory",
) -> int:
    """Intelligently link a new memory to active goals using GoalLinkingAgent.

    Falls back to heuristic tag matching if agent is disabled or fails.
    Idempotent via DB upsert constraint on goal_memory_links.

    Returns:
        Number of links created/updated.
    """

    async def _run() -> int:
        # Idempotency lock (best-effort)
        lock_key = f"goal:link_memory:{org_id}:{memory_id}"
        if not await _acquire_idempotency_lock(lock_key, ttl_seconds=300):
            logger.info(f"Idempotency lock held for {lock_key}; skipping")
            return 0

        async with async_session_factory() as db:
            async with db.begin():
                await set_tenant_context(
                    db,
                    initiator_user_id,
                    org_id,
                    roles=roles,
                    clearance_level=int(clearance_level or 0),
                    justification=justification,
                )

                memory = await db.get(MemoryMetadata, memory_id)
                if not memory or memory.organization_id != org_id:
                    return 0

                # Ensure the actor can read this memory (fail closed).
                perm = PermissionChecker(db)
                decision = await perm.check_memory_access(
                    initiator_user_id,
                    org_id,
                    memory_id,
                    action="read",
                    clearance_level=int(clearance_level or 0),
                )
                if not decision.allowed:
                    return 0

                # Candidate goals: recent active/proposed/blocked.
                res = await db.execute(
                    select(Goal)
                    .where(and_(Goal.organization_id == org_id, Goal.status.in_(["proposed", "active", "blocked"])))
                    .order_by(Goal.updated_at.desc())
                    .limit(25)
                )
                goals = res.scalars().all()

                if not goals:
                    return 0

                # Build memory context for agent
                memory_context = {
                    "id": memory.id,
                    "title": getattr(memory, "title", None) or "",
                    "content_preview": (getattr(memory, "content", None) or "")[:500],
                    "tags": getattr(memory, "tags", None) or [],
                    "classification": getattr(memory, "classification", None) or "internal",
                }

                # Build goals context with nodes
                active_goals = []
                for g in goals:
                    nodes_res = await db.execute(select(GoalNode).where(GoalNode.goal_id == g.id))
                    nodes = nodes_res.scalars().all()

                    active_goals.append({
                        "id": g.id,
                        "title": g.title,
                        "description": g.description,
                        "goal_type": g.goal_type,
                        "tags": g.tags or [],
                        "nodes": [
                            {"id": n.id, "title": n.title, "node_type": n.node_type, "status": n.status}
                            for n in nodes
                        ],
                    })

                # Try LLM agent first
                agent = GoalLinkingAgent()
                output = await agent.suggest_links(memory=memory_context, active_goals=active_goals)

                svc = GoalService(db)
                linked = 0

                # Use agent suggestions if confidence is high enough
                if output.confidence >= 0.50 and output.links:
                    for link in output.links:
                        if link.confidence < 0.50:
                            continue

                        # Validate goal_id belongs to our candidates
                        if link.goal_id not in [g["id"] for g in active_goals]:
                            continue

                        await svc.link_memory(
                            org_id=org_id,
                            actor_user_id=initiator_user_id,
                            goal_id=link.goal_id,
                            memory_id=memory_id,
                            link_type=link.link_type,
                            confidence=link.confidence,
                            node_id=link.node_id,
                            linked_by="agent",
                        )
                        linked += 1
                else:
                    # Fallback to heuristic tag matching
                    mem_tags = set([t.strip().lower() for t in (memory.tags or []) if t])

                    for goal in goals:
                        goal_tags = set([t.strip().lower() for t in (goal.tags or []) if t])
                        overlap = mem_tags.intersection(goal_tags)
                        if not overlap:
                            continue

                        confidence = min(1.0, 0.55 + 0.1 * float(len(overlap)))
                        link_type = "progress" if ("progress" in mem_tags or "milestone" in mem_tags) else "evidence"

                        await svc.link_memory(
                            org_id=org_id,
                            actor_user_id=initiator_user_id,
                            goal_id=goal.id,
                            memory_id=memory_id,
                            link_type=link_type,
                            confidence=confidence,
                            node_id=None,
                            linked_by="auto",
                        )
                        linked += 1

                return linked

    return int(_run_async(_run()) or 0)


@celery_app.task(name="app.tasks.goals.goal_progress_recompute_task")
def goal_progress_recompute_task(
    *,
    org_id: str,
    goal_id: str,
    initiator_user_id: str,
    roles: str = "",
    clearance_level: int = 0,
    justification: str = "goal_progress_recompute",
) -> str:
    """Recompute and persist progress summary into goal.metadata.

    Returns:
        goal_id or "skipped".
    """

    async def _run() -> str:
        async with async_session_factory() as db:
            async with db.begin():
                await set_tenant_context(
                    db,
                    initiator_user_id,
                    org_id,
                    roles=roles,
                    clearance_level=int(clearance_level or 0),
                    justification=justification,
                )

                svc = GoalService(db)
                goal, nodes, _edges, _links, progress = await svc.get_goal_detail_bundle(goal_id=goal_id, org_id=org_id)

                goal.extra_metadata = dict(getattr(goal, "extra_metadata", None) or {})
                goal.extra_metadata["progress"] = {
                    "percent_complete": progress.percent_complete,
                    "completed_nodes": progress.completed_nodes,
                    "total_nodes": progress.total_nodes,
                    "confidence": progress.confidence,
                    "computed_at": datetime.now(timezone.utc).isoformat(),
                }
                goal.updated_at = datetime.now(timezone.utc)
                await db.flush()

                await svc._log(
                    org_id=org_id,
                    goal_id=goal.id,
                    node_id=None,
                    actor_type="system",
                    actor_id=initiator_user_id,
                    action="recompute_progress",
                    details={"progress": goal.extra_metadata["progress"]},
                )

                return goal.id

    return str(_run_async(_run()) or "skipped")


@celery_app.task(name="app.tasks.goals.goal_blocker_detection_task")
def goal_blocker_detection_task(
    *,
    org_id: str,
    goal_id: str,
    initiator_user_id: str,
    roles: str = "",
    clearance_level: int = 0,
    justification: str = "goal_blocker_detection",
) -> str:
    """Compute blockers and persist summary into goal.metadata."""

    async def _run() -> str:
        async with async_session_factory() as db:
            async with db.begin():
                await set_tenant_context(
                    db,
                    initiator_user_id,
                    org_id,
                    roles=roles,
                    clearance_level=int(clearance_level or 0),
                    justification=justification,
                )

                nav = GoalNavigator(db)
                blockers = await nav.compute_blockers(org_id=org_id, goal_id=goal_id)

                goal = await db.get(Goal, goal_id)
                if not goal or goal.organization_id != org_id:
                    return "skipped"

                goal.extra_metadata = dict(getattr(goal, "extra_metadata", None) or {})
                blocker_info = {
                    "blocked_node_ids": [b.id for b in blockers],
                    "count": len(blockers),
                    "computed_at": datetime.now(timezone.utc).isoformat(),
                }
                goal.extra_metadata["blockers"] = blocker_info
                goal.updated_at = datetime.now(timezone.utc)
                await db.flush()

                await GoalService(db)._log(
                    org_id=org_id,
                    goal_id=goal.id,
                    node_id=None,
                    actor_type="system",
                    actor_id=initiator_user_id,
                    action="detect_blockers",
                    details={"blockers": blocker_info},
                )
                
                # Escalate/notify if blockers found
                if blockers:
                    # Log escalation event
                    await GoalService(db)._log(
                        org_id=org_id,
                        goal_id=goal.id,
                        node_id=None,
                        actor_type="system",
                        actor_id=initiator_user_id,
                        action="escalate_blockers",
                        details={
                            "reason": f"Detected {len(blockers)} blocked node(s) requiring attention",
                            "blocked_node_ids": [b.id for b in blockers],
                            "blocked_node_titles": [b.title for b in blockers if b.title],
                        },
                    )
                    
                    # Update goal status to "blocked" if it's currently "active"
                    if goal.status == "active":
                        goal.status = "blocked"
                        goal.status_reason = f"Auto-blocked: {len(blockers)} node(s) detected as blocked"
                        await db.flush()

                return goal.id

    return str(_run_async(_run()) or "skipped")
