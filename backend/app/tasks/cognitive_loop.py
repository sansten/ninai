"""Cognitive Loop Celery tasks.

Implements the cognitive_loop_task(session_id) entrypoint described in the AGI path doc.

Important:
- Celery tasks are synchronous entrypoints; we bridge into async via asyncio.run.
- Tenant context (RLS) MUST be set for every DB session.
- No RBAC bypass: the task runs as the initiating user (or a service user only if explicitly supplied).
"""

from __future__ import annotations

import asyncio

from celery.utils.log import get_task_logger

from app.core.celery_app import celery_app
from app.core.database import async_session_factory, set_tenant_context
from app.core.config import settings
from app.services.cognitive_loop.repository import CognitiveLoopRepository
from app.services.cognitive_loop.orchestrator import LoopOrchestrator, OrchestratorConfig
from app.services.cognitive_loop.evidence_service import CognitiveEvidenceService
from app.services.cognitive_loop.evaluation_report_service import EvaluationReportService
from app.services.cognitive_loop.planner_agent import PlannerAgent
from app.services.cognitive_loop.executor_agent import ExecutorAgent
from app.services.cognitive_loop.critic_agent import CriticAgent
from app.services.cognitive_loop.tools import register_builtin_tools
from app.services.cognitive_tooling.tool_registry import ToolRegistry
from app.services.cognitive_tooling.policy_guard import PolicyGuard, ToolContext
from app.services.cognitive_tooling.tool_call_log_service import ToolCallLogService
from app.services.cognitive_tooling.tool_invoker import ToolInvoker
from app.services.agent_scheduler_service import AgentSchedulerService
from app.services.memory_service import MemoryService
from app.services.permission_checker import PermissionChecker
from app.services.self_model_service import SelfModelService
from app.services.simulation_report_service import SimulationReportService
from app.services.simulation_service import SimulationService
from app.tasks.self_model import self_model_recompute_task
from app.services.goal_service import GoalService
from app.tasks.goals import goal_progress_recompute_task


logger = get_task_logger(__name__)


def _broker_enabled() -> bool:
    broker = celery_app.conf.broker_url
    return bool(broker) and not str(broker).startswith("memory://")


@celery_app.task(name="app.tasks.cognitive_loop.cognitive_loop_task")
def cognitive_loop_task(
    *,
    org_id: str,
    session_id: str,
    initiator_user_id: str,
    process_id: str | None = None,
    roles: str = "",
    clearance_level: int = 0,
    justification: str = "cognitive_loop",
    max_iterations: int = 3,
) -> str:
    """Run the CognitiveLoop for a previously-created session."""

    async def _run() -> str:
        async with async_session_factory() as db:
            scheduler = AgentSchedulerService(db)
            status: str = "failed"
            error_reason: str = ""

            try:
                async with db.begin():
                    await set_tenant_context(
                        db,
                        initiator_user_id,
                        org_id,
                        roles=roles,
                        clearance_level=int(clearance_level or 0),
                        justification=justification,
                    )

                    repo = CognitiveLoopRepository(db)

                    memory_service = MemoryService(
                        db,
                        user_id=initiator_user_id,
                        org_id=org_id,
                        clearance_level=int(clearance_level or 0),
                    )

                    evidence = CognitiveEvidenceService(memory_service)

                    # Tooling setup
                    registry = ToolRegistry()
                    register_builtin_tools(registry=registry, memory_service=memory_service)

                    permission_checker = PermissionChecker(db)
                    guard = PolicyGuard(permission_checker)
                    log_service = ToolCallLogService(db)
                    invoker = ToolInvoker(registry=registry, guard=guard, log_service=log_service)

                    executor = ExecutorAgent(tool_invoker=invoker)

                    simulator = SimulationService()
                    simulation_reports = SimulationReportService(db)

                    # SelfModel summary influences planning/evidence. Fail-closed to defaults.
                    self_model_summary: dict = {}
                    try:
                        sm = SelfModelService(db)
                        summary = await sm.get_planner_summary(org_id=org_id)
                        self_model_summary = {
                            "unreliable_tools": list(summary.unreliable_tools or []),
                            "low_confidence_domains": list(summary.low_confidence_domains or []),
                            "recommended_evidence_multiplier": int(summary.recommended_evidence_multiplier or 1),
                        }
                    except Exception:
                        self_model_summary = {}

                    orchestrator = LoopOrchestrator(
                        repo=repo,
                        evidence=evidence,
                        planner=PlannerAgent(),
                        simulator=simulator,
                        simulation_reports=simulation_reports,
                        executor=executor,
                        critic=CriticAgent(),
                        available_tools=["memory.search"],
                        self_model_summary=self_model_summary,
                        config=OrchestratorConfig(max_iterations=int(max_iterations or 3)),
                    )

                    status = await orchestrator.run(
                        session_id=session_id,
                        tool_ctx=ToolContext(
                            user_id=initiator_user_id,
                            org_id=org_id,
                            classification=None,
                            clearance_level=int(clearance_level or 0),
                            justification=justification,
                        ),
                    )

                    if status in {"succeeded", "failed", "aborted"}:
                        await EvaluationReportService(db).generate_for_session(session_id=session_id)

                        # If session is attached to a goal, link evidence memories for traceability.
                        try:
                            sess = await repo.get_session(session_id)
                            goal_id = str(getattr(sess, "goal_id", "") or "").strip() if sess else ""
                            if goal_id:
                                # If succeeded, mark a single in-progress node as done.
                                if status == "succeeded":
                                    try:
                                        from sqlalchemy import select
                                        from app.models.goal import GoalNode

                                        order_by = (
                                            GoalNode.priority.desc(),
                                            GoalNode.ordering.asc(),
                                            GoalNode.created_at.asc(),
                                        )

                                        # 1) Prefer completing an explicitly in-progress node.
                                        nres = await db.execute(
                                            select(GoalNode)
                                            .where(
                                                GoalNode.organization_id == org_id,
                                                GoalNode.goal_id == goal_id,
                                                GoalNode.status == "in_progress",
                                            )
                                            .order_by(*order_by)
                                            .limit(1)
                                        )
                                        node = nres.scalar_one_or_none()

                                        # 2) Otherwise, complete a single todo node assigned to the initiator.
                                        if node is None:
                                            nres2 = await db.execute(
                                                select(GoalNode)
                                                .where(
                                                    GoalNode.organization_id == org_id,
                                                    GoalNode.goal_id == goal_id,
                                                    GoalNode.status == "todo",
                                                    GoalNode.assigned_to_user_id == initiator_user_id,
                                                )
                                                .order_by(*order_by)
                                                .limit(1)
                                            )
                                            node = nres2.scalar_one_or_none()

                                        # 3) Otherwise, complete any single todo node.
                                        if node is None:
                                            nres3 = await db.execute(
                                                select(GoalNode)
                                                .where(
                                                    GoalNode.organization_id == org_id,
                                                    GoalNode.goal_id == goal_id,
                                                    GoalNode.status == "todo",
                                                )
                                                .order_by(*order_by)
                                                .limit(1)
                                            )
                                            node = nres3.scalar_one_or_none()

                                        if node is not None and str(getattr(node, "status", "")) != "done":
                                            svc = GoalService(db)
                                            await svc.update_node_status(
                                                node_id=str(node.id),
                                                org_id=org_id,
                                                actor_user_id=initiator_user_id,
                                                status_value="done",
                                            )
                                    except Exception:
                                        logger.exception("GoalGraph node completion update failed")

                                latest = await EvaluationReportService(db).get_latest_for_session(session_id=session_id)
                                evidence_ids = list((latest.report or {}).get("evidence_memory_ids") or []) if latest else []
                                svc = GoalService(db)
                                for mid in evidence_ids:
                                    if not mid:
                                        continue
                                    await svc.link_memory(
                                        org_id=org_id,
                                        actor_user_id=initiator_user_id,
                                        goal_id=goal_id,
                                        memory_id=str(mid),
                                        link_type="evidence",
                                        confidence=0.7,
                                        node_id=None,
                                        linked_by="agent",
                                    )

                                # Progress recompute can be slightly heavier; enqueue if broker exists.
                                if _broker_enabled():
                                    goal_progress_recompute_task.delay(
                                        org_id=org_id,
                                        goal_id=goal_id,
                                        initiator_user_id=initiator_user_id,
                                        roles=roles,
                                        clearance_level=int(clearance_level or 0),
                                        justification="cognitive_loop_goal_progress",
                                    )
                        except Exception:
                            logger.exception("GoalGraph post-run linking failed")

                        # SelfModel event ingestion is cheap + idempotent.
                        try:
                            await SelfModelService(db).ingest_from_session(org_id=org_id, session_id=session_id)
                        except Exception:
                            logger.exception("SelfModel ingest_from_session failed")

                        # Recompute runs in its own task to keep the loop responsive.
                        if _broker_enabled():
                            self_model_recompute_task.delay(org_id=org_id)

            except Exception as exc:  # noqa: BLE001
                error_reason = str(exc)
                raise
            finally:
                if process_id:
                    try:
                        if status == "succeeded":
                            await scheduler.mark_succeeded(process_id=process_id, scopes={"scheduler.update"})
                        elif status in {"failed", "aborted"}:
                            await scheduler.mark_failed(
                                process_id=process_id,
                                reason=error_reason or status,
                                scopes={"scheduler.update"},
                            )
                        else:
                            await scheduler.mark_blocked(process_id=process_id, reason=status, scopes={"scheduler.update"})
                    except Exception:
                        logger.exception("AgentProcess finalization failed")

            return status

    try:
        return asyncio.run(_run())
    except RuntimeError as exc:
        # Only handle the specific case where asyncio.run() is called from within
        # an already-running event loop. Other RuntimeErrors (e.g. asyncpg
        # cross-loop errors) should surface to the caller.
        if "asyncio.run() cannot be called from a running event loop" not in str(exc):
            raise

        # In a running loop we can't blockingly run_until_complete. Execute the
        # coroutine in a dedicated thread with its own event loop.
        import threading

        result: dict[str, str] = {}
        error: dict[str, BaseException] = {}

        def _thread_target() -> None:
            try:
                result["value"] = asyncio.run(_run())
            except BaseException as e:  # noqa: BLE001
                error["error"] = e

        t = threading.Thread(target=_thread_target, name="cognitive_loop_task_asyncio")
        t.start()
        t.join()

        if "error" in error:
            raise error["error"]
        return result["value"]
