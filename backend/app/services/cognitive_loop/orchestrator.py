"""LoopOrchestrator for CognitiveLoop.

Runs a bounded Planner -> Executor -> Critic loop and persists each iteration.

Security notes:
- Evidence retrieval must be RLS-safe and authorized.
- Tool calls must go through PolicyGuard/ToolInvoker.

Idempotency (best-effort):
- If the session is already succeeded/failed/aborted, no work is done.
- If an iteration row already exists with non-empty critique_json, it is treated
  as completed and used to decide next action.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.schemas.cognitive import ExecutorOutput, PlannerOutput, CriticOutput
from app.services.cognitive_loop.repository import CognitiveLoopRepository
from app.services.cognitive_loop.evidence_service import CognitiveEvidenceService
from app.services.cognitive_loop.planner_agent import PlannerAgent
from app.services.cognitive_loop.executor_agent import ExecutorAgent
from app.services.cognitive_loop.critic_agent import CriticAgent
from app.services.simulation_report_service import SimulationReportService
from app.services.simulation_service import SimulationService
from app.services.cognitive_tooling.policy_guard import ToolContext
from app.services.adaptive_strategy_service import AdaptiveStrategyService
from app.services.strategy_learning_service import StrategyLearningService, _classify_goal_type
from app.services.cognitive_context_aggregator import CognitiveContextAggregator


@dataclass(frozen=True)
class OrchestratorConfig:
    max_iterations: int = 3


class LoopOrchestrator:
    def __init__(
        self,
        *,
        repo: CognitiveLoopRepository,
        evidence: CognitiveEvidenceService,
        planner: PlannerAgent,
        simulator: SimulationService,
        simulation_reports: SimulationReportService,
        executor: ExecutorAgent,
        critic: CriticAgent,
        available_tools: list[str],
        self_model_summary: dict[str, Any] | None = None,
        adaptive_strategy: AdaptiveStrategyService | None = None,
        strategy_learning: StrategyLearningService | None = None,
        context_aggregator: CognitiveContextAggregator | None = None,
        config: OrchestratorConfig | None = None,
    ) -> None:
        self.repo = repo
        self.evidence = evidence
        self.planner = planner
        self.simulator = simulator
        self.simulation_reports = simulation_reports
        self.executor = executor
        self.critic = critic
        self.available_tools = available_tools
        self.self_model_summary = self_model_summary or {}
        self.adaptive_strategy = adaptive_strategy
        self.strategy_learning = strategy_learning
        self.context_aggregator = context_aggregator
        self.config = config or OrchestratorConfig()

    def _apply_safe_simulation_patch(self, *, plan: PlannerOutput, patch: dict[str, Any]) -> PlannerOutput:
        """Apply a conservative subset of Simulation recommended patches.

        Safe patch rules enforced here (subset):
        - Only allow adding steps that use already-available tools.
        - Never introduce a tool outside available_tools.
        - Allow removing steps by step_id.

        If anything looks off, fail closed by returning the original plan.
        """

        try:
            remove_steps = set((patch or {}).get("remove_steps") or [])
            add_steps = list((patch or {}).get("add_steps") or [])
        except Exception:
            return plan

        allowed_tools = set(self.available_tools or [])
        new_steps = [s for s in (plan.steps or []) if s.step_id not in remove_steps]
        existing_ids = {s.step_id for s in new_steps}

        for s in add_steps:
            if not isinstance(s, dict):
                continue
            step_id = str(s.get("step_id") or "").strip()
            if not step_id or step_id in existing_ids:
                continue
            tool = s.get("tool")
            if tool is not None and str(tool) not in allowed_tools:
                # unsafe: introducing unknown tool
                continue

            new_steps.append(
                {
                    "step_id": step_id,
                    "action": str(s.get("action") or ""),
                    "tool": tool,
                    "tool_input_hint": dict(s.get("tool_input_hint") or {}),
                    "expected_output": str(s.get("expected_output") or ""),
                    "success_criteria": list(s.get("success_criteria") or []),
                    "risk_notes": [],
                }
            )
            existing_ids.add(step_id)

        # Keep required_tools consistent with steps (but do not add unknown tools).
        required = list(plan.required_tools or [])
        required_set = set(required)
        for s in new_steps:
            tool = getattr(s, "tool", None) if not isinstance(s, dict) else s.get("tool")
            if tool and tool in allowed_tools and tool not in required_set:
                required.append(tool)
                required_set.add(tool)

        return PlannerOutput(
            objective=plan.objective,
            assumptions=list(plan.assumptions or []),
            constraints=list(plan.constraints or []),
            required_tools=required,
            steps=[x.model_dump() if hasattr(x, "model_dump") else x for x in new_steps],
            stop_conditions=list(plan.stop_conditions or []),
            confidence=float(plan.confidence or 0.0),
        )

    async def run(self, *, session_id: str, tool_ctx: ToolContext) -> str:
        sess = await self.repo.get_session(session_id)
        if sess is None:
            raise ValueError(f"CognitiveSession not found: {session_id}")

        if sess.status in ("succeeded", "failed", "aborted"):
            return sess.status

        await self.repo.save_session_status(sess, "running")

        goal = str(getattr(sess, "goal", "") or "").strip()
        if not goal:
            await self.repo.save_session_status(sess, "failed")
            return "failed"

        # Pre-classify goal type and domain for strategy learning
        goal_type = _classify_goal_type(goal)
        session_domain = str((getattr(sess, "context_snapshot", None) or {}).get("domain", "general")).strip() or "general"

        # Fetch strategy hints and cognitive context sequentially. Both helpers
        # use the same AsyncSession under the hood, and concurrent DB use on a
        # single session can leave transactions in a failed state.
        @asynccontextmanager
        async def _optional_savepoint():
            begin_nested = getattr(getattr(self.repo, "session", None), "begin_nested", None)
            if callable(begin_nested):
                async with begin_nested():
                    yield
                return
            yield

        async def _fetch_strategy_hints() -> dict[str, Any] | None:
            if not self.strategy_learning:
                return None
            try:
                async with _optional_savepoint():
                    return await self.strategy_learning.get_strategy_hints(
                        org_id=tool_ctx.org_id,
                        goal_type=goal_type,
                        domain=session_domain,
                    )
            except Exception:
                return None

        async def _fetch_cognitive_context() -> dict[str, Any] | None:
            if not self.context_aggregator:
                return None
            try:
                async with _optional_savepoint():
                    return await self.context_aggregator.aggregate(
                        org_id=tool_ctx.org_id,
                        goal=goal,
                    ) or None
            except Exception:
                return None

        strategy_hints = await _fetch_strategy_hints()
        cognitive_context = await _fetch_cognitive_context()

        final_status = "failed"

        for iteration_num in range(1, int(self.config.max_iterations) + 1):
            t0 = datetime.now(timezone.utc)

            existing = await self.repo.get_iteration(session_id=session_id, iteration_num=iteration_num)
            if existing is not None and (existing.critique_json or {}):
                evaluation = str(existing.evaluation or "")
                if evaluation == "pass":
                    final_status = "succeeded"
                    break
                if evaluation in ("retry", "needs_evidence"):
                    continue
                final_status = "failed"
                break

            iteration = existing or await self.repo.create_iteration(session_id=session_id, iteration_num=iteration_num)

            base_limit = 10
            multiplier = 1
            try:
                multiplier = int((self.self_model_summary or {}).get("recommended_evidence_multiplier") or 1)
            except Exception:
                multiplier = 1
            multiplier = max(1, min(3, multiplier))
            evidence_limit = max(1, min(30, base_limit * multiplier))

            evidence_cards = await self.evidence.retrieve_evidence(goal=goal, limit=evidence_limit, hybrid=True)

            # Fetch adaptive tool recommendations for this iteration
            tool_recommendations: list[dict[str, Any]] = []
            if self.adaptive_strategy:
                try:
                    tool_recommendations = await self.adaptive_strategy.get_tool_recommendations(
                        org_id=tool_ctx.org_id,
                        goal_id=session_id,
                        goal_type=goal_type,
                        domain=session_domain,
                        limit=5,
                    )
                except Exception:
                    tool_recommendations = []

            plan: PlannerOutput = await self.planner.plan(
                goal=goal,
                evidence_cards=evidence_cards,
                available_tools=self.available_tools,
                self_model_summary=self.self_model_summary,
                tool_recommendations=tool_recommendations,
                strategy_hints=strategy_hints,
                cognitive_context=cognitive_context,
            )

            simulation = self.simulator.simulate_plan(plan=plan, evidence_cards=evidence_cards)
            # Persist for auditability; do not block the loop on failures.
            simulation_report_id: str | None = None
            try:
                row = await self.simulation_reports.create(
                    org_id=tool_ctx.org_id,
                    session_id=session_id,
                    report={
                        "iteration_num": int(iteration_num),
                        "planner_confidence": float(plan.confidence or 0.0),
                        "simulation": simulation.model_dump(),
                    },
                )
                simulation_report_id = str(row.id)
            except Exception:
                simulation_report_id = None

            plan = self._apply_safe_simulation_patch(plan=plan, patch=simulation.recommended_plan_patch.model_dump())

            execution: ExecutorOutput = await self.executor.execute(
                session_id=session_id,
                iteration_id=str(iteration.id),
                plan=plan,
                ctx=tool_ctx,
            )

            critique: CriticOutput = await self.critic.critique(
                goal=goal,
                plan=plan.model_dump(),
                execution=execution.model_dump(),
                evidence_cards=evidence_cards,
                simulation=simulation.model_dump(),
            )

            evaluation = critique.evaluation

            t1 = datetime.now(timezone.utc)
            metrics: dict[str, Any] = {
                "duration_ms": (t1 - t0).total_seconds() * 1000.0,
                "tool_steps": sum(1 for s in plan.steps if s.tool),
                "errors_count": len(execution.errors or []),
                "confidence": float(getattr(critique, "confidence", 0.0) or 0.0),
                "evidence_memory_ids": [str(c.get("id")) for c in (evidence_cards or []) if c.get("id")],
                "simulation_report_id": simulation_report_id,
                "simulation": {
                    "success_probability": float(simulation.plan_risk.success_probability),
                    "policy_violation_probability": float(simulation.plan_risk.policy_violation_probability),
                    "tool_failure_probability": float(simulation.plan_risk.tool_failure_probability),
                    "confidence": float(simulation.confidence),
                },
            }

            await self.repo.finalize_iteration(
                iteration=iteration,
                plan_json=plan.model_dump(),
                execution_json=execution.model_dump(),
                critique_json=critique.model_dump(),
                evaluation=evaluation,
                metrics=metrics,
                finished_at=t1,
            )

            if evaluation == "pass":
                final_status = "succeeded"
                break
            if evaluation in ("retry", "needs_evidence"):
                final_status = "failed"  # unless later passes
                continue
            final_status = "failed"
            break

        await self.repo.save_session_status(sess, final_status)

        # Post-session: ingest outcome into strategy library (fire-and-forget)
        if self.strategy_learning and final_status in ("succeeded", "failed", "aborted"):
            try:
                await self.strategy_learning.ingest_session_outcome(
                    org_id=tool_ctx.org_id,
                    session_id=session_id,
                )
            except Exception:
                pass  # non-blocking; learning failure must not affect session outcome

        return final_status
