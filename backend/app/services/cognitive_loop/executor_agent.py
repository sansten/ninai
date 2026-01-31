"""ExecutorAgent for CognitiveLoop.

Executes a PlannerOutput deterministically:
- For each step: if tool is provided, call ToolInvoker (PolicyGuard enforced)
- Writes tool_call_logs via ToolInvoker
- Produces strict ExecutorOutput (safe summaries; no sensitive raw outputs)

Design notes:
- Denied tools do not crash execution; step status becomes "denied".
- Tool failures do not crash by default (swallow_exceptions=True) and are surfaced in output.
"""

from __future__ import annotations

from typing import Any

from app.schemas.cognitive import ExecutorOutput, ExecutorStepResult, PlannerOutput
from app.services.cognitive_tooling.policy_guard import ToolContext
from app.services.cognitive_tooling.tool_invoker import ToolInvoker


class ExecutorAgent:
    name = "executor_agent"
    version = "v1"

    def __init__(self, *, tool_invoker: ToolInvoker):
        self.tool_invoker = tool_invoker

    @staticmethod
    def _compute_overall_status(step_results: list[ExecutorStepResult]) -> str:
        # success if no denied/failed (skipped ok)
        if any(r.status == "failed" for r in step_results):
            return "failed"
        if any(r.status in ("denied",) for r in step_results):
            return "partial"
        if any(r.status == "skipped" for r in step_results):
            return "success"
        return "success"

    async def execute(
        self,
        *,
        session_id: str,
        iteration_id: str,
        plan: PlannerOutput,
        ctx: ToolContext,
    ) -> ExecutorOutput:
        step_results: list[ExecutorStepResult] = []
        errors: list[str] = []

        for step in plan.steps:
            tool_name = step.tool
            if not tool_name:
                step_results.append(
                    ExecutorStepResult(
                        step_id=step.step_id,
                        status="skipped",
                        tool_name=None,
                        tool_call_id=None,
                        summary="No tool for step; skipped.",
                        artifacts={},
                    )
                )
                continue

            tool_input_hint: dict[str, Any] = dict(step.tool_input_hint or {})

            try:
                result = await self.tool_invoker.invoke(
                    session_id=session_id,
                    iteration_id=iteration_id,
                    tool_name=tool_name,
                    tool_input=tool_input_hint,
                    ctx=ctx,
                    swallow_exceptions=True,
                )
            except Exception as e:
                # Should be rare since swallow_exceptions=True, but be defensive.
                msg = f"Tool '{tool_name}' raised: {type(e).__name__}: {e}"
                errors.append(msg)
                step_results.append(
                    ExecutorStepResult(
                        step_id=step.step_id,
                        status="failed",
                        tool_name=tool_name,
                        tool_call_id=None,
                        summary=msg,
                        artifacts={},
                    )
                )
                continue

            if result.status == "success":
                step_results.append(
                    ExecutorStepResult(
                        step_id=step.step_id,
                        status="success",
                        tool_name=tool_name,
                        tool_call_id=result.tool_call_id,
                        summary=f"Tool '{tool_name}' succeeded.",
                        artifacts={},
                    )
                )
            elif result.status == "denied":
                step_results.append(
                    ExecutorStepResult(
                        step_id=step.step_id,
                        status="denied",
                        tool_name=tool_name,
                        tool_call_id=result.tool_call_id,
                        summary=f"Tool '{tool_name}' denied: {result.denial_reason or 'denied'}",
                        artifacts={},
                    )
                )
                if result.denial_reason:
                    errors.append(f"{tool_name} denied: {result.denial_reason}")
            else:
                step_results.append(
                    ExecutorStepResult(
                        step_id=step.step_id,
                        status="failed",
                        tool_name=tool_name,
                        tool_call_id=result.tool_call_id,
                        summary=f"Tool '{tool_name}' failed: {result.error or 'failed'}",
                        artifacts={},
                    )
                )
                if result.error:
                    errors.append(f"{tool_name} failed: {result.error}")

        overall_status = self._compute_overall_status(step_results)
        return ExecutorOutput(step_results=step_results, overall_status=overall_status, errors=errors)
