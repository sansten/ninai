"""ExecutorAgent for CognitiveLoop.

Executes a PlannerOutput deterministically:
- For each step: if tool is provided, call ToolInvoker (PolicyGuard enforced)
- Writes tool_call_logs via ToolInvoker
- Produces strict ExecutorOutput (safe summaries; no sensitive raw outputs)

Phase 47 additions:
- Retry logic: retryable step failures are retried up to max_retries times
  with exponential backoff (default 2 retries; 1 s, 2 s waits).
- Rollback: if a step declares a rollback_tool in tool_input_hint and it
  fails after all retries, the rollback tool is invoked automatically.
  Rolled-back steps are surfaced as status="rolled_back" in the output.

Design notes:
- Denied tools do not crash execution; step status becomes "denied".
- Tool failures do not crash by default (swallow_exceptions=True) and are surfaced in output.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.schemas.cognitive import ExecutorOutput, ExecutorStepResult, PlannerOutput
from app.services.cognitive_tooling.policy_guard import ToolContext
from app.services.cognitive_tooling.tool_invoker import ToolInvoker

_BACKOFF_BASE = 1.0   # seconds; doubles per retry, capped at 30 s
_DEFAULT_MAX_RETRIES = 2


def _backoff(attempt: int, base: float = _BACKOFF_BASE) -> float:
    return min(base * (2 ** attempt), 30.0)


class ExecutorAgent:
    name = "executor_agent"
    version = "v2"   # bumped for Phase 47 retry/rollback

    def __init__(
        self,
        *,
        tool_invoker: ToolInvoker,
        max_retries: int = _DEFAULT_MAX_RETRIES,
    ):
        self.tool_invoker = tool_invoker
        self.max_retries = max(0, int(max_retries))

    @staticmethod
    def _compute_overall_status(step_results: list[ExecutorStepResult]) -> str:
        # success if no denied/failed (skipped/rolled_back ok)
        if any(r.status == "failed" for r in step_results):
            return "failed"
        if any(r.status == "denied" for r in step_results):
            return "partial"
        return "success"

    async def _invoke_step(
        self,
        *,
        session_id: str,
        iteration_id: str,
        tool_name: str,
        tool_input: dict[str, Any],
        ctx: ToolContext,
    ) -> Any:
        """Invoke a tool once, swallowing exceptions."""
        return await self.tool_invoker.invoke(
            session_id=session_id,
            iteration_id=iteration_id,
            tool_name=tool_name,
            tool_input=tool_input,
            ctx=ctx,
            swallow_exceptions=True,
        )

    async def _invoke_with_retry(
        self,
        *,
        session_id: str,
        iteration_id: str,
        tool_name: str,
        tool_input: dict[str, Any],
        ctx: ToolContext,
    ) -> tuple[Any, int]:
        """Invoke a tool with retry on failure.  Returns (result, attempts_used)."""
        last_result = None
        for attempt in range(self.max_retries + 1):
            try:
                result = await self._invoke_step(
                    session_id=session_id,
                    iteration_id=iteration_id,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    ctx=ctx,
                )
            except Exception as exc:
                # swallow_exceptions=True means this is very rare
                class _FakeResult:
                    status = "failed"
                    tool_call_id = None
                    denial_reason = None
                    error = f"{type(exc).__name__}: {exc}"
                result = _FakeResult()

            last_result = result
            if result.status in ("success", "denied"):
                return result, attempt + 1
            # failed — retry if attempts remain
            if attempt < self.max_retries:
                await asyncio.sleep(_backoff(attempt))

        return last_result, self.max_retries + 1

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
            rollback_tool: str | None = tool_input_hint.pop("rollback_tool", None)

            result, attempts = await self._invoke_with_retry(
                session_id=session_id,
                iteration_id=iteration_id,
                tool_name=tool_name,
                tool_input=tool_input_hint,
                ctx=ctx,
            )

            attempt_note = f" (attempt {attempts}/{self.max_retries + 1})" if attempts > 1 else ""

            if result.status == "success":
                step_results.append(
                    ExecutorStepResult(
                        step_id=step.step_id,
                        status="success",
                        tool_name=tool_name,
                        tool_call_id=result.tool_call_id,
                        summary=f"Tool '{tool_name}' succeeded{attempt_note}.",
                        artifacts={"attempts": attempts},
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
                # Step failed — attempt rollback if declared
                if rollback_tool:
                    rb_result, rb_attempts = await self._invoke_with_retry(
                        session_id=session_id,
                        iteration_id=iteration_id,
                        tool_name=rollback_tool,
                        tool_input={"rolled_back_step": step.step_id},
                        ctx=ctx,
                    )
                    rb_ok = rb_result.status == "success"
                    step_results.append(
                        ExecutorStepResult(
                            step_id=step.step_id,
                            status="rolled_back",
                            tool_name=tool_name,
                            tool_call_id=result.tool_call_id,
                            summary=(
                                f"Tool '{tool_name}' failed{attempt_note}; "
                                f"rollback via '{rollback_tool}' "
                                f"{'succeeded' if rb_ok else 'also failed'}."
                            ),
                            artifacts={
                                "attempts": attempts,
                                "rollback_tool": rollback_tool,
                                "rollback_ok": rb_ok,
                            },
                        )
                    )
                else:
                    step_results.append(
                        ExecutorStepResult(
                            step_id=step.step_id,
                            status="failed",
                            tool_name=tool_name,
                            tool_call_id=result.tool_call_id,
                            summary=f"Tool '{tool_name}' failed{attempt_note}: {result.error or 'failed'}",
                            artifacts={"attempts": attempts},
                        )
                    )
                err = getattr(result, "error", None)
                if err:
                    errors.append(f"{tool_name} failed: {err}")

        overall_status = self._compute_overall_status(step_results)
        return ExecutorOutput(step_results=step_results, overall_status=overall_status, errors=errors)
