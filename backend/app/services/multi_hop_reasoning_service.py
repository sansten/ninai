"""Multi-Hop Reasoning Chain Service — Gap F.

The planner decomposes goals into independent parallel subtasks.  It cannot
chain steps: "first check X, use that result to query Y, then decide Z."
Intermediate outputs are discarded rather than fed into the next step.

This service closes that gap:
  - Accepts a chain definition: list of ChainStep with an input_template that
    can reference prior steps' outputs via {step_0_output}, {step_1_output} etc.
  - Executes each step in sequence using CognitiveGatewayService verbs.
  - Passes the previous step's output as context into the next step's query.
  - Returns a ChainResult with per-step outputs and a final synthesis.

Design rules:
- Pure service: no DB session.  Each step calls CognitiveGatewayService which
  is already a pure service.
- Steps can be: "plan" (decompose sub-goal), "decide" (evaluate evidence),
  or "read" (retrieve relevant memories).
- Maximum chain depth is capped at _MAX_HOPS to prevent runaway execution.
- Each step records its input, output, and confidence so the chain is auditable.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from app.services.cognitive_gateway_service import CognitiveGatewayService


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_HOPS = 6       # cap chain depth
_STEP_TYPES = frozenset({"plan", "decide", "read"})


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ChainStep:
    """One step in a multi-hop reasoning chain."""
    step_type: str          # "plan" | "decide" | "read"
    input_template: str     # may contain {step_N_output} placeholders
    # Optional static content / enrichment for decide steps
    content: str = ""
    enrichment: dict[str, Any] = field(default_factory=dict)
    memories: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class StepResult:
    """Output of one executed chain step."""
    index: int
    step_type: str
    resolved_input: str     # input after placeholder substitution
    output: str             # primary output string
    confidence: float
    raw: dict[str, Any]     # full gateway result as dict


@dataclass
class ChainResult:
    """Full output of a multi-hop reasoning execution."""
    goal: str
    steps_executed: int
    step_results: list[StepResult]
    final_output: str       # last step's output
    chain_confidence: float  # geometric mean of step confidences
    truncated: bool          # True if chain was capped at _MAX_HOPS


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class MultiHopReasoningService:
    """Execute a sequential chain of reasoning steps.

    Each step can reference prior outputs via {step_N_output} in its
    input_template.  This allows goals like:

        Step 0 (read):   "find all incidents related to database latency"
        Step 1 (decide): "given {step_0_output}, is this a systemic failure?"
        Step 2 (plan):   "plan remediation for {step_1_output}"

    Usage:
        svc = MultiHopReasoningService()
        result = svc.execute(
            goal="investigate and remediate the database latency spike",
            steps=[
                ChainStep("read", "database latency incidents", memories=pool),
                ChainStep("decide", "is {step_0_output} systemic?", content="{step_0_output}"),
                ChainStep("plan", "plan remediation for {step_1_output}"),
            ],
        )
    """

    def __init__(self, *, gateway: CognitiveGatewayService | None = None) -> None:
        self._gateway = gateway or CognitiveGatewayService()

    def execute(
        self,
        *,
        goal: str,
        steps: list[ChainStep],
    ) -> ChainResult:
        """Execute the chain, bridging to async gateway calls.

        Parameters
        ----------
        goal:   High-level goal driving the chain (for logging/audit).
        steps:  Ordered list of ChainStep definitions.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # Called from within a running event loop (e.g. pytest-asyncio).
            # Schedule a coroutine and block via a thread so we don't nest loops.
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self._execute_async(goal=goal, steps=steps))
                return future.result()
        else:
            return asyncio.run(self._execute_async(goal=goal, steps=steps))

    async def _execute_async(
        self,
        *,
        goal: str,
        steps: list[ChainStep],
    ) -> ChainResult:
        if not steps:
            return ChainResult(
                goal=goal,
                steps_executed=0,
                step_results=[],
                final_output="",
                chain_confidence=0.0,
                truncated=False,
            )

        step_outputs: list[str] = []
        step_results: list[StepResult] = []
        truncated = False

        for idx, step in enumerate(steps[:_MAX_HOPS]):
            if idx >= _MAX_HOPS:
                truncated = True
                break

            resolved_input = self._resolve_template(
                step.input_template, step_outputs
            )
            resolved_content = self._resolve_template(step.content, step_outputs)

            sr = await self._execute_step(
                index=idx,
                step=step,
                resolved_input=resolved_input,
                resolved_content=resolved_content,
            )
            step_results.append(sr)
            step_outputs.append(sr.output)

        if len(steps) > _MAX_HOPS:
            truncated = True

        final_output = step_results[-1].output if step_results else ""
        chain_confidence = self._chain_confidence(step_results)

        return ChainResult(
            goal=goal,
            steps_executed=len(step_results),
            step_results=step_results,
            final_output=final_output,
            chain_confidence=chain_confidence,
            truncated=truncated,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_template(template: str, prior_outputs: list[str]) -> str:
        """Substitute {step_N_output} placeholders with actual prior outputs."""
        result = template
        for i, output in enumerate(prior_outputs):
            result = result.replace(f"{{step_{i}_output}}", output)
        return result

    async def _execute_step(
        self,
        *,
        index: int,
        step: ChainStep,
        resolved_input: str,
        resolved_content: str,
    ) -> StepResult:
        step_type = step.step_type if step.step_type in _STEP_TYPES else "plan"

        plan_method = self._gateway.plan
        decide_method = self._gateway.decide
        read_method = self._gateway.read

        if step_type == "plan":
            call = plan_method(goal=resolved_input)
            gw_result = (await call) if asyncio.iscoroutine(call) else call
            output = " → ".join(
                s.get("title", s.get("description", "")) for s in gw_result.steps
            )
            confidence = gw_result.confidence
            raw: dict[str, Any] = {
                "steps": gw_result.steps,
                "step_count": gw_result.step_count,
                "blocking_step": gw_result.blocking_step,
            }

        elif step_type == "decide":
            content = resolved_content or resolved_input
            enrichment = dict(step.enrichment)
            call = decide_method(content=content, enrichment=enrichment)
            gw_result = (await call) if asyncio.iscoroutine(call) else call
            output = f"{gw_result.decision}: {gw_result.action_recommended or ''}"
            confidence = gw_result.confidence
            raw = {
                "decision": gw_result.decision,
                "action_recommended": gw_result.action_recommended,
                "agents_run": gw_result.agents_run,
            }

        else:  # "read"
            call = read_method(
                query=resolved_input,
                memories=list(step.memories),
                limit=5,
            )
            gw_result = (await call) if asyncio.iscoroutine(call) else call
            output = "; ".join(
                str(m.get("content", ""))[:120] for m in gw_result.memories[:3]
            )
            confidence = 0.70
            raw = {"memories": gw_result.memories, "count": len(gw_result.memories)}

        return StepResult(
            index=index,
            step_type=step_type,
            resolved_input=resolved_input,
            output=output.strip() or resolved_input,
            confidence=confidence,
            raw=raw,
        )

    @staticmethod
    def _chain_confidence(step_results: list[StepResult]) -> float:
        """Geometric mean of per-step confidences (penalises weak links)."""
        if not step_results:
            return 0.0
        product = 1.0
        for sr in step_results:
            product *= max(0.01, sr.confidence)
        return round(product ** (1.0 / len(step_results)), 4)
