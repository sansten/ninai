"""Meta-Cognitive Planning Agent — Phase 36.

Executive control layer that decides how deeply and carefully the agent should
think before acting.  Wraps MetaCognitiveService with the standard agent
interface, LLM fallback, and structured output validation.

Inputs (via context["memory"]["enrichment"]):
  - query:              str  — the incoming question or task
  - uncertainty_level:  str  — "low" | "medium" | "high" from UncertaintyReportingAgent
  - budget_confidence:  float — from AdaptiveEnrichmentBudgetAgent
  - credibility_score:  float — from CredibilityAgent

Inputs (via context["runtime"]):
  - time_budget_seconds: int — wall-clock deadline for this run (default 30)

Outputs:
  - reasoning_mode:      "heuristic" | "mixed" | "deliberative" | "escalate"
  - budget_allocated:    int   — max retrieval items approved
  - confidence_threshold: float — minimum confidence before answer is trusted
  - seek_validation:     bool  — should the answer be cross-checked?
  - complexity_score:    float — estimated query complexity 0..1
  - strategy_selected:   str   — mirrors reasoning_mode
  - rationale:           str   — "heuristic" | "llm"
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.agents.llm.ollama_breaker import create_ollama_client
from app.core.config import settings
from app.services.meta_cognitive_service import MetaCognitiveService


# ---------------------------------------------------------------------------
# Reasoning-mode aliases
# ---------------------------------------------------------------------------

_VALID_MODES = frozenset({"heuristic", "mixed", "deliberative", "escalate"})

_DEFAULT_CONFIDENCE_THRESHOLD = 0.70
_DEFAULT_TIME_BUDGET = 30


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _uncertainty_to_complexity_boost(uncertainty_level: str) -> float:
    """Map UncertaintyReportingAgent level to a complexity addend."""
    level = str(uncertainty_level or "").strip().lower()
    if level == "high":
        return 0.20
    if level == "medium":
        return 0.10
    return 0.0


def _derive_confidence_threshold(
    complexity: float,
    budget_confidence: float,
    uncertainty_level: str,
) -> float:
    """Compute the confidence threshold the agent must hit before trusting output.

    Higher complexity and lower budget_confidence → stricter threshold.
    """
    base = _DEFAULT_CONFIDENCE_THRESHOLD
    # Raise bar for complex queries
    base += round(complexity * 0.15, 4)
    # Lower bar slightly when we have high budget confidence (well-calibrated run)
    base -= round((budget_confidence or 0.0) * 0.05, 4)
    # Raise bar further for high-uncertainty queries
    level = str(uncertainty_level or "").strip().lower()
    if level == "high":
        base += 0.05
    return round(min(0.95, max(0.50, base)), 4)


def run_heuristic(
    query: str,
    uncertainty_level: str,
    budget_confidence: float,
    time_budget_seconds: int,
) -> dict[str, Any]:
    """Heuristic meta-cognitive plan — no LLM required."""
    svc = MetaCognitiveService()

    import asyncio

    async def _compute() -> dict[str, Any]:
        complexity_raw = await svc.estimate_query_complexity(query)
        boost = _uncertainty_to_complexity_boost(uncertainty_level)
        complexity = round(min(1.0, complexity_raw + boost), 3)

        plan = await svc.select_strategy(
            query=query,
            complexity=complexity,
            time_budget_seconds=time_budget_seconds,
            confidence_threshold=_derive_confidence_threshold(
                complexity, budget_confidence, uncertainty_level
            ),
        )

        return {
            "reasoning_mode": plan["strategy_selected"],
            "budget_allocated": plan["retrieval_budget"],
            "confidence_threshold": plan["confidence_threshold"],
            "seek_validation": plan["verification_required"],
            "complexity_score": complexity,
            "strategy_selected": plan["strategy_selected"],
            "rationale": "heuristic",
        }

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, _compute())
                return future.result(timeout=5)
        return loop.run_until_complete(_compute())
    except Exception:
        # Safe fallback — conservative mixed strategy
        return _safe_default(query)


async def run_heuristic_async(
    query: str,
    uncertainty_level: str,
    budget_confidence: float,
    time_budget_seconds: int,
) -> dict[str, Any]:
    """Async variant used inside the agent's run() method."""
    svc = MetaCognitiveService()

    complexity_raw = await svc.estimate_query_complexity(query)
    boost = _uncertainty_to_complexity_boost(uncertainty_level)
    complexity = round(min(1.0, complexity_raw + boost), 3)

    plan = await svc.select_strategy(
        query=query,
        complexity=complexity,
        time_budget_seconds=time_budget_seconds,
        confidence_threshold=_derive_confidence_threshold(
            complexity, budget_confidence, uncertainty_level
        ),
    )

    return {
        "reasoning_mode": plan["strategy_selected"],
        "budget_allocated": plan["retrieval_budget"],
        "confidence_threshold": plan["confidence_threshold"],
        "seek_validation": plan["verification_required"],
        "complexity_score": complexity,
        "strategy_selected": plan["strategy_selected"],
        "rationale": "heuristic",
    }


def _safe_default(query: str) -> dict[str, Any]:
    """Return a safe conservative default when all else fails."""
    return {
        "reasoning_mode": "mixed",
        "budget_allocated": 20,
        "confidence_threshold": _DEFAULT_CONFIDENCE_THRESHOLD,
        "seek_validation": False,
        "complexity_score": 0.5,
        "strategy_selected": "mixed",
        "rationale": "heuristic",
    }


def _parse_llm_output(resp: Any) -> dict[str, Any] | None:
    """Validate and normalize LLM JSON output; return None if invalid."""
    if not isinstance(resp, dict):
        return None

    mode = resp.get("reasoning_mode") or resp.get("strategy_selected")
    if mode not in _VALID_MODES:
        return None

    budget = resp.get("budget_allocated")
    if not isinstance(budget, int) or budget < 1:
        return None

    threshold = resp.get("confidence_threshold")
    if not isinstance(threshold, (int, float)) or not (0.0 <= float(threshold) <= 1.0):
        return None

    return {
        "reasoning_mode": str(mode),
        "budget_allocated": int(budget),
        "confidence_threshold": round(float(threshold), 4),
        "seek_validation": bool(resp.get("seek_validation", False)),
        "complexity_score": round(float(resp.get("complexity_score", 0.5)), 3),
        "strategy_selected": str(mode),
        "rationale": "llm",
    }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class MetaCognitivePlanningAgent(BaseAgent):
    """Phase 36: Decide reasoning depth and retrieval budget per query.

    Simple queries get fast heuristic paths; hard or irreversible queries
    trigger deep deliberative reasoning with validation gating.
    """

    name = "MetaCognitivePlanningAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return ["UncertaintyReportingAgent", "AdaptiveEnrichmentBudgetAgent"]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}

        mode = outputs.get("reasoning_mode")
        if mode not in _VALID_MODES:
            raise ValueError(f"reasoning_mode must be one of {_VALID_MODES}, got {mode!r}")

        budget = outputs.get("budget_allocated")
        if not isinstance(budget, int) or budget < 1:
            raise ValueError("budget_allocated must be a positive int")

        threshold = outputs.get("confidence_threshold")
        if not isinstance(threshold, (int, float)) or not (0.0 <= float(threshold) <= 1.0):
            raise ValueError("confidence_threshold must be a float in [0, 1]")

        if not isinstance(outputs.get("seek_validation"), bool):
            raise ValueError("seek_validation must be a bool")

        complexity = outputs.get("complexity_score")
        if not isinstance(complexity, (int, float)) or not (0.0 <= float(complexity) <= 1.0):
            raise ValueError("complexity_score must be a float in [0, 1]")

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")

        memory = context.get("memory") or {}
        enrichment: dict[str, Any] = memory.get("enrichment") or {}

        query: str = str(enrichment.get("query") or context.get("query") or "")
        uncertainty_level: str = str(enrichment.get("uncertainty_level") or "low")
        budget_confidence: float = float(enrichment.get("budget_confidence") or 0.0)
        time_budget_seconds: int = int(
            (context.get("runtime") or {}).get("time_budget_seconds", _DEFAULT_TIME_BUDGET)
            or _DEFAULT_TIME_BUDGET
        )

        strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        outputs: dict[str, Any]

        if strategy == "heuristic":
            outputs = await run_heuristic_async(
                query, uncertainty_level, budget_confidence, time_budget_seconds
            )
        else:
            prompt = (
                "You are a meta-cognitive controller for an enterprise AI OS. "
                "Output JSON only. Do not hallucinate.\n\n"
                "Decide how deeply to reason about the query below.\n\n"
                f"QUERY: {query}\n"
                f"UNCERTAINTY_LEVEL: {uncertainty_level}\n"
                f"BUDGET_CONFIDENCE: {budget_confidence}\n"
                f"TIME_BUDGET_SECONDS: {time_budget_seconds}\n\n"
                "Choose reasoning_mode from: heuristic | mixed | deliberative | escalate.\n"
                "  heuristic   — fast, low-cost, use for simple lookups\n"
                "  mixed       — balanced, default\n"
                "  deliberative — deep reasoning, multi-step validation\n"
                "  escalate    — route to human review; high-risk/irreversible\n\n"
                "Return JSON with keys:\n"
                "  reasoning_mode:       str   (one of the four above)\n"
                "  budget_allocated:     int   (max retrieval items, 6-50)\n"
                "  confidence_threshold: float (0.50-0.95)\n"
                "  seek_validation:      bool\n"
                "  complexity_score:     float (0-1)\n"
                "  strategy_selected:    str   (same as reasoning_mode)\n"
            )
            try:
                client = create_ollama_client(
                    base_url=str(getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")),
                    model=str(getattr(settings, "OLLAMA_MODEL", "qwen2.5:7b")),
                    timeout_seconds=float(getattr(settings, "OLLAMA_TIMEOUT_SECONDS", 5.0)),
                    max_concurrency=int(getattr(settings, "OLLAMA_MAX_CONCURRENCY", 2)),
                )
                resp = await client.complete_json(
                    prompt=prompt,
                    schema_hint={},
                    tool_event_sink=context.get("tool_event_sink"),
                )
                parsed = _parse_llm_output(resp)
            except Exception:
                parsed = None

            if parsed is not None:
                outputs = parsed
            else:
                outputs = await run_heuristic_async(
                    query, uncertainty_level, budget_confidence, time_budget_seconds
                )

        finished_at = datetime.now(timezone.utc)

        result = AgentResult(
            agent_name=self.name,
            agent_version=self.version,
            memory_id=memory_id,
            status="success",
            confidence=float(outputs.get("confidence_threshold", _DEFAULT_CONFIDENCE_THRESHOLD)),
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=started_at,
            finished_at=finished_at,
            trace_id=str(trace_id) if trace_id else None,
        )
        self.validate_outputs(result)
        return result
