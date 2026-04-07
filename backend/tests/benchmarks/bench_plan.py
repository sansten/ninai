"""E2: Plan quality benchmark for CognitiveGatewayService.plan()."""
from __future__ import annotations

from typing import Any

# Curated goal set — chosen to exercise the keyword template branches in _heuristic_plan():
#   "report"|"summarise"  → 3-step report plan
#   "investigate"|"debug" → 4-step investigation plan
#   "escalate"|"incident" → 4-step escalation plan
#   default               → 4-step generic plan
#
PLAN_TASKS: list[tuple[str, dict]] = [
    ("summarise last week security events", {"domain": "security"}),
    ("investigate and debug the auth service crash", {"domain": "ops"}),
    ("escalate P1 payment outage to engineering", {"domain": "engineering"}),
    ("deploy new payment service to production", {"domain": "engineering"}),
    ("onboard new enterprise customer account", {"domain": "sales"}),
    ("resolve ongoing incident in checkout flow", {"domain": "ops"}),
    ("prepare quarterly business report for board", {"domain": "executive"}),
    ("debug memory leak in the recommendation engine", {"domain": "engineering"}),
]

STEP_QUALITY_FLOOR = 0.80  # 80% of plans must pass all quality checks

_ACTION_VERBS = frozenset({
    "retrieve", "assemble", "generate", "search", "identify", "check",
    "synthesise", "synthesize", "assign", "dispatch", "open", "decompose",
    "execute", "validate", "escalate", "deploy", "prepare", "debug",
    "investigate", "resolve", "onboard", "summarise", "summarize",
})


def _check_plan_quality(goal: str, steps: list[dict]) -> dict[str, bool]:
    """Run 4 quality checks on a plan result."""
    # has_steps
    has_steps = len(steps) >= 1

    # no_duplicate_steps: all action text unique (case-insensitive)
    texts = [str(s.get("action") or "").strip().lower() for s in steps]
    no_duplicates = len(texts) == len(set(texts))

    # actionable: each step action starts with a known action verb
    actionable = all(
        any(str(s.get("action") or "").lower().startswith(v) for v in _ACTION_VERBS)
        for s in steps
    )

    # has_goal: the goal string on the result is non-empty
    has_goal = bool(goal.strip())

    return {
        "has_steps": has_steps,
        "no_duplicate_steps": no_duplicates,
        "actionable": actionable,
        "has_goal": has_goal,
    }


async def run(*, mode: str, strategy: str, **kwargs: Any) -> dict[str, Any]:
    from app.services.cognitive_gateway_service import (
        CognitiveGatewayCapabilities,
        CognitiveGatewayService,
    )

    caps = CognitiveGatewayCapabilities.full()
    svc = CognitiveGatewayService(capabilities=caps)

    passing = 0
    results = []
    for goal, context in PLAN_TASKS:
        plan_result = await svc.plan(goal=goal, context=context)
        checks = _check_plan_quality(plan_result.goal, plan_result.steps)
        plan_passed = all(checks.values())
        if plan_passed:
            passing += 1
        results.append(
            {
                "goal": goal[:50],
                "passed": plan_passed,
                "step_count": plan_result.step_count,
                "checks": checks,
            }
        )

    pass_rate = passing / len(PLAN_TASKS)
    return {
        "benchmark": "plan_quality",
        "mode": mode,
        "strategy": strategy,
        "pass_rate": round(pass_rate, 4),
        "quality_floor": STEP_QUALITY_FLOOR,
        "passed": pass_rate >= STEP_QUALITY_FLOOR,
        "task_count": len(PLAN_TASKS),
        "passing_plans": passing,
        "results": results,
        "status": "ok" if pass_rate >= STEP_QUALITY_FLOOR else "below_floor",
    }
