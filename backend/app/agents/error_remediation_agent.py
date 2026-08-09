"""Phase 83: ErrorRemediationAgent — auto-remediate high-severity errors.

Inspects enriched error events from known monitoring/alerting sources
(Sentry, PagerDuty, OpsGenie) and decides between automated dispatch,
human review routing, or ignore.

Decision rules:
  - error_source not in _ERROR_SOURCES  → ignore
  - severity not in _HIGH_SEVERITIES    → ignore
  - playbook found AND confidence >= 0.7 → dispatch (autonomous action)
  - otherwise                            → review (human review queue)

Inputs (via context["memory"]["enrichment"]):
  - error_source:         str — "sentry" | "pagerduty" | "opsgenie"
  - severity:             str — "critical" | "high" | "p1" | "p2" | "medium" | "low"
  - error_type:           str | None
  - playbook_candidates:  list[dict] — each with id + success_rate/confidence
  - playbook_confidence:  float | None — override for confidence

Outputs:
  - action:               "dispatch" | "review" | "ignore"
  - error_source:         str
  - severity:             str
  - playbook_id:          str | None
  - playbook_confidence:  float
  - routing_reason:       str
  - confidence:           float 0..1
  - rationale:            "heuristic"
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ERROR_SOURCES = frozenset({"sentry", "pagerduty", "opsgenie"})
_HIGH_SEVERITIES = frozenset({"high", "critical", "p1", "p2"})
_DISPATCH_CONFIDENCE_THRESHOLD = 0.70


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _best_playbook(candidates: list[dict]) -> tuple[str | None, float]:
    """Return the (playbook_id, confidence) of the best candidate by success rate."""
    if not candidates:
        return None, 0.0
    best = max(
        candidates,
        key=lambda c: float(c.get("success_rate") or c.get("confidence") or 0.0),
    )
    pb_id = best.get("id") or best.get("playbook_id") or None
    conf = float(best.get("success_rate") or best.get("confidence") or 0.0)
    return pb_id, conf


def run_heuristic(enrichment: dict) -> dict[str, Any]:
    """Determine remediation action from enrichment dict."""
    error_source = str(enrichment.get("error_source") or "").lower().strip()
    severity = str(enrichment.get("severity") or "").lower().strip()
    candidates: list[dict] = enrichment.get("playbook_candidates") or []
    override_confidence = enrichment.get("playbook_confidence")

    # Step 1: check source and severity
    if error_source not in _ERROR_SOURCES:
        return {
            "action": "ignore",
            "error_source": error_source,
            "severity": severity,
            "playbook_id": None,
            "playbook_confidence": 0.0,
            "routing_reason": f"source {error_source!r} not in monitored error sources",
            "confidence": 0.85,
            "rationale": "heuristic",
        }
    if severity not in _HIGH_SEVERITIES:
        return {
            "action": "ignore",
            "error_source": error_source,
            "severity": severity,
            "playbook_id": None,
            "playbook_confidence": 0.0,
            "routing_reason": f"severity {severity!r} below high threshold",
            "confidence": 0.85,
            "rationale": "heuristic",
        }

    # Step 2: look up best playbook
    playbook_id, playbook_conf = _best_playbook(candidates)
    if override_confidence is not None:
        try:
            playbook_conf = float(override_confidence)
        except (TypeError, ValueError):
            # A non-numeric override (e.g. a different agent's unrelated
            # value landing under the same enrichment key) must not crash
            # remediation for a high-severity error — keep the
            # heuristically-computed confidence instead.
            pass

    # Step 3: decide dispatch vs. review
    if playbook_id and playbook_conf >= _DISPATCH_CONFIDENCE_THRESHOLD:
        action = "dispatch"
        reason = f"playbook {playbook_id!r} matched with confidence {playbook_conf:.2f}"
        agent_confidence = 0.85
    elif playbook_id:
        action = "review"
        reason = (
            f"playbook {playbook_id!r} confidence {playbook_conf:.2f} "
            f"below threshold {_DISPATCH_CONFIDENCE_THRESHOLD}"
        )
        agent_confidence = 0.75
    else:
        action = "review"
        reason = "no matching playbook found — routing to human review"
        agent_confidence = 0.70

    return {
        "action": action,
        "error_source": error_source,
        "severity": severity,
        "playbook_id": playbook_id,
        "playbook_confidence": round(playbook_conf, 4),
        "routing_reason": reason,
        "confidence": agent_confidence,
        "rationale": "heuristic",
    }


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class ErrorRemediationAgent(BaseAgent):
    """Auto-remediate high-severity errors from monitored error sources."""

    name = "ErrorRemediationAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return ["PlaybookAgent", "AutonomousActionAgent", "HumanReviewQueueAgent"]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}

        if outputs.get("action") not in {"dispatch", "review", "ignore"}:
            raise ValueError("action must be one of dispatch/review/ignore")
        if not isinstance(outputs.get("error_source"), str):
            raise ValueError("error_source must be a str")
        if not isinstance(outputs.get("severity"), str):
            raise ValueError("severity must be a str")

        confidence = outputs.get("confidence", 0)
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            raise ValueError("confidence must be a float in [0, 1]")

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")

        enrichment: dict = (context.get("memory") or {}).get("enrichment") or {}
        outputs = run_heuristic(enrichment)
        finished_at = datetime.now(timezone.utc)

        result = AgentResult(
            agent_name=self.name,
            agent_version=self.version,
            memory_id=memory_id,
            status="success",
            confidence=float(outputs["confidence"]),
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=started_at,
            finished_at=finished_at,
            trace_id=str(trace_id) if trace_id else None,
        )
        self.validate_outputs(result)
        return result
