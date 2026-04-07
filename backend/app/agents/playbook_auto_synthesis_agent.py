"""PlaybookAutoSynthesisAgent — Phase 81.

Scans a batch of AutonomousGoalOutcome records supplied in enrichment and
auto-synthesizes new Playbook candidates for recurring high-success patterns.

Qualifying pattern:
  - Same impact-description fingerprint appears >= _MIN_OCCURRENCES times.
  - Fraction of "valuable" outcomes within the cluster >= _SUCCESS_RATE_FLOOR.

Inputs (via context["memory"]["enrichment"]):
  - outcome_records: list[dict] — each dict may have:
        outcome_type:         "valuable" | "not_valuable" | "premature"
        impact_description:   str | None
        goal_id:              str | None

Outputs:
  - synthesized_count: int — number of playbooks synthesized
  - patterns_found:    int — clusters that qualified
  - playbooks:         list[dict] — synthesized playbook descriptors
  - confidence:        float
  - rationale:         "heuristic"
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_OCCURRENCES = 3
_SUCCESS_RATE_FLOOR = 0.85
_VALUABLE = "valuable"
_MAX_STEP_DESCRIPTION_LEN = 80


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _fingerprint(impact_description: str) -> str:
    """Stable 16-hex-char hash for a normalised impact description."""
    tokens = re.sub(r"[^a-z0-9 ]", " ", impact_description.lower()).split()
    key = " ".join(sorted(set(tokens)))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _generate_steps(records: list[dict]) -> list[str]:
    """Derive generic actionable steps from a cluster of outcome records."""
    descriptions = [
        r.get("impact_description", "")
        for r in records
        if r.get("impact_description")
    ]
    unique_descs = list(dict.fromkeys(descriptions))[:3]

    steps: list[str] = ["identify the recurring pattern"]
    for desc in unique_descs:
        first = desc.split(".")[0].strip()[:_MAX_STEP_DESCRIPTION_LEN]
        if first:
            steps.append(f"apply: {first}")
    steps.append("validate outcome and mark valuable")
    return steps


def synthesize_playbooks(outcome_records: list[dict]) -> list[dict]:
    """Group outcome records by fingerprint and return qualifying playbooks."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for record in outcome_records:
        desc = record.get("impact_description") or ""
        key = _fingerprint(desc) if desc else "generic"
        buckets[key].append(record)

    playbooks: list[dict] = []
    for key, records in buckets.items():
        total = len(records)
        if total < _MIN_OCCURRENCES:
            continue
        valuable_count = sum(1 for r in records if r.get("outcome_type") == _VALUABLE)
        success_rate = valuable_count / total
        if success_rate < _SUCCESS_RATE_FLOOR:
            continue
        representative = records[0]
        raw_title = representative.get("impact_description") or "pattern"
        title = f"Auto-synthesized playbook: {raw_title[:60]}"
        steps = _generate_steps(records)
        playbooks.append(
            {
                "title": title,
                "signature_hash": key,
                "steps": steps,
                "success_rate": round(success_rate, 4),
                "evidence": {
                    "outcome_count": total,
                    "valuable_count": valuable_count,
                },
                "problem_signature": {"fingerprint": key},
            }
        )
    return playbooks


def run_heuristic(outcome_records: list[dict]) -> dict[str, Any]:
    playbooks = synthesize_playbooks(outcome_records)
    patterns_found = len(playbooks)
    confidence = min(0.90, 0.50 + patterns_found * 0.10) if patterns_found else 0.40
    return {
        "synthesized_count": len(playbooks),
        "patterns_found": patterns_found,
        "playbooks": playbooks,
        "confidence": round(confidence, 4),
        "rationale": "heuristic",
    }


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class PlaybookAutoSynthesisAgent(BaseAgent):
    """Auto-synthesize playbooks from recurring high-success outcome patterns."""

    name = "PlaybookAutoSynthesisAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return ["PlaybookExecutionTrackerAgent"]

    def validate_outputs(self, result: AgentResult) -> None:
        assert isinstance(result.outputs.get("synthesized_count"), int)
        assert isinstance(result.outputs.get("patterns_found"), int)
        assert isinstance(result.outputs.get("playbooks"), list)
        assert 0.0 <= result.outputs.get("confidence", 0) <= 1.0

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")

        enrichment: dict = (context.get("memory") or {}).get("enrichment") or {}
        outcome_records: list[dict] = enrichment.get("outcome_records") or []

        outputs = run_heuristic(outcome_records)
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
