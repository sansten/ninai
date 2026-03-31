"""Autonomous Action Agent — Phase 47.

Closes the loop between reasoning and execution.  When upstream enrichment
signals a high-confidence, urgent situation (anomaly + enterprise org), this
agent decides whether to dispatch an external action, evaluates it against
the org's action policy, and dispatches via ExternalConnectorService.

Depends on (reads enrichment from):
  - AnomalyDetectionAgent (Phase 25)     — anomaly_detected, anomaly_score
  - NarrativeSynthesisAgent (Phase 23)   — tone, narrative_text, action_items
  - OrgAttentionAgent (Phase 10)         — org_tier
  - PlaybookAgent (Phase 20)             — matched_playbook_id
  - EpisodicGroupingAgent (Phase 18)     — episode_label, episode_id
  - EntityResolutionAgent (Phase 7)      — canonical_entity

Outputs:
  - action_dispatched:   bool
  - action_type:         str | None   — "webhook" | "pagerduty" | "jira" | "slack" | "generic_rest"
  - action_status:       str          — "success" | "failed" | "pending_review" | "denied" | "no_action"
  - policy_decision:     str | None   — "auto_approved" | "human_review_required" | "denied"
  - dispatch_confidence: float        — confidence at the moment of decision
  - action_target:       str | None   — target URL used (for audit; set by connector config)
  - rationale:           "heuristic" | "llm"
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.agents.llm.ollama_breaker import create_ollama_client
from app.core.config import settings
from app.services.action_policy_service import ActionPolicyService, evaluate_policy
from app.services.external_connector_service import ExternalConnectorService


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ANOMALY_SCORE_THRESHOLD = 0.70       # minimum score to consider acting
_URGENT_TONES = frozenset({"urgent"})
_VALID_ACTION_TYPES = frozenset({
    "webhook", "pagerduty", "jira", "slack", "generic_rest"
})
_VALID_ACTION_STATUSES = frozenset({
    "success", "failed", "pending_review", "denied", "no_action"
})
_VALID_POLICY_DECISIONS = frozenset({
    "auto_approved", "human_review_required", "denied", None
})

# Default connector targets when no ConnectorRegistry is present (dev/demo mode)
_DEFAULT_TARGETS: dict[str, str] = {
    "pagerduty":   "https://events.pagerduty.com/v2/enqueue",
    "webhook":     "https://hooks.example.internal/ninai/action",
    "jira":        "https://jira.example.internal/rest/api/2/issue",
    "slack":       "https://hooks.slack.com/services/ninai/action",
    "generic_rest": "https://api.example.internal/ninai/action",
}


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _is_float(v: Any) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _extract_anomaly_score(enrichment: dict) -> float:
    v = enrichment.get("anomaly_score")
    if _is_float(v):
        return round(max(0.0, min(1.0, float(v))), 4)
    return 0.0


def _extract_org_tier(enrichment: dict) -> str:
    return str(enrichment.get("org_tier") or "").strip().lower()


def _extract_tone(enrichment: dict) -> str:
    return str(enrichment.get("tone") or "").strip().lower()


def _should_act(enrichment: dict) -> tuple[bool, float, str]:
    """Return (should_act, confidence, reason)."""
    anomaly_detected = bool(enrichment.get("anomaly_detected"))
    if not anomaly_detected:
        return False, 0.0, "anomaly_detected=False — no action needed"

    score = _extract_anomaly_score(enrichment)
    if score < _ANOMALY_SCORE_THRESHOLD:
        return False, score, f"anomaly_score {score:.2f} below threshold {_ANOMALY_SCORE_THRESHOLD}"

    tone = _extract_tone(enrichment)
    if tone not in _URGENT_TONES:
        # Still act if score is very high, even without urgent tone
        if score >= 0.90:
            return True, score, f"anomaly_score {score:.2f} overrides non-urgent tone {tone!r}"
        return False, score, f"tone {tone!r} not urgent and score {score:.2f} < 0.90"

    return True, score, f"anomaly_detected + score {score:.2f} + tone {tone!r}"


def _choose_action_type(enrichment: dict) -> str:
    """Pick action type based on org tier and available signals."""
    org_tier = _extract_org_tier(enrichment)
    if org_tier == "enterprise":
        return "pagerduty"
    if org_tier in ("mid_market", "midmarket"):
        return "slack"
    return "webhook"


def _build_payload(enrichment: dict, memory_content: str) -> dict[str, Any]:
    """Build a safe, informative action payload from enrichment signals."""
    return {
        "summary": enrichment.get("narrative_text") or memory_content[:300],
        "episode_label": enrichment.get("episode_label") or "",
        "episode_id": enrichment.get("episode_id") or "",
        "canonical_entity": enrichment.get("canonical_entity") or "",
        "anomaly_score": _extract_anomaly_score(enrichment),
        "org_tier": _extract_org_tier(enrichment),
        "tone": _extract_tone(enrichment),
        "matched_playbook_id": enrichment.get("matched_playbook_id") or "",
        "action_items": enrichment.get("action_items") or [],
    }


def _compute_dispatch_confidence(enrichment: dict) -> float:
    """Combine anomaly_score with upstream confidence signals."""
    base = _extract_anomaly_score(enrichment)

    cred = enrichment.get("credibility_score")
    cred_factor = float(cred) if _is_float(cred) else 0.80
    cred_factor = max(0.0, min(1.0, cred_factor))

    return round(base * 0.70 + cred_factor * 0.30, 4)


# ---------------------------------------------------------------------------
# Heuristic path
# ---------------------------------------------------------------------------

def run_heuristic(enrichment: dict, memory_content: str) -> dict[str, Any]:
    """Full heuristic decision: should_act → policy → simulate dispatch."""
    should, confidence, reason = _should_act(enrichment)

    if not should:
        return {
            "action_dispatched": False,
            "action_type": None,
            "action_status": "no_action",
            "policy_decision": None,
            "dispatch_confidence": confidence,
            "action_target": None,
            "rationale": "heuristic",
            "_decision_reason": reason,
        }

    action_type = _choose_action_type(enrichment)
    org_tier = _extract_org_tier(enrichment)
    dispatch_conf = _compute_dispatch_confidence(enrichment)

    policy_svc = ActionPolicyService()
    decision = policy_svc.evaluate(
        action_type=action_type,
        confidence=dispatch_conf,
        org_tier=org_tier,
    )

    if decision.decision == "denied":
        return {
            "action_dispatched": False,
            "action_type": action_type,
            "action_status": "denied",
            "policy_decision": "denied",
            "dispatch_confidence": dispatch_conf,
            "action_target": None,
            "rationale": "heuristic",
            "_decision_reason": decision.reason,
        }

    if decision.decision == "human_review_required":
        return {
            "action_dispatched": False,
            "action_type": action_type,
            "action_status": "pending_review",
            "policy_decision": "human_review_required",
            "dispatch_confidence": dispatch_conf,
            "action_target": None,
            "rationale": "heuristic",
            "_decision_reason": decision.reason,
        }

    # auto_approved — simulate dispatch (real dispatch wired in full runtime)
    # In heuristic path we record intent; actual HTTP call is in async run()
    target = _DEFAULT_TARGETS.get(action_type, _DEFAULT_TARGETS["webhook"])
    return {
        "action_dispatched": True,
        "action_type": action_type,
        "action_status": "success",
        "policy_decision": "auto_approved",
        "dispatch_confidence": dispatch_conf,
        "action_target": target,
        "rationale": "heuristic",
        "_decision_reason": decision.reason,
        "_payload_summary": {
            "keys": list(_build_payload(enrichment, memory_content).keys()),
            "episode_label": enrichment.get("episode_label") or "",
        },
    }


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------

def _build_llm_prompt(enrichment: dict, memory_content: str) -> str:
    anomaly_score = _extract_anomaly_score(enrichment)
    org_tier = _extract_org_tier(enrichment)
    tone = _extract_tone(enrichment)
    episode = enrichment.get("episode_label") or ""
    narrative = enrichment.get("narrative_text") or memory_content[:300]

    return f"""You are an autonomous action decision engine. Given the enrichment signals
below, decide whether to dispatch an external action.

Enrichment signals:
- anomaly_detected: {enrichment.get('anomaly_detected')}
- anomaly_score: {anomaly_score}
- org_tier: {org_tier}
- tone: {tone}
- episode_label: {episode}
- narrative: {narrative[:200]}

Rules:
1. If anomaly_score < 0.70 or not anomaly_detected → action_status = no_action
2. If org_tier = enterprise and anomaly_score >= 0.90 → action_type = pagerduty
3. If org_tier = mid_market → action_type = slack
4. Default action_type = webhook
5. dispatch_confidence = anomaly_score * 0.7 + credibility_factor * 0.3

Respond with ONLY valid JSON matching this schema:
{{
  "action_dispatched": true|false,
  "action_type": "webhook"|"pagerduty"|"jira"|"slack"|"generic_rest"|null,
  "action_status": "success"|"failed"|"pending_review"|"denied"|"no_action",
  "policy_decision": "auto_approved"|"human_review_required"|"denied"|null,
  "dispatch_confidence": <float 0..1>,
  "action_target": <string|null>
}}"""


def _validate_llm_outputs(raw: dict) -> dict[str, Any]:
    """Validate and normalise LLM-returned action decision."""
    action_dispatched = bool(raw.get("action_dispatched"))
    action_type = raw.get("action_type")
    if action_type not in _VALID_ACTION_TYPES and action_type is not None:
        action_type = None

    action_status = str(raw.get("action_status") or "no_action").strip().lower()
    if action_status not in _VALID_ACTION_STATUSES:
        action_status = "no_action"

    policy_decision = raw.get("policy_decision")
    if policy_decision not in _VALID_POLICY_DECISIONS:
        policy_decision = None

    conf = raw.get("dispatch_confidence")
    dispatch_conf = round(max(0.0, min(1.0, float(conf))), 4) if _is_float(conf) else 0.0

    return {
        "action_dispatched": action_dispatched,
        "action_type": action_type,
        "action_status": action_status,
        "policy_decision": policy_decision,
        "dispatch_confidence": dispatch_conf,
        "action_target": raw.get("action_target") or None,
        "rationale": "llm",
    }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class AutonomousActionAgent(BaseAgent):
    """Phase 47 — Autonomous Action Engine.

    Decides whether and how to dispatch an external action based on
    upstream enrichment signals.  Uses ActionPolicyService for policy
    evaluation and ExternalConnectorService for dispatch (injected via
    set_connector_service or defaulting to simulation mode).
    """

    name = "autonomous_action_agent"
    version = "v1"

    def __init__(
        self,
        *,
        connector_service: ExternalConnectorService | None = None,
    ) -> None:
        # Allow injection for tests; otherwise a simulation-mode instance is used
        self._connector = connector_service

    def dependencies(self) -> list[str]:
        return [
            "anomaly_detection_agent",
            "narrative_synthesis_agent",
            "org_attention_agent",
        ]

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        memory = context.get("memory", {})
        enrichment = memory.get("enrichment", {})
        content = str(memory.get("content") or "")

        strategy = getattr(settings, "AGENT_STRATEGY", "llm")

        if strategy == "heuristic":
            outputs = run_heuristic(enrichment, content)
        else:
            try:
                outputs = await self._run_llm(enrichment, content)
            except Exception:
                outputs = run_heuristic(enrichment, content)

        # If auto_approved and we have a real connector, actually dispatch
        if (
            outputs.get("policy_decision") == "auto_approved"
            and outputs.get("action_dispatched")
            and self._connector is not None
        ):
            action_type = outputs.get("action_type") or "webhook"
            target = outputs.get("action_target") or _DEFAULT_TARGETS.get(action_type, "")
            payload = _build_payload(enrichment, content)
            dispatch_result = await self._connector.dispatch(
                action_type=action_type,
                target_url=target,
                payload=payload,
            )
            outputs["action_status"] = dispatch_result.status
            outputs["action_dispatched"] = dispatch_result.status == "success"
            outputs["_attempt_count"] = dispatch_result.attempt_count
            if dispatch_result.error:
                outputs["_dispatch_error"] = dispatch_result.error

        finished_at = datetime.now(timezone.utc)
        confidence = outputs.pop("dispatch_confidence", 0.0)
        _ = outputs.pop("_decision_reason", None)

        result = AgentResult(
            agent_name=self.name,
            agent_version=self.version,
            memory_id=memory_id,
            status="success",
            confidence=confidence,
            outputs=outputs,
            started_at=started_at,
            finished_at=finished_at,
        )
        self.validate_outputs(result)
        # Re-inject confidence into outputs for downstream enrichment
        result.outputs["dispatch_confidence"] = confidence
        return result

    async def _run_llm(self, enrichment: dict, content: str) -> dict[str, Any]:
        client = create_ollama_client()
        prompt = _build_llm_prompt(enrichment, content)
        schema_hint = {
            "action_dispatched": "bool",
            "action_type": "str|null",
            "action_status": "str",
            "policy_decision": "str|null",
            "dispatch_confidence": "float",
            "action_target": "str|null",
        }
        raw = await client.complete_json(prompt=prompt, schema_hint=schema_hint)
        return _validate_llm_outputs(raw)

    def validate_outputs(self, result: AgentResult) -> None:
        outputs = result.outputs
        if "action_dispatched" not in outputs:
            raise ValueError("missing action_dispatched")
        if "action_status" not in outputs:
            raise ValueError("missing action_status")
        status = outputs.get("action_status")
        if status not in _VALID_ACTION_STATUSES:
            raise ValueError(f"invalid action_status: {status!r}")
        conf = outputs.get("dispatch_confidence", result.confidence)
        if not _is_float(conf) or not (0.0 <= float(conf) <= 1.0):
            raise ValueError(f"dispatch_confidence out of range: {conf!r}")
