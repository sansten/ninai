"""Action Policy Service — Phase 47.

Evaluates whether an autonomous action is permitted given the current
enrichment confidence, action type, and per-org policy configuration.

Policy hierarchy (first matching rule wins):
1. Confidence below minimum threshold → denied
2. Action type not in org's permitted set → denied
3. Confidence >= auto_approve_threshold AND org_tier in auto_approve_tiers
   AND action_type in auto_approvable_types → auto_approved
4. Fallthrough → human_review_required

Callers should treat this as a pure synchronous service — no DB access here.
DB-backed org config can be injected via the org_config dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Defaults — overridable per org via org_config
# ---------------------------------------------------------------------------

_DEFAULT_MIN_CONFIDENCE: float = 0.70
_DEFAULT_AUTO_APPROVE_THRESHOLD: float = 0.90
_DEFAULT_PERMITTED_TYPES: frozenset[str] = frozenset({
    "webhook", "pagerduty", "jira", "slack", "generic_rest"
})
_DEFAULT_AUTO_APPROVABLE_TYPES: frozenset[str] = frozenset({
    "webhook", "pagerduty",
})
_DEFAULT_AUTO_APPROVE_TIERS: frozenset[str] = frozenset({
    "enterprise",
})

_VALID_ACTION_TYPES: frozenset[str] = frozenset({
    "webhook", "pagerduty", "jira", "slack", "generic_rest"
})
_VALID_DECISIONS: frozenset[str] = frozenset({
    "auto_approved", "human_review_required", "denied"
})


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolicyDecision:
    decision: str        # "auto_approved" | "human_review_required" | "denied"
    reason: str
    confidence_threshold_met: bool
    min_confidence_used: float
    auto_approve_threshold_used: float


# ---------------------------------------------------------------------------
# Pure evaluation helpers
# ---------------------------------------------------------------------------

def _is_float(v: Any) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _resolve_permitted_types(org_config: dict | None) -> frozenset[str]:
    if org_config:
        raw = org_config.get("permitted_action_types")
        if isinstance(raw, (list, tuple, set, frozenset)):
            parsed = frozenset(str(t).strip() for t in raw if t)
            if parsed:
                return parsed & _VALID_ACTION_TYPES
    return _DEFAULT_PERMITTED_TYPES


def _resolve_auto_approvable_types(org_config: dict | None) -> frozenset[str]:
    if org_config:
        raw = org_config.get("auto_approvable_action_types")
        if isinstance(raw, (list, tuple, set, frozenset)):
            parsed = frozenset(str(t).strip() for t in raw if t)
            if parsed:
                return parsed & _VALID_ACTION_TYPES
    return _DEFAULT_AUTO_APPROVABLE_TYPES


def _resolve_auto_approve_tiers(org_config: dict | None) -> frozenset[str]:
    if org_config:
        raw = org_config.get("auto_approve_tiers")
        if isinstance(raw, (list, tuple, set, frozenset)):
            parsed = frozenset(str(t).strip().lower() for t in raw if t)
            if parsed:
                return parsed
    return _DEFAULT_AUTO_APPROVE_TIERS


def _resolve_min_confidence(org_config: dict | None) -> float:
    if org_config:
        v = org_config.get("min_confidence")
        if _is_float(v):
            return max(0.0, min(1.0, float(v)))
    return _DEFAULT_MIN_CONFIDENCE


def _resolve_auto_approve_threshold(org_config: dict | None) -> float:
    if org_config:
        v = org_config.get("auto_approve_threshold")
        if _is_float(v):
            return max(0.0, min(1.0, float(v)))
    return _DEFAULT_AUTO_APPROVE_THRESHOLD


def evaluate_policy(
    *,
    action_type: str,
    confidence: float,
    org_tier: str,
    org_config: dict | None = None,
) -> PolicyDecision:
    """Pure function — evaluate whether an action is permitted.

    Parameters
    ----------
    action_type:  one of the _VALID_ACTION_TYPES
    confidence:   enrichment confidence at decision time (0..1)
    org_tier:     "enterprise" | "mid_market" | "smb" | etc.
    org_config:   optional per-org overrides (loaded from DB by caller)
    """
    min_conf = _resolve_min_confidence(org_config)
    auto_thresh = _resolve_auto_approve_threshold(org_config)
    permitted = _resolve_permitted_types(org_config)
    auto_approvable = _resolve_auto_approvable_types(org_config)
    auto_tiers = _resolve_auto_approve_tiers(org_config)

    # Normalise
    safe_confidence = float(confidence) if _is_float(confidence) else 0.0
    safe_confidence = max(0.0, min(1.0, safe_confidence))
    safe_tier = str(org_tier or "").strip().lower()
    safe_type = str(action_type or "").strip().lower()

    # Rule 1 — unknown action type
    if safe_type not in _VALID_ACTION_TYPES:
        return PolicyDecision(
            decision="denied",
            reason=f"Unknown action type: {action_type!r}",
            confidence_threshold_met=False,
            min_confidence_used=min_conf,
            auto_approve_threshold_used=auto_thresh,
        )

    # Rule 2 — confidence too low
    if safe_confidence < min_conf:
        return PolicyDecision(
            decision="denied",
            reason=f"Confidence {safe_confidence:.2f} below minimum {min_conf:.2f}",
            confidence_threshold_met=False,
            min_confidence_used=min_conf,
            auto_approve_threshold_used=auto_thresh,
        )

    # Rule 3 — action type not permitted for this org
    if safe_type not in permitted:
        return PolicyDecision(
            decision="denied",
            reason=f"Action type {action_type!r} not in org permitted set",
            confidence_threshold_met=True,
            min_confidence_used=min_conf,
            auto_approve_threshold_used=auto_thresh,
        )

    # Rule 4 — auto-approve path
    if (
        safe_confidence >= auto_thresh
        and safe_tier in auto_tiers
        and safe_type in auto_approvable
    ):
        return PolicyDecision(
            decision="auto_approved",
            reason=(
                f"Confidence {safe_confidence:.2f} >= threshold {auto_thresh:.2f}, "
                f"org_tier={org_tier!r}, action_type={action_type!r}"
            ),
            confidence_threshold_met=True,
            min_confidence_used=min_conf,
            auto_approve_threshold_used=auto_thresh,
        )

    # Rule 5 — fallthrough: human review required
    return PolicyDecision(
        decision="human_review_required",
        reason=(
            f"Confidence {safe_confidence:.2f} or tier {org_tier!r} does not meet "
            f"auto-approve criteria (threshold={auto_thresh:.2f}, "
            f"auto_approvable={sorted(auto_approvable)}, "
            f"auto_tiers={sorted(auto_tiers)})"
        ),
        confidence_threshold_met=True,
        min_confidence_used=min_conf,
        auto_approve_threshold_used=auto_thresh,
    )


# ---------------------------------------------------------------------------
# Service class (thin wrapper for DI)
# ---------------------------------------------------------------------------

class ActionPolicyService:
    """Stateless policy evaluator — thread-safe, no DB access."""

    def evaluate(
        self,
        *,
        action_type: str,
        confidence: float,
        org_tier: str,
        org_config: dict | None = None,
    ) -> PolicyDecision:
        return evaluate_policy(
            action_type=action_type,
            confidence=confidence,
            org_tier=org_tier,
            org_config=org_config,
        )
