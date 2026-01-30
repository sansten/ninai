"""PolicyGuard for Cognitive Loop tool invocations.

All tool calls MUST go through this guard.
It enforces:
- RBAC via PermissionChecker
- Optional scope constraints
- Optional clearance/justification requirements

This module does not execute tools; it only authorizes.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.permission_checker import AccessDecision, PermissionChecker
from app.services.cognitive_tooling.tool_registry import ToolSpec


_CLASSIFICATION_ORDER = {
    "public": 0,
    "internal": 1,
    "confidential": 2,
    "restricted": 3,
}


@dataclass(frozen=True)
class ToolContext:
    user_id: str
    org_id: str
    scope: str | None = None
    scope_id: str | None = None
    classification: str | None = None
    clearance_level: int = 0
    justification: str | None = None
    self_model: dict | None = None  # Optional SelfModel profile for reliability checks


class PolicyGuard:
    def __init__(self, permission_checker: PermissionChecker):
        self.permission_checker = permission_checker

    async def authorize(self, *, tool: ToolSpec, ctx: ToolContext) -> AccessDecision:
        # Check SelfModel reliability for dynamic justification requirement
        requires_justification = tool.require_justification
        reliability_warning = None
        
        if ctx.self_model and isinstance(ctx.self_model, dict):
            tool_reliability = ctx.self_model.get("tool_reliability") or {}
            if isinstance(tool_reliability, dict):
                tool_stats = tool_reliability.get(tool.name)
                if isinstance(tool_stats, dict):
                    success_rate = tool_stats.get("success_rate_30d")
                    sample_size = tool_stats.get("sample_size_30d")
                    
                    # If tool has low reliability (< 80%) and enough samples (>= 3), require justification
                    if success_rate is not None and sample_size is not None:
                        try:
                            rate = float(success_rate)
                            n = int(sample_size)
                            if n >= 3 and rate < 0.80:
                                requires_justification = True
                                reliability_warning = f"Tool {tool.name} has low reliability ({rate:.1%}) - justification required"
                        except (ValueError, TypeError):
                            pass
        
        # Justification requirement (original or SelfModel-adjusted)
        if requires_justification and not (ctx.justification or "").strip():
            return AccessDecision(
                allowed=False,
                reason=reliability_warning or "Justification required",
                method="policy",
                details={
                    "tool": tool.name,
                    "reason": "missing_justification",
                    "reliability_adjusted": reliability_warning is not None,
                },
            )

        # Clearance requirement
        if int(ctx.clearance_level or 0) < int(tool.min_clearance_level or 0):
            return AccessDecision(
                allowed=False,
                reason="Insufficient clearance",
                method="policy",
                details={
                    "tool": tool.name,
                    "reason": "insufficient_clearance",
                    "min_clearance": int(tool.min_clearance_level or 0),
                    "clearance": int(ctx.clearance_level or 0),
                },
            )

        # Scope restriction
        if tool.allowed_scopes is not None:
            if (ctx.scope or "") not in set(tool.allowed_scopes):
                return AccessDecision(
                    allowed=False,
                    reason="Scope not allowed for tool",
                    method="policy",
                    details={"tool": tool.name, "reason": "scope_not_allowed", "scope": ctx.scope},
                )

        # Optional classification restriction (tool may decide based on ctx.classification)
        if ctx.classification:
            if ctx.classification not in _CLASSIFICATION_ORDER:
                return AccessDecision(
                    allowed=False,
                    reason="Unknown classification",
                    method="policy",
                    details={"tool": tool.name, "reason": "unknown_classification", "classification": ctx.classification},
                )

        # RBAC permissions
        for perm in tool.required_permissions:
            decision = await self.permission_checker.check_permission(
                ctx.user_id,
                ctx.org_id,
                perm,
            )
            if not decision.allowed:
                # Carry forward the underlying decision but annotate.
                details = dict(decision.details or {})
                details.update({"tool": tool.name, "required_permission": perm})
                return AccessDecision(
                    allowed=False,
                    reason=decision.reason or f"Missing permission: {perm}",
                    method=decision.method or "rbac",
                    details=details,
                )

        # Build final decision with reliability warnings
        final_details = {"tool": tool.name}
        final_reason = "Allowed"
        
        if reliability_warning:
            final_details["reliability_warning"] = reliability_warning
            final_reason = f"Allowed (Warning: {reliability_warning})"
        
        return AccessDecision(
            allowed=True,
            reason=final_reason,
            method="rbac",
            details=final_details,
        )
