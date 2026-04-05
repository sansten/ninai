"""Cognitive GraphQL resolvers."""

from __future__ import annotations

import strawberry

from app.middleware.tenant_context import TenantContext
from app.services.cognitive_gateway_service import (
    CognitiveGatewayCapabilities,
    CognitiveGatewayService,
)


@strawberry.type
class GqlDecideResult:
    decision: str
    confidence: float
    tone: str
    action_recommended: bool


async def resolve_cognitive_decide(
    info: strawberry.Info,
    situation: str,
) -> GqlDecideResult:
    ctx = info.context
    tenant = ctx["tenant"]
    if not isinstance(tenant, TenantContext):
        raise ValueError("Invalid tenant context")

    gateway = CognitiveGatewayService(capabilities=CognitiveGatewayCapabilities.full())
    result = await gateway.decide(
        content=situation,
        enrichment={},
        context_id=None,
        org_id=tenant.org_id,
    )

    return GqlDecideResult(
        decision=result.decision,
        confidence=result.confidence,
        tone=result.tone,
        action_recommended=result.action_recommended,
    )
