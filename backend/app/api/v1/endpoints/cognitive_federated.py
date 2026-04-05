"""Federated Cognitive Learning endpoint (Feature 24.8).

Exposes the FederatedCognitiveLearningService as a single POST endpoint:

  POST /cognitive/federated/synthesize

Cross-org weight aggregation and multi-metric benchmarking without
raw data sharing. All peer values are DP-noised before any aggregate
is computed or returned.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, status

from app.middleware.tenant_context import TenantContext, require_org_admin
from app.services.federated_cognitive_learning_service import (
    FederatedCognitiveLearningService,
)

router = APIRouter()


@router.post("/synthesize")
async def federated_synthesize(
    payload: dict[str, Any] = Body(...),
    tenant: TenantContext = Depends(require_org_admin()),
) -> dict[str, Any]:
    """Federated weight aggregation + cross-org benchmark insights.

    Request body:
      agent_name               (str, required)
      org_weight_submissions   (list[dict], optional)
        Each item: {weights: dict, sample_count: int, sensitivity_tier: str}
      org_metrics              (dict[str, float], optional)
      peer_metric_submissions  (list[dict[str, float]], optional)
      sensitivity_tier         (str, optional — "high"|"medium"|"low")

    Returns:
      aggregated_weights, benchmark_insights, sharing_recommendation,
      total_privacy_budget_spent, federation_confidence
    """
    agent_name = str(payload.get("agent_name") or "").strip()
    if not agent_name:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="agent_name is required",
        )

    svc = FederatedCognitiveLearningService()
    result = svc.synthesize(
        agent_name=agent_name,
        org_weight_submissions=list(payload.get("org_weight_submissions") or []),
        org_metrics=dict(payload.get("org_metrics") or {}),
        peer_metric_submissions=list(payload.get("peer_metric_submissions") or []),
        sensitivity_tier=str(payload.get("sensitivity_tier") or "medium"),
    )

    return {
        "agent_name": agent_name,
        "aggregated_weights": [
            {
                "agent_name": aw.agent_name,
                "global_weights": aw.global_weights,
                "contributing_orgs": aw.contributing_orgs,
                "aggregation_epsilon": aw.aggregation_epsilon,
                "privacy_budget_spent": aw.privacy_budget_spent,
            }
            for aw in result.aggregated_weights
        ],
        "benchmark_insights": [
            {
                "metric": b.metric,
                "org_value": b.org_value,
                "global_private_mean": b.global_private_mean,
                "percentile_rank": b.percentile_rank,
                "gap_to_p75": b.gap_to_p75,
                "status": b.status,
            }
            for b in result.benchmark_insights
        ],
        "sharing_recommendation": result.sharing_recommendation,
        "total_privacy_budget_spent": result.total_privacy_budget_spent,
        "federation_confidence": result.federation_confidence,
        "org_id": tenant.org_id,
    }
