"""Tenant Evaluation Metrics Service — Phase 90.

Defines concrete, measurable outcome metrics per enterprise tenant type.
Bridges the gap between "we have 85 agent phases" and "did they help?"

Tenant types and their primary metrics:

  sales        — lead_conversion_rate, follow_up_completion_rate, pipeline_accuracy
  engineering  — bug_resolution_rate, knowledge_reuse_rate, deploy_success_rate
  support      — ticket_deflection_rate, first_response_accuracy, escalation_rate
  research     — synthesis_accuracy, citation_relevance_rate, hypothesis_hit_rate
  operations   — task_completion_rate, process_adherence_rate, anomaly_precision
  default      — memory_hit_rate, goal_completion_rate, action_success_rate

Usage::

    svc = TenantMetricsService()
    metrics = svc.get_metrics(TenantType.engineering)
    result = svc.evaluate(db, org_id="acme", tenant_type=TenantType.engineering,
                          start=..., end=..., outcomes=outcome_list)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TenantType(str, Enum):
    sales = "sales"
    engineering = "engineering"
    support = "support"
    research = "research"
    operations = "operations"
    default = "default"


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    description: str
    unit: str                    # "rate", "count", "score", "seconds"
    target: float                # desired value (e.g. 0.80 for 80% rate)
    higher_is_better: bool       # False for metrics like escalation_rate
    outcome_types: list[str]     # goal_type values counted as relevant
    success_outcomes: list[str]  # outcome statuses counted as success


# ---------------------------------------------------------------------------
# Metric catalogue
# ---------------------------------------------------------------------------

_METRICS: dict[TenantType, list[MetricDefinition]] = {

    TenantType.sales: [
        MetricDefinition(
            name="lead_conversion_rate",
            description="Fraction of lead-related actions that resulted in a conversion outcome",
            unit="rate",
            target=0.25,
            higher_is_better=True,
            outcome_types=["lead", "prospect", "opportunity"],
            success_outcomes=["converted", "success", "won"],
        ),
        MetricDefinition(
            name="follow_up_completion_rate",
            description="Fraction of follow-up tasks completed before their due date",
            unit="rate",
            target=0.85,
            higher_is_better=True,
            outcome_types=["follow_up", "outreach", "reminder"],
            success_outcomes=["completed", "success", "done"],
        ),
        MetricDefinition(
            name="pipeline_accuracy",
            description="Fraction of Ninai deal-stage predictions that matched actual outcomes",
            unit="rate",
            target=0.70,
            higher_is_better=True,
            outcome_types=["deal_stage_prediction", "forecast"],
            success_outcomes=["correct", "success", "matched"],
        ),
    ],

    TenantType.engineering: [
        MetricDefinition(
            name="bug_resolution_rate",
            description="Fraction of bug-related actions that resulted in a confirmed fix",
            unit="rate",
            target=0.80,
            higher_is_better=True,
            outcome_types=["bug", "defect", "incident", "error"],
            success_outcomes=["resolved", "fixed", "success", "closed"],
        ),
        MetricDefinition(
            name="knowledge_reuse_rate",
            description="Fraction of queries that matched an existing memory/playbook rather than triggering fresh LLM inference",
            unit="rate",
            target=0.60,
            higher_is_better=True,
            outcome_types=["knowledge_query", "doc_lookup", "how_to"],
            success_outcomes=["retrieved", "playbook_hit", "success", "matched"],
        ),
        MetricDefinition(
            name="deploy_success_rate",
            description="Fraction of deployment-related autonomous actions that completed without rollback",
            unit="rate",
            target=0.95,
            higher_is_better=True,
            outcome_types=["deploy", "release", "migration"],
            success_outcomes=["deployed", "success", "completed"],
        ),
    ],

    TenantType.support: [
        MetricDefinition(
            name="ticket_deflection_rate",
            description="Fraction of support queries resolved by memory retrieval without human escalation",
            unit="rate",
            target=0.55,
            higher_is_better=True,
            outcome_types=["support_query", "ticket", "help_request"],
            success_outcomes=["deflected", "self_served", "resolved", "success"],
        ),
        MetricDefinition(
            name="first_response_accuracy",
            description="Fraction of first Ninai responses rated correct by the support agent",
            unit="rate",
            target=0.75,
            higher_is_better=True,
            outcome_types=["first_response", "initial_answer"],
            success_outcomes=["accurate", "correct", "success", "approved"],
        ),
        MetricDefinition(
            name="escalation_rate",
            description="Fraction of tickets that required human escalation after Ninai response",
            unit="rate",
            target=0.20,
            higher_is_better=False,
            outcome_types=["escalation", "human_handoff"],
            success_outcomes=["escalated", "handed_off"],
        ),
    ],

    TenantType.research: [
        MetricDefinition(
            name="synthesis_accuracy",
            description="Fraction of research summaries rated factually accurate by domain experts",
            unit="rate",
            target=0.85,
            higher_is_better=True,
            outcome_types=["synthesis", "summary", "literature_review"],
            success_outcomes=["accurate", "verified", "success", "approved"],
        ),
        MetricDefinition(
            name="citation_relevance_rate",
            description="Fraction of retrieved sources rated relevant to the research question",
            unit="rate",
            target=0.70,
            higher_is_better=True,
            outcome_types=["citation", "source_retrieval", "reference_search"],
            success_outcomes=["relevant", "useful", "success", "accepted"],
        ),
        MetricDefinition(
            name="hypothesis_hit_rate",
            description="Fraction of Ninai-generated hypotheses later confirmed by experimental results",
            unit="rate",
            target=0.30,
            higher_is_better=True,
            outcome_types=["hypothesis", "prediction", "research_question"],
            success_outcomes=["confirmed", "validated", "success", "supported"],
        ),
    ],

    TenantType.operations: [
        MetricDefinition(
            name="task_completion_rate",
            description="Fraction of autonomous tasks completed without manual intervention",
            unit="rate",
            target=0.90,
            higher_is_better=True,
            outcome_types=["task", "workflow", "process_step"],
            success_outcomes=["completed", "success", "done", "executed"],
        ),
        MetricDefinition(
            name="process_adherence_rate",
            description="Fraction of actions that followed the defined playbook without deviation",
            unit="rate",
            target=0.85,
            higher_is_better=True,
            outcome_types=["process", "playbook_execution", "sop"],
            success_outcomes=["adherent", "compliant", "success", "followed"],
        ),
        MetricDefinition(
            name="anomaly_precision",
            description="Fraction of flagged anomalies that were confirmed as genuine issues",
            unit="rate",
            target=0.75,
            higher_is_better=True,
            outcome_types=["anomaly", "alert", "detection"],
            success_outcomes=["confirmed", "true_positive", "success", "verified"],
        ),
    ],

    TenantType.default: [
        MetricDefinition(
            name="memory_hit_rate",
            description="Fraction of queries that found relevant memories in the store",
            unit="rate",
            target=0.70,
            higher_is_better=True,
            outcome_types=["query", "search", "retrieve"],
            success_outcomes=["hit", "found", "success", "retrieved"],
        ),
        MetricDefinition(
            name="goal_completion_rate",
            description="Fraction of goals that reached a terminal success state",
            unit="rate",
            target=0.75,
            higher_is_better=True,
            outcome_types=["goal", "objective", "task"],
            success_outcomes=["completed", "achieved", "success", "done"],
        ),
        MetricDefinition(
            name="action_success_rate",
            description="Fraction of autonomous actions that succeeded without retry or rollback",
            unit="rate",
            target=0.85,
            higher_is_better=True,
            outcome_types=["action", "execution"],
            success_outcomes=["success", "succeeded", "completed"],
        ),
    ],
}


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

@dataclass
class MetricResult:
    metric: MetricDefinition
    value: float
    total: int
    successes: int
    meets_target: bool
    gap: float                       # value - target (negative = below target)


@dataclass
class TenantEvaluationReport:
    org_id: str
    tenant_type: TenantType
    window_start: datetime
    window_end: datetime
    metrics: list[MetricResult] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        """Mean of normalised metric values (0–1 per metric)."""
        if not self.metrics:
            return 0.0
        scores = []
        for m in self.metrics:
            if m.metric.higher_is_better:
                scores.append(min(1.0, m.value / max(m.metric.target, 1e-9)))
            else:
                # Lower is better: score = target / value (capped at 1)
                scores.append(min(1.0, m.metric.target / max(m.value, 1e-9)) if m.value > 0 else 1.0)
        return sum(scores) / len(scores)

    @property
    def metrics_meeting_target(self) -> int:
        return sum(1 for m in self.metrics if m.meets_target)


class TenantMetricsService:
    """Retrieves metric definitions and evaluates them against outcome data."""

    def get_metrics(self, tenant_type: TenantType | str) -> list[MetricDefinition]:
        """Return the metric definitions for the given tenant type."""
        if isinstance(tenant_type, str):
            try:
                tenant_type = TenantType(tenant_type.lower())
            except ValueError:
                tenant_type = TenantType.default
        return list(_METRICS.get(tenant_type, _METRICS[TenantType.default]))

    def all_tenant_types(self) -> list[TenantType]:
        return list(TenantType)

    def evaluate(
        self,
        *,
        org_id: str,
        tenant_type: TenantType | str,
        window_start: datetime,
        window_end: datetime,
        outcomes: list[dict[str, Any]],
    ) -> TenantEvaluationReport:
        """Compute metric values from a list of outcome dicts.

        Each outcome dict must contain:
          - goal_type (str)   : type of the action / goal
          - status (str)      : terminal status of the action
          - org_id (str)      : tenant scope (filtered by org_id)

        Returns a TenantEvaluationReport with per-metric results.
        """
        if isinstance(tenant_type, str):
            try:
                tenant_type = TenantType(tenant_type.lower())
            except ValueError:
                tenant_type = TenantType.default

        # Filter to this org
        org_outcomes = [o for o in outcomes if str(o.get("org_id", "")) == str(org_id)]

        metric_defs = self.get_metrics(tenant_type)
        results: list[MetricResult] = []

        for metric in metric_defs:
            relevant = [
                o for o in org_outcomes
                if str(o.get("goal_type", "")).lower() in {t.lower() for t in metric.outcome_types}
            ]
            total = len(relevant)
            if total == 0:
                value = 0.0
                successes = 0
            else:
                successes = sum(
                    1 for o in relevant
                    if str(o.get("status", "")).lower() in {s.lower() for s in metric.success_outcomes}
                )
                value = successes / total

            if metric.higher_is_better:
                meets_target = value >= metric.target
            else:
                meets_target = value <= metric.target

            results.append(MetricResult(
                metric=metric,
                value=value,
                total=total,
                successes=successes,
                meets_target=meets_target,
                gap=value - metric.target if metric.higher_is_better else metric.target - value,
            ))

        return TenantEvaluationReport(
            org_id=str(org_id),
            tenant_type=tenant_type,
            window_start=window_start,
            window_end=window_end,
            metrics=results,
        )
