"""Proof scorecard service - Phase 54 Slice 1.

Deterministic scorecard and ROI computations used for productization proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from typing import Any


@dataclass(frozen=True)
class CognitiveOsScorecard:
    """Aggregated operational impact metrics for a tenant/time window."""

    lead_time_gain_pct: float
    sla_avoidance_rate: float
    mttr_delta_pct: float
    false_escalation_reduction_pct: float
    incidents_count: int
    score: float


@dataclass(frozen=True)
class MonthlyImpactReport:
    """Monthly ROI summary for proof and stakeholder reporting."""

    month: str
    tenant_id: str
    incidents_count: int
    lead_time_saved_hours: float
    mttr_saved_hours: float
    avoided_sla_penalty: float
    estimated_savings: float
    operating_cost: float
    net_impact: float
    roi_pct: float


class ProofScorecardService:
    """Compute reproducible scorecards and ROI baselines from raw logs."""

    def compute_scorecard(
        self,
        *,
        records: list[dict[str, Any]],
        baseline: dict[str, float],
    ) -> CognitiveOsScorecard:
        """Compute core Cognitive OS scorecard fields from raw operational logs."""
        total = len(records)
        if total == 0:
            return CognitiveOsScorecard(
                lead_time_gain_pct=0.0,
                sla_avoidance_rate=0.0,
                mttr_delta_pct=0.0,
                false_escalation_reduction_pct=0.0,
                incidents_count=0,
                score=0.0,
            )

        lead_times = [float(r.get("lead_time_hours", 0.0) or 0.0) for r in records]
        mttr_vals = [float(r.get("mttr_hours", 0.0) or 0.0) for r in records]
        avoided_count = sum(1 for r in records if bool(r.get("avoided_sla_breach")))
        false_count = sum(1 for r in records if bool(r.get("false_escalation")))

        avg_lead = sum(lead_times) / total
        avg_mttr = sum(mttr_vals) / total
        fp_rate = false_count / total

        baseline_lead = float(baseline.get("lead_time_hours", 0.0) or 0.0)
        baseline_mttr = float(baseline.get("mttr_hours", 0.0) or 0.0)
        baseline_fp_rate = float(baseline.get("false_escalation_rate", 0.0) or 0.0)

        lead_gain_pct = self._improvement_pct(baseline_lead, avg_lead)
        mttr_delta_pct = self._improvement_pct(baseline_mttr, avg_mttr)
        fp_reduction_pct = self._improvement_pct(baseline_fp_rate, fp_rate)
        sla_avoidance_rate = avoided_count / total

        # Weighted impact score (0..100)
        score = (
            0.30 * lead_gain_pct
            + 0.25 * (sla_avoidance_rate * 100.0)
            + 0.25 * mttr_delta_pct
            + 0.20 * fp_reduction_pct
        )

        return CognitiveOsScorecard(
            lead_time_gain_pct=round(lead_gain_pct, 2),
            sla_avoidance_rate=round(sla_avoidance_rate, 4),
            mttr_delta_pct=round(mttr_delta_pct, 2),
            false_escalation_reduction_pct=round(fp_reduction_pct, 2),
            incidents_count=total,
            score=round(max(0.0, score), 2),
        )

    def compute_monthly_roi_report(
        self,
        *,
        tenant_id: str,
        month: str,
        records: list[dict[str, Any]],
        baseline: dict[str, float],
        labor_cost_per_hour: float = 120.0,
        false_escalation_cost: float = 250.0,
        monthly_operating_cost: float = 3000.0,
    ) -> MonthlyImpactReport:
        """Compute a deterministic monthly ROI report from incident records."""
        total = len(records)
        lead_times = [float(r.get("lead_time_hours", 0.0) or 0.0) for r in records]
        mttr_vals = [float(r.get("mttr_hours", 0.0) or 0.0) for r in records]
        false_count = sum(1 for r in records if bool(r.get("false_escalation")))
        avoided_count = sum(1 for r in records if bool(r.get("avoided_sla_breach")))

        baseline_lead = float(baseline.get("lead_time_hours", 0.0) or 0.0)
        baseline_mttr = float(baseline.get("mttr_hours", 0.0) or 0.0)
        baseline_fp_rate = float(baseline.get("false_escalation_rate", 0.0) or 0.0)
        avg_lead = (sum(lead_times) / total) if total else 0.0
        avg_mttr = (sum(mttr_vals) / total) if total else 0.0

        lead_time_saved_hours = max(0.0, baseline_lead - avg_lead) * total
        mttr_saved_hours = max(0.0, baseline_mttr - avg_mttr) * total

        baseline_false_count = baseline_fp_rate * total
        false_reduction_count = max(0.0, baseline_false_count - false_count)

        avoided_sla_penalty = float(baseline.get("sla_penalty_per_breach", 0.0) or 0.0) * avoided_count
        estimated_savings = (
            (lead_time_saved_hours + mttr_saved_hours) * labor_cost_per_hour
            + false_reduction_count * false_escalation_cost
            + avoided_sla_penalty
        )
        net_impact = estimated_savings - monthly_operating_cost
        roi_pct = (net_impact / monthly_operating_cost * 100.0) if monthly_operating_cost > 0 else 0.0

        return MonthlyImpactReport(
            month=month,
            tenant_id=tenant_id,
            incidents_count=total,
            lead_time_saved_hours=round(lead_time_saved_hours, 2),
            mttr_saved_hours=round(mttr_saved_hours, 2),
            avoided_sla_penalty=round(avoided_sla_penalty, 2),
            estimated_savings=round(estimated_savings, 2),
            operating_cost=round(monthly_operating_cost, 2),
            net_impact=round(net_impact, 2),
            roi_pct=round(roi_pct, 2),
        )

    @staticmethod
    def reproducibility_hash(*, records: list[dict[str, Any]], baseline: dict[str, float]) -> str:
        """Create deterministic hash proving scorecard reproducibility from raw logs."""
        normalized_records = [ProofScorecardService._normalize_record(r) for r in records]
        payload = {
            "records": sorted(normalized_records, key=lambda x: x.get("incident_id", "")),
            "baseline": dict(sorted((baseline or {}).items())),
            "schema_version": 1,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _improvement_pct(baseline_val: float, observed_val: float) -> float:
        if baseline_val <= 0:
            return 0.0
        return ((baseline_val - observed_val) / baseline_val) * 100.0

    @staticmethod
    def _normalize_record(r: dict[str, Any]) -> dict[str, Any]:
        ts = r.get("timestamp")
        if isinstance(ts, datetime):
            ts = ts.isoformat()
        return {
            "incident_id": str(r.get("incident_id") or ""),
            "timestamp": str(ts or ""),
            "lead_time_hours": float(r.get("lead_time_hours", 0.0) or 0.0),
            "mttr_hours": float(r.get("mttr_hours", 0.0) or 0.0),
            "avoided_sla_breach": bool(r.get("avoided_sla_breach")),
            "false_escalation": bool(r.get("false_escalation")),
        }
