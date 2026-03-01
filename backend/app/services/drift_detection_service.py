"""Drift detection service for memory quality regression monitoring."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.drift_report import DriftReport
from app.models.eval_run import EvalRun


class DriftDetectionService:
    """Service for detecting memory quality drift (regressions) over time.
    
    Compares metrics between baseline and current eval runs to flag:
    - Precision/recall drops
    - Increased cross-tenant leak rates
    - Policy violation increases
    - Reduced topk Jaccard stability
    - Latency regressions
    
    Assigns severity levels for alerting:
    - none: No significant drift
    - low: <5% degradation
    - medium: 5-10% degradation
    - high: 10-20% degradation
    - critical: >20% degradation or any leak rate increase
    """

    # Drift thresholds (percentage change)
    THRESHOLDS = {
        "low": 0.05,      # 5%
        "medium": 0.08,   # 8%
        "high": 0.10,     # 10%
        "critical": 0.20, # 20%
    }

    def __init__(self, session: AsyncSession, org_id: str):
        self.session = session
        self.org_id = org_id

    async def compute_drift(
        self, baseline_run_id: str, current_run_id: str
    ) -> str:
        """Compute drift between baseline and current eval runs.
        
        Args:
            baseline_run_id: ID of the baseline eval run
            current_run_id: ID of the current eval run
            
        Returns:
            ID of the created drift report
        """
        # Fetch both runs
        result = await self.session.execute(
            select(EvalRun).where(
                EvalRun.id.in_([baseline_run_id, current_run_id]),
                EvalRun.organization_id == self.org_id,
            )
        )
        runs = {run.id: run for run in result.scalars().all()}
        
        if len(runs) != 2:
            raise ValueError("Could not find both baseline and current eval runs")
        
        baseline = runs[baseline_run_id]
        current = runs[current_run_id]

        # Compute delta metrics
        delta = self._compute_delta_metrics(baseline.metrics, current.metrics)
        
        # Determine severity and flagged issues
        severity, flagged_issues = self._assess_severity(delta)

        # Create drift report
        drift_report = DriftReport(
            id=str(uuid4()),
            organization_id=self.org_id,
            baseline_run_id=baseline_run_id,
            current_run_id=current_run_id,
            delta=delta,
            severity=severity,
            flagged_issues=flagged_issues,
            created_at=datetime.utcnow(),
        )
        
        self.session.add(drift_report)
        await self.session.flush()
        return drift_report.id

    def _compute_delta_metrics(
        self, baseline: dict[str, Any], current: dict[str, Any]
    ) -> dict[str, Any]:
        """Compute delta (change) between baseline and current metrics.
        
        Args:
            baseline: Baseline metrics
            current: Current metrics
            
        Returns:
            Dictionary of delta values (percentage change)
        """
        delta = {}
        
        # Metrics where lower is worse (precision, recall, ndcg, mrr, stability)
        positive_metrics = [
            "precision_at_1", "precision_at_3", "precision_at_5", "precision_at_10",
            "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
            "ndcg_at_1", "ndcg_at_3", "ndcg_at_5", "ndcg_at_10",
            "mrr", "topk_jaccard_stability",
        ]
        
        # Metrics where higher is worse (leak rate, violations, stale/contradiction rates, latency)
        negative_metrics = [
            "cross_tenant_leak_rate", "policy_violation_rate",
            "stale_recall_rate", "contradiction_recall_rate",
            "latency_p50", "latency_p95", "latency_p99",
        ]
        
        # Compute deltas for positive metrics (drop is bad)
        for metric in positive_metrics:
            if metric in baseline and metric in current:
                baseline_val = baseline[metric]
                current_val = current[metric]
                
                if baseline_val > 0:
                    # Negative delta = degradation
                    delta[metric] = (current_val - baseline_val) / baseline_val
                else:
                    delta[metric] = 0.0
        
        # Compute deltas for negative metrics (increase is bad)
        for metric in negative_metrics:
            if metric in baseline and metric in current:
                baseline_val = baseline[metric]
                current_val = current[metric]
                
                if baseline_val > 0:
                    # Positive delta = degradation for these metrics
                    delta[metric] = (current_val - baseline_val) / baseline_val
                elif current_val > 0:
                    # Baseline was 0, now non-zero = bad
                    delta[metric] = 1.0
                else:
                    delta[metric] = 0.0
        
        return delta

    def _assess_severity(
        self, delta: dict[str, Any]
    ) -> tuple[str, list[str]]:
        """Assess drift severity and flag specific issues.
        
        Args:
            delta: Delta metrics
            
        Returns:
            Tuple of (severity_level, list_of_flagged_issues)
        """
        flagged_issues = []
        max_severity = "none"
        severity_levels = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}

        # Critical: Any leak rate increase
        if delta.get("cross_tenant_leak_rate", 0.0) > 0:
            flagged_issues.append("cross_tenant_leak_increase")
            max_severity = "critical"

        # Check precision drops
        precision_drops = [
            abs(delta.get(f"precision_at_{k}", 0.0))
            for k in [1, 3, 5, 10]
            if delta.get(f"precision_at_{k}", 0.0) < 0
        ]
        if precision_drops:
            max_drop = max(precision_drops)
            if max_drop > self.THRESHOLDS["critical"]:
                flagged_issues.append("precision_drop_critical")
                max_severity = self._update_severity(max_severity, "critical", severity_levels)
            elif max_drop > self.THRESHOLDS["high"]:
                flagged_issues.append("precision_drop_high")
                max_severity = self._update_severity(max_severity, "high", severity_levels)
            elif max_drop > self.THRESHOLDS["medium"]:
                flagged_issues.append("precision_drop_medium")
                max_severity = self._update_severity(max_severity, "medium", severity_levels)
            elif max_drop > self.THRESHOLDS["low"]:
                flagged_issues.append("precision_drop_low")
                max_severity = self._update_severity(max_severity, "low", severity_levels)

        # Check recall drops
        recall_drops = [
            abs(delta.get(f"recall_at_{k}", 0.0))
            for k in [1, 3, 5, 10]
            if delta.get(f"recall_at_{k}", 0.0) < 0
        ]
        if recall_drops:
            max_drop = max(recall_drops)
            if max_drop > self.THRESHOLDS["critical"]:
                flagged_issues.append("recall_drop_critical")
                max_severity = self._update_severity(max_severity, "critical", severity_levels)
            elif max_drop > self.THRESHOLDS["high"]:
                flagged_issues.append("recall_drop")
                max_severity = self._update_severity(max_severity, "high", severity_levels)
            elif max_drop > self.THRESHOLDS["medium"]:
                max_severity = self._update_severity(max_severity, "medium", severity_levels)

        # Check stability drops
        stability_delta = delta.get("topk_jaccard_stability", 0.0)
        if stability_delta < -self.THRESHOLDS["medium"]:
            flagged_issues.append("stability_drop")
            max_severity = self._update_severity(max_severity, "medium", severity_levels)

        # Check policy violations increase
        policy_delta = delta.get("policy_violation_rate", 0.0)
        if policy_delta > self.THRESHOLDS["low"]:
            flagged_issues.append("policy_violation_increase")
            max_severity = self._update_severity(max_severity, "high", severity_levels)

        # Check latency regressions
        latency_p95_delta = delta.get("latency_p95", 0.0)
        if latency_p95_delta > self.THRESHOLDS["high"]:
            flagged_issues.append("latency_regression")
            max_severity = self._update_severity(max_severity, "medium", severity_levels)

        return max_severity, flagged_issues

    def _update_severity(
        self, current: str, new: str, levels: dict[str, int]
    ) -> str:
        """Update severity to the higher level.
        
        Args:
            current: Current severity
            new: New severity to consider
            levels: Severity level mapping
            
        Returns:
            Higher severity level
        """
        return current if levels[current] > levels[new] else new

    async def get_drift_report(self, drift_report_id: str) -> DriftReport | None:
        """Get a drift report by ID.
        
        Args:
            drift_report_id: ID of the drift report
            
        Returns:
            DriftReport or None if not found
        """
        result = await self.session.execute(
            select(DriftReport).where(
                DriftReport.id == drift_report_id,
                DriftReport.organization_id == self.org_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_drift_reports(
        self, severity: str | None = None, limit: int = 50
    ) -> list[DriftReport]:
        """List drift reports for the organization.
        
        Args:
            severity: Optional severity level to filter by
            limit: Maximum number of reports to return
            
        Returns:
            List of drift reports
        """
        query = select(DriftReport).where(DriftReport.organization_id == self.org_id)
        
        if severity:
            query = query.where(DriftReport.severity == severity)
        
        query = query.order_by(DriftReport.created_at.desc()).limit(limit)
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
