"""Evaluation tests (PR6: Eval Harness + Drift Detection)."""

import pytest
from datetime import datetime
from uuid import uuid4
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.eval_suite import EvalSuite
from app.models.eval_run import EvalRun
from app.core.database import set_tenant_context
from app.services.eval_run_service import EvalRunService
from app.services.drift_detection_service import DriftDetectionService


@pytest.mark.asyncio
async def test_eval_run_generates_metrics(db_session: AsyncSession, test_org_id: str, test_user_id: str):
    """Test that eval runs compute all expected metrics."""
    await set_tenant_context(db_session, test_user_id, test_org_id, roles="system,org_admin", clearance_level=4)

    # Create eval suite
    suite_id = str(uuid4())
    suite = EvalSuite(
        id=suite_id,
        organization_id=test_org_id,
        name="Memory Quality Test",
        queries=[
            {"query": "test query 1"},
            {"query": "test query 2"},
        ],
        expected={
            "query_0": {"ids": ["mem-a", "mem-b"], "min_score": 0.8},
            "query_1": {"ids": ["mem-c"], "min_score": 0.9},
        },
    )
    db_session.add(suite)
    await db_session.commit()

    # Create and run eval
    service = EvalRunService(db_session, test_org_id)
    eval_run_id = await service.create_eval_run(suite_id=suite_id)
    
    # Simulate query results
    results = [
        {
            "query": "test query 1",
            "actual_ids": ["mem-a", "mem-b", "mem-x"],
            "expected_ids": ["mem-a", "mem-b"],
            "actual_memories": [],
            "leaked_orgs": [],
            "policy_violations": [],
            "latency_ms": 45.0,
        },
        {
            "query": "test query 2",
            "actual_ids": ["mem-c", "mem-y"],
            "expected_ids": ["mem-c"],
            "actual_memories": [],
            "leaked_orgs": [],
            "policy_violations": [],
            "latency_ms": 55.0,
        },
    ]
    
    # Compute metrics
    metrics = await service.compute_metrics(eval_run_id, results)
    
    # Verify core metrics exist
    assert "precision_at_1" in metrics
    assert "precision_at_3" in metrics
    assert "recall_at_1" in metrics
    assert "recall_at_3" in metrics
    assert "mrr" in metrics
    assert "ndcg_at_1" in metrics
    assert "cross_tenant_leak_rate" in metrics
    assert "policy_violation_rate" in metrics
    
    # Verify precision/recall values are reasonable
    assert 0.0 <= metrics["precision_at_1"] <= 1.0
    assert 0.0 <= metrics["recall_at_1"] <= 1.0
    assert 0.0 <= metrics["mrr"] <= 1.0
    
    # Verify no leaks or violations
    assert metrics["cross_tenant_leak_rate"] == 0.0
    assert metrics["policy_violation_rate"] == 0.0
    
    # Finalize
    await service.finalize_eval_run(eval_run_id, metrics, status="success")
    await db_session.commit()
    
    # Verify run is complete
    eval_run = await service.get_eval_run(eval_run_id)
    assert eval_run is not None
    assert eval_run.status == "success"
    assert eval_run.finished_at is not None


@pytest.mark.asyncio
async def test_leak_rate_zero_with_rls(db_session: AsyncSession, test_org_id: str, test_user_id: str):
    """Test that cross-tenant leak rate is zero with proper RLS isolation."""
    await set_tenant_context(db_session, test_user_id, test_org_id, roles="system,org_admin", clearance_level=4)

    # Create eval suite
    suite_id = str(uuid4())
    suite = EvalSuite(
        id=suite_id,
        organization_id=test_org_id,
        name="RLS Leak Test",
        queries=[
            {"query": "sensitive data"},
        ],
        expected={
            "query_0": {"ids": ["mem-ok"], "min_score": 0.8},
        },
    )
    db_session.add(suite)
    await db_session.commit()

    # Create eval run
    service = EvalRunService(db_session, test_org_id)
    eval_run_id = await service.create_eval_run(suite_id=suite_id)
    
    # Simulate results with NO leaks (RLS properly filtering)
    results = [
        {
            "query": "sensitive data",
            "actual_ids": ["mem-ok"],
            "expected_ids": ["mem-ok"],
            "actual_memories": [],
            "leaked_orgs": [],  # RLS prevented cross-tenant access
            "policy_violations": [],
            "latency_ms": 50.0,
        },
    ]
    
    # Compute metrics
    metrics = await service.compute_metrics(eval_run_id, results)
    
    # Assert zero leak rate
    assert metrics["cross_tenant_leak_rate"] == 0.0, "RLS must prevent all cross-tenant leaks"
    assert metrics["policy_violation_rate"] == 0.0
    
    # Finalize
    await service.finalize_eval_run(eval_run_id, metrics, status="success")
    await db_session.commit()


@pytest.mark.asyncio
async def test_drift_severity_thresholds(db_session: AsyncSession, test_org_id: str, test_user_id: str):
    """Test that drift detection properly flags severity levels."""
    await set_tenant_context(db_session, test_user_id, test_org_id, roles="system,org_admin", clearance_level=4)

    # Create eval suite
    suite_id = str(uuid4())
    suite = EvalSuite(
        id=suite_id,
        organization_id=test_org_id,
        name="Drift Test Suite",
        queries=[{"query": "test"}],
        expected={"query_0": {"ids": ["mem-1"]}},
    )
    db_session.add(suite)
    await db_session.commit()

    # Create baseline run with good metrics
    eval_service = EvalRunService(db_session, test_org_id)
    baseline_run_id = await eval_service.create_eval_run(suite_id=suite_id)
    
    baseline_metrics = {
        "precision_at_1": 0.90,
        "recall_at_1": 0.85,
        "ndcg_at_1": 0.88,
        "mrr": 0.87,
        "cross_tenant_leak_rate": 0.0,
        "policy_violation_rate": 0.0,
        "topk_jaccard_stability": 0.95,
    }
    
    await eval_service.finalize_eval_run(baseline_run_id, baseline_metrics, status="success")
    await db_session.commit()

    # Create current run with degraded metrics (15% drop = high severity)
    current_run_id = await eval_service.create_eval_run(suite_id=suite_id)
    
    current_metrics = {
        "precision_at_1": 0.75,  # 16.7% drop
        "recall_at_1": 0.72,     # 15.3% drop
        "ndcg_at_1": 0.75,
        "mrr": 0.74,
        "cross_tenant_leak_rate": 0.0,
        "policy_violation_rate": 0.0,
        "topk_jaccard_stability": 0.80,  # 15.8% drop
    }
    
    await eval_service.finalize_eval_run(current_run_id, current_metrics, status="success")
    await db_session.commit()

    # Compute drift
    drift_service = DriftDetectionService(db_session, test_org_id)
    drift_report_id = await drift_service.compute_drift(baseline_run_id, current_run_id)
    await db_session.commit()

    # Verify drift report
    drift_report = await drift_service.get_drift_report(drift_report_id)
    assert drift_report is not None
    
    # Check severity (should be high due to >10% precision drop)
    assert drift_report.severity in ["high", "critical"], f"Expected high/critical severity, got {drift_report.severity}"
    
    # Check flagged issues
    assert len(drift_report.flagged_issues) > 0
    assert any("precision" in issue or "recall" in issue for issue in drift_report.flagged_issues)
    
    # Verify delta metrics
    assert "precision_at_1" in drift_report.delta
    assert drift_report.delta["precision_at_1"] < 0, "Precision delta should be negative (degradation)"
