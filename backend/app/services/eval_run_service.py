"""Evaluation service for memory retrieval quality metrics."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.eval_run import EvalRun
from app.models.eval_suite import EvalSuite


class EvalRunService:
    """Service for running evaluation suites and computing memory quality metrics.
    
    Computes:
    - precision@k, recall@k: Retrieval accuracy
    - MRR (Mean Reciprocal Rank): First relevant result position
    - NDCG@k (Normalized Discounted Cumulative Gain): Ranked quality
    - cross_tenant_leak_rate: Cross-org data leakage
    - policy_violation_rate: Access control failures
    - stale_recall_rate: Outdated memory recall
    - contradiction_recall_rate: Contradicted fact recall
    - topk_jaccard_stability: Result consistency
    - latency metrics: p50, p95, p99
    """

    def __init__(self, session: AsyncSession, org_id: str):
        self.session = session
        self.org_id = org_id

    async def create_eval_run(
        self, *, suite_id: str, config: dict[str, Any] | None = None
    ) -> str:
        """Create a new evaluation run for a suite.
        
        Args:
            suite_id: ID of the evaluation suite to run
            config: Configuration for this run (k values, thresholds, etc.)
            
        Returns:
            ID of the created eval run
        """
        if config is None:
            config = {"k_values": [1, 3, 5, 10], "leak_check_enabled": True}

        eval_run = EvalRun(
            id=str(uuid4()),
            organization_id=self.org_id,
            suite_id=suite_id,
            started_at=datetime.utcnow(),
            config=config,
            metrics={},
            status="running",
        )
        self.session.add(eval_run)
        await self.session.flush()
        return eval_run.id

    async def compute_metrics(
        self, eval_run_id: str, results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Compute evaluation metrics from query results.
        
        Args:
            eval_run_id: ID of the eval run
            results: List of query results with actual vs expected
            
        Returns:
            Dictionary of computed metrics
        """
        # Get eval run to fetch config
        result = await self.session.execute(
            select(EvalRun).where(EvalRun.id == eval_run_id)
        )
        eval_run = result.scalar_one()
        k_values = eval_run.config.get("k_values", [1, 3, 5, 10])

        metrics = {}
        
        # Compute precision and recall at k
        for k in k_values:
            precision_sum = 0.0
            recall_sum = 0.0
            
            for query_result in results:
                actual_ids = set(query_result["actual_ids"][:k])
                expected_ids = set(query_result["expected_ids"])
                
                if len(actual_ids) > 0:
                    precision_sum += len(actual_ids & expected_ids) / len(actual_ids)
                if len(expected_ids) > 0:
                    recall_sum += len(actual_ids & expected_ids) / len(expected_ids)
            
            num_queries = len(results) if results else 1
            metrics[f"precision_at_{k}"] = precision_sum / num_queries
            metrics[f"recall_at_{k}"] = recall_sum / num_queries

        # Compute MRR (Mean Reciprocal Rank)
        mrr_sum = 0.0
        for query_result in results:
            expected_ids = set(query_result["expected_ids"])
            actual_ids = query_result["actual_ids"]
            
            for rank, actual_id in enumerate(actual_ids, start=1):
                if actual_id in expected_ids:
                    mrr_sum += 1.0 / rank
                    break
        
        metrics["mrr"] = mrr_sum / len(results) if results else 0.0

        # Compute NDCG@k
        for k in k_values:
            ndcg_sum = 0.0
            for query_result in results:
                actual_ids = query_result["actual_ids"][:k]
                expected_ids = set(query_result["expected_ids"])
                
                # DCG (Discounted Cumulative Gain)
                dcg = sum(
                    (1 if actual_id in expected_ids else 0) / (i + 1)
                    for i, actual_id in enumerate(actual_ids)
                )
                
                # IDCG (Ideal DCG)
                idcg = sum(1 / (i + 1) for i in range(min(k, len(expected_ids))))
                
                ndcg_sum += dcg / idcg if idcg > 0 else 0.0
            
            metrics[f"ndcg_at_{k}"] = ndcg_sum / len(results) if results else 0.0

        # Compute cross-tenant leak rate
        leak_count = sum(
            1 for query_result in results if query_result.get("leaked_orgs", [])
        )
        metrics["cross_tenant_leak_rate"] = leak_count / len(results) if results else 0.0

        # Compute policy violation rate
        violation_count = sum(
            1 for query_result in results if query_result.get("policy_violations", [])
        )
        metrics["policy_violation_rate"] = violation_count / len(results) if results else 0.0

        # Compute stale recall rate
        stale_count = sum(
            1 for query_result in results 
            if any(mem.get("is_stale", False) for mem in query_result.get("actual_memories", []))
        )
        metrics["stale_recall_rate"] = stale_count / len(results) if results else 0.0

        # Compute contradiction recall rate
        contradiction_count = sum(
            1 for query_result in results 
            if any(mem.get("has_contradiction", False) for mem in query_result.get("actual_memories", []))
        )
        metrics["contradiction_recall_rate"] = contradiction_count / len(results) if results else 0.0

        # Compute top-k Jaccard stability (if baseline exists)
        if "baseline_results" in results[0] if results else {}:
            jaccard_sum = 0.0
            for query_result in results:
                actual = set(query_result["actual_ids"][:10])
                baseline = set(query_result.get("baseline_results", [])[:10])
                
                if len(actual | baseline) > 0:
                    jaccard_sum += len(actual & baseline) / len(actual | baseline)
            
            metrics["topk_jaccard_stability"] = jaccard_sum / len(results) if results else 0.0

        # Compute latency metrics
        latencies = [query_result["latency_ms"] for query_result in results if "latency_ms" in query_result]
        if latencies:
            latencies_sorted = sorted(latencies)
            n = len(latencies_sorted)
            metrics["latency_p50"] = latencies_sorted[int(n * 0.5)]
            metrics["latency_p95"] = latencies_sorted[int(n * 0.95)]
            metrics["latency_p99"] = latencies_sorted[int(n * 0.99)]

        return metrics

    async def finalize_eval_run(
        self, eval_run_id: str, metrics: dict[str, Any], status: str = "success", error: str | None = None
    ) -> None:
        """Mark eval run as completed and save metrics.
        
        Args:
            eval_run_id: ID of the eval run
            metrics: Computed metrics
            status: Final status (success/failure)
            error: Error message if failed
        """
        result = await self.session.execute(
            select(EvalRun).where(EvalRun.id == eval_run_id)
        )
        eval_run = result.scalar_one()
        
        eval_run.finished_at = datetime.utcnow()
        eval_run.metrics = metrics
        eval_run.status = status
        if error:
            eval_run.error_message = error
        
        await self.session.flush()

    async def get_eval_run(self, eval_run_id: str) -> EvalRun | None:
        """Get an eval run by ID.
        
        Args:
            eval_run_id: ID of the eval run
            
        Returns:
            EvalRun or None if not found
        """
        result = await self.session.execute(
            select(EvalRun).where(
                EvalRun.id == eval_run_id,
                EvalRun.organization_id == self.org_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_eval_runs(self, suite_id: str | None = None, limit: int = 50) -> list[EvalRun]:
        """List eval runs for the organization.
        
        Args:
            suite_id: Optional suite ID to filter by
            limit: Maximum number of runs to return
            
        Returns:
            List of eval runs
        """
        query = select(EvalRun).where(EvalRun.organization_id == self.org_id)
        
        if suite_id:
            query = query.where(EvalRun.suite_id == suite_id)
        
        query = query.order_by(EvalRun.started_at.desc()).limit(limit)
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
