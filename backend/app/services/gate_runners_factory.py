"""Gate Runners Factory — Real Gate E evaluators for AutoResearch benchmarking.

Provides factory functions to create gate runners (decide, plan, explain) that
wrap CognitiveGatewayService verbs and extract numerical scores for benchmarking.

Each gate runner:
- Accepts an org_id
- Calls the corresponding CognitiveGatewayService verb with synthetic content
- Extracts and returns the confidence score as a benchmark metric

Usage:
    riders = GateRunnersFactory.create_runners(gateway_svc)
    harness = AutoResearchBenchmarkHarness(gate_runners=runners)
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from app.services.cognitive_gateway_service import CognitiveGatewayService

logger = logging.getLogger(__name__)

GateRunner = Callable[[str], Awaitable[float | dict[str, Any]]]


class GateRunnersFactory:
    """Factory for creating real gate runners from CognitiveGatewayService."""

    @staticmethod
    async def run_decide_gate(gateway: CognitiveGatewayService, org_id: str) -> float:
        """Run decide gate: anomaly detection confidence.

        Simulates a decision task to evaluate the org's anomaly detection pipeline.
        Returns the confidence score (0.0-1.0) as a benchmark metric.
        """
        try:
            # Synthetic anomaly detection input
            sample_content = (
                "System response time increased from 100ms to 500ms over the last hour. "
                "Error rate spiked from 0.1% to 5%. Database connection pool exhausted. "
                "User reports indicate intermittent timeouts."
            )

            result = await gateway.decide(
                content=sample_content,
                enrichment={
                    "severity": "high",
                    "domain": "performance",
                },
                org_id=org_id,
            )

            return result.confidence
        except Exception as e:
            logger.warning(f"decide gate failed for org {org_id}: {e}")
            return 0.0

    @staticmethod
    async def run_plan_gate(gateway: CognitiveGatewayService, org_id: str) -> float:
        """Run plan gate: goal decomposition confidence.

        Simulates a planning task to evaluate the org's goal decomposition pipeline.
        Returns the confidence score (0.0-1.0) as a benchmark metric.
        """
        try:
            # Synthetic goal planning input
            sample_goal = (
                "Investigate the recent spike in database latency, identify root causes, "
                "and develop a remediation strategy."
            )

            result = await gateway.plan(
                goal=sample_goal,
                context={
                    "domain": "performance",
                    "priority": "high",
                },
                org_id=org_id,
            )

            return result.confidence
        except Exception as e:
            logger.warning(f"plan gate failed for org {org_id}: {e}")
            return 0.0

    @staticmethod
    async def run_explain_gate(gateway: CognitiveGatewayService, org_id: str) -> float:
        """Run explain gate: audit trail completeness confidence.

        Simulates an explainability task to evaluate the org's audit trail pipeline.
        Returns the confidence score (0.0-1.0) as a benchmark metric.
        """
        try:
            # Synthetic explain input — Use a synthetic memory ID
            sample_memory_id = "synthetic_memory_bench_001"
            sample_audit_records = [
                {
                    "timestamp": "2025-01-01T12:00:00Z",
                    "agent": "anomaly_detection",
                    "action": "flagged_high_latency",
                    "confidence": 0.95,
                },
                {
                    "timestamp": "2025-01-01T12:05:00Z",
                    "agent": "goal_decomposition",
                    "action": "planned_investigation",
                    "confidence": 0.87,
                },
            ]

            result = await gateway.explain(
                memory_id=sample_memory_id,
                audit_records=sample_audit_records,
                org_id=org_id,
            )

            return result.confidence
        except Exception as e:
            logger.warning(f"explain gate failed for org {org_id}: {e}")
            return 0.0

    @staticmethod
    def create_runners(
        gateway: CognitiveGatewayService,
    ) -> dict[str, GateRunner]:
        """Create a mapping of gate names to async runners.

        Args:
            gateway: Configured CognitiveGatewayService instance.

        Returns:
            Dict mapping gate names ("decide", "plan", "explain") to async runners.
            Each runner accepts org_id and returns float confidence score.
        """
        return {
            "decide": lambda org_id: GateRunnersFactory.run_decide_gate(gateway, org_id),
            "plan": lambda org_id: GateRunnersFactory.run_plan_gate(gateway, org_id),
            "explain": lambda org_id: GateRunnersFactory.run_explain_gate(gateway, org_id),
        }
