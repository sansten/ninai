"""Bounded auto-research agent for tenant-scoped parameter experiments."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.models.cognitive_experiment_ledger import CognitiveExperimentLedger
from app.services.auto_research_benchmark_harness import AutoResearchBenchmarkHarness, BenchmarkSummary
from app.services.config_snapshot_service import AUTO_RESEARCH_PARAMETER_REGISTRY, ConfigSnapshotService, ParameterSpec
from app.services.cognitive_gateway_service import CognitiveGatewayService


def _round_score(value: float) -> float:
    return round(float(value), 6)


def _candidate_values(current_value: float, spec: ParameterSpec) -> list[float]:
    midpoint = (spec.min_value + spec.max_value) / 2.0
    deltas = [spec.step, -spec.step] if current_value <= midpoint else [-spec.step, spec.step]
    candidates: list[float] = []
    for delta in deltas:
        candidate = round(max(spec.min_value, min(spec.max_value, current_value + delta)), 6)
        if candidate != round(current_value, 6) and candidate not in candidates:
            candidates.append(candidate)
    return candidates


class AutoResearchAgent(BaseAgent):
    name = "AutoResearchAgent"
    version = "v1"

    def __init__(
        self,
        *,
        session: AsyncSession | None = None,
        config_service: ConfigSnapshotService | None = None,
        benchmark_harness: AutoResearchBenchmarkHarness | None = None,
        gateway: CognitiveGatewayService | None = None,
    ):
        self.session = session
        self.config_service = config_service
        self.benchmark_harness = benchmark_harness
        self.gateway = gateway

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}
        if not isinstance(outputs.get("selected_parameter"), str):
            raise ValueError("selected_parameter must be a string")
        if not isinstance(outputs.get("experiments"), list):
            raise ValueError("experiments must be a list")
        if not isinstance(outputs.get("accepted"), bool):
            raise ValueError("accepted must be a bool")
        for key in ("baseline_score", "final_score", "score_delta"):
            if not isinstance(outputs.get(key), (int, float)):
                raise ValueError(f"{key} must be numeric")

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        enrichment = (context.get("memory") or {}).get("enrichment") or {}
        tenant = context.get("tenant") or {}
        org_id = str(tenant.get("org_id") or enrichment.get("org_id") or "").strip()

        if not org_id:
            finished_at = datetime.now(timezone.utc)
            return AgentResult(
                agent_name=self.name,
                agent_version=self.version,
                memory_id=memory_id,
                status="failed",
                confidence=0.0,
                outputs={},
                warnings=[],
                errors=["org_id is required for auto research"],
                started_at=started_at,
                finished_at=finished_at,
                trace_id=trace_id,
            )

        session = self.session or enrichment.get("session")
        if session is None and (self.config_service is None or self.benchmark_harness is None):
            raise ValueError("AutoResearchAgent requires a session or explicit service dependencies")

        config_service = self.config_service or ConfigSnapshotService(session)
        harness = self.benchmark_harness or AutoResearchBenchmarkHarness(session=session, gateway=self.gateway)

        requested_keys = [
            key for key in (enrichment.get("parameter_keys") or []) if key in AUTO_RESEARCH_PARAMETER_REGISTRY
        ]
        if not requested_keys:
            requested_keys = list(AUTO_RESEARCH_PARAMETER_REGISTRY)

        selected_parameter = str(enrichment.get("parameter_key") or requested_keys[0])
        if selected_parameter not in AUTO_RESEARCH_PARAMETER_REGISTRY:
            raise ValueError(f"Unknown auto-research parameter: {selected_parameter}")

        min_improvement = float(enrichment.get("min_improvement") or 0.01)
        baseline_snapshot = await config_service.snapshot(org_id, [selected_parameter])
        baseline_summary = await harness.evaluate(org_id, label="baseline")
        spec = AUTO_RESEARCH_PARAMETER_REGISTRY[selected_parameter]
        current_value = baseline_snapshot[selected_parameter]

        experiments: list[dict[str, Any]] = []
        accepted = False
        applied_value = current_value
        final_summary = baseline_summary

        for candidate_value in _candidate_values(current_value, spec):
            await config_service.set_parameter_value(org_id, selected_parameter, candidate_value)
            candidate_summary = await harness.evaluate(org_id, label=f"candidate:{selected_parameter}")
            delta = _round_score(candidate_summary.composite_score - baseline_summary.composite_score)
            accepted = delta >= min_improvement
            status = "accepted" if accepted else "reverted"
            experiments.append(
                {
                    "parameter_key": selected_parameter,
                    "baseline_value": current_value,
                    "candidate_value": candidate_value,
                    "baseline_score": baseline_summary.composite_score,
                    "candidate_score": candidate_summary.composite_score,
                    "score_delta": delta,
                    "status": status,
                    "benchmark_summary": asdict(candidate_summary),
                }
            )
            await self._record_experiment(
                org_id=org_id,
                parameter_key=selected_parameter,
                baseline_value=current_value,
                candidate_value=candidate_value,
                baseline_summary=baseline_summary,
                candidate_summary=candidate_summary,
                status=status,
                trace_id=trace_id,
            )
            if accepted:
                applied_value = candidate_value
                final_summary = candidate_summary
                break
            await config_service.restore_snapshot(org_id, baseline_snapshot)

        if not accepted:
            await config_service.restore_snapshot(org_id, baseline_snapshot)

        finished_at = datetime.now(timezone.utc)
        outputs = {
            "selected_parameter": selected_parameter,
            "baseline_score": baseline_summary.composite_score,
            "final_score": final_summary.composite_score,
            "score_delta": _round_score(final_summary.composite_score - baseline_summary.composite_score),
            "accepted": accepted,
            "applied_value": applied_value,
            "baseline_value": current_value,
            "experiments": experiments,
            "baseline_benchmark": asdict(baseline_summary),
            "final_benchmark": asdict(final_summary),
            "rationale": "heuristic",
        }
        result = AgentResult(
            agent_name=self.name,
            agent_version=self.version,
            memory_id=memory_id,
            status="success",
            confidence=0.8 if accepted else 0.6,
            outputs=outputs,
            warnings=[] if experiments else ["no candidate values available for selected parameter"],
            errors=[],
            started_at=started_at,
            finished_at=finished_at,
            trace_id=trace_id,
        )
        self.validate_outputs(result)
        return result

    async def _record_experiment(
        self,
        *,
        org_id: str,
        parameter_key: str,
        baseline_value: float,
        candidate_value: float,
        baseline_summary: BenchmarkSummary,
        candidate_summary: BenchmarkSummary,
        status: str,
        trace_id: str | None,
    ) -> None:
        if self.session is None:
            return
        self.session.add(
            CognitiveExperimentLedger(
                org_id=org_id,
                agent_name=self.name,
                parameter_key=parameter_key,
                baseline_value=baseline_value,
                candidate_value=candidate_value,
                baseline_score=baseline_summary.composite_score,
                candidate_score=candidate_summary.composite_score,
                score_delta=_round_score(candidate_summary.composite_score - baseline_summary.composite_score),
                status=status,
                benchmark_summary={
                    "trace_id": trace_id,
                    "baseline": asdict(baseline_summary),
                    "candidate": asdict(candidate_summary),
                },
                rationale="bounded single-parameter experiment",
            )
        )
        await self.session.flush()