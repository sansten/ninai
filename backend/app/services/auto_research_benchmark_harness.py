"""Benchmark harness for bounded auto-research experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from statistics import mean
from time import perf_counter
from typing import Any, Awaitable, Callable, Mapping

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.benchmark_run import BenchmarkRun
from app.services.cognitive_gateway_service import CognitiveGatewayService
from app.services.gate_runners_factory import GateRunnersFactory


GateRunner = Callable[[str], Awaitable[float | dict[str, Any]]]


@dataclass
class GateScore:
    name: str
    score: float
    details: dict[str, Any]


@dataclass
class BenchmarkSummary:
    composite_score: float
    gate_scores: dict[str, float]
    gates_completed: int
    duration_seconds: float
    budget_exceeded: bool
    results: list[dict[str, Any]]


class AutoResearchBenchmarkHarness:
    def __init__(
        self,
        *,
        gate_runners: Mapping[str, GateRunner] | None = None,
        gateway: CognitiveGatewayService | None = None,
        session: AsyncSession | None = None,
        budget_seconds: float = 60.0,
    ):
        # If gateway is provided and no gate_runners, create real runners from gateway
        if gateway is not None and not gate_runners:
            gate_runners = GateRunnersFactory.create_runners(gateway)
        self.gate_runners = dict(gate_runners or {})
        self.session = session
        self.budget_seconds = float(budget_seconds)

    async def evaluate(self, org_id: str, *, label: str = "candidate") -> BenchmarkSummary:
        started = perf_counter()
        gate_scores: list[GateScore] = []
        gate_map = self.gate_runners or {
            "decide": self._default_runner,
            "plan": self._default_runner,
            "explain": self._default_runner,
        }

        for gate_name in gate_map:
            elapsed = perf_counter() - started
            if elapsed > self.budget_seconds:
                break
            gate_scores.append(await self._run_gate(gate_name, org_id))

        composite_score = round(mean([item.score for item in gate_scores]) if gate_scores else 0.0, 6)
        duration_seconds = round(perf_counter() - started, 6)
        summary = BenchmarkSummary(
            composite_score=composite_score,
            gate_scores={item.name: item.score for item in gate_scores},
            gates_completed=len(gate_scores),
            duration_seconds=duration_seconds,
            budget_exceeded=duration_seconds > self.budget_seconds,
            results=[asdict(item) for item in gate_scores],
        )

        if self.session is not None:
            self.session.add(
                BenchmarkRun(
                    run_at=datetime.now(timezone.utc),
                    mode="auto_research",
                    strategy=label,
                    dataset="gate_e",
                    VLLM_MODEL=None,
                    duration_seconds=summary.duration_seconds,
                    composite_score=summary.composite_score,
                    results=summary.results,
                )
            )
            await self.session.flush()

        return summary

    async def _run_gate(self, gate_name: str, org_id: str) -> GateScore:
        runner = self.gate_runners.get(gate_name, self._default_runner)
        raw = await runner(org_id)
        if isinstance(raw, dict):
            score = float(raw.get("score") or 0.0)
            details = dict(raw.get("details") or {})
        else:
            score = float(raw)
            details = {}
        return GateScore(name=gate_name, score=round(score, 6), details=details)

    async def _default_runner(self, org_id: str) -> dict[str, Any]:
        return {"score": 0.0, "details": {"org_id": org_id, "reason": "no gate runner configured"}}