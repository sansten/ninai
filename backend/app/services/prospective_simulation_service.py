"""Prospective Simulation Service (Feature 24.9).

Exposes EpisodicFutureSimulationAgent's heuristic as a clean async service
with the roadmap-specified API contract:

  scenario           — natural-language "what if" question
  horizon_days       — how far into the future to simulate
  variables_to_watch — metric names to project at each simulation step

Adds per-variable projections on top of the agent's episode timeline,
so callers get a structured forecast table alongside the episode narrative.

Research: Episodic future simulation / mental time travel (Atance &
O'Neill 2001; Schacter et al. 2012).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from app.services.enterprise_fallbacks import future_run_heuristic as _run_simulation


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class VariableProjection:
    variable: str
    baseline: float           # assumed 1.0 (normalized) unless caller provides current_metrics
    projected_values: list[float]   # one value per simulation step
    trend: str                # "increasing" | "decreasing" | "stable"
    peak_risk_step: int | None


@dataclass
class SimulationTimelineEntry:
    step: int
    horizon_date: str          # ISO date at step
    event_description: str
    probability: float
    severity_change: str       # "increase" | "decrease" | "stable"
    entities_affected: list[str]
    variable_projections: dict[str, float]   # variable → projected value at step


@dataclass
class ProspectiveSimulationResult:
    scenario: str
    horizon_days: int
    variables_to_watch: list[str]
    simulation_timeline: list[SimulationTimelineEntry]
    variable_summaries: list[VariableProjection]
    success_probability: float
    risk_events: list[dict]
    recommended_precautions: list[str]
    confidence: float
    simulated_at: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STEPS_PER_WEEK = 3          # approximately; horizon_days drives step count
_MAX_STEPS = 10
_MIN_STEPS = 1


def _steps_for_horizon(horizon_days: int) -> int:
    """Convert horizon_days to simulation step count (capped at MAX_STEPS)."""
    weeks = max(1, horizon_days / 7.0)
    return max(_MIN_STEPS, min(_MAX_STEPS, round(weeks * _STEPS_PER_WEEK)))


def _date_at_step(start: datetime, step: int, total_steps: int, horizon_days: int) -> str:
    """ISO date for a simulation step interpolated over horizon_days."""
    if total_steps <= 1:
        days_offset = horizon_days
    else:
        days_offset = round(step * horizon_days / (total_steps - 1))
    return (start + timedelta(days=days_offset)).date().isoformat()


def _variable_impact(
    severity_change: str,
    probability: float,
    variable: str,
) -> float:
    """Estimate variable impact multiplier for one simulation step.

    severity_change=increase → variable grows if it is a risk-type metric
    (incident_rate, error_rate, latency) or shrinks if it is a quality metric
    (customer_satisfaction, quality, availability).

    All deltas are scaled by the episode's probability.
    """
    risk_vars = {"incident_rate", "error_rate", "latency", "failure_rate", "outage_rate"}
    quality_vars = {"customer_satisfaction", "quality", "availability", "uptime", "nps"}

    var_lower = variable.lower().replace(" ", "_")

    if severity_change == "increase":
        if any(rv in var_lower for rv in risk_vars):
            return 1.0 + 0.15 * probability           # risk metrics rise
        if any(qv in var_lower for qv in quality_vars):
            return 1.0 - 0.12 * probability           # quality metrics fall
        return 1.0 + 0.05 * probability               # neutral lean up
    elif severity_change == "decrease":
        if any(rv in var_lower for rv in risk_vars):
            return 1.0 - 0.10 * probability           # risk metrics fall (good)
        if any(qv in var_lower for qv in quality_vars):
            return 1.0 + 0.08 * probability           # quality metrics rise
        return 1.0 - 0.03 * probability               # neutral lean down
    else:
        return 1.0                                     # stable


def _project_variables(
    episodes: list[dict[str, Any]],
    variables: list[str],
    baseline: float = 1.0,
) -> dict[str, list[float]]:
    """Compute projected value per variable per simulation step."""
    projections: dict[str, list[float]] = {v: [] for v in variables}

    for var in variables:
        current = baseline
        for ep in episodes:
            mult = _variable_impact(
                severity_change=str(ep.get("severity_change") or "stable"),
                probability=float(ep.get("probability") or 0.5),
                variable=var,
            )
            current = round(max(0.0, current * mult), 4)
            projections[var].append(current)

    return projections


def _variable_summaries(
    projections: dict[str, list[float]],
    baseline: float = 1.0,
) -> list[VariableProjection]:
    summaries: list[VariableProjection] = []
    for var, values in projections.items():
        if not values:
            continue
        final = values[-1]
        if final > baseline * 1.05:
            trend = "increasing"
        elif final < baseline * 0.95:
            trend = "decreasing"
        else:
            trend = "stable"

        # Peak risk step: step with maximum deviation from baseline
        peak_step: int | None = None
        max_dev = 0.0
        for i, v in enumerate(values):
            dev = abs(v - baseline)
            if dev > max_dev:
                max_dev = dev
                peak_step = i

        summaries.append(VariableProjection(
            variable=var,
            baseline=baseline,
            projected_values=values,
            trend=trend,
            peak_risk_step=peak_step,
        ))
    return summaries


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class ProspectiveSimulationService:
    """Feature 24.9: Mental time-travel simulation from a scenario description.

    Maps the public API contract to EpisodicFutureSimulationAgent's heuristic
    and layers variable projections over the episode timeline.
    """

    def simulate(
        self,
        *,
        scenario: str,
        horizon_days: int,
        variables_to_watch: list[str],
        historical_episodes: list[dict[str, Any]] | None = None,
        current_metrics: dict[str, float] | None = None,
    ) -> ProspectiveSimulationResult:
        """Run a prospective simulation from a natural-language scenario.

        Args:
            scenario: "What if …" question
            horizon_days: Forecast window in days
            variables_to_watch: Metrics to project (e.g. ["incident_rate"])
            historical_episodes: Optional past episode feed to improve quality
            current_metrics: Optional current metric values as baselines
                             (defaults to 1.0 per variable if absent)
        """
        horizon_days = max(1, int(horizon_days))
        steps = _steps_for_horizon(horizon_days)
        variables = list(variables_to_watch or [])
        _metrics = dict(current_metrics or {})

        # Map new API → agent contract
        current_state = {
            "entities": variables,
            "metrics": _metrics,
        }
        agent_outputs = _run_simulation(
            current_state=current_state,
            planned_action=scenario,
            historical_episodes=list(historical_episodes or []),
            simulation_steps=steps,
        )

        episodes: list[dict[str, Any]] = list(agent_outputs.get("simulated_episodes") or [])
        now = datetime.now(timezone.utc)

        # Project variables across simulation steps
        projections = _project_variables(episodes, variables)
        var_summaries = _variable_summaries(projections)

        # Build timeline
        timeline: list[SimulationTimelineEntry] = []
        for ep in episodes:
            step_idx = int(ep.get("step") or 0)
            var_at_step = {v: projections[v][step_idx] for v in variables if projections.get(v)}
            timeline.append(SimulationTimelineEntry(
                step=step_idx,
                horizon_date=_date_at_step(now, step_idx, steps, horizon_days),
                event_description=str(ep.get("event_description") or ""),
                probability=float(ep.get("probability") or 0.0),
                severity_change=str(ep.get("severity_change") or "stable"),
                entities_affected=list(ep.get("entities_affected") or []),
                variable_projections=var_at_step,
            ))

        return ProspectiveSimulationResult(
            scenario=scenario,
            horizon_days=horizon_days,
            variables_to_watch=variables,
            simulation_timeline=timeline,
            variable_summaries=var_summaries,
            success_probability=float(agent_outputs.get("success_probability") or 0.0),
            risk_events=list(agent_outputs.get("risk_events") or []),
            recommended_precautions=list(agent_outputs.get("recommended_precautions") or []),
            confidence=float(agent_outputs.get("confidence") or 0.5),
            simulated_at=now.isoformat(),
        )
