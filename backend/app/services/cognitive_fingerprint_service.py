"""Cognitive Fingerprinting Service (Feature 24.7).

Tracks the statistical distribution of each agent's numeric output fields
using Welford's online algorithm (numerically stable running mean + variance).

Detects anomalous agent outputs by comparing against the established
fingerprint baseline using Z-score thresholding.

Use cases:
  - Prompt injection detection (sudden shift in agent confidence or tone)
  - Model drift early warning (gradual drift in anomaly_score baseline)
  - Data poisoning signals (spike in uncertainty from normally steady agents)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

_DEFAULT_ANOMALY_THRESHOLD = 2.5   # Z-score threshold for flagging
_MIN_SAMPLES_FOR_DETECTION = 5     # Suppress alerts until baseline is established


@dataclass
class _FieldStats:
    """Welford online mean/variance tracker for a single numeric field."""

    n: int = 0
    mean: float = 0.0
    _M2: float = 0.0  # sum of squared deviations from running mean

    def update(self, value: float) -> None:
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self._M2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.n < 2:
            return 0.0
        return self._M2 / (self.n - 1)

    @property
    def stddev(self) -> float:
        return math.sqrt(self.variance)

    def z_score(self, value: float) -> float:
        if self.n < _MIN_SAMPLES_FOR_DETECTION:
            return 0.0
        sd = self.stddev
        if sd == 0.0:
            # Zero variance: any deviation from the fixed point is maximally anomalous.
            return 0.0 if value == self.mean else 99.0
        return (value - self.mean) / sd


@dataclass
class FingerprintAlert:
    agent_name: str
    field: str
    current_value: float
    expected_mean: float
    z_score: float


@dataclass
class FingerprintResult:
    alerts: list[FingerprintAlert]
    anomalous: bool
    agent_name: str
    sample_count: int


class CognitiveFingerprintService:
    """Per-agent output distribution tracker and anomaly detector.

    Fingerprints are stored in-memory (per service instance) and are
    suitable for single-process deployments and unit tests.  In a
    multi-worker deployment the state can be externalized to Redis by
    wrapping _agent_fingerprints with a distributed store.
    """

    def __init__(self, *, anomaly_threshold: float = _DEFAULT_ANOMALY_THRESHOLD) -> None:
        self._threshold = anomaly_threshold
        # {agent_name: {field_name: _FieldStats}}
        self._agent_fingerprints: dict[str, dict[str, _FieldStats]] = {}

    def _get_or_create_field(self, agent_name: str, field_name: str) -> _FieldStats:
        bucket = self._agent_fingerprints.setdefault(agent_name, {})
        if field_name not in bucket:
            bucket[field_name] = _FieldStats()
        return bucket[field_name]

    def _extract_numeric(self, output_fields: dict[str, Any]) -> dict[str, float]:
        """Extract all numeric (int/float) values from output_fields."""
        numerics: dict[str, float] = {}
        for key, val in output_fields.items():
            if isinstance(val, bool):
                numerics[key] = float(val)
            elif isinstance(val, (int, float)):
                numerics[key] = float(val)
        return numerics

    def update_fingerprint(self, agent_name: str, output_fields: dict[str, Any]) -> None:
        """Record a new observation for an agent's output.

        Extracts all numeric fields from output_fields and updates the
        running Welford statistics for each tracked field.
        """
        for field_name, value in self._extract_numeric(output_fields).items():
            stats = self._get_or_create_field(agent_name, field_name)
            stats.update(value)

    def detect_anomaly(
        self, agent_name: str, output_fields: dict[str, Any]
    ) -> FingerprintResult:
        """Compare a fresh agent output against its established fingerprint.

        Returns a FingerprintResult with any fields whose Z-score exceeds
        the configured threshold.  The fingerprint is NOT updated here —
        callers should call update_fingerprint() separately once the
        output is accepted as legitimate.
        """
        alerts: list[FingerprintAlert] = []
        agent_bucket = self._agent_fingerprints.get(agent_name, {})

        for field_name, value in self._extract_numeric(output_fields).items():
            stats = agent_bucket.get(field_name)
            if stats is None or stats.n < _MIN_SAMPLES_FOR_DETECTION:
                continue
            z = stats.z_score(value)
            if abs(z) >= self._threshold:
                alerts.append(
                    FingerprintAlert(
                        agent_name=agent_name,
                        field=field_name,
                        current_value=value,
                        expected_mean=round(stats.mean, 4),
                        z_score=round(z, 4),
                    )
                )

        max_n = max((s.n for s in agent_bucket.values()), default=0)
        return FingerprintResult(
            alerts=alerts,
            anomalous=len(alerts) > 0,
            agent_name=agent_name,
            sample_count=max_n,
        )

    def get_fingerprint(self, agent_name: str) -> dict[str, dict[str, float]]:
        """Return current fingerprint stats for an agent (for inspection/debugging)."""
        bucket = self._agent_fingerprints.get(agent_name, {})
        return {
            field_name: {
                "mean": round(stats.mean, 4),
                "stddev": round(stats.stddev, 4),
                "n": stats.n,
            }
            for field_name, stats in bucket.items()
        }
