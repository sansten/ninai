"""Confidence ensembling service (Phase 70)."""

from __future__ import annotations


class ConfidenceEnsembleService:
    WEIGHTS = {
        "credibility": 0.30,
        "anomaly": 0.20,
        "uncertainty": 0.25,
        "hypothesis": 0.15,
        "calibration": 0.10,
    }

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def ensemble(self, *, signals: dict[str, float]) -> float:
        src = dict(signals or {})
        adjusted: dict[str, float] = {}

        for key in self.WEIGHTS:
            raw = self._clamp(float(src.get(key, 0.5)))
            if key in {"anomaly", "uncertainty"}:
                adjusted[key] = 1.0 - raw
            else:
                adjusted[key] = raw

        score = sum(self.WEIGHTS[k] * adjusted[k] for k in self.WEIGHTS)
        return round(self._clamp(score), 4)

    def ensemble_label(self, score: float) -> str:
        value = self._clamp(score)
        if value >= 0.8:
            return "high_confidence"
        if value >= 0.6:
            return "moderate_confidence"
        if value >= 0.4:
            return "low_confidence"
        return "unreliable"

    def missing_signal_impact(self, *, missing_key: str) -> float:
        key = str(missing_key or "").strip().lower()
        return float(self.WEIGHTS.get(key, 0.0))

    def dominant_signal(self, *, signals: dict[str, float]) -> str:
        src = dict(signals or {})
        contributions: dict[str, float] = {}
        for key, weight in self.WEIGHTS.items():
            raw = self._clamp(float(src.get(key, 0.5)))
            if key in {"anomaly", "uncertainty"}:
                raw = 1.0 - raw
            contributions[key] = weight * raw

        return max(contributions.items(), key=lambda kv: kv[1])[0]
