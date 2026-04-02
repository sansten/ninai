"""Meta-learning service (Phase 74)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.meta_learning_config import MetaLearningConfig


class MetaLearningService:
    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, float(value)))

    @staticmethod
    def _as_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def calibration_error(self, outcome_history: list[dict]) -> float:
        history = outcome_history or []
        if not history:
            return 0.0

        errors: list[float] = []
        for row in history:
            predicted = self._as_float(row.get("predicted_confidence"), 0.0)
            actual = self._as_float(row.get("actual_outcome"), 0.0)
            errors.append(abs(predicted - actual))

        return sum(errors) / len(errors)

    def recommended_alpha(self, *, calibration_error: float, current_alpha: float) -> float:
        current = self._clamp(current_alpha, 0.05, 0.5)
        if calibration_error > 0.25:
            return min(0.5, current + 0.05)
        if calibration_error < 0.05:
            return max(0.05, current - 0.02)
        return current

    def _noise_duplicate_rate(self, outcome_history: list[dict]) -> float:
        rates = [
            self._as_float(row.get("noise_duplicate_rate"), -1.0)
            for row in (outcome_history or [])
            if row.get("noise_duplicate_rate") is not None
        ]
        if not rates:
            return 0.0
        return sum(rates) / len(rates)

    async def get_config(self, *, db: AsyncSession, org_id: str) -> MetaLearningConfig:
        stmt = select(MetaLearningConfig).where(MetaLearningConfig.org_id == org_id)
        config = (await db.execute(stmt)).scalar_one_or_none()
        if config is not None:
            return config

        config = MetaLearningConfig(org_id=org_id)
        db.add(config)
        await db.flush()
        return config

    async def tune(
        self,
        *,
        db: AsyncSession,
        org_id: str,
        outcome_history: list[dict],
    ) -> MetaLearningConfig:
        config = await self.get_config(db=db, org_id=org_id)

        history = outcome_history or []
        sample_count = len(history)
        error = self.calibration_error(history)

        if error > 0.15 and sample_count >= 20:
            if error > 0.25:
                config.ema_alpha = self._clamp(float(config.ema_alpha) + 0.05, 0.05, 0.5)
            else:
                config.ema_alpha = self._clamp(float(config.ema_alpha) - 0.02, 0.05, 0.5)

        noise_duplicate_rate = self._noise_duplicate_rate(history)
        if noise_duplicate_rate > 0.3:
            config.noise_threshold = self._clamp(float(config.noise_threshold) - 0.05, 0.5, 0.99)
        elif noise_duplicate_rate < 0.05:
            config.noise_threshold = self._clamp(float(config.noise_threshold) + 0.05, 0.5, 0.99)

        config.tuning_iteration = int(config.tuning_iteration or 0) + 1
        config.last_tuned = datetime.now(timezone.utc)

        await db.flush()
        return config
