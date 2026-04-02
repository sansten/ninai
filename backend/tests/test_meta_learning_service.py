from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.meta_learning_config import MetaLearningConfig
from app.services.meta_learning_service import MetaLearningService


def _history(
    *,
    n: int,
    predicted: float,
    actual: float,
    noise_duplicate_rate: float | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for i in range(n):
        row = {
            "predicted_confidence": predicted,
            "actual_outcome": actual,
            "timestamp": datetime(2026, 4, 2, 12, 0, i % 60, tzinfo=timezone.utc).isoformat(),
        }
        if noise_duplicate_rate is not None:
            row["noise_duplicate_rate"] = noise_duplicate_rate
        rows.append(row)
    return rows


@pytest.mark.asyncio
class TestGetConfig:
    async def test_get_config_creates_default_when_missing(self, db_session: AsyncSession, test_org_id: str):
        svc = MetaLearningService()
        config = await svc.get_config(db=db_session, org_id=test_org_id)

        assert config.org_id == test_org_id
        assert config.ema_alpha == 0.25
        assert config.noise_threshold == 0.85
        assert config.confidence_floor == 0.4
        assert config.decay_half_life_days == 30
        assert config.calibration_window == 100
        assert config.tuning_iteration == 0

    async def test_get_config_returns_existing_config(self, db_session: AsyncSession, test_org_id: str):
        svc = MetaLearningService()
        first = await svc.get_config(db=db_session, org_id=test_org_id)
        second = await svc.get_config(db=db_session, org_id=test_org_id)

        assert second.id == first.id

    async def test_unique_per_org(self, db_session: AsyncSession, test_org_id: str):
        svc = MetaLearningService()
        await svc.get_config(db=db_session, org_id=test_org_id)
        await svc.get_config(db=db_session, org_id=test_org_id)

        rows = list((await db_session.execute(select(MetaLearningConfig))).scalars().all())
        assert len(rows) == 1


class TestCalibrationError:
    def test_all_predictions_correct_zero(self):
        svc = MetaLearningService()
        error = svc.calibration_error(_history(n=10, predicted=1.0, actual=1.0))
        assert error == 0.0

    def test_all_predictions_half_off(self):
        svc = MetaLearningService()
        error = svc.calibration_error(_history(n=6, predicted=0.9, actual=0.4))
        assert error == 0.5

    def test_empty_history_returns_zero(self):
        svc = MetaLearningService()
        assert svc.calibration_error([]) == 0.0

    def test_mixed_history_mean_absolute_error(self):
        svc = MetaLearningService()
        rows = [
            {"predicted_confidence": 0.9, "actual_outcome": 0.8},
            {"predicted_confidence": 0.2, "actual_outcome": 0.6},
            {"predicted_confidence": 0.5, "actual_outcome": 0.5},
        ]
        assert svc.calibration_error(rows) == pytest.approx((0.1 + 0.4 + 0.0) / 3)

    def test_invalid_values_treated_zero(self):
        svc = MetaLearningService()
        rows = [{"predicted_confidence": "x", "actual_outcome": 1.0}]
        assert svc.calibration_error(rows) == 1.0


class TestRecommendedAlpha:
    def test_error_above_point_twenty_five_increases(self):
        svc = MetaLearningService()
        assert svc.recommended_alpha(calibration_error=0.3, current_alpha=0.25) == 0.3

    def test_error_below_point_zero_five_decreases(self):
        svc = MetaLearningService()
        assert svc.recommended_alpha(calibration_error=0.01, current_alpha=0.25) == 0.23

    def test_error_within_tolerance_no_change(self):
        svc = MetaLearningService()
        assert svc.recommended_alpha(calibration_error=0.1, current_alpha=0.25) == 0.25

    def test_alpha_clamped_maximum(self):
        svc = MetaLearningService()
        assert svc.recommended_alpha(calibration_error=0.9, current_alpha=0.49) == 0.5

    def test_alpha_clamped_minimum(self):
        svc = MetaLearningService()
        assert svc.recommended_alpha(calibration_error=0.01, current_alpha=0.05) == 0.05


@pytest.mark.asyncio
class TestTune:
    async def test_tune_high_error_increases_ema_alpha(self, db_session: AsyncSession, test_org_id: str):
        svc = MetaLearningService()
        updated = await svc.tune(
            db=db_session,
            org_id=test_org_id,
            outcome_history=_history(n=25, predicted=0.9, actual=0.5),
        )
        assert updated.ema_alpha == pytest.approx(0.30)

    async def test_tune_medium_error_decreases_ema_alpha(self, db_session: AsyncSession, test_org_id: str):
        svc = MetaLearningService()
        updated = await svc.tune(
            db=db_session,
            org_id=test_org_id,
            outcome_history=_history(n=20, predicted=0.7, actual=0.5),
        )
        assert updated.ema_alpha == pytest.approx(0.23)

    async def test_tune_low_error_stays_current_alpha(self, db_session: AsyncSession, test_org_id: str):
        svc = MetaLearningService()
        updated = await svc.tune(
            db=db_session,
            org_id=test_org_id,
            outcome_history=_history(n=25, predicted=0.52, actual=0.5),
        )
        assert updated.ema_alpha == pytest.approx(0.25)

    async def test_tune_less_than_twenty_samples_no_alpha_tuning(self, db_session: AsyncSession, test_org_id: str):
        svc = MetaLearningService()
        updated = await svc.tune(
            db=db_session,
            org_id=test_org_id,
            outcome_history=_history(n=19, predicted=0.9, actual=0.4),
        )
        assert updated.ema_alpha == pytest.approx(0.25)

    async def test_tune_clamps_alpha_to_point_five(self, db_session: AsyncSession, test_org_id: str):
        svc = MetaLearningService()
        config = await svc.get_config(db=db_session, org_id=test_org_id)
        config.ema_alpha = 0.49
        await db_session.flush()

        updated = await svc.tune(
            db=db_session,
            org_id=test_org_id,
            outcome_history=_history(n=25, predicted=0.95, actual=0.1),
        )
        assert updated.ema_alpha == pytest.approx(0.5)

    async def test_tune_clamps_alpha_to_point_zero_five(self, db_session: AsyncSession, test_org_id: str):
        svc = MetaLearningService()
        config = await svc.get_config(db=db_session, org_id=test_org_id)
        config.ema_alpha = 0.05
        await db_session.flush()

        updated = await svc.tune(
            db=db_session,
            org_id=test_org_id,
            outcome_history=_history(n=20, predicted=0.7, actual=0.5),
        )
        assert updated.ema_alpha == pytest.approx(0.05)

    async def test_noise_threshold_tightened_when_duplicate_rate_high(self, db_session: AsyncSession, test_org_id: str):
        svc = MetaLearningService()
        updated = await svc.tune(
            db=db_session,
            org_id=test_org_id,
            outcome_history=_history(n=25, predicted=0.5, actual=0.5, noise_duplicate_rate=0.4),
        )
        assert updated.noise_threshold == pytest.approx(0.80)

    async def test_noise_threshold_loosened_when_duplicate_rate_low(self, db_session: AsyncSession, test_org_id: str):
        svc = MetaLearningService()
        updated = await svc.tune(
            db=db_session,
            org_id=test_org_id,
            outcome_history=_history(n=25, predicted=0.5, actual=0.5, noise_duplicate_rate=0.01),
        )
        assert updated.noise_threshold == pytest.approx(0.90)

    async def test_noise_threshold_clamped_minimum(self, db_session: AsyncSession, test_org_id: str):
        svc = MetaLearningService()
        config = await svc.get_config(db=db_session, org_id=test_org_id)
        config.noise_threshold = 0.5
        await db_session.flush()

        updated = await svc.tune(
            db=db_session,
            org_id=test_org_id,
            outcome_history=_history(n=25, predicted=0.5, actual=0.5, noise_duplicate_rate=0.9),
        )
        assert updated.noise_threshold == pytest.approx(0.5)

    async def test_noise_threshold_clamped_maximum(self, db_session: AsyncSession, test_org_id: str):
        svc = MetaLearningService()
        config = await svc.get_config(db=db_session, org_id=test_org_id)
        config.noise_threshold = 0.99
        await db_session.flush()

        updated = await svc.tune(
            db=db_session,
            org_id=test_org_id,
            outcome_history=_history(n=25, predicted=0.5, actual=0.5, noise_duplicate_rate=0.0),
        )
        assert updated.noise_threshold == pytest.approx(0.99)

    async def test_tuning_iteration_increments_each_call(self, db_session: AsyncSession, test_org_id: str):
        svc = MetaLearningService()
        one = await svc.tune(db=db_session, org_id=test_org_id, outcome_history=_history(n=25, predicted=0.5, actual=0.5))
        first_iteration = one.tuning_iteration
        two = await svc.tune(db=db_session, org_id=test_org_id, outcome_history=_history(n=25, predicted=0.5, actual=0.5))
        assert first_iteration == 1
        assert two.tuning_iteration == 2

    async def test_last_tuned_is_updated(self, db_session: AsyncSession, test_org_id: str):
        svc = MetaLearningService()
        config = await svc.get_config(db=db_session, org_id=test_org_id)
        before = config.last_tuned

        updated = await svc.tune(db=db_session, org_id=test_org_id, outcome_history=_history(n=25, predicted=0.5, actual=0.5))
        assert updated.last_tuned >= before

    async def test_tune_with_empty_history_still_increments_iteration(self, db_session: AsyncSession, test_org_id: str):
        svc = MetaLearningService()
        updated = await svc.tune(db=db_session, org_id=test_org_id, outcome_history=[])
        assert updated.tuning_iteration == 1


class TestServiceSanity:
    def test_calibration_error_non_negative(self):
        svc = MetaLearningService()
        assert svc.calibration_error(_history(n=5, predicted=0.2, actual=0.9)) >= 0.0

    def test_recommended_alpha_with_out_of_range_current(self):
        svc = MetaLearningService()
        value = svc.recommended_alpha(calibration_error=0.1, current_alpha=9.0)
        assert value == 0.5

    def test_recommended_alpha_with_negative_current(self):
        svc = MetaLearningService()
        value = svc.recommended_alpha(calibration_error=0.1, current_alpha=-9.0)
        assert value == 0.05
