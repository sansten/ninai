from __future__ import annotations

import pytest

from app.utils.eval_early_stop import EarlyStopConfig, update_plateau_streak


def test_early_stop_disabled_when_no_thresholds():
    cfg = EarlyStopConfig(mrr_delta=None, recall_at_k_delta=None, patience=2)
    streak, stop = update_plateau_streak(current_streak=5, delta_mrr=0.0, delta_recall_at_k=0.0, config=cfg)
    assert streak == 0
    assert stop is False


def test_early_stop_plateau_with_patience_two():
    cfg = EarlyStopConfig(mrr_delta=0.001, recall_at_k_delta=0.01, patience=2)

    # First plateau
    streak, stop = update_plateau_streak(current_streak=0, delta_mrr=0.0, delta_recall_at_k=0.0, config=cfg)
    assert streak == 1
    assert stop is False

    # Second plateau triggers stop
    streak, stop = update_plateau_streak(current_streak=streak, delta_mrr=0.0005, delta_recall_at_k=0.005, config=cfg)
    assert streak == 2
    assert stop is True


def test_early_stop_resets_streak_on_improvement():
    cfg = EarlyStopConfig(mrr_delta=0.001, recall_at_k_delta=None, patience=2)

    streak, stop = update_plateau_streak(current_streak=1, delta_mrr=0.01, delta_recall_at_k=0.0, config=cfg)
    assert streak == 0
    assert stop is False


def test_early_stop_patience_validation():
    cfg = EarlyStopConfig(mrr_delta=0.001, recall_at_k_delta=None, patience=0)
    with pytest.raises(ValueError):
        update_plateau_streak(current_streak=0, delta_mrr=0.0, delta_recall_at_k=0.0, config=cfg)
