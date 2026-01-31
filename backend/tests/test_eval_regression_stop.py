from __future__ import annotations

from app.utils.eval_regression_stop import RegressionStopConfig, should_stop_on_regression


def test_regression_stop_disabled_when_no_thresholds():
    cfg = RegressionStopConfig(mrr_drop=None, recall_at_k_drop=None)
    assert should_stop_on_regression(delta_mrr=-1.0, delta_recall_at_k=-1.0, config=cfg) is False


def test_regression_stop_triggers_on_mrr_drop():
    cfg = RegressionStopConfig(mrr_drop=0.01, recall_at_k_drop=None)
    assert should_stop_on_regression(delta_mrr=-0.02, delta_recall_at_k=0.0, config=cfg) is True
    assert should_stop_on_regression(delta_mrr=-0.009, delta_recall_at_k=0.0, config=cfg) is False


def test_regression_stop_triggers_on_recall_drop():
    cfg = RegressionStopConfig(mrr_drop=None, recall_at_k_drop=0.05)
    assert should_stop_on_regression(delta_mrr=0.0, delta_recall_at_k=-0.06, config=cfg) is True
    assert should_stop_on_regression(delta_mrr=0.0, delta_recall_at_k=-0.049, config=cfg) is False
