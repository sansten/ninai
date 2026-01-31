from __future__ import annotations

from app.services.self_model_service import ema, p95


def test_ema_bounds_and_smoothing():
    assert ema(previous=None, observed=0.2, alpha=0.5) == 0.2

    # smoothing
    v = ema(previous=1.0, observed=0.0, alpha=0.2)
    assert 0.7 < v < 1.0

    # bounds
    assert ema(previous=0.5, observed=5.0, alpha=0.5) == 0.75
    assert ema(previous=0.5, observed=-3.0, alpha=0.5) == 0.25


def test_p95_nearest_rank():
    assert p95([]) is None
    assert p95([1.0]) == 1.0

    vals = [1, 2, 3, 4, 100]
    out = p95(vals)
    assert out in {4.0, 100.0}
