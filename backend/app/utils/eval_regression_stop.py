from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RegressionStopConfig:
    mrr_drop: Optional[float] = None
    recall_at_k_drop: Optional[float] = None


def should_stop_on_regression(*, delta_mrr: float, delta_recall_at_k: float, config: RegressionStopConfig) -> bool:
    """Return True if a negative delta exceeds configured drop thresholds.

    Thresholds are positive numbers representing the maximum allowed drop.
    Example: mrr_drop=0.01 stops if delta_mrr <= -0.01.

    If both thresholds are None, regression stop is disabled.
    """

    if config.mrr_drop is None and config.recall_at_k_drop is None:
        return False

    stop = False
    if config.mrr_drop is not None:
        stop = stop or (float(delta_mrr) <= -float(config.mrr_drop))
    if config.recall_at_k_drop is not None:
        stop = stop or (float(delta_recall_at_k) <= -float(config.recall_at_k_drop))
    return stop
