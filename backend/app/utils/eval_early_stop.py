from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EarlyStopConfig:
    mrr_delta: Optional[float] = None
    recall_at_k_delta: Optional[float] = None
    patience: int = 2


def update_plateau_streak(
    *,
    current_streak: int,
    delta_mrr: float,
    delta_recall_at_k: float,
    config: EarlyStopConfig,
) -> tuple[int, bool]:
    """Update plateau streak and decide whether to stop.

    Plateau is defined as: for each configured metric threshold, the delta-from-prev
    is strictly less than the threshold. If no thresholds are configured, early stop
    is disabled.
    """

    if config.mrr_delta is None and config.recall_at_k_delta is None:
        return 0, False

    if int(config.patience) < 1:
        raise ValueError("patience must be >= 1")

    plateau = True
    if config.mrr_delta is not None:
        plateau = plateau and (float(delta_mrr) < float(config.mrr_delta))
    if config.recall_at_k_delta is not None:
        plateau = plateau and (float(delta_recall_at_k) < float(config.recall_at_k_delta))

    new_streak = current_streak + 1 if plateau else 0
    return new_streak, new_streak >= int(config.patience)
