from __future__ import annotations

import random
from typing import Iterable, TypeVar


T = TypeVar("T")


def split_deterministic(items: Iterable[T], train_frac: float, seed: int) -> tuple[list[T], list[T]]:
    """Deterministically split items into (train, test) with a seeded RNG.

    - Preserves the original item identities (no copying).
    - Shuffles with a local RNG so the split is stable for a given seed.
    """

    if not (0.0 < float(train_frac) < 1.0):
        raise ValueError("train_frac must be between 0 and 1")

    items_list = list(items)
    rng = random.Random(int(seed))
    rng.shuffle(items_list)

    train_n = int(round(len(items_list) * float(train_frac)))
    train_n = max(0, min(len(items_list), train_n))
    train = items_list[:train_n]
    test = items_list[train_n:]
    return train, test
