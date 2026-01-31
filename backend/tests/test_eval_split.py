from __future__ import annotations

import pytest

from app.utils.eval_split import split_deterministic


def test_split_deterministic_is_stable_for_seed():
    items = list(range(10))
    a_train, a_test = split_deterministic(items, train_frac=0.6, seed=123)
    b_train, b_test = split_deterministic(items, train_frac=0.6, seed=123)

    assert a_train == b_train
    assert a_test == b_test
    assert sorted(a_train + a_test) == items


def test_split_deterministic_changes_with_seed():
    items = list(range(10))
    a_train, _ = split_deterministic(items, train_frac=0.6, seed=1)
    b_train, _ = split_deterministic(items, train_frac=0.6, seed=2)
    assert a_train != b_train


def test_split_deterministic_validates_frac():
    with pytest.raises(ValueError):
        split_deterministic([1, 2, 3], train_frac=0.0, seed=1)
    with pytest.raises(ValueError):
        split_deterministic([1, 2, 3], train_frac=1.0, seed=1)
