from __future__ import annotations

import pytest

from app.services.reward_propagation_service import RewardPropagationService


class TestPropagateBackwards:
    def test_first_step_full_reward_second_discounted(self):
        svc = RewardPropagationService()
        out = svc.propagate_backwards(
            outcome_reward=1.0,
            causal_chain=[
                {"step_id": "s1", "step_type": "decision"},
                {"step_id": "s2", "step_type": "memory"},
            ],
        )
        assert out[0]["reward_signal"] == 1.0
        assert out[1]["reward_signal"] == pytest.approx(0.8)

    def test_stops_below_min_reward(self):
        svc = RewardPropagationService()
        chain = [{"step_id": f"s{i}", "step_type": "x"} for i in range(40)]
        out = svc.propagate_backwards(outcome_reward=0.02, causal_chain=chain)
        assert len(out) < len(chain)
        assert abs(out[-1]["reward_signal"]) >= svc.MIN_REWARD

    def test_negative_outcome_sets_negative_credit_type(self):
        svc = RewardPropagationService()
        out = svc.propagate_backwards(
            outcome_reward=-1.0,
            causal_chain=[{"step_id": "s1", "step_type": "x"}, {"step_id": "s2", "step_type": "x"}],
        )
        assert all(row["credit_type"] == "negative" for row in out)

    def test_empty_chain_returns_empty(self):
        svc = RewardPropagationService()
        assert svc.propagate_backwards(outcome_reward=1.0, causal_chain=[]) == []

    def test_contains_expected_keys(self):
        svc = RewardPropagationService()
        out = svc.propagate_backwards(
            outcome_reward=1.0,
            causal_chain=[{"step_id": "s1", "step_type": "decision"}],
        )
        assert set(out[0]) == {"step_id", "step_type", "reward_signal", "credit_type"}

    def test_step_fields_stringified(self):
        svc = RewardPropagationService()
        out = svc.propagate_backwards(
            outcome_reward=1.0,
            causal_chain=[{"step_id": 123, "step_type": 456}],
        )
        assert out[0]["step_id"] == "123"
        assert out[0]["step_type"] == "456"

    def test_rounding_to_4_decimals(self):
        svc = RewardPropagationService()
        out = svc.propagate_backwards(
            outcome_reward=0.333333,
            causal_chain=[{"step_id": "s1", "step_type": "x"}],
        )
        assert out[0]["reward_signal"] == round(out[0]["reward_signal"], 4)

    def test_zero_outcome_yields_no_records(self):
        svc = RewardPropagationService()
        out = svc.propagate_backwards(
            outcome_reward=0.0,
            causal_chain=[{"step_id": "s1", "step_type": "x"}],
        )
        assert out == []


class TestAggregateCredits:
    def test_multiple_records_same_step_summed(self):
        svc = RewardPropagationService()
        result = svc.aggregate_credits(
            reward_records=[
                {"step_id": "s1", "reward_signal": 0.5},
                {"step_id": "s1", "reward_signal": 0.2},
            ]
        )
        assert result["s1"] == 0.7

    def test_multiple_steps_aggregated(self):
        svc = RewardPropagationService()
        result = svc.aggregate_credits(
            reward_records=[
                {"step_id": "a", "reward_signal": 0.3},
                {"step_id": "b", "reward_signal": -0.1},
            ]
        )
        assert result == {"a": 0.3, "b": -0.1}

    def test_ignores_missing_step_id(self):
        svc = RewardPropagationService()
        result = svc.aggregate_credits(reward_records=[{"reward_signal": 1.0}])
        assert result == {}

    def test_empty_records(self):
        svc = RewardPropagationService()
        assert svc.aggregate_credits(reward_records=[]) == {}

    def test_rounding_applied(self):
        svc = RewardPropagationService()
        result = svc.aggregate_credits(
            reward_records=[
                {"step_id": "s1", "reward_signal": 0.33333},
                {"step_id": "s1", "reward_signal": 0.33333},
            ]
        )
        assert result["s1"] == 0.6667


class TestTopCreditedSteps:
    def test_returns_at_most_top_k(self):
        svc = RewardPropagationService()
        result = svc.top_credited_steps(credits={str(i): float(i) for i in range(10)}, top_k=5)
        assert len(result) == 5

    def test_sorted_by_abs_credit_desc(self):
        svc = RewardPropagationService()
        result = svc.top_credited_steps(credits={"a": 0.2, "b": -0.9, "c": 0.5}, top_k=3)
        assert [r["step_id"] for r in result] == ["b", "c", "a"]

    def test_negative_top_k_returns_empty(self):
        svc = RewardPropagationService()
        assert svc.top_credited_steps(credits={"a": 1.0}, top_k=-1) == []

    def test_default_top_k_five(self):
        svc = RewardPropagationService()
        result = svc.top_credited_steps(credits={str(i): float(i) for i in range(7)})
        assert len(result) == 5

    def test_empty_credits_returns_empty(self):
        svc = RewardPropagationService()
        assert svc.top_credited_steps(credits={}, top_k=5) == []

    def test_output_shape(self):
        svc = RewardPropagationService()
        result = svc.top_credited_steps(credits={"x": 1.0}, top_k=1)
        assert set(result[0]) == {"step_id", "credit"}


class TestUpdateMemoryImportance:
    def test_ema_formula_correct(self):
        svc = RewardPropagationService()
        updated = svc.update_memory_importance(memory_id="m1", current_importance=0.4, credit=0.7, alpha=0.1)
        expected = (1 - 0.1) * 0.4 + 0.1 * (0.7 + 0.5)
        assert updated == round(expected, 4)

    def test_clamped_at_one(self):
        svc = RewardPropagationService()
        updated = svc.update_memory_importance(memory_id="m1", current_importance=0.99, credit=10.0, alpha=1.0)
        assert updated == 1.0

    def test_clamped_at_zero(self):
        svc = RewardPropagationService()
        updated = svc.update_memory_importance(memory_id="m1", current_importance=0.01, credit=-10.0, alpha=1.0)
        assert updated == 0.0

    def test_alpha_clamped_low(self):
        svc = RewardPropagationService()
        updated = svc.update_memory_importance(memory_id="m1", current_importance=0.4, credit=0.5, alpha=-1.0)
        assert updated == 0.4

    def test_alpha_clamped_high(self):
        svc = RewardPropagationService()
        updated = svc.update_memory_importance(memory_id="m1", current_importance=0.4, credit=0.5, alpha=2.0)
        assert updated == 1.0

    def test_current_importance_clamped(self):
        svc = RewardPropagationService()
        updated = svc.update_memory_importance(memory_id="m1", current_importance=-9.0, credit=0.0, alpha=0.5)
        assert updated == 0.25


class TestServiceSanity:
    def test_constants_expected_values(self):
        assert RewardPropagationService.PROPAGATION_DISCOUNT == 0.8
        assert RewardPropagationService.MIN_REWARD == 0.01

    def test_top_credited_tie_break_by_step_id_desc(self):
        svc = RewardPropagationService()
        result = svc.top_credited_steps(credits={"a": 1.0, "b": -1.0}, top_k=2)
        assert result[0]["step_id"] == "b"

    def test_propagate_backwards_preserves_order(self):
        svc = RewardPropagationService()
        chain = [{"step_id": "s1", "step_type": "a"}, {"step_id": "s2", "step_type": "b"}]
        out = svc.propagate_backwards(outcome_reward=1.0, causal_chain=chain)
        assert [r["step_id"] for r in out] == ["s1", "s2"]

    def test_aggregate_handles_none_signal(self):
        svc = RewardPropagationService()
        result = svc.aggregate_credits(reward_records=[{"step_id": "s1", "reward_signal": None}])
        assert result["s1"] == 0.0
