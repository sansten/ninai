"""Reward signal propagation service (Phase 78)."""

from __future__ import annotations

from typing import Any


class RewardPropagationService:
    PROPAGATION_DISCOUNT = 0.8
    MIN_REWARD = 0.01

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(value)))

    def propagate_backwards(
        self,
        *,
        outcome_reward: float,
        causal_chain: list[dict],
    ) -> list[dict]:
        reward = float(outcome_reward)
        chain = list(causal_chain or [])
        results: list[dict] = []

        for i, step in enumerate(chain):
            discounted = reward * (self.PROPAGATION_DISCOUNT ** i)
            if abs(discounted) < self.MIN_REWARD:
                break

            results.append(
                {
                    "step_id": str(step.get("step_id", "")),
                    "step_type": str(step.get("step_type", "")),
                    "reward_signal": round(discounted, 4),
                    "credit_type": "positive" if discounted > 0 else "negative",
                }
            )

        return results

    def aggregate_credits(self, *, reward_records: list[dict]) -> dict[str, float]:
        totals: dict[str, float] = {}
        for row in reward_records or []:
            step_id = str(row.get("step_id", ""))
            if not step_id:
                continue
            value = float(row.get("reward_signal", 0.0) or 0.0)
            totals[step_id] = totals.get(step_id, 0.0) + value
        return {key: round(value, 4) for key, value in totals.items()}

    def top_credited_steps(
        self,
        *,
        credits: dict[str, float],
        top_k: int = 5,
    ) -> list[dict]:
        limit = max(0, int(top_k if top_k is not None else 5))
        rows = [
            {"step_id": str(step_id), "credit": float(credit)}
            for step_id, credit in (credits or {}).items()
        ]
        rows.sort(key=lambda item: (abs(float(item["credit"])), str(item["step_id"])), reverse=True)
        return rows[:limit]

    def update_memory_importance(
        self,
        *,
        memory_id: str,
        current_importance: float,
        credit: float,
        alpha: float = 0.1,
    ) -> float:
        _ = memory_id
        a = self._clamp(alpha, 0.0, 1.0)
        current = self._clamp(current_importance, 0.0, 1.0)
        target = float(credit) + 0.5
        updated = (1.0 - a) * current + a * target
        return round(self._clamp(updated, 0.0, 1.0), 4)
