from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.meta_agent import CalibrationProfile
from app.models.memory_feedback import MemoryFeedback
from app.core.config import settings
from app.core.redis import RedisClient
from app.services.meta_agent.confidence_aggregator import normalize_signal_weights


class CalibrationService:
    CACHE_PREFIX = "meta:calibration_profile"
    CACHE_TTL_SECONDS = 300  # 5m per spec

    def _cache_key(self, org_id: str) -> str:
        return f"{self.CACHE_PREFIX}:{org_id}"

    def _redis_enabled(self) -> bool:
        return bool(getattr(settings, "REDIS_HOST", None))

    async def _cache_get(self, *, org_id: str) -> dict | None:
        if not self._redis_enabled():
            return None
        try:
            return await RedisClient.get_json(self._cache_key(org_id))
        except Exception:
            return None

    async def _cache_set(self, *, org_id: str, value: dict) -> None:
        if not self._redis_enabled():
            return
        try:
            await RedisClient.set_json(self._cache_key(org_id), value, ttl=self.CACHE_TTL_SECONDS)
        except Exception:
            return

    async def _cache_delete(self, *, org_id: str) -> None:
        if not self._redis_enabled():
            return
        try:
            await RedisClient.delete(self._cache_key(org_id))
        except Exception:
            return

    def _serialize_profile(self, profile: CalibrationProfile) -> dict:
        # Avoid triggering IO (lazy load) while serializing.
        updated_at = getattr(profile, "__dict__", {}).get("updated_at")
        return {
            "organization_id": profile.organization_id,
            "promotion_threshold": float(profile.promotion_threshold or 0.75),
            "conflict_escalation_threshold": float(profile.conflict_escalation_threshold or 0.60),
            "drift_threshold": float(profile.drift_threshold or 0.20),
            "signal_weights": normalize_signal_weights(profile.signal_weights or {}),
            "learning_rate": float(profile.learning_rate or 0.05),
            "updated_at": updated_at.isoformat() if updated_at else None,
        }

    def _deserialize_profile(self, payload: dict) -> CalibrationProfile | None:
        try:
            if not isinstance(payload, dict):
                return None
            org_id = payload.get("organization_id")
            if not org_id:
                return None

            updated_at = payload.get("updated_at")
            dt = None
            if isinstance(updated_at, str) and updated_at:
                try:
                    dt = datetime.fromisoformat(updated_at)
                except ValueError:
                    dt = None

            return CalibrationProfile(
                organization_id=str(org_id),
                promotion_threshold=float(payload.get("promotion_threshold", 0.75)),
                conflict_escalation_threshold=float(payload.get("conflict_escalation_threshold", 0.60)),
                drift_threshold=float(payload.get("drift_threshold", 0.20)),
                signal_weights=normalize_signal_weights(payload.get("signal_weights") or {}),
                learning_rate=float(payload.get("learning_rate", 0.05)),
                updated_at=dt or datetime.utcnow(),
            )
        except Exception:
            return None

    async def get_profile_for_read(self, session: AsyncSession, *, org_id: str) -> CalibrationProfile:
        """Fast read path: prefer Redis cache (TTL 5m), fall back to DB."""
        cached = await self._cache_get(org_id=org_id)
        if cached is not None:
            profile = self._deserialize_profile(cached)
            if profile is not None:
                return profile

        profile = await self.get_or_create_profile(session, org_id=org_id)
        await self._cache_set(org_id=org_id, value=self._serialize_profile(profile))
        return profile
    async def get_or_create_profile(self, session: AsyncSession, *, org_id: str) -> CalibrationProfile:
        res = await session.execute(select(CalibrationProfile).where(CalibrationProfile.organization_id == org_id))
        profile = res.scalar_one_or_none()
        if profile is not None:
            profile.signal_weights = normalize_signal_weights(profile.signal_weights or {})
            return profile

        profile = CalibrationProfile(
            organization_id=org_id,
            promotion_threshold=0.75,
            conflict_escalation_threshold=0.60,
            drift_threshold=0.20,
            signal_weights=normalize_signal_weights({}),
            learning_rate=0.05,
        )
        session.add(profile)
        await session.flush()
        await self._cache_set(org_id=org_id, value=self._serialize_profile(profile))
        return profile

    async def update_profile(
        self,
        session: AsyncSession,
        *,
        org_id: str,
        signal_weights: dict[str, float] | None,
        learning_rate: float | None = None,
        promotion_threshold: float | None = None,
        conflict_escalation_threshold: float | None = None,
        drift_threshold: float | None = None,
    ) -> CalibrationProfile:
        profile = await self.get_or_create_profile(session, org_id=org_id)

        if signal_weights is not None:
            profile.signal_weights = normalize_signal_weights(signal_weights)

        if learning_rate is not None:
            profile.learning_rate = float(learning_rate)
        if promotion_threshold is not None:
            profile.promotion_threshold = float(promotion_threshold)
        if conflict_escalation_threshold is not None:
            profile.conflict_escalation_threshold = float(conflict_escalation_threshold)
        if drift_threshold is not None:
            profile.drift_threshold = float(drift_threshold)

        await session.flush()
        await self._cache_set(org_id=org_id, value=self._serialize_profile(profile))
        return profile

    async def learn_from_feedback(
        self,
        session: AsyncSession,
        *,
        org_id: str,
        lookback_days_current: int = 7,
        lookback_days_baseline: int = 30,
        min_feedback_samples: int = 10,
    ) -> CalibrationProfile:
        """Spec: calibration learning triggered by feedback_events.

        Uses a simple rolling correction-rate heuristic derived from MemoryFeedback.
        We keep the update deterministic, normalized, and capped per cycle.
        """

        profile = await self.get_or_create_profile(session, org_id=org_id)

        now = datetime.utcnow()
        current_since = now - timedelta(days=lookback_days_current)
        baseline_since = now - timedelta(days=lookback_days_baseline)

        def _counts_query(since: datetime):
            return (
                select(
                    func.count().label("total"),
                    func.coalesce(
                        func.sum(
                            case(
                                (
                                    MemoryFeedback.feedback_type == "classification_override",
                                    1,
                                ),
                                else_=0,
                            )
                        ),
                        0,
                    ).label("overrides"),
                )
                .select_from(MemoryFeedback)
                .where(
                    MemoryFeedback.organization_id == org_id,
                    MemoryFeedback.created_at >= since,
                    MemoryFeedback.target_agent.is_not(None),
                )
            )

        baseline_res = await session.execute(_counts_query(baseline_since))
        baseline_total, baseline_overrides = baseline_res.one()
        baseline_total = int(baseline_total or 0)
        baseline_overrides = int(baseline_overrides or 0)

        current_res = await session.execute(_counts_query(current_since))
        current_total, current_overrides = current_res.one()
        current_total = int(current_total or 0)
        current_overrides = int(current_overrides or 0)

        if baseline_total < min_feedback_samples or current_total < min_feedback_samples:
            return profile

        baseline_rate = baseline_overrides / max(1, baseline_total)
        current_rate = current_overrides / max(1, current_total)

        # Positive delta means improvement (fewer corrections recently than baseline).
        improvement_delta = float(baseline_rate - current_rate)

        deltas: dict[str, float] = {
            "w_agent_confidence": improvement_delta,
            "w_evidence_strength": -improvement_delta,
        }

        profile.signal_weights = self._apply_learning_update(
            profile.signal_weights or {},
            deltas=deltas,
            learning_rate=float(profile.learning_rate or 0.05),
        )

        await session.flush()
        await self._cache_set(org_id=org_id, value=self._serialize_profile(profile))
        return profile

    def _apply_learning_update(
        self,
        signal_weights: dict[str, float],
        *,
        deltas: dict[str, float],
        learning_rate: float,
        max_abs_change_per_cycle: float | None = None,
    ) -> dict[str, float]:
        weights = normalize_signal_weights(signal_weights)
        lr = float(learning_rate)
        cap = float(max_abs_change_per_cycle) if max_abs_change_per_cycle is not None else lr

        updated = dict(weights)
        for key, delta in (deltas or {}).items():
            if key not in updated:
                continue
            change = lr * float(delta)
            if change > cap:
                change = cap
            elif change < -cap:
                change = -cap
            updated[key] = max(0.0, float(updated[key]) + change)

        return normalize_signal_weights(updated)
