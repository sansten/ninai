"""Confidence Calibration Service.

Raw confidence scores produced by HypothesisService, UncertaintyResolutionService,
and CognitiveGatewayService are not inherently calibrated — a score of 0.80 may
not correspond to 80 % actual accuracy.  This service closes that gap by:

1. Collecting historical (predicted_confidence, actual_outcome) pairs from
   ActionExecutionRecord (the same source used by StrategyEvolutionService).
2. Binning predictions into N decile buckets and computing observed accuracy
   per bucket.
3. Storing a correction mapping: raw_bucket → calibrated_value.
4. Exposing calibrate(raw_score, agent_type) which applies the correction and
   returns a calibrated float.

Design rules:
- Pure service: DB session is optional; without it the service operates in
  passthrough mode (returns raw score unchanged) so it is safely usable in
  tests and in-process without a live DB.
- Fallback: if fewer than min_samples observations are available for a bucket,
  the raw score is returned unchanged.
- Thread-safe reads: the calibration table is a plain dict replaced atomically
  on rebuild; no locking needed for concurrent reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_N_BUCKETS = 10          # decile buckets
_MIN_SAMPLES = 5         # minimum observations to trust a bucket
_SMOOTHING = 0.10        # blend weight toward raw score when samples are sparse


@dataclass
class CalibrationBucket:
    """One decile bucket of historical predictions."""
    lower: float          # inclusive lower bound
    upper: float          # exclusive upper bound (1.0 bucket is inclusive)
    count: int = 0
    successes: int = 0

    @property
    def observed_rate(self) -> float | None:
        if self.count < _MIN_SAMPLES:
            return None
        return self.successes / self.count


@dataclass
class CalibrationReport:
    """Summary of the current calibration state."""
    agent_type: str
    buckets: list[CalibrationBucket]
    total_samples: int
    calibrated_buckets: int   # number of buckets with enough samples


def _bucket_index(score: float) -> int:
    """Map a score in [0, 1] to a bucket index 0..(N-1)."""
    idx = int(score * _N_BUCKETS)
    return min(idx, _N_BUCKETS - 1)


class ConfidenceCalibrationService:
    """Calibration table builder and score corrector.

    Usage (with DB — nightly rebuild):
        svc = ConfidenceCalibrationService(session=db)
        await svc.rebuild(org_id="org-1", agent_type="hypothesis")
        calibrated = svc.calibrate(raw_score=0.75, agent_type="hypothesis")

    Usage (passthrough — no DB):
        svc = ConfidenceCalibrationService()
        calibrated = svc.calibrate(0.75)  # returns 0.75 unchanged
    """

    def __init__(self, *, session: Any = None) -> None:
        self._session = session
        # agent_type -> list[CalibrationBucket]
        self._tables: dict[str, list[CalibrationBucket]] = {}

    # ------------------------------------------------------------------
    # Build calibration table from historical outcomes
    # ------------------------------------------------------------------

    def _empty_buckets(self) -> list[CalibrationBucket]:
        buckets = []
        for i in range(_N_BUCKETS):
            lower = i / _N_BUCKETS
            upper = (i + 1) / _N_BUCKETS
            buckets.append(CalibrationBucket(lower=lower, upper=upper))
        return buckets

    def ingest_outcomes(
        self,
        *,
        agent_type: str,
        outcomes: list[dict[str, Any]],
    ) -> None:
        """Feed (predicted_confidence, success) pairs into the calibration table.

        Each dict must contain:
            predicted_confidence: float  — the raw score at decision time
            success: bool                — whether the outcome was positive
        """
        buckets = self._empty_buckets()
        for obs in outcomes:
            try:
                score = float(obs.get("predicted_confidence") or 0.5)
                success = bool(obs.get("success"))
            except (TypeError, ValueError):
                continue
            idx = _bucket_index(score)
            buckets[idx].count += 1
            if success:
                buckets[idx].successes += 1
        self._tables[agent_type] = buckets

    async def rebuild(
        self,
        *,
        org_id: str,
        agent_type: str = "default",
        window_days: int = 30,
    ) -> CalibrationReport:
        """Rebuild the calibration table from ActionExecutionRecord history.

        Requires a live DB session.  Each record's payload_summary is checked
        for a ``predicted_confidence`` field written by the action engine; the
        record status is used as the success signal.
        """
        if self._session is None:
            return self.report(agent_type=agent_type)

        from datetime import datetime, timedelta, timezone
        from sqlalchemy import and_, select
        from app.models.action_execution_record import ActionExecutionRecord

        now = datetime.now(timezone.utc)
        start = now - timedelta(days=max(1, int(window_days)))

        stmt = select(
            ActionExecutionRecord.status,
            ActionExecutionRecord.payload_summary,
        ).where(
            and_(
                ActionExecutionRecord.organization_id == org_id,
                ActionExecutionRecord.created_at >= start,
                ActionExecutionRecord.created_at < now,
            )
        )
        res = await self._session.execute(stmt)
        rows = res.all()

        outcomes: list[dict[str, Any]] = []
        for status, payload_summary in rows:
            payload = payload_summary or {}
            raw_conf = payload.get("predicted_confidence")
            if raw_conf is None:
                continue
            success = str(status).lower() in {"success", "completed", "done"}
            outcomes.append({"predicted_confidence": raw_conf, "success": success})

        self.ingest_outcomes(agent_type=agent_type, outcomes=outcomes)
        return self.report(agent_type=agent_type)

    # ------------------------------------------------------------------
    # Apply calibration
    # ------------------------------------------------------------------

    def calibrate(self, raw_score: float, agent_type: str = "default") -> float:
        """Return a calibrated confidence score.

        If no calibration table exists for agent_type, or if the relevant
        bucket has fewer than min_samples, the raw score is returned unchanged.

        Blending formula (for well-populated buckets):
            calibrated = (1 - _SMOOTHING) * observed_rate + _SMOOTHING * raw_score

        The smoothing term prevents the correction from over-fitting sparse data
        and keeps scores from collapsing to 0 or 1 at the extremes.
        """
        try:
            raw = float(raw_score)
        except (TypeError, ValueError):
            return 0.5

        raw = max(0.0, min(1.0, raw))
        buckets = self._tables.get(agent_type)
        if not buckets:
            return round(raw, 4)

        idx = _bucket_index(raw)
        bucket = buckets[idx]
        observed = bucket.observed_rate

        if observed is None:
            return round(raw, 4)

        calibrated = (1.0 - _SMOOTHING) * observed + _SMOOTHING * raw
        return round(max(0.0, min(1.0, calibrated)), 4)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def report(self, *, agent_type: str = "default") -> CalibrationReport:
        """Return the current calibration table as a report."""
        buckets = self._tables.get(agent_type, self._empty_buckets())
        calibrated = sum(1 for b in buckets if b.count >= _MIN_SAMPLES)
        return CalibrationReport(
            agent_type=agent_type,
            buckets=buckets,
            total_samples=sum(b.count for b in buckets),
            calibrated_buckets=calibrated,
        )

    def is_calibrated(self, agent_type: str = "default") -> bool:
        """True if at least one bucket has enough samples."""
        buckets = self._tables.get(agent_type, [])
        return any(b.count >= _MIN_SAMPLES for b in buckets)
