"""Memory Resolution Write-Back Service — Gap B.

After UncertaintyResolutionService or HypothesisService complete their
analysis, the originating memory remains in its stale anomalous/uncertain
state forever.  Nothing closes the loop.

This service closes it:
  resolved report → patch memory's extra_metadata with:
    - resolution_status: "resolved" | "escalate" | "human_review" | "monitor"
    - resolution_confidence: float
    - resolved_at: ISO timestamp
    - root_cause_entity: str (from HypothesisReport.confirmed_hypothesis)
    - uncertainty_level: str (from UncertaintyResolutionReport)
    - anomaly_score: float (from HypothesisReport)

Design rules:
- Pure service: takes a MemoryService instance injected by the caller.
- Operates on UncertaintyResolutionReport and HypothesisReport directly.
- Never overwrites content or title; only merges into extra_metadata.
- Does NOT raise on failure; returns a WriteBackResult with ok=False so
  callers can fire-and-forget without try/except.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.services.uncertainty_resolution_service import UncertaintyResolutionReport
from app.services.hypothesis_service import HypothesisReport


@dataclass
class WriteBackResult:
    memory_id: str
    ok: bool
    error: str | None = None
    patch: dict[str, Any] | None = None


class MemoryResolutionWriteBackService:
    """Patches a memory's extra_metadata after a resolution or hypothesis pass.

    Typical usage (fire-and-forget after resolution):
        wb = MemoryResolutionWriteBackService(memory_service=mem_svc)
        await wb.apply_resolution(memory_id=mid, report=resolution_report)

    Typical usage after hypothesis analysis:
        await wb.apply_hypothesis(memory_id=mid, report=hypothesis_report)
    """

    def __init__(self, *, memory_service: Any) -> None:
        """
        Parameters
        ----------
        memory_service:  A MemoryService instance with update_memory().
                         Type is Any to avoid coupling to DB session at import time.
        """
        self._svc = memory_service

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def apply_resolution(
        self,
        *,
        memory_id: str,
        report: UncertaintyResolutionReport,
    ) -> WriteBackResult:
        """Patch memory with uncertainty resolution outcome."""
        patch = {
            "resolution_status": report.recommended_action,
            "resolution_confidence": report.resolution_confidence,
            "uncertainty_level_at_resolution": report.uncertainty_level,
            "uncertainty_score_at_resolution": report.uncertainty_score,
            "still_unresolved_fields": report.still_unresolved,
            "resolved_at": datetime.now(timezone.utc).isoformat(),
        }
        return await self._patch(memory_id=memory_id, patch=patch)

    async def apply_hypothesis(
        self,
        *,
        memory_id: str,
        report: HypothesisReport,
    ) -> WriteBackResult:
        """Patch memory with hypothesis analysis outcome."""
        patch: dict[str, Any] = {
            "hypothesis_action": report.recommended_action,
            "anomaly_score_at_analysis": report.anomaly_score,
            "anomaly_type_at_analysis": report.anomaly_type,
            "hypothesis_evidence_count": report.evidence_count,
            "hypothesis_analysed_at": datetime.now(timezone.utc).isoformat(),
        }
        if report.confirmed_hypothesis is not None:
            patch["root_cause_entity"] = report.confirmed_hypothesis.entity
            patch["root_cause_confidence"] = report.confirmed_hypothesis.confirmed_confidence
            patch["root_cause_path"] = report.confirmed_hypothesis.causal_path
            patch["counterfactual_hint"] = report.confirmed_hypothesis.counterfactual_hint
        return await self._patch(memory_id=memory_id, patch=patch)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _patch(self, *, memory_id: str, patch: dict[str, Any]) -> WriteBackResult:
        """Merge patch into the memory's extra_metadata via MemoryUpdate."""
        try:
            from app.schemas.memory import MemoryUpdate

            # Read existing extra_metadata first so we merge rather than overwrite.
            # MemoryService.update_memory replaces extra_metadata wholesale — we
            # need to fetch the current value first.
            current_meta: dict[str, Any] = {}
            try:
                mem = await self._svc.get_memory(memory_id)
                if mem is not None:
                    current_meta = dict(mem.extra_metadata or {})
            except Exception:
                current_meta = {}

            merged = {**current_meta, **patch}
            update = MemoryUpdate(extra_metadata=merged)
            await self._svc.update_memory(memory_id=memory_id, data=update)
            return WriteBackResult(memory_id=memory_id, ok=True, patch=patch)
        except Exception as exc:
            return WriteBackResult(
                memory_id=memory_id,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                patch=patch,
            )
