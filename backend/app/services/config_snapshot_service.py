"""Snapshot and mutation helpers for bounded auto-research parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.meta_learning_config import MetaLearningConfig
from app.models.org_feedback_learning_config import OrgFeedbackLearningConfig
from app.models.org_llm_config import OrgLlmConfig


@dataclass(frozen=True)
class ParameterSpec:
    key: str
    source: str
    location: str | tuple[str, str]
    default: float
    min_value: float
    max_value: float
    step: float


AUTO_RESEARCH_PARAMETER_REGISTRY: dict[str, ParameterSpec] = {
    "meta_learning.confidence_floor": ParameterSpec(
        key="meta_learning.confidence_floor",
        source="meta_learning",
        location="confidence_floor",
        default=0.4,
        min_value=0.1,
        max_value=0.9,
        step=0.05,
    ),
    "meta_learning.noise_threshold": ParameterSpec(
        key="meta_learning.noise_threshold",
        source="meta_learning",
        location="noise_threshold",
        default=0.85,
        min_value=0.5,
        max_value=0.99,
        step=0.05,
    ),
    "feedback.thresholds.acceptance": ParameterSpec(
        key="feedback.thresholds.acceptance",
        source="feedback_config",
        location=("updated_thresholds", "acceptance"),
        default=0.7,
        min_value=0.4,
        max_value=0.95,
        step=0.05,
    ),
    "feedback.heuristic_weights.recency": ParameterSpec(
        key="feedback.heuristic_weights.recency",
        source="feedback_config",
        location=("heuristic_weights", "recency"),
        default=1.0,
        min_value=0.2,
        max_value=2.0,
        step=0.1,
    ),
    "org_llm.temperature": ParameterSpec(
        key="org_llm.temperature",
        source="org_llm",
        location=("config", "temperature"),
        default=0.2,
        min_value=0.0,
        max_value=1.0,
        step=0.05,
    ),
    "org_llm.top_p": ParameterSpec(
        key="org_llm.top_p",
        source="org_llm",
        location=("config", "top_p"),
        default=0.9,
        min_value=0.1,
        max_value=1.0,
        step=0.05,
    ),
}


def _clamp(value: float, spec: ParameterSpec) -> float:
    return round(max(spec.min_value, min(spec.max_value, float(value))), 6)


class ConfigSnapshotService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def snapshot(self, org_id: str, param_keys: list[str] | None = None) -> dict[str, float]:
        keys = param_keys or list(AUTO_RESEARCH_PARAMETER_REGISTRY)
        snapshot: dict[str, float] = {}
        for key in keys:
            snapshot[key] = await self.get_parameter_value(org_id, key)
        return snapshot

    async def get_parameter_value(self, org_id: str, key: str) -> float:
        spec = AUTO_RESEARCH_PARAMETER_REGISTRY[key]
        record = await self._get_record(org_id, spec)
        return _clamp(self._extract_value(record, spec), spec)

    async def set_parameter_value(self, org_id: str, key: str, value: float) -> float:
        spec = AUTO_RESEARCH_PARAMETER_REGISTRY[key]
        record = await self._get_record(org_id, spec)
        bounded = _clamp(value, spec)
        self._assign_value(record, spec, bounded)
        await self.session.flush()
        return bounded

    async def restore_snapshot(self, org_id: str, values: dict[str, float]) -> None:
        for key, value in values.items():
            await self.set_parameter_value(org_id, key, value)

    async def _get_record(self, org_id: str, spec: ParameterSpec) -> Any:
        if spec.source == "meta_learning":
            return await self._get_or_create_meta_learning(org_id)
        if spec.source == "feedback_config":
            return await self._get_or_create_feedback_config(org_id)
        if spec.source == "org_llm":
            return await self._get_or_create_org_llm(org_id)
        raise KeyError(f"Unsupported source: {spec.source}")

    async def _get_or_create_meta_learning(self, org_id: str) -> MetaLearningConfig:
        result = await self.session.execute(select(MetaLearningConfig).where(MetaLearningConfig.org_id == org_id))
        record = result.scalar_one_or_none()
        if record is None:
            record = MetaLearningConfig(org_id=org_id)
            self.session.add(record)
            await self.session.flush()
        return record

    async def _get_or_create_feedback_config(self, org_id: str) -> OrgFeedbackLearningConfig:
        result = await self.session.execute(
            select(OrgFeedbackLearningConfig).where(OrgFeedbackLearningConfig.organization_id == org_id)
        )
        record = result.scalar_one_or_none()
        if record is None:
            record = OrgFeedbackLearningConfig(organization_id=org_id)
            self.session.add(record)
            await self.session.flush()
        return record

    async def _get_or_create_org_llm(self, org_id: str) -> OrgLlmConfig:
        result = await self.session.execute(select(OrgLlmConfig).where(OrgLlmConfig.organization_id == org_id))
        record = result.scalar_one_or_none()
        if record is None:
            record = OrgLlmConfig(organization_id=org_id)
            self.session.add(record)
            await self.session.flush()
        return record

    def _extract_value(self, record: Any, spec: ParameterSpec) -> float:
        if isinstance(spec.location, str):
            return float(getattr(record, spec.location, spec.default) or spec.default)

        root_name, nested_key = spec.location
        payload = getattr(record, root_name, None) or {}
        return float(payload.get(nested_key, spec.default))

    def _assign_value(self, record: Any, spec: ParameterSpec, value: float) -> None:
        if isinstance(spec.location, str):
            setattr(record, spec.location, value)
            return

        root_name, nested_key = spec.location
        payload = dict(getattr(record, root_name, None) or {})
        payload[nested_key] = value
        setattr(record, root_name, payload)