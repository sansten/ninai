"""Feature Readiness Service — Phase 28.

Evaluates whether each enrichment agent has enough data in a tenant's store to
produce meaningful output.  When data conditions are not met the feature should
be greyed out in the UI so it does not mislead users with noise.

Design principles
-----------------
- Readiness is evaluated per-tenant from live DB data (memory count, age, source
  diversity) rather than from agent runs, so it works before any agent has run.
- Each agent exposes a ``ReadinessSpec`` with minimum thresholds and human-readable
  descriptions.  All specs live here to avoid touching agent files.
- Results are stored in ``FeatureReadinessConfig`` so the API can return them
  instantly without re-querying and so admins can inspect past evaluations.
- Admins may force-enable or force-disable any agent via the ``enabled`` flag,
  which overrides the computed readiness for UI purposes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.feature_readiness import FeatureReadinessConfig
from app.models.memory import MemoryMetadata


# ---------------------------------------------------------------------------
# Readiness spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReadinessSpec:
    """Minimum data conditions for an agent to produce meaningful output."""

    description: str
    """Human-readable explanation of what this agent does."""

    min_memories: int = 1
    """Minimum number of stored memories required."""

    min_days_active: int = 0
    """Minimum days since the tenant's first memory was created."""

    min_sources: int = 1
    """Minimum number of distinct source_type values required."""

    default_params: dict[str, Any] = field(default_factory=dict)
    """Default tuneable parameters exposed to the UI."""


# ---------------------------------------------------------------------------
# Agent specs (one per enrichment agent, Phases 6-27)
# ---------------------------------------------------------------------------

AGENT_SPECS: dict[str, ReadinessSpec] = {
    "SemanticNormalizationAgent": ReadinessSpec(
        description="Normalises text and tags so downstream agents see consistent signals.",
        min_memories=1,
        min_days_active=0,
        min_sources=1,
        default_params={"min_text_length": 10},
    ),
    "EntityResolutionAgent": ReadinessSpec(
        description="Identifies and deduplicates named entities across memories. "
                    "Needs sufficient volume to detect recurring entity names.",
        min_memories=50,
        min_days_active=7,
        min_sources=1,
        default_params={"similarity_threshold": 0.85, "max_aliases": 5},
    ),
    "ContextAmplifierAgent": ReadinessSpec(
        description="Enriches memories with domain context from related memories. "
                    "Requires a small base of existing memories to draw context from.",
        min_memories=20,
        min_days_active=3,
        min_sources=1,
        default_params={"context_window": 10},
    ),
    "SiloPropagationAgent": ReadinessSpec(
        description="Propagates signals across organisational silos. "
                    "Requires data from at least two distinct sources.",
        min_memories=100,
        min_days_active=14,
        min_sources=2,
        default_params={"propagation_depth": 2},
    ),
    "OrgAttentionAgent": ReadinessSpec(
        description="Ranks organisational importance of memories by attention signals. "
                    "Needs cross-silo activity to distinguish high-attention topics.",
        min_memories=50,
        min_days_active=7,
        min_sources=2,
        default_params={"attention_decay_hours": 72},
    ),
    "ProactiveMemoryPushAgent": ReadinessSpec(
        description="Proactively surfaces relevant memories before they are queried. "
                    "Requires enough history to detect recurring patterns.",
        min_memories=100,
        min_days_active=14,
        min_sources=1,
        default_params={"push_threshold": 0.70},
    ),
    "CausalReasoningAgent": ReadinessSpec(
        description="Infers cause-and-effect links between memories. "
                    "Needs a corpus of events to reason over.",
        min_memories=50,
        min_days_active=7,
        min_sources=1,
        default_params={"min_causal_confidence": 0.60},
    ),
    "ConflictDetectionAgent": ReadinessSpec(
        description="Detects contradictory information across memories. "
                    "Most useful when data arrives from multiple sources.",
        min_memories=20,
        min_days_active=3,
        min_sources=2,
        default_params={"conflict_severity_threshold": "medium"},
    ),
    "AdaptiveConflictResolutionAgent": ReadinessSpec(
        description="Automatically resolves detected conflicts using learned patterns. "
                    "Requires prior conflict detections to learn from.",
        min_memories=30,
        min_days_active=7,
        min_sources=2,
        default_params={"resolution_confidence_min": 0.70},
    ),
    "MemoryDecayAgent": ReadinessSpec(
        description="Ages memories by relevance over time. "
                    "Requires enough historical depth to compute meaningful decay curves.",
        min_memories=100,
        min_days_active=30,
        min_sources=1,
        default_params={"half_life_days": 90, "min_decay_factor": 0.10},
    ),
    "MemoryConsolidationAgent": ReadinessSpec(
        description="Merges near-duplicate memories to reduce redundancy. "
                    "Needs volume so duplicates statistically appear.",
        min_memories=100,
        min_days_active=14,
        min_sources=1,
        default_params={"consolidation_similarity": 0.90},
    ),
    "TemporalReasoningAgent": ReadinessSpec(
        description="Reasons about time-ordered sequences of events. "
                    "Needs a corpus with temporal references.",
        min_memories=50,
        min_days_active=7,
        min_sources=1,
        default_params={"temporal_window_days": 30},
    ),
    "EpisodicGroupingAgent": ReadinessSpec(
        description="Clusters memories into discrete episodes or sessions. "
                    "Requires enough temporal breadth to form meaningful episode boundaries.",
        min_memories=50,
        min_days_active=14,
        min_sources=1,
        default_params={"episode_gap_hours": 4, "min_episode_size": 3},
    ),
    "CredibilityAgent": ReadinessSpec(
        description="Scores the trustworthiness of memories by source tier and signals. "
                    "Benefits from source diversity to distinguish credibility levels.",
        min_memories=20,
        min_days_active=3,
        min_sources=2,
        default_params={"default_credibility": 0.50, "high_tier_boost": 0.20},
    ),
    "PlaybookAgent": ReadinessSpec(
        description="Extracts reusable procedural playbooks from memories. "
                    "Needs enough procedural content to detect patterns.",
        min_memories=30,
        min_days_active=7,
        min_sources=1,
        default_params={"min_playbook_confidence": 0.65},
    ),
    "GoalDecompositionAgent": ReadinessSpec(
        description="Decomposes high-level goals into actionable sub-tasks from memory. "
                    "Needs content containing goal-like language.",
        min_memories=30,
        min_days_active=7,
        min_sources=1,
        default_params={"max_subtasks": 8},
    ),
    "UncertaintyReportingAgent": ReadinessSpec(
        description="Aggregates uncertainty signals from all prior agents. "
                    "Useful once any enrichment agents have run.",
        min_memories=20,
        min_days_active=3,
        min_sources=1,
        default_params={"uncertainty_review_threshold": "high"},
    ),
    "NarrativeSynthesisAgent": ReadinessSpec(
        description="Synthesises memories into human-readable narratives. "
                    "Useful even on small datasets.",
        min_memories=20,
        min_days_active=3,
        min_sources=1,
        default_params={"max_narrative_length": 500},
    ),
    "FeedbackIntegrationAgent": ReadinessSpec(
        description="Incorporates human review verdicts back into enrichment. "
                    "Requires users to have submitted feedback.",
        min_memories=50,
        min_days_active=14,
        min_sources=1,
        default_params={"feedback_weight": 0.25},
    ),
    "AnomalyDetectionAgent": ReadinessSpec(
        description="Detects unusual patterns by comparing against learned baselines. "
                    "Requires enough history to establish a baseline.",
        min_memories=100,
        min_days_active=14,
        min_sources=1,
        default_params={
            "credibility_spike_threshold": 0.25,
            "uncertainty_surge_threshold": 0.80,
            "conflict_flood_threshold": 5,
        },
    ),
    "QueryIntelligenceAgent": ReadinessSpec(
        description="Enriches the read path with intent detection and dynamic filters. "
                    "Useful from the first memory.",
        min_memories=1,
        min_days_active=0,
        min_sources=1,
        default_params={},
    ),
}


# ---------------------------------------------------------------------------
# Data snapshot
# ---------------------------------------------------------------------------

async def _fetch_data_snapshot(org_id: str, db: AsyncSession) -> dict[str, Any]:
    """Query the DB once and return the raw data facts for readiness evaluation."""
    count_q = select(func.count()).select_from(MemoryMetadata).where(
        MemoryMetadata.organization_id == org_id
    )
    count_result = await db.execute(count_q)
    memory_count: int = count_result.scalar_one() or 0

    min_date_q = select(func.min(MemoryMetadata.created_at)).where(
        MemoryMetadata.organization_id == org_id
    )
    min_date_result = await db.execute(min_date_q)
    first_memory_at: Optional[datetime] = min_date_result.scalar_one()

    if first_memory_at and first_memory_at.tzinfo is None:
        first_memory_at = first_memory_at.replace(tzinfo=timezone.utc)

    active_days = 0
    if first_memory_at:
        delta = datetime.now(timezone.utc) - first_memory_at
        active_days = max(0, delta.days)

    source_q = select(func.count(MemoryMetadata.source_type.distinct())).where(
        MemoryMetadata.organization_id == org_id
    )
    source_result = await db.execute(source_q)
    source_count: int = source_result.scalar_one() or 0

    return {
        "memory_count": memory_count,
        "active_days": active_days,
        "source_count": source_count,
    }


# ---------------------------------------------------------------------------
# Readiness evaluation
# ---------------------------------------------------------------------------

def _evaluate_spec(spec: ReadinessSpec, snapshot: dict[str, Any]) -> dict[str, Any]:
    """Return a readiness state dict for a single agent."""
    memory_count = snapshot["memory_count"]
    active_days = snapshot["active_days"]
    source_count = snapshot["source_count"]

    reasons: list[str] = []
    blocking: list[str] = []

    # Memory count
    if memory_count >= spec.min_memories:
        reasons.append(f"memory_count={memory_count} >= {spec.min_memories}")
    else:
        blocking.append(
            f"need {spec.min_memories} memories, have {memory_count}"
        )

    # Active days
    if active_days >= spec.min_days_active:
        reasons.append(f"active_days={active_days} >= {spec.min_days_active}")
    else:
        blocking.append(
            f"need {spec.min_days_active} days of activity, have {active_days}"
        )

    # Source diversity
    if source_count >= spec.min_sources:
        reasons.append(f"source_count={source_count} >= {spec.min_sources}")
    else:
        blocking.append(
            f"need {spec.min_sources} distinct sources, have {source_count}"
        )

    is_ready = len(blocking) == 0

    return {
        "is_ready": is_ready,
        "reasons": reasons,
        "blocking_reasons": blocking,
        "data_snapshot": snapshot,
    }


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class FeatureReadinessService:
    """Evaluate and persist feature readiness state for all enrichment agents."""

    async def _get_or_create_config(
        self,
        org_id: str,
        agent_name: str,
        db: AsyncSession,
    ) -> FeatureReadinessConfig:
        stmt = select(FeatureReadinessConfig).where(
            FeatureReadinessConfig.organization_id == org_id,
            FeatureReadinessConfig.agent_name == agent_name,
        )
        result = await db.execute(stmt)
        config = result.scalar_one_or_none()
        if config is None:
            config = FeatureReadinessConfig(
                organization_id=org_id,
                agent_name=agent_name,
                enabled=True,
                custom_params=None,
                readiness_state=None,
                evaluated_at=None,
            )
            db.add(config)
            await db.flush()
        return config

    async def evaluate_all(
        self,
        org_id: str,
        db: AsyncSession,
    ) -> list[dict[str, Any]]:
        """Run readiness evaluation for all agents and persist results."""
        snapshot = await _fetch_data_snapshot(org_id, db)
        now = datetime.now(timezone.utc)
        results: list[dict[str, Any]] = []

        for agent_name, spec in AGENT_SPECS.items():
            state = _evaluate_spec(spec, snapshot)
            config = await self._get_or_create_config(org_id, agent_name, db)
            config.readiness_state = state
            config.evaluated_at = now

            effective_params = dict(spec.default_params)
            if config.custom_params:
                effective_params.update(config.custom_params)

            results.append(
                _build_response_item(agent_name, spec, config, effective_params)
            )

        await db.commit()
        return results

    async def get_all(
        self,
        org_id: str,
        db: AsyncSession,
    ) -> list[dict[str, Any]]:
        """Return persisted readiness configs without re-evaluating.

        Falls back to evaluate_all if no rows exist yet.
        """
        stmt = select(FeatureReadinessConfig).where(
            FeatureReadinessConfig.organization_id == org_id
        )
        result = await db.execute(stmt)
        configs = {row.agent_name: row for row in result.scalars().all()}

        if not configs:
            return await self.evaluate_all(org_id, db)

        items: list[dict[str, Any]] = []
        for agent_name, spec in AGENT_SPECS.items():
            config = configs.get(agent_name)
            if config is None:
                # New agent added after initial evaluation — synthesise on-the-fly.
                items.append(
                    _build_uneval_item(agent_name, spec)
                )
                continue

            effective_params = dict(spec.default_params)
            if config.custom_params:
                effective_params.update(config.custom_params)

            items.append(
                _build_response_item(agent_name, spec, config, effective_params)
            )

        return items

    async def update_config(
        self,
        org_id: str,
        agent_name: str,
        enabled: Optional[bool],
        custom_params: Optional[dict[str, Any]],
        notes: Optional[str],
        db: AsyncSession,
    ) -> dict[str, Any]:
        """Apply admin overrides for a single agent."""
        if agent_name not in AGENT_SPECS:
            raise ValueError(f"Unknown agent: {agent_name!r}")

        config = await self._get_or_create_config(org_id, agent_name, db)
        spec = AGENT_SPECS[agent_name]

        if enabled is not None:
            config.enabled = enabled
        if custom_params is not None:
            # Merge into existing overrides rather than replace
            existing = dict(config.custom_params or {})
            existing.update(custom_params)
            config.custom_params = existing
        if notes is not None:
            config.notes = notes

        await db.commit()

        effective_params = dict(spec.default_params)
        if config.custom_params:
            effective_params.update(config.custom_params)

        return _build_response_item(agent_name, spec, config, effective_params)


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------

def _build_response_item(
    agent_name: str,
    spec: ReadinessSpec,
    config: FeatureReadinessConfig,
    effective_params: dict[str, Any],
) -> dict[str, Any]:
    state = config.readiness_state or {}
    is_ready = bool(state.get("is_ready", False))
    return {
        "agent_name": agent_name,
        "description": spec.description,
        "enabled": config.enabled,
        "is_ready": is_ready,
        "active": config.enabled and is_ready,
        "readiness_state": state,
        "params": effective_params,
        "notes": config.notes,
        "evaluated_at": config.evaluated_at,
        "thresholds": {
            "min_memories": spec.min_memories,
            "min_days_active": spec.min_days_active,
            "min_sources": spec.min_sources,
        },
    }


def _build_uneval_item(agent_name: str, spec: ReadinessSpec) -> dict[str, Any]:
    return {
        "agent_name": agent_name,
        "description": spec.description,
        "enabled": True,
        "is_ready": False,
        "active": False,
        "readiness_state": {},
        "params": dict(spec.default_params),
        "notes": None,
        "evaluated_at": None,
        "thresholds": {
            "min_memories": spec.min_memories,
            "min_days_active": spec.min_days_active,
            "min_sources": spec.min_sources,
        },
    }
