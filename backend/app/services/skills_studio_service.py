"""Skills Studio service.

Stores per-organization skill customizations for non-core agents in app_settings.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.registry import list_registered_agent_classes
from app.models.app_setting import AppSetting


SKILLS_STUDIO_KEY_PREFIX = "skills_studio"

# Safety-critical/infra-critical agents remain core and not user-editable.
CORE_AGENT_NAMES: set[str] = {
    "OrchestrationBusAgent",
    "HumanReviewQueueAgent",
    "AuditTrailAgent",
    "AutonomousActionAgent",
    "AdaptiveConflictResolutionAgent",
    "ConflictDetectionAgent",
    "UncertaintyReportingAgent",
    "CredibilityAgent",
    "MemoryDecayAgent",
    "MemoryConsolidationAgent",
    "FeedbackIntegrationAgent",
    "QueryIntelligenceAgent",
    "WorldModelAgent",
    "GoalDecompositionAgent",
}


DEFAULT_SKILL = {
    "enabled": False,
    "instructions": "",
    "parameters": {},
}


def _setting_key(org_id: str) -> str:
    return f"{SKILLS_STUDIO_KEY_PREFIX}:{org_id}"


def list_agent_names() -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for cls in list_registered_agent_classes():
        name = getattr(cls, "name", cls.__name__)
        if name not in seen:
            names.append(name)
            seen.add(name)
    names.sort()
    return names


def partition_agents() -> tuple[list[str], list[str]]:
    all_agents = list_agent_names()
    core = [name for name in all_agents if name in CORE_AGENT_NAMES]
    non_core = [name for name in all_agents if name not in CORE_AGENT_NAMES]
    return core, non_core


async def load_skills_studio_state(db: AsyncSession, org_id: str) -> dict[str, Any]:
    key = _setting_key(org_id)
    row = (await db.execute(select(AppSetting).where(AppSetting.key == key))).scalar_one_or_none()

    if row is None or not isinstance(row.value, dict):
        return {
            "draft": {},
            "published": {},
            "last_published_at": None,
            "last_published_by_user_id": None,
        }

    value = row.value
    return {
        "draft": value.get("draft") if isinstance(value.get("draft"), dict) else {},
        "published": value.get("published") if isinstance(value.get("published"), dict) else {},
        "last_published_at": value.get("last_published_at"),
        "last_published_by_user_id": value.get("last_published_by_user_id"),
    }


async def save_skills_studio_state(
    db: AsyncSession,
    *,
    org_id: str,
    state: dict[str, Any],
    updated_by_user_id: str | None,
) -> None:
    key = _setting_key(org_id)
    row = (await db.execute(select(AppSetting).where(AppSetting.key == key))).scalar_one_or_none()

    payload = {
        "draft": state.get("draft", {}),
        "published": state.get("published", {}),
        "last_published_at": state.get("last_published_at"),
        "last_published_by_user_id": state.get("last_published_by_user_id"),
    }

    if row is None:
        row = AppSetting(key=key, value=payload, updated_by_user_id=updated_by_user_id)
        db.add(row)
    else:
        row.value = payload
        row.updated_by_user_id = updated_by_user_id

    await db.flush()


def normalize_skill_payload(skill: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(skill, dict):
        return dict(DEFAULT_SKILL)

    enabled = bool(skill.get("enabled", False))
    instructions = str(skill.get("instructions") or "")
    parameters = skill.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {}

    return {
        "enabled": enabled,
        "instructions": instructions,
        "parameters": parameters,
    }


def build_agent_skill_rows(state: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    core, non_core = partition_agents()
    draft = state.get("draft", {})
    published = state.get("published", {})

    core_rows = [{"agent_name": name, "is_core": True} for name in core]

    non_core_rows: list[dict[str, Any]] = []
    for name in non_core:
        non_core_rows.append(
            {
                "agent_name": name,
                "is_core": False,
                "skill": normalize_skill_payload(draft.get(name)),
                "published_skill": normalize_skill_payload(published.get(name)),
            }
        )

    return core_rows, non_core_rows


def mark_published(state: dict[str, Any], user_id: str | None) -> dict[str, Any]:
    state["published"] = dict(state.get("draft", {}))
    state["last_published_at"] = datetime.now(timezone.utc).isoformat()
    state["last_published_by_user_id"] = user_id
    return state
