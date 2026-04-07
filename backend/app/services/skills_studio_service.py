"""Skills Studio service.

Stores per-organization skill customizations for non-core agents in app_settings.

Model summary:
- Every non-core agent has a baseline immutable `v1` version.
- Editors (developers/admins) can save drafts and submit for approval.
- Admins approve submissions into new immutable versions (v2, v3, ...).
- Admins can rollback active skill version to v1 (or any prior version).
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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_version_record() -> dict[str, Any]:
    return {
        "version": "v1",
        "skill": dict(DEFAULT_SKILL),
        "approved_at": _now_iso(),
        "approved_by_user_id": None,
        "source": "baseline",
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
        state = {
            "draft": {},
            "submissions": {},
            "versions": {},
            "active_versions": {},
            "last_published_at": None,
            "last_published_by_user_id": None,
        }
        return ensure_baseline_versions(state)

    value = row.value
    state = {
        "draft": value.get("draft") if isinstance(value.get("draft"), dict) else {},
        "submissions": value.get("submissions") if isinstance(value.get("submissions"), dict) else {},
        "versions": value.get("versions") if isinstance(value.get("versions"), dict) else {},
        "active_versions": value.get("active_versions") if isinstance(value.get("active_versions"), dict) else {},
        "last_published_at": value.get("last_published_at"),
        "last_published_by_user_id": value.get("last_published_by_user_id"),
    }

    # Backward compatibility: import legacy published snapshots if present.
    legacy_published = value.get("published") if isinstance(value.get("published"), dict) else {}
    state = ensure_baseline_versions(state, legacy_published=legacy_published)
    return state


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
        "submissions": state.get("submissions", {}),
        "versions": state.get("versions", {}),
        "active_versions": state.get("active_versions", {}),
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
    state = ensure_baseline_versions(state)
    core, non_core = partition_agents()
    draft = state.get("draft", {})
    submissions = state.get("submissions", {})
    versions_map = state.get("versions", {})
    active_versions = state.get("active_versions", {})

    core_rows = [{"agent_name": name, "is_core": True} for name in core]

    non_core_rows: list[dict[str, Any]] = []
    for name in non_core:
        versions = versions_map.get(name) if isinstance(versions_map.get(name), list) else []
        active_version = str(active_versions.get(name) or "v1")
        active_skill = resolve_version_skill(versions, active_version) or dict(DEFAULT_SKILL)
        draft_skill = normalize_skill_payload(draft.get(name) if name in draft else active_skill)
        submitted = submissions.get(name) if isinstance(submissions.get(name), dict) else None

        non_core_rows.append(
            {
                "agent_name": name,
                "is_core": False,
                "skill": draft_skill,
                "published_skill": normalize_skill_payload(active_skill),
                "active_version": active_version,
                "submitted": submitted,
                "versions": versions,
            }
        )

    return core_rows, non_core_rows


def ensure_baseline_versions(
    state: dict[str, Any],
    *,
    legacy_published: dict[str, Any] | None = None,
) -> dict[str, Any]:
    core, non_core = partition_agents()
    _ = core

    versions_map = state.setdefault("versions", {})
    active_versions = state.setdefault("active_versions", {})
    draft = state.setdefault("draft", {})
    submissions = state.setdefault("submissions", {})

    for name in non_core:
        versions = versions_map.get(name)
        if not isinstance(versions, list) or len(versions) == 0:
            versions = [_default_version_record()]
            versions_map[name] = versions

        if not active_versions.get(name):
            active_versions[name] = "v1"

        if name not in draft:
            active_skill = resolve_version_skill(versions, str(active_versions[name])) or dict(DEFAULT_SKILL)
            draft[name] = normalize_skill_payload(active_skill)

        if name not in submissions or not isinstance(submissions.get(name), dict):
            submissions.pop(name, None)

        # One-time legacy import from old "published" snapshot into v2 if useful.
        if legacy_published and name in legacy_published and len(versions) == 1:
            legacy_skill = normalize_skill_payload(legacy_published.get(name))
            if legacy_skill != DEFAULT_SKILL:
                versions.append(
                    {
                        "version": "v2",
                        "skill": legacy_skill,
                        "approved_at": _now_iso(),
                        "approved_by_user_id": None,
                        "source": "legacy_import",
                    }
                )
                active_versions[name] = "v2"
                draft[name] = legacy_skill

    return state


def parse_version_num(version: str) -> int:
    v = str(version or "").strip().lower()
    if not v.startswith("v"):
        return 0
    try:
        return int(v[1:])
    except Exception:
        return 0


def next_version_label(versions: list[dict[str, Any]]) -> str:
    highest = 1
    for item in versions:
        if isinstance(item, dict):
            highest = max(highest, parse_version_num(str(item.get("version"))))
    return f"v{highest + 1}"


def resolve_version_skill(versions: list[dict[str, Any]], target_version: str) -> dict[str, Any] | None:
    for item in versions:
        if not isinstance(item, dict):
            continue
        if str(item.get("version")) == str(target_version):
            return normalize_skill_payload(item.get("skill") if isinstance(item.get("skill"), dict) else None)
    return None


def submit_agent_skill(
    state: dict[str, Any],
    *,
    agent_name: str,
    submitted_by_user_id: str | None,
) -> dict[str, Any]:
    state = ensure_baseline_versions(state)
    draft = state.setdefault("draft", {})
    submissions = state.setdefault("submissions", {})

    current = normalize_skill_payload(draft.get(agent_name))
    submissions[agent_name] = {
        "skill": current,
        "submitted_at": _now_iso(),
        "submitted_by_user_id": submitted_by_user_id,
        "status": "submitted",
    }
    return submissions[agent_name]


def approve_agent_skill(
    state: dict[str, Any],
    *,
    agent_name: str,
    approved_by_user_id: str | None,
) -> dict[str, Any]:
    state = ensure_baseline_versions(state)
    submissions = state.setdefault("submissions", {})
    versions_map = state.setdefault("versions", {})
    active_versions = state.setdefault("active_versions", {})

    pending = submissions.get(agent_name)
    if not isinstance(pending, dict):
        raise ValueError("No submitted draft found for agent")

    versions = versions_map.get(agent_name) if isinstance(versions_map.get(agent_name), list) else []
    if not versions:
        versions = [_default_version_record()]
        versions_map[agent_name] = versions

    next_label = next_version_label(versions)
    approved_skill = normalize_skill_payload(pending.get("skill") if isinstance(pending.get("skill"), dict) else None)
    record = {
        "version": next_label,
        "skill": approved_skill,
        "approved_at": _now_iso(),
        "approved_by_user_id": approved_by_user_id,
        "source": "approval",
    }
    versions.append(record)
    active_versions[agent_name] = next_label
    submissions.pop(agent_name, None)
    state["last_published_at"] = record["approved_at"]
    state["last_published_by_user_id"] = approved_by_user_id
    return record


def rollback_agent_skill(
    state: dict[str, Any],
    *,
    agent_name: str,
    target_version: str,
    approved_by_user_id: str | None,
) -> dict[str, Any]:
    state = ensure_baseline_versions(state)
    versions_map = state.setdefault("versions", {})
    active_versions = state.setdefault("active_versions", {})

    versions = versions_map.get(agent_name) if isinstance(versions_map.get(agent_name), list) else []
    if not versions:
        raise ValueError("No versions found for agent")

    target_skill = resolve_version_skill(versions, target_version)
    if target_skill is None:
        raise ValueError(f"Version {target_version} not found")

    next_label = next_version_label(versions)
    record = {
        "version": next_label,
        "skill": target_skill,
        "approved_at": _now_iso(),
        "approved_by_user_id": approved_by_user_id,
        "source": f"rollback:{target_version}",
    }
    versions.append(record)
    active_versions[agent_name] = next_label
    state["last_published_at"] = record["approved_at"]
    state["last_published_by_user_id"] = approved_by_user_id
    return record


def mark_published(state: dict[str, Any], user_id: str | None) -> dict[str, Any]:
    """Backward-compatible publish-all helper.

    Converts all current drafts into new versions and marks them active.
    """
    state = ensure_baseline_versions(state)
    draft = state.get("draft", {})
    versions_map = state.setdefault("versions", {})
    active_versions = state.setdefault("active_versions", {})

    for agent_name, skill in draft.items():
        versions = versions_map.get(agent_name) if isinstance(versions_map.get(agent_name), list) else []
        if not versions:
            versions = [_default_version_record()]
            versions_map[agent_name] = versions
        label = next_version_label(versions)
        versions.append(
            {
                "version": label,
                "skill": normalize_skill_payload(skill if isinstance(skill, dict) else None),
                "approved_at": _now_iso(),
                "approved_by_user_id": user_id,
                "source": "publish_all",
            }
        )
        active_versions[agent_name] = label

    state["last_published_at"] = _now_iso()
    state["last_published_by_user_id"] = user_id
    return state
