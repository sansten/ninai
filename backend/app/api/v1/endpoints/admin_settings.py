"""Admin settings endpoints.

Exposes runtime-editable configuration (DB-backed) to system administrators.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import TenantContext, require_org_admin
from app.models.org_feedback_learning_config import OrgFeedbackLearningConfig
from app.schemas.admin_settings import (
    AuthConfig,
    AuthConfigResponse,
    AuthConfigUpdate,
    CognitiveAutonomyConfig,
    CognitiveAutonomyConfigResponse,
    CognitiveAutonomyConfigUpdate,
    EnvSetting,
    EnvSettingsResponse,
)
from app.services.app_settings_service import get_effective_auth_config, update_auth_config_overrides
from app.services.cognitive_autonomy_control_service import get_cognitive_autonomy_control_service
from app.services.skills_studio_service import (
    CORE_AGENT_NAMES,
    build_agent_skill_rows,
    list_agent_names,
    load_skills_studio_state,
    mark_published,
    normalize_skill_payload,
    save_skills_studio_state,
)


class FeedbackLearningResponse(BaseModel):
    updated_thresholds: dict
    stopwords: list
    heuristic_weights: dict
    calibration_delta: dict
    last_agent_version: Optional[str] = None
    updated_at: Optional[str] = None


class AgentSkillDraft(BaseModel):
    enabled: bool = False
    instructions: str = ""
    parameters: dict[str, Any] = {}


class AgentSkillRow(BaseModel):
    agent_name: str
    is_core: bool = False
    skill: AgentSkillDraft
    published_skill: AgentSkillDraft


class CoreAgentRow(BaseModel):
    agent_name: str
    is_core: bool = True


class SkillsStudioResponse(BaseModel):
    total_agents: int
    core_agents: list[CoreAgentRow]
    non_core_agents: list[AgentSkillRow]
    last_published_at: Optional[str] = None
    last_published_by_user_id: Optional[str] = None


class UpdateSkillRequest(BaseModel):
    enabled: Optional[bool] = None
    instructions: Optional[str] = None
    parameters: Optional[dict[str, Any]] = None


class PublishSkillsResponse(BaseModel):
    status: str
    published_count: int
    last_published_at: str


router = APIRouter()


@router.get("/settings/auth", response_model=AuthConfigResponse)
async def get_auth_settings(
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    effective, overrides = await get_effective_auth_config(db)
    return AuthConfigResponse(effective=AuthConfig(**effective), overrides=overrides)


@router.put("/settings/auth", response_model=AuthConfigResponse)
async def put_auth_settings(
    body: AuthConfigUpdate,
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    patch: dict[str, Any] = body.model_dump(exclude_unset=True)
    overrides = await update_auth_config_overrides(
        db,
        patch=patch,
        updated_by_user_id=tenant.user_id,
    )
    await db.commit()

    effective, _ = await get_effective_auth_config(db)
    return AuthConfigResponse(effective=AuthConfig(**effective), overrides=overrides)


_SENSITIVE_SUBSTRINGS = (
    "PASSWORD",
    "SECRET",
    "TOKEN",
    "PRIVATE",
    "KEY",
)

_ALLOWED_EXCEPTIONS = {
    # Not secrets; keep visible.
    "OIDC_CLIENT_ID",
    "JWT_ALGORITHM",
    "API_KEY",  # not present today, but common; still consider masking in prod
}


def _is_sensitive(key: str) -> bool:
    if key in _ALLOWED_EXCEPTIONS:
        return False
    upper = key.upper()
    return any(s in upper for s in _SENSITIVE_SUBSTRINGS)


def _stringify(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return ",".join(str(v) for v in value)
    return str(value)


@router.get("/settings/env", response_model=EnvSettingsResponse)
async def get_env_settings(
    tenant: TenantContext = Depends(require_org_admin()),
):
    # NOTE: This is read-only; changing .env values still requires redeploy/restart.
    raw = settings.model_dump()

    items: list[EnvSetting] = []
    for k in sorted(raw.keys()):
        sensitive = _is_sensitive(k)
        value = None if sensitive else _stringify(raw.get(k))
        if sensitive and raw.get(k) is not None:
            value = "***"
        items.append(
            EnvSetting(
                key=k,
                value=value,
                is_sensitive=sensitive,
                requires_restart=True,
            )
        )

    return EnvSettingsResponse(items=items)


@router.get("/cognitive-autonomy", response_model=CognitiveAutonomyConfigResponse)
async def get_cognitive_autonomy_settings(
    tenant: TenantContext = Depends(require_org_admin()),
):
    svc = get_cognitive_autonomy_control_service()
    snap = svc.snapshot(org_id=tenant.org_id)

    global_cfg = snap.get("global") or {}
    org_cfg = snap.get("org")
    effective = snap.get("effective") or {}

    return CognitiveAutonomyConfigResponse(
        org_id=tenant.org_id,
        global_config=CognitiveAutonomyConfig(
            enabled=bool(global_cfg.get("enabled", True)),
            reason=global_cfg.get("reason"),
        ),
        org_config=(
            CognitiveAutonomyConfig(
                enabled=bool(org_cfg.get("enabled", True)),
                reason=org_cfg.get("reason"),
            )
            if org_cfg
            else None
        ),
        effective=CognitiveAutonomyConfig(
            enabled=bool(effective.get("enabled", True)),
            reason=effective.get("reason"),
        ),
    )


@router.put("/cognitive-autonomy", response_model=CognitiveAutonomyConfigResponse)
async def put_cognitive_autonomy_settings(
    body: CognitiveAutonomyConfigUpdate,
    tenant: TenantContext = Depends(require_org_admin()),
):
    svc = get_cognitive_autonomy_control_service()

    if body.global_enabled is not None:
        svc.set_global(enabled=bool(body.global_enabled), reason=body.global_reason)

    if body.enabled is not None:
        svc.set_org(tenant.org_id, enabled=bool(body.enabled), reason=body.reason)

    snap = svc.snapshot(org_id=tenant.org_id)
    global_cfg = snap.get("global") or {}
    org_cfg = snap.get("org")
    effective = snap.get("effective") or {}

    return CognitiveAutonomyConfigResponse(
        org_id=tenant.org_id,
        global_config=CognitiveAutonomyConfig(
            enabled=bool(global_cfg.get("enabled", True)),
            reason=global_cfg.get("reason"),
        ),
        org_config=(
            CognitiveAutonomyConfig(
                enabled=bool(org_cfg.get("enabled", True)),
                reason=org_cfg.get("reason"),
            )
            if org_cfg
            else None
        ),
        effective=CognitiveAutonomyConfig(
            enabled=bool(effective.get("enabled", True)),
            reason=effective.get("reason"),
        ),
    )


@router.get("/feedback-learning", response_model=FeedbackLearningResponse)
async def get_feedback_learning_config(
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    """Return the current feedback learning calibration for this organisation."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
    result = await db.execute(
        select(OrgFeedbackLearningConfig).where(
            OrgFeedbackLearningConfig.organization_id == tenant.org_id
        )
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No calibration data found for this organisation")
    return FeedbackLearningResponse(
        updated_thresholds=row.updated_thresholds or {},
        stopwords=row.stopwords or [],
        heuristic_weights=row.heuristic_weights or {},
        calibration_delta=row.calibration_delta or {},
        last_agent_version=row.last_agent_version,
        updated_at=row.updated_at.isoformat() if row.updated_at else None,
    )


@router.delete("/feedback-learning", status_code=status.HTTP_204_NO_CONTENT)
async def reset_feedback_learning_config(
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    """Delete the feedback learning calibration row for this organisation, forcing a fresh start."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
    await db.execute(
        delete(OrgFeedbackLearningConfig).where(
            OrgFeedbackLearningConfig.organization_id == tenant.org_id
        )
    )
    await db.commit()


@router.get("/skills-studio", response_model=SkillsStudioResponse)
async def get_skills_studio_settings(
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
    state = await load_skills_studio_state(db, tenant.org_id)
    core_rows, non_core_rows = build_agent_skill_rows(state)

    return SkillsStudioResponse(
        total_agents=len(list_agent_names()),
        core_agents=[CoreAgentRow(**row) for row in core_rows],
        non_core_agents=[AgentSkillRow(**row) for row in non_core_rows],
        last_published_at=state.get("last_published_at"),
        last_published_by_user_id=state.get("last_published_by_user_id"),
    )


@router.put("/skills-studio/{agent_name}", response_model=AgentSkillRow)
async def update_agent_skill(
    agent_name: str,
    body: UpdateSkillRequest,
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    all_agents = set(list_agent_names())
    if agent_name not in all_agents:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown agent")
    if agent_name in CORE_AGENT_NAMES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Core agents are not editable in Skills Studio",
        )

    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
    state = await load_skills_studio_state(db, tenant.org_id)

    draft = state.setdefault("draft", {})
    existing = normalize_skill_payload(draft.get(agent_name))

    patch = body.model_dump(exclude_unset=True)
    if "enabled" in patch:
        existing["enabled"] = bool(patch["enabled"])
    if "instructions" in patch:
        existing["instructions"] = str(patch["instructions"] or "")
    if "parameters" in patch:
        existing["parameters"] = patch["parameters"] if isinstance(patch["parameters"], dict) else {}

    draft[agent_name] = existing
    await save_skills_studio_state(
        db,
        org_id=tenant.org_id,
        state=state,
        updated_by_user_id=tenant.user_id,
    )
    await db.commit()

    published = normalize_skill_payload(state.get("published", {}).get(agent_name))
    return AgentSkillRow(
        agent_name=agent_name,
        is_core=False,
        skill=AgentSkillDraft(**existing),
        published_skill=AgentSkillDraft(**published),
    )


@router.post("/skills-studio/publish", response_model=PublishSkillsResponse)
async def publish_skills_studio(
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
    state = await load_skills_studio_state(db, tenant.org_id)

    state = mark_published(state, tenant.user_id)
    await save_skills_studio_state(
        db,
        org_id=tenant.org_id,
        state=state,
        updated_by_user_id=tenant.user_id,
    )
    await db.commit()

    return PublishSkillsResponse(
        status="published",
        published_count=len(state.get("published", {})),
        last_published_at=state.get("last_published_at"),
    )
