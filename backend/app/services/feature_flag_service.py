"""FeatureFlagService - check and manage per-org feature flags."""

from __future__ import annotations

from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.org_feature_flag import OrgFeatureFlag


class FeatureFlagService:
    def __init__(self, session: AsyncSession, org_id: str):
        self.session = session
        self.org_id = org_id

    async def is_enabled(self, flag_name: str) -> bool:
        res = await self.session.execute(
            select(OrgFeatureFlag).where(
                OrgFeatureFlag.organization_id == self.org_id,
                OrgFeatureFlag.flag_name == flag_name,
            )
        )
        row = res.scalar_one_or_none()
        return bool(row and row.enabled)

    async def set_flag(self, *, flag_name: str, enabled: bool, rollout_pct: int = 100) -> OrgFeatureFlag:
        res = await self.session.execute(
            select(OrgFeatureFlag).where(
                OrgFeatureFlag.organization_id == self.org_id,
                OrgFeatureFlag.flag_name == flag_name,
            )
        )
        row = res.scalar_one_or_none()
        if not row:
            row = OrgFeatureFlag(id=str(uuid4()), organization_id=self.org_id, flag_name=flag_name)
            self.session.add(row)

        row.enabled = enabled
        row.rollout_pct = max(0, min(100, rollout_pct))
        await self.session.flush()
        return row

    async def list_flags(self) -> list[dict]:
        res = await self.session.execute(
            select(OrgFeatureFlag).where(OrgFeatureFlag.organization_id == self.org_id)
        )
        return [
            {"flag_name": r.flag_name, "enabled": r.enabled, "rollout_pct": r.rollout_pct}
            for r in res.scalars().all()
        ]
