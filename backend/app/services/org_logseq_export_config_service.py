"""Org Logseq export config service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import inspect
from typing import Optional
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.org_logseq_export_config import OrgLogseqExportConfig


@dataclass(frozen=True)
class EffectiveLogseqExportConfig:
    export_base_dir: str
    org_export_dir: str
    override_base_dir: Optional[str]
    last_nightly_export_at: Optional[datetime]


class OrgLogseqExportConfigService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_config(self, *, organization_id: str) -> Optional[OrgLogseqExportConfig]:
        res = await self.session.execute(
            select(OrgLogseqExportConfig).where(OrgLogseqExportConfig.organization_id == organization_id)
        )
        row = res.scalar_one_or_none()
        if inspect.isawaitable(row):
            row = await row
        return row

    async def upsert_config(
        self,
        *,
        organization_id: str,
        export_base_dir: Optional[str],
        updated_by_user_id: Optional[str],
    ) -> OrgLogseqExportConfig:
        cleaned = None
        if export_base_dir is not None:
            v = str(export_base_dir).strip()
            cleaned = v or None

        stmt = (
            insert(OrgLogseqExportConfig)
            .values(
                {
                    "id": str(uuid4()),
                    "organization_id": organization_id,
                    "export_base_dir": cleaned,
                    "updated_by_user_id": updated_by_user_id,
                }
            )
            .on_conflict_do_update(
                index_elements=["organization_id"],
                set_={
                    "export_base_dir": cleaned,
                    "updated_by_user_id": updated_by_user_id,
                },
            )
            .returning(OrgLogseqExportConfig)
        )

        res = await self.session.execute(stmt)
        row = res.scalar_one()
        if inspect.isawaitable(row):
            row = await row
        await self.session.flush()
        return row

    async def update_last_nightly_export_at(
        self,
        *,
        organization_id: str,
        last_nightly_export_at: datetime,
    ) -> None:
        stmt = (
            insert(OrgLogseqExportConfig)
            .values(
                {
                    "id": str(uuid4()),
                    "organization_id": organization_id,
                    "last_nightly_export_at": last_nightly_export_at,
                }
            )
            .on_conflict_do_update(
                index_elements=["organization_id"],
                set_={
                    "last_nightly_export_at": last_nightly_export_at,
                },
            )
        )

        await self.session.execute(stmt)
        await self.session.flush()
