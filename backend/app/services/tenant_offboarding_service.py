"""Tenant offboarding service for GDPR erasure and data export."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.org_subscription import OrgSubscription
from app.models.organization import Organization


@dataclass
class OffboardingReport:
    org_id: str
    memories_deleted: int
    users_anonymized: int
    subscription_canceled: bool
    export_path: str | None
    completed_at: str


class TenantOffboardingService:
    """Implements GDPR Art. 17 deletion with optional pre-deletion export."""

    TABLES_TO_PURGE = [
        "memories",
        "memory_edges",
        "memory_attachments",
        "cognitive_sessions",
        "agent_runs",
        "audit_events",
        "usage_events",
        "webhook_subscriptions",
        "webhook_outbox_events",
        "dpa_acceptances",
        "org_data_residency",
        "org_feature_flags",
    ]

    def __init__(self, session: AsyncSession):
        self.session = session

    async def export_org_data(self, org_id: str, export_dir: str = "/tmp") -> str | None:
        """Export known org-scoped tables to a JSON file before deletion."""
        path = os.path.join(
            export_dir,
            f"ninai_export_{org_id}_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json",
        )
        export: dict = {
            "org_id": org_id,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "tables": {},
        }

        for table in self.TABLES_TO_PURGE:
            try:
                result = await self.session.execute(
                    text(f"SELECT * FROM {table} WHERE organization_id = :org"),
                    {"org": org_id},
                )
                export["tables"][table] = [dict(r._mapping) for r in result.all()]
            except Exception:
                export["tables"][table] = []

        with open(path, "w", encoding="utf-8") as f:
            json.dump(export, f, default=str)
        return path

    async def delete_org_data(self, org_id: str) -> dict[str, int]:
        """Delete org-scoped data and anonymize related users."""
        counts: dict[str, int] = {}

        for table in self.TABLES_TO_PURGE:
            try:
                result = await self.session.execute(
                    text(f"DELETE FROM {table} WHERE organization_id = :org"),
                    {"org": org_id},
                )
                counts[table] = int(result.rowcount or 0)
            except Exception:
                counts[table] = 0

        anonymize_result = await self.session.execute(
            text(
                """
                UPDATE users SET
                    email = concat('deleted_', id, '@deleted.invalid'),
                    full_name = 'Deleted User',
                    hashed_password = 'DELETED',
                    is_active = false
                WHERE id IN (
                    SELECT user_id FROM user_roles WHERE organization_id = :org
                )
                """
            ),
            {"org": org_id},
        )
        counts["users_anonymized"] = int(anonymize_result.rowcount or 0)
        return counts

    async def offboard(
        self,
        org_id: str,
        *,
        export_first: bool = True,
        export_dir: str = "/tmp",
    ) -> OffboardingReport:
        """Offboard org by optional export + purge + metadata updates."""
        export_path = None
        if export_first:
            export_path = await self.export_org_data(org_id, export_dir)

        sub_res = await self.session.execute(
            select(OrgSubscription).where(OrgSubscription.organization_id == org_id)
        )
        sub = sub_res.scalar_one_or_none()
        subscription_canceled = False
        if sub:
            sub.status = "canceled"
            subscription_canceled = True

        counts = await self.delete_org_data(org_id)

        org_res = await self.session.execute(select(Organization).where(Organization.id == org_id))
        org = org_res.scalar_one_or_none()
        if org:
            org.settings = {
                **(org.settings or {}),
                "offboarded": True,
                "offboarded_at": datetime.now(timezone.utc).isoformat(),
            }

        return OffboardingReport(
            org_id=org_id,
            memories_deleted=counts.get("memories", 0),
            users_anonymized=counts.get("users_anonymized", 0),
            subscription_canceled=subscription_canceled,
            export_path=export_path,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
