"""Tenant provisioning service for self-service onboarding."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_password_hash
from app.models.organization import Organization
from app.models.org_subscription import OrgSubscription
from app.models.user import Role, User, UserRole


@dataclass
class ProvisionResult:
    org_id: str
    admin_user_id: str
    subscription_id: str | None
    provisioned_at: str


class TenantProvisioningService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def provision(
        self,
        *,
        org_name: str,
        admin_email: str,
        admin_password: str,
        plan: str = "community",
        seats: int = 0,
        region: str = "us-central1",
        stripe_customer_id: str | None = None,
    ) -> ProvisionResult:
        org_id = str(uuid4())
        user_id = str(uuid4())

        org = Organization(
            id=org_id,
            name=org_name,
            slug=self._slugify(org_name),
            settings={"region": region, "plan": plan},
        )
        self.session.add(org)
        await self.session.flush()

        user = User(
            id=user_id,
            email=admin_email,
            hashed_password=get_password_hash(admin_password),
            full_name="Admin",
            is_active=True,
        )
        self.session.add(user)
        await self.session.flush()

        await self._assign_org_admin_role(user_id, org_id)

        sub_id: str | None = None
        if plan != "community" and stripe_customer_id:
            from app.services.billing_service import BillingService

            billing = BillingService(self.session)
            sub = await billing.create_subscription(
                org_id=org_id,
                stripe_customer_id=stripe_customer_id,
                plan=plan,
                seats=seats,
                email=admin_email,
            )
            sub_id = sub.id

        return ProvisionResult(
            org_id=org_id,
            admin_user_id=user_id,
            subscription_id=sub_id,
            provisioned_at=datetime.now(timezone.utc).isoformat(),
        )

    async def deprovision(self, *, org_id: str, reason: str = "canceled") -> dict:
        res = await self.session.execute(select(Organization).where(Organization.id == org_id))
        org = res.scalar_one_or_none()
        if not org:
            return {"status": "not_found"}

        org.settings = {
            **(org.settings or {}),
            "deprovisioned": True,
            "deprovision_reason": reason,
        }
        return {"status": "deprovisioned", "org_id": org_id}

    @staticmethod
    def _slugify(name: str) -> str:
        return re.sub(r"[^a-z0-9-]", "-", name.lower().strip())[:50]

    async def _assign_org_admin_role(self, user_id: str, org_id: str) -> None:
        role_res = await self.session.execute(
            select(Role).where(Role.organization_id == org_id, Role.name == "org_admin")
        )
        role = role_res.scalar_one_or_none()

        if not role:
            role = Role(
                id=str(uuid4()),
                name="org_admin",
                display_name="Org Admin",
                description="Organization administrator",
                permissions=["*"],
                organization_id=org_id,
                is_system=True,
                is_default=False,
            )
            self.session.add(role)
            await self.session.flush()

        self.session.add(
            UserRole(
                id=str(uuid4()),
                user_id=user_id,
                role_id=role.id,
                organization_id=org_id,
            )
        )
