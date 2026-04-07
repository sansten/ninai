"""Self-service signup endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.tenant_provisioning_service import TenantProvisioningService

router = APIRouter(prefix="/onboarding", tags=["onboarding"])


class SignupRequest(BaseModel):
    org_name: str
    admin_email: EmailStr
    admin_password: str
    plan: str = "community"
    seats: int = 0
    region: str = "us-central1"


class SignupResponse(BaseModel):
    org_id: str
    admin_user_id: str
    plan: str
    provisioned_at: str


@router.post("/signup", response_model=SignupResponse, status_code=status.HTTP_201_CREATED)
async def signup(body: SignupRequest, db: AsyncSession = Depends(get_db)):
    if len(body.admin_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    if body.plan not in ("community", "enterprise_self", "enterprise_managed"):
        raise HTTPException(status_code=400, detail="Invalid plan")

    svc = TenantProvisioningService(db)
    result = await svc.provision(
        org_name=body.org_name,
        admin_email=body.admin_email,
        admin_password=body.admin_password,
        plan=body.plan,
        seats=body.seats,
        region=body.region,
    )
    await db.commit()

    return SignupResponse(
        org_id=result.org_id,
        admin_user_id=result.admin_user_id,
        plan=body.plan,
        provisioned_at=result.provisioned_at,
    )
