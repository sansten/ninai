"""
Public self-service signup flow — no authentication required.

POST /api/v1/signup                 Create org + first admin user, send verification email
GET  /api/v1/signup/verify?token=   Verify email, activate org, return JWT
POST /api/v1/signup/resend          Resend verification email (always 202 to prevent enumeration)
"""

from __future__ import annotations

import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, EmailStr, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.core.security import (
    create_access_token,
    create_refresh_token,
    get_password_hash,
)
from app.models.org_subscription import OrgSubscription
from app.models.organization import Organization
from app.models.user import Role, User, UserRole
from app.services.email_service import email_service

router = APIRouter(tags=["Signup"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    org_name: str
    org_slug: Optional[str] = None
    ref: Optional[str] = None

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 10:
            raise ValueError("Password must be at least 10 characters")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        return v


class SignupResponse(BaseModel):
    message: str
    org_id: str


class VerifyResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    org_id: str
    user_id: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:50]


async def _slug_taken(db: AsyncSession, slug: str) -> bool:
    r = await db.execute(select(Organization.id).where(Organization.slug == slug))
    return r.scalar_one_or_none() is not None


async def _email_taken(db: AsyncSession, email: str) -> bool:
    r = await db.execute(select(User.id).where(User.email == email))
    return r.scalar_one_or_none() is not None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/signup", response_model=SignupResponse, status_code=201)
async def signup(body: SignupRequest, db: AsyncSession = Depends(get_db)):
    """Create a new tenant workspace and its first admin user."""
    if await _email_taken(db, body.email):
        raise HTTPException(409, "An account with that email already exists")

    slug = body.org_slug or _make_slug(body.org_name)
    if await _slug_taken(db, slug):
        slug = f"{slug}-{secrets.token_hex(3)}"

    # Org starts as pending until email verified
    org = Organization(
        name=body.org_name,
        slug=slug,
        status="pending_verification",
        signup_ref=body.ref,
        settings={},
    )
    db.add(org)
    await db.flush()  # generates org.id

    # Create trial subscription
    trial_ends = datetime.now(timezone.utc) + timedelta(days=settings.TRIAL_DAYS)
    sub = OrgSubscription(
        organization_id=org.id,
        plan="trial",
        status="trialing",
        trial_ends_at=trial_ends,
        seat_limit=settings.TRIAL_MAX_USERS,
        seat_count=1,
    )
    db.add(sub)

    # Create the first user as org_admin
    user = User(
        email=body.email,
        hashed_password=get_password_hash(body.password),
        full_name=body.full_name,
        role="org_admin",
        is_active=True,
    )
    db.add(user)
    await db.flush()  # generates user.id

    # Assign the system org_admin Role (if it exists)
    role_result = await db.execute(
        select(Role).where(Role.name == "org_admin", Role.is_system.is_(True))
    )
    role_obj = role_result.scalar_one_or_none()
    if role_obj:
        db.add(UserRole(user_id=user.id, role_id=role_obj.id, organization_id=org.id))

    # Store verification token inside org.settings (no extra table needed)
    token = secrets.token_urlsafe(48)
    expires = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
    org.settings = {
        **org.settings,
        "_vt": token,       # verification token
        "_vt_exp": expires,  # expiry ISO string
        "_vt_uid": user.id,  # user to activate
    }

    await db.commit()

    # Send email (fire-and-forget — failure logged but not raised)
    await email_service.send_verification(body.email, token, body.org_name)

    return SignupResponse(
        message="Verification email sent. Please check your inbox.", org_id=org.id
    )


@router.get("/signup/verify", response_model=VerifyResponse)
async def verify_email(token: str = Query(...), db: AsyncSession = Depends(get_db)):
    """Verify email address and activate the workspace."""
    # Find orgs with matching token (no index — low volume, JSONB scan is fine)
    result = await db.execute(
        select(Organization).where(Organization.status == "pending_verification")
    )
    orgs = result.scalars().all()

    org = next((o for o in orgs if o.settings.get("_vt") == token), None)
    if not org:
        raise HTTPException(400, "Invalid or expired verification token")

    exp_str = org.settings.get("_vt_exp")
    if exp_str and datetime.now(timezone.utc) > datetime.fromisoformat(exp_str):
        raise HTTPException(400, "Verification token has expired. Request a new one.")

    user_id = org.settings.get("_vt_uid")
    if not user_id:
        raise HTTPException(500, "Verification state corrupted — contact support")

    # Activate org and scrub token from settings
    org.status = "active"
    cleaned = {k: v for k, v in org.settings.items() if not k.startswith("_vt")}
    org.settings = cleaned

    await db.commit()

    user_result = await db.execute(select(User).where(User.id == user_id))
    user = user_result.scalar_one_or_none()
    if not user:
        raise HTTPException(500, "User record missing after verification")

    await email_service.send_welcome(user.email, org.name)

    access = create_access_token(user_id=user.id, org_id=org.id, roles=["org_admin"])
    refresh = create_refresh_token(user_id=user.id, org_id=org.id, roles=["org_admin"])

    return VerifyResponse(
        access_token=access, refresh_token=refresh, org_id=org.id, user_id=user.id
    )


@router.post("/signup/resend", status_code=202)
async def resend_verification(email: EmailStr, db: AsyncSession = Depends(get_db)):
    """Resend verification email. Always 202 to prevent email enumeration."""
    user_r = await db.execute(select(User).where(User.email == email))
    user = user_r.scalar_one_or_none()
    if not user:
        return {"message": "If that email exists, a new verification link was sent"}

    org_r = await db.execute(
        select(Organization).where(
            Organization.status == "pending_verification",
            Organization.settings["_vt_uid"].astext == user.id,
        )
    )
    org = org_r.scalar_one_or_none()
    if org:
        token = org.settings.get("_vt") or secrets.token_urlsafe(48)
        await email_service.send_verification(email, token, org.name)

    return {"message": "If that email exists, a new verification link was sent"}
