"""API key service.

Creates and validates org-scoped API keys.
"""

from __future__ import annotations

import secrets
from datetime import datetime

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_password_hash, verify_password
from app.models.api_key import ApiKey
from app.models.user import Role, User, UserRole


class ApiKeyService:
    PREFIX_LEN = 12

    @classmethod
    def generate_key(cls) -> str:
        token = secrets.token_urlsafe(32)
        return f"nk_{token}"

    @classmethod
    async def create_api_key(
        cls,
        *,
        session: AsyncSession,
        organization_id: str,
        user_id: str,
        name: str,
    ) -> tuple[ApiKey, str]:
        raw = cls.generate_key()
        prefix = raw[: cls.PREFIX_LEN]
        key_hash = get_password_hash(raw)

        api_key = ApiKey(
            organization_id=organization_id,
            user_id=user_id,
            name=name,
            prefix=prefix,
            key_hash=key_hash,
        )
        session.add(api_key)
        await session.flush()
        return api_key, raw

    @classmethod
    async def authenticate_api_key(
        cls,
        session: AsyncSession,
        presented_key: str,
    ) -> tuple[str, str, list[str], int]:
        if not presented_key or len(presented_key) < cls.PREFIX_LEN:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

        prefix = presented_key[: cls.PREFIX_LEN]
        res = await session.execute(
            select(ApiKey).where(ApiKey.prefix == prefix, ApiKey.revoked_at.is_(None))
        )
        candidates = list(res.scalars().all())

        match: ApiKey | None = None
        for candidate in candidates:
            if verify_password(presented_key, candidate.key_hash):
                match = candidate
                break

        if not match:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

        user_res = await session.execute(select(User).where(User.id == match.user_id))
        user = user_res.scalar_one_or_none()
        if not user or not user.is_active:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive")

        now = datetime.utcnow()
        roles_res = await session.execute(
            select(Role.name)
            .select_from(UserRole)
            .join(Role, Role.id == UserRole.role_id)
            .where(
                UserRole.user_id == match.user_id,
                UserRole.organization_id == match.organization_id,
                (UserRole.expires_at.is_(None) | (UserRole.expires_at > now)),
            )
        )
        role_names = [r for (r,) in roles_res.all()]

        match.last_used_at = datetime.utcnow()
        await session.flush()

        return match.user_id, match.organization_id, role_names, int(user.clearance_level or 0)
