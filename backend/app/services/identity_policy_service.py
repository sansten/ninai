"""
IdentityPolicyService
======================

Applies the three-layer policy to produce the final actor context for a memory write.

Precedence (highest to lowest):
  1. OrgIdentityPolicy.mandate_actor_identity → FULL always
  2. call_anonymous=True flag on the request  → ANONYMOUS (if org permits)
  3. UserIdentityPreference.preference
  4. Default: FULL

The output (ResolvedActorContext) is what actually gets written to the memory row
and the soft-anonymity audit record.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional, TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user_identity_preference import IdentityMode, UserIdentityPreference
from app.models.org_identity_policy import OrgIdentityPolicy

if TYPE_CHECKING:
    from app.services.identity_resolver_service import ResolvedIdentity


logger = logging.getLogger(__name__)


@dataclass
class ResolvedActorContext:
    """What actually gets written to the memory row."""

    actor_id: Optional[str]
    """None in ROLE_ONLY and ANONYMOUS modes."""

    actor_type: Optional[str]
    """None in ANONYMOUS mode."""

    role: Optional[str]
    """None in ANONYMOUS mode."""

    department: Optional[str]
    """From AD/SCIM when available. None in ANONYMOUS mode."""

    display_name: Optional[str]
    """None in ROLE_ONLY and ANONYMOUS modes."""

    mode_applied: str
    """full | role_only | anonymous — the effective mode used."""

    identity_confidence: float
    """Confidence score from IdentityResolverService."""

    mandate_was_active: bool
    """True when the org mandate forced FULL regardless of user preference."""


# Default policy returned when no DB row exists for an org (safe, non-crashing).
# Uses SimpleNamespace to avoid SQLAlchemy ORM instrumentation side-effects.
_DEFAULT_ORG_POLICY = SimpleNamespace(
    mandate_actor_identity=False,
    allowed_modes=["full", "role_only", "anonymous"],
    enrich_from_directory=True,
    audit_trail_always=True,
)


_POLICY_CACHE_TTL = 60  # seconds


async def _redis_get(key: str):
    try:
        from app.core.redis import RedisClient
        rc = await RedisClient.get_client()
        return await rc.get(key)
    except Exception:
        return None


async def _redis_set(key: str, value: str, ttl: int = _POLICY_CACHE_TTL):
    try:
        from app.core.redis import RedisClient
        rc = await RedisClient.get_client()
        await rc.set(key, value, ex=ttl)
    except Exception:
        pass


async def get_org_policy(db: AsyncSession, org_id: str):
    """Load org policy from Redis cache (60 s TTL) then DB; return a safe default when no row exists."""
    import json
    cache_key = f"org_policy:{org_id}"
    cached = await _redis_get(cache_key)
    if cached:
        try:
            return SimpleNamespace(**json.loads(cached))
        except Exception:
            pass

    try:
        row = await db.scalar(
            select(OrgIdentityPolicy).where(OrgIdentityPolicy.org_id == org_id)
        )
        result = row if row is not None else _DEFAULT_ORG_POLICY
        payload = {
            "mandate_actor_identity": bool(result.mandate_actor_identity),
            "allowed_modes": list(result.allowed_modes or []),
            "enrich_from_directory": bool(result.enrich_from_directory),
            "audit_trail_always": bool(result.audit_trail_always),
        }
        await _redis_set(cache_key, json.dumps(payload))
        return result
    except Exception as exc:
        logger.warning("Failed to load OrgIdentityPolicy for org=%s: %s", org_id, exc)
        return _DEFAULT_ORG_POLICY


async def get_user_pref(
    db: AsyncSession, user_id: str
) -> Optional[UserIdentityPreference]:
    """Load user identity preference from Redis cache (60 s TTL) then DB."""
    import json
    cache_key = f"user_pref:{user_id}"
    cached = await _redis_get(cache_key)
    if cached:
        try:
            data = json.loads(cached)
            if data is None:
                return None
            return SimpleNamespace(**data)  # type: ignore[return-value]
        except Exception:
            pass

    try:
        row = await db.scalar(
            select(UserIdentityPreference).where(
                UserIdentityPreference.user_id == user_id
            )
        )
        if row is None:
            await _redis_set(cache_key, "null")
            return None
        payload = {
            "user_id": str(row.user_id),
            "preference": str(row.preference) if row.preference else "full",
        }
        await _redis_set(cache_key, json.dumps(payload))
        return row
    except Exception as exc:
        logger.warning(
            "Failed to load UserIdentityPreference for user=%s: %s", user_id, exc
        )
        return None


class IdentityPolicyService:
    """Applies three-layer policy to produce the final actor context."""

    async def resolve(
        self,
        *,
        resolved_identity: "ResolvedIdentity",
        org_policy: OrgIdentityPolicy,
        user_pref: Optional[UserIdentityPreference],
        call_anonymous: bool,
    ) -> ResolvedActorContext:
        """Determine the effective IdentityMode and build the actor context.

        Args:
            resolved_identity: Output of IdentityResolverService.resolve().
            org_policy: OrgIdentityPolicy for this org (use get_org_policy()).
            user_pref: UserIdentityPreference for this user (may be None).
            call_anonymous: True when the API caller set anonymous=True on the request.

        Returns:
            ResolvedActorContext ready to be written to the memory row.
        """
        mandate_active = org_policy.mandate_actor_identity
        allowed = list(org_policy.allowed_modes or ["full", "role_only", "anonymous"])

        if mandate_active:
            mode = IdentityMode.FULL
        elif call_anonymous and IdentityMode.ANONYMOUS.value in allowed:
            mode = IdentityMode.ANONYMOUS
        elif user_pref and user_pref.preference in allowed:
            mode = IdentityMode(user_pref.preference)
        else:
            mode = IdentityMode.FULL

        return self._build_context(
            resolved_identity, mode, mandate_active
        )

    def _build_context(
        self,
        identity: "ResolvedIdentity",
        mode: IdentityMode,
        mandate_active: bool,
    ) -> ResolvedActorContext:
        if mode == IdentityMode.FULL:
            return ResolvedActorContext(
                actor_id=identity.actor_id,
                actor_type=identity.actor_type,
                role=identity.role,
                department=identity.department,
                display_name=identity.display_name,
                mode_applied="full",
                identity_confidence=identity.confidence,
                mandate_was_active=mandate_active,
            )
        elif mode == IdentityMode.ROLE_ONLY:
            return ResolvedActorContext(
                actor_id=None,
                actor_type=identity.actor_type,
                role=identity.role,
                department=identity.department,
                display_name=None,
                mode_applied="role_only",
                identity_confidence=identity.confidence,
                mandate_was_active=mandate_active,
            )
        else:  # ANONYMOUS
            return ResolvedActorContext(
                actor_id=None,
                actor_type=None,
                role=None,
                department=None,
                display_name=None,
                mode_applied="anonymous",
                identity_confidence=0.0,
                mandate_was_active=mandate_active,
            )
