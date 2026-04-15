"""
IdentityResolverService
========================

Resolves actor identity from auth context (JWT / API key claims),
optionally enriched from an AD/SCIM directory cache.

Resolution order:
  1. JWT claims (actor_id, roles_string) — always authoritative
  2. Redis-cached AD/SCIM record for this (org_id, user_id) — TTL 15 min
  3. Live AD/SCIM lookup if cache miss and org has enrich_from_directory=True
  4. Fallback: derive role from roles_string; no department/location

Never raises. Returns a safe fallback (anonymous) on any failure.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ResolvedIdentity:
    """Fully resolved actor identity for a single request."""

    actor_id: str
    """user_id from JWT (authoritative — never overridden)."""

    actor_type: str
    """employee | bot | anonymous."""

    display_name: str
    """Human-readable name. Defaults to actor_id when AD is unavailable."""

    role: str
    """Primary role key (e.g. employee_engineering, employee_leadership)."""

    department: Optional[str]
    """Department from AD/SCIM. None when AD is skipped or unavailable."""

    location: Optional[str]
    """Office location from AD/SCIM. None unless org enables location enrichment."""

    manager_id: Optional[str]
    """Manager's user_id from AD/SCIM."""

    confidence: float
    """
    1.0 = JWT + live AD record
    0.8 = from Redis cache
    0.5 = JWT fallback (no AD)
    0.0 = anonymous / service account without config
    """


class IdentityResolverService:
    """
    Resolves actor identity from auth context, optionally enriched from AD/SCIM.

    Hard timeout of 200ms on AD lookups — memory writes must never block on AD.
    """

    CACHE_TTL = 900  # 15 minutes

    def __init__(self, redis_client, ad_client=None) -> None:
        self._redis = redis_client
        self._ad = ad_client

    async def resolve(
        self,
        user_id: str,
        org_id: str,
        roles_string: str,
    ) -> ResolvedIdentity:
        """Resolve identity for the current request actor.

        Args:
            user_id: User UUID from JWT / API key service record.
            org_id: Organisation UUID (for cache key scoping).
            roles_string: Comma-separated role names from JWT claims.

        Returns:
            ResolvedIdentity with guaranteed non-None scalar fields.
        """
        if not user_id or user_id == "anonymous":
            return ResolvedIdentity(
                actor_id="anonymous",
                actor_type="anonymous",
                display_name="anonymous",
                role="anonymous",
                department=None,
                location=None,
                manager_id=None,
                confidence=0.0,
            )

        cache_key = f"identity:{org_id}:{user_id}"

        # 1. Check Redis cache
        try:
            cached = await self._redis.get(cache_key)
            if cached:
                data = json.loads(cached)
                return ResolvedIdentity(**data, confidence=0.8)
        except Exception as exc:
            logger.warning("Redis identity cache lookup failed: %s", exc)

        # 2. Try AD/SCIM enrichment
        if self._ad is not None:
            try:
                ad_record = await self._ad.get_user(user_id)
                if ad_record:
                    resolved = ResolvedIdentity(
                        actor_id=user_id,
                        actor_type=self._detect_actor_type(user_id, roles_string, ad_record),
                        display_name=ad_record.get("displayName") or user_id,
                        role=ad_record.get("jobTitle") or self._role_from_jwt(roles_string),
                        department=ad_record.get("department"),
                        location=ad_record.get("officeLocation"),
                        manager_id=ad_record.get("managerId"),
                        confidence=1.0,
                    )
                    try:
                        cache_data = asdict(resolved)
                        # Remove confidence before caching; re-added on cache hit
                        cache_data.pop("confidence", None)
                        await self._redis.setex(
                            cache_key, self.CACHE_TTL, json.dumps(cache_data)
                        )
                    except Exception as exc:
                        logger.warning("Redis identity cache write failed: %s", exc)
                    return resolved
            except Exception as exc:
                logger.warning("AD/SCIM enrichment failed for user_id=%s: %s", user_id, exc)

        # 3. JWT fallback
        return ResolvedIdentity(
            actor_id=user_id,
            actor_type=self._detect_actor_type(user_id, roles_string, None),
            display_name=user_id,
            role=self._role_from_jwt(roles_string),
            department=None,
            location=None,
            manager_id=None,
            confidence=0.5,
        )

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _detect_actor_type(
        user_id: str,
        roles_string: str,
        ad_record: Optional[dict],
    ) -> str:
        """Detect whether this actor is a human employee or a service bot."""
        if ad_record and ad_record.get("servicePrincipalType"):
            return "bot"
        if roles_string and ("bot" in roles_string or "service" in roles_string):
            return "bot"
        return "employee"

    @staticmethod
    def _role_from_jwt(roles_string: str) -> str:
        """Map JWT roles_string to a normalized role key.

        Takes the most specific match from a priority list.
        """
        if not roles_string:
            return "anonymous"
        priority = [
            "system_admin",
            "org_admin",
            "engineering",
            "support",
            "leadership",
            "admin",
        ]
        for r in priority:
            if r in roles_string:
                return f"employee_{r}" if not r.startswith("employee_") else r
        parts = [p.strip() for p in roles_string.split(",") if p.strip()]
        return parts[0] or "employee_general"
