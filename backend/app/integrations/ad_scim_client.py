"""
AD/SCIM Client Stub
====================

Thin async client for Azure AD (Microsoft Graph) or SCIM 2.0 endpoint.

Configure via env vars:
  AD_SCIM_ENDPOINT   — e.g. https://graph.microsoft.com/v1.0 or https://your-idp/scim/v2
  AD_SCIM_TOKEN      — Bearer token or client_credentials token
  AD_SCIM_TENANT_ID  — Azure AD tenant ID (Graph only)

Returns dict with keys: displayName, jobTitle, department, officeLocation,
managerId, servicePrincipalType.
Returns None if user not found or integration not configured.

Integration is fully opt-in: when AD_SCIM_ENDPOINT is not set in config,
IdentityResolverService skips the AD call and falls back to JWT claims.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ADSCIMClient:
    """
    Async client for Azure AD (Microsoft Graph) or SCIM 2.0.

    This is a skeleton implementation. Wire up `get_user()` to your identity
    provider. The IdentityResolverService calls this with a hard 200ms timeout
    and falls back to JWT claims on any failure.
    """

    def __init__(self, endpoint: str, token: str) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._token = token

    async def get_user(self, user_id: str) -> Optional[dict]:
        """
        Fetch a user record from the directory.

        Returns a dict with these keys (all optional except None check):
          displayName       — Human-readable name
          jobTitle          — Job title / role text
          department        — Department name
          officeLocation    — Physical or virtual office location
          managerId         — Manager's user_id
          servicePrincipalType — Non-null for service accounts / bots

        Returns None if user not found or the request fails.
        """
        import asyncio
        try:
            import httpx
        except ImportError:
            logger.warning("httpx not installed; AD/SCIM enrichment disabled")
            return None

        url = f"{self._endpoint}/users/{user_id}"
        headers = {"Authorization": f"Bearer {self._token}"}

        try:
            async with httpx.AsyncClient(timeout=0.2) as client:
                response = await client.get(url, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "displayName": data.get("displayName"),
                        "jobTitle": data.get("jobTitle"),
                        "department": data.get("department"),
                        "officeLocation": data.get("officeLocation"),
                        "managerId": (data.get("manager", {}) or {}).get("id"),
                        "servicePrincipalType": data.get("servicePrincipalType"),
                    }
                if response.status_code == 404:
                    return None
                logger.warning(
                    "AD/SCIM returned status %d for user_id=%s",
                    response.status_code,
                    user_id,
                )
                return None
        except Exception as exc:
            logger.warning("AD/SCIM lookup failed for user_id=%s: %s", user_id, exc)
            return None
