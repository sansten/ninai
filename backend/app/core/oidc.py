"""OIDC utilities.

Implements Option A (recommended): validate OIDC ID tokens using issuer discovery + JWKS.

This module intentionally keeps things minimal and provider-agnostic.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from jose import jwt


@dataclass
class OidcProviderMetadata:
    issuer: str
    jwks_uri: str


class OidcError(Exception):
    pass


_DISCOVERY_CACHE: dict[str, tuple[float, OidcProviderMetadata]] = {}
_JWKS_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


async def _fetch_json(url: str, timeout: float = 10.0) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()


def _well_known_url(issuer: str) -> str:
    issuer = issuer.rstrip("/")
    return f"{issuer}/.well-known/openid-configuration"


async def get_provider_metadata(issuer: str, cache_ttl_seconds: int = 300) -> OidcProviderMetadata:
    now = time.time()
    cached = _DISCOVERY_CACHE.get(issuer)
    if cached and (now - cached[0]) < cache_ttl_seconds:
        return cached[1]

    data = await _fetch_json(_well_known_url(issuer))
    if "issuer" not in data or "jwks_uri" not in data:
        raise OidcError("Invalid OIDC discovery document")

    meta = OidcProviderMetadata(issuer=data["issuer"], jwks_uri=data["jwks_uri"])
    _DISCOVERY_CACHE[issuer] = (now, meta)
    return meta


async def get_jwks(jwks_uri: str, cache_ttl_seconds: int = 300) -> dict[str, Any]:
    now = time.time()
    cached = _JWKS_CACHE.get(jwks_uri)
    if cached and (now - cached[0]) < cache_ttl_seconds:
        return cached[1]

    jwks = await _fetch_json(jwks_uri)
    if "keys" not in jwks:
        raise OidcError("Invalid JWKS document")

    _JWKS_CACHE[jwks_uri] = (now, jwks)
    return jwks


def _pick_jwk(jwks: dict[str, Any], kid: Optional[str]) -> dict[str, Any]:
    keys = jwks.get("keys") or []
    if not isinstance(keys, list) or not keys:
        raise OidcError("JWKS contains no keys")

    if kid:
        for key in keys:
            if key.get("kid") == kid:
                return key

    # Fallback: if there's a single key, use it.
    if len(keys) == 1:
        return keys[0]

    raise OidcError("Unable to select signing key (kid mismatch)")


def _extract_email(claims: dict[str, Any]) -> Optional[str]:
    for field in ("email", "preferred_username", "upn"):
        value = claims.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return None


def _extract_name(claims: dict[str, Any]) -> Optional[str]:
    for field in ("name", "given_name"):
        value = claims.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


async def verify_id_token(
    *,
    id_token: str,
    issuer: str,
    client_id: str,
    audience: Optional[str] = None,
    leeway_seconds: int = 60,
) -> dict[str, Any]:
    """Verify an OIDC ID token and return claims.

    Raises OidcError on verification errors.
    """
    try:
        header = jwt.get_unverified_header(id_token)
    except Exception as e:  # noqa: BLE001
        raise OidcError("Invalid token header") from e

    kid = header.get("kid")
    alg = header.get("alg")
    if not alg:
        raise OidcError("Missing token alg")

    meta = await get_provider_metadata(issuer)
    jwks = await get_jwks(meta.jwks_uri)
    jwk = _pick_jwk(jwks, kid)

    expected_aud = audience or client_id

    try:
        claims = jwt.decode(
            id_token,
            jwk,
            algorithms=[alg],
            audience=expected_aud,
            issuer=meta.issuer,
            options={"leeway": leeway_seconds},
        )
    except Exception as e:  # noqa: BLE001
        raise OidcError("Token verification failed") from e

    if not isinstance(claims, dict):
        raise OidcError("Invalid token claims")

    return claims


def parse_group_to_role_mapping(raw_json: Optional[str]) -> dict[str, str]:
    if not raw_json:
        return {}
    try:
        data = json.loads(raw_json)
    except Exception as e:  # noqa: BLE001
        raise OidcError("OIDC_GROUP_TO_ROLE_JSON must be valid JSON") from e

    if not isinstance(data, dict):
        raise OidcError("OIDC_GROUP_TO_ROLE_JSON must be an object")

    mapping: dict[str, str] = {}
    for k, v in data.items():
        if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
            mapping[k.strip()] = v.strip()
    return mapping


def extract_email_and_name(claims: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    return _extract_email(claims), _extract_name(claims)
