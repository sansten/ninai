"""Signed license tokens (tamper-evident) for Enterprise entitlements.

Important:
- This makes the *token itself* tamper-proof via Ed25519 signature verification.
- No purely local check can be perfectly "tamper-proof" against an operator who can
  modify server code/binaries; enforcement is therefore *tamper-resistant*.

Token format (compact, offline-friendly):

  ninai1.<base64url(payload_json)>.<base64url(ed25519_signature)>

The signature covers the literal bytes: b"ninai1." + base64url(payload_json).

The payload is JSON with at least:
- org_id: str
- features: list[str]
- iat: int (unix seconds)
- exp: int (unix seconds)
Optionally:
- nbf: int (unix seconds)
- license_id, plan, seats, etc.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
    PrivateFormat,
    NoEncryption,
    load_pem_private_key,
    load_pem_public_key,
)


TOKEN_PREFIX = "ninai1"


class LicenseTokenError(ValueError):
    pass


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * ((4 - (len(data) % 4)) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("ascii"))


def _now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _signing_input(prefix: str, payload_b64: str) -> bytes:
    return (prefix + "." + payload_b64).encode("ascii")


@dataclass(frozen=True)
class LicenseClaims:
    org_id: str
    features: tuple[str, ...]
    iat: int
    exp: int
    nbf: int | None = None
    license_id: str | None = None
    plan: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "LicenseClaims":
        org_id = str(data.get("org_id") or "")
        features_raw = data.get("features") or []
        if not isinstance(features_raw, list) or not all(isinstance(x, str) for x in features_raw):
            raise LicenseTokenError("Invalid 'features' in license payload")

        def _as_int(name: str) -> int:
            v = data.get(name)
            if v is None:
                raise LicenseTokenError(f"Missing '{name}' in license payload")
            try:
                return int(v)
            except Exception as e:
                raise LicenseTokenError(f"Invalid '{name}' in license payload") from e

        iat = _as_int("iat")
        exp = _as_int("exp")
        nbf = data.get("nbf")
        nbf_int = int(nbf) if nbf is not None else None

        if not org_id:
            raise LicenseTokenError("Missing 'org_id' in license payload")
        if exp <= 0 or iat <= 0:
            raise LicenseTokenError("Invalid 'iat'/'exp' in license payload")

        return cls(
            org_id=org_id,
            features=tuple(sorted(set(features_raw))),
            iat=iat,
            exp=exp,
            nbf=nbf_int,
            license_id=(str(data.get("license_id")) if data.get("license_id") is not None else None),
            plan=(str(data.get("plan")) if data.get("plan") is not None else None),
        )


def generate_keypair_pem() -> tuple[bytes, bytes]:
    """Generate a new Ed25519 keypair.

    Intended for tests and internal tooling. Production license keys should be
    generated and stored securely outside the repo.
    """

    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()

    priv_pem = priv.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
    pub_pem = pub.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    return priv_pem, pub_pem


def sign_license_token(*, private_key_pem: bytes, payload: Mapping[str, Any], prefix: str = TOKEN_PREFIX) -> str:
    """Create a signed license token.

    Note: This does not enforce any product policy; it only signs the payload.
    """

    priv = load_pem_private_key(private_key_pem, password=None)
    if not isinstance(priv, Ed25519PrivateKey):
        raise LicenseTokenError("private_key_pem is not an Ed25519 key")

    payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload_b64 = _b64url_encode(payload_json)

    signature = priv.sign(_signing_input(prefix, payload_b64))
    sig_b64 = _b64url_encode(signature)

    return f"{prefix}.{payload_b64}.{sig_b64}"


def verify_license_token(
    *,
    token: str,
    public_key_pem: bytes,
    now_ts: int | None = None,
    clock_skew_seconds: int = 300,
    prefix: str = TOKEN_PREFIX,
) -> LicenseClaims:
    """Verify a license token signature and validate time-based claims.

    Raises LicenseTokenError on any validation failure.
    """

    if now_ts is None:
        now_ts = _now_ts()

    parts = token.split(".")
    if len(parts) != 3:
        raise LicenseTokenError("Invalid token format")

    token_prefix, payload_b64, sig_b64 = parts
    if token_prefix != prefix:
        raise LicenseTokenError("Invalid token prefix")

    try:
        payload_json = _b64url_decode(payload_b64)
        payload = json.loads(payload_json.decode("utf-8"))
    except Exception as e:
        raise LicenseTokenError("Invalid token payload") from e

    try:
        signature = _b64url_decode(sig_b64)
    except Exception as e:
        raise LicenseTokenError("Invalid token signature encoding") from e

    pub = load_pem_public_key(public_key_pem)
    if not isinstance(pub, Ed25519PublicKey):
        raise LicenseTokenError("public_key_pem is not an Ed25519 key")

    try:
        pub.verify(signature, _signing_input(prefix, payload_b64))
    except Exception as e:
        raise LicenseTokenError("Invalid token signature") from e

    claims = LicenseClaims.from_mapping(payload)

    # Time validation with skew.
    if claims.nbf is not None and now_ts + clock_skew_seconds < claims.nbf:
        raise LicenseTokenError("Token not yet valid")

    if now_ts - clock_skew_seconds > claims.exp:
        raise LicenseTokenError("Token expired")

    return claims
