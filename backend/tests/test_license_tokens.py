import pytest

import base64
import json

from app.core.license_token import (
    LicenseTokenError,
    generate_keypair_pem,
    sign_license_token,
    verify_license_token,
)


def _payload(org_id: str, features: list[str], iat: int, exp: int, **extra):
    p = {"org_id": org_id, "features": features, "iat": iat, "exp": exp}
    p.update(extra)
    return p


def test_verify_license_token_rejects_tampering():
    priv_pem, pub_pem = generate_keypair_pem()

    token = sign_license_token(
        private_key_pem=priv_pem,
        payload=_payload(
            org_id="org-123",
            features=["enterprise.autoevalbench"],
            iat=1_700_000_000,
            exp=1_900_000_000,
        ),
    )

    # Tamper with the payload (but keep it valid JSON) without resigning.
    prefix, payload_b64, sig_b64 = token.split(".")

    padding = "=" * ((4 - (len(payload_b64) % 4)) % 4)
    payload_json = base64.urlsafe_b64decode((payload_b64 + padding).encode("ascii")).decode("utf-8")
    payload = json.loads(payload_json)
    payload["org_id"] = "org-CHANGED"  # modification should break signature verification

    new_payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    new_payload_b64 = base64.urlsafe_b64encode(new_payload_json).decode("ascii").rstrip("=")
    tampered = f"{prefix}.{new_payload_b64}.{sig_b64}"

    with pytest.raises(LicenseTokenError, match="signature"):
        verify_license_token(token=tampered, public_key_pem=pub_pem, now_ts=1_800_000_000)


def test_verify_license_token_expiry_and_nbf():
    priv_pem, pub_pem = generate_keypair_pem()

    token = sign_license_token(
        private_key_pem=priv_pem,
        payload=_payload(
            org_id="org-123",
            features=["enterprise.drift_detection"],
            iat=100,
            exp=200,
            nbf=150,
        ),
    )

    # Too early (before nbf, even with skew=0)
    with pytest.raises(LicenseTokenError, match="not yet valid"):
        verify_license_token(token=token, public_key_pem=pub_pem, now_ts=149, clock_skew_seconds=0)

    # Valid window
    claims = verify_license_token(token=token, public_key_pem=pub_pem, now_ts=160, clock_skew_seconds=0)
    assert claims.org_id == "org-123"
    assert "enterprise.drift_detection" in claims.features

    # Expired
    with pytest.raises(LicenseTokenError, match="expired"):
        verify_license_token(token=token, public_key_pem=pub_pem, now_ts=201, clock_skew_seconds=0)
