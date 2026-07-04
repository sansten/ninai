"""Tests for optional inbound-webhook HMAC verification on
POST /connectors/inbound/{connector_type}.

Previously the only gate on this endpoint was require_org_admin() JWT/API-key
auth — there was no way to verify the payload actually came from the named
external system, despite the docstring calling it a webhook receiver "from
external systems". _verify_inbound_signature adds an opt-in HMAC check
(NINAI_INBOUND_WEBHOOK_SECRET / _<TYPE>), mirroring the Stripe webhook
pattern already used in billing.py. It's a no-op when unconfigured, so
existing internally-proxied callers are unaffected.
"""
from __future__ import annotations

import hashlib
import hmac

import pytest
from fastapi import HTTPException

from app.api.v1.endpoints.connectors import _verify_inbound_signature


class TestVerifyInboundSignature:
    def test_noop_when_no_secret_configured(self, monkeypatch):
        monkeypatch.delenv("NINAI_INBOUND_WEBHOOK_SECRET", raising=False)
        monkeypatch.delenv("NINAI_INBOUND_WEBHOOK_SECRET_GITHUB", raising=False)
        # Must not raise even with no signature at all — backward compatible.
        _verify_inbound_signature("github", b'{"foo": "bar"}', None)

    def test_rejects_missing_signature_when_generic_secret_configured(self, monkeypatch):
        monkeypatch.setenv("NINAI_INBOUND_WEBHOOK_SECRET", "sekrit")
        with pytest.raises(HTTPException) as exc:
            _verify_inbound_signature("github", b'{"foo": "bar"}', None)
        assert exc.value.status_code == 400

    def test_rejects_bad_signature(self, monkeypatch):
        monkeypatch.setenv("NINAI_INBOUND_WEBHOOK_SECRET", "sekrit")
        with pytest.raises(HTTPException) as exc:
            _verify_inbound_signature("github", b'{"foo": "bar"}', "sha256=deadbeef")
        assert exc.value.status_code == 400

    def test_accepts_valid_signature(self, monkeypatch):
        monkeypatch.setenv("NINAI_INBOUND_WEBHOOK_SECRET", "sekrit")
        body = b'{"foo": "bar"}'
        expected = hmac.new(b"sekrit", body, hashlib.sha256).hexdigest()
        # Must not raise.
        _verify_inbound_signature("github", body, f"sha256={expected}")

    def test_per_connector_secret_takes_precedence_over_generic(self, monkeypatch):
        monkeypatch.setenv("NINAI_INBOUND_WEBHOOK_SECRET", "generic-secret")
        monkeypatch.setenv("NINAI_INBOUND_WEBHOOK_SECRET_GITHUB", "github-only-secret")
        body = b'{"foo": "bar"}'

        # Signed with the generic secret — must be rejected because github
        # has its own more-specific secret configured.
        wrong_sig = hmac.new(b"generic-secret", body, hashlib.sha256).hexdigest()
        with pytest.raises(HTTPException):
            _verify_inbound_signature("github", body, f"sha256={wrong_sig}")

        # Signed with the connector-specific secret — accepted.
        right_sig = hmac.new(b"github-only-secret", body, hashlib.sha256).hexdigest()
        _verify_inbound_signature("github", body, f"sha256={right_sig}")

    def test_connector_type_without_specific_secret_falls_back_to_generic(self, monkeypatch):
        monkeypatch.setenv("NINAI_INBOUND_WEBHOOK_SECRET", "generic-secret")
        monkeypatch.delenv("NINAI_INBOUND_WEBHOOK_SECRET_SLACK", raising=False)
        body = b'{"foo": "bar"}'
        sig = hmac.new(b"generic-secret", body, hashlib.sha256).hexdigest()
        _verify_inbound_signature("slack", body, f"sha256={sig}")
