"""Regression tests for the cross-tenant tenant_id trust bug in v2_router.py.

_resolve_tenant used to return a caller-supplied tenant_id unconditionally,
so any authenticated user from org A could pass tenant_id="org-B" on
/v2/interact and read/write org B's memory graph. The fix requires the
tenant_id to match the caller's own org unless NINAI_BENCH_MODE is on
(the LongMemEval/LoCoMo harness's synthetic per-question tenants) or the
caller is a superuser.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.v2.api import v2_router


class _User:
    def __init__(self, organization_id: str, is_superuser: bool = False) -> None:
        self.organization_id = organization_id
        self.is_superuser = is_superuser


class TestResolveTenant:
    def test_no_tenant_falls_back_to_own_org(self):
        user = _User("org-a")
        assert v2_router._resolve_tenant(None, user) == "org-a"

    def test_tenant_matching_own_org_is_allowed(self):
        user = _User("org-a")
        assert v2_router._resolve_tenant("org-a", user) == "org-a"

    def test_cross_tenant_request_is_rejected_by_default(self, monkeypatch):
        monkeypatch.setattr(v2_router, "_BENCH_MODE", False)
        user = _User("org-a")
        with pytest.raises(HTTPException) as exc_info:
            v2_router._resolve_tenant("org-b", user)
        assert exc_info.value.status_code == 403

    def test_cross_tenant_request_allowed_in_bench_mode(self, monkeypatch):
        monkeypatch.setattr(v2_router, "_BENCH_MODE", True)
        user = _User("org-a")
        assert v2_router._resolve_tenant("org-b", user) == "org-b"

    def test_cross_tenant_request_allowed_for_superuser(self, monkeypatch):
        monkeypatch.setattr(v2_router, "_BENCH_MODE", False)
        user = _User("org-a", is_superuser=True)
        assert v2_router._resolve_tenant("org-b", user) == "org-b"

    def test_dict_current_user_cross_tenant_rejected(self, monkeypatch):
        monkeypatch.setattr(v2_router, "_BENCH_MODE", False)
        with pytest.raises(HTTPException) as exc_info:
            v2_router._resolve_tenant("org-b", {"org_id": "org-a"})
        assert exc_info.value.status_code == 403

    def test_dict_current_user_matching_org_allowed(self, monkeypatch):
        monkeypatch.setattr(v2_router, "_BENCH_MODE", False)
        assert v2_router._resolve_tenant("org-a", {"org_id": "org-a"}) == "org-a"


class TestAuthFailsClosed:
    def test_auth_dependency_is_configured(self):
        """The import-failure fallback used to leave _AUTH = None, which made
        current_user=_AUTH default to no dependency at all (unauthenticated).
        It must always be a real Depends() object."""
        from fastapi.params import Depends as DependsMarker

        assert isinstance(v2_router._AUTH, DependsMarker)
