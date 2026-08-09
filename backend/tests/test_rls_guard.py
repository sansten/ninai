"""Tests for the ORM RLS defense-in-depth loader criteria (rls_guard.py).

Focus: attach_org_filter used to silently continue when with_loader_criteria
failed for a model, leaving that model with NO ORM-level org isolation while
get_org_filter_status() still reported active=True with no way to tell the
coverage was partial. failed_models now surfaces that gap.
"""
from __future__ import annotations

import app.services.rls_guard as rls_guard


class _FakeSession:
    def __init__(self) -> None:
        self.info: dict = {}


class _FakeModel:
    """Hashable stand-in for a SQLAlchemy model class in loader_criteria dicts."""

    def __init__(self, name: str) -> None:
        self.__name__ = name


def test_attach_org_filter_sets_active_and_empty_failed_models(monkeypatch):
    monkeypatch.setattr(
        rls_guard, "_get_tenant_models", lambda: {_FakeModel("Event"): "org_col"}
    )
    monkeypatch.setattr(rls_guard, "with_loader_criteria", lambda *a, **k: "criteria")

    session = _FakeSession()
    rls_guard.attach_org_filter(session, "org-1", "user-1")

    status = rls_guard.get_org_filter_status(session)
    assert status is not None
    assert status["active"] is True
    assert status["failed_models"] == []
    assert status["model_count"] == 1


def test_attach_org_filter_records_failed_models_when_criteria_raises(monkeypatch):
    good_model = _FakeModel("Event")
    bad_model = _FakeModel("BrokenModel")
    monkeypatch.setattr(
        rls_guard,
        "_get_tenant_models",
        lambda: {good_model: "org_col", bad_model: "org_col"},
    )

    def _fake_loader_criteria(model_class, *a, **k):
        if model_class is bad_model:
            raise RuntimeError("boom")
        return "criteria"

    monkeypatch.setattr(rls_guard, "with_loader_criteria", _fake_loader_criteria)

    session = _FakeSession()
    rls_guard.attach_org_filter(session, "org-1", "user-1")

    status = rls_guard.get_org_filter_status(session)
    assert status is not None
    assert status["active"] is True
    # The one model that succeeded is still attached — one bad model must not
    # disable filtering for every other tenant model.
    assert status["model_count"] == 1
    # But the failure is now visible, not silent.
    assert status["failed_models"] == ["BrokenModel"]


def test_attach_org_filter_empty_org_id_uses_deny_all_sentinel(monkeypatch):
    monkeypatch.setattr(
        rls_guard, "_get_tenant_models", lambda: {_FakeModel("Event"): "org_col"}
    )
    monkeypatch.setattr(rls_guard, "with_loader_criteria", lambda *a, **k: "criteria")

    session = _FakeSession()
    rls_guard.attach_org_filter(session, "", "user-1")

    assert session.info["org_id"] == "00000000-0000-0000-0000-000000000000"


def test_get_org_filter_status_returns_none_when_not_attached():
    session = _FakeSession()
    assert rls_guard.get_org_filter_status(session) is None
