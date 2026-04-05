from __future__ import annotations

from app.core import local_mode


def test_edge_deployment_enables_local_first(monkeypatch):
    monkeypatch.setattr(local_mode.settings, "DEPLOYMENT_MODE", "edge", raising=False)
    monkeypatch.setattr(local_mode.settings, "LOCAL_FIRST_MODE", False, raising=False)

    assert local_mode.is_edge_deployment() is True
    assert local_mode.is_local_first_mode() is True
    assert local_mode.external_federation_enabled() is False


def test_explicit_local_first_disables_federation(monkeypatch):
    monkeypatch.setattr(local_mode.settings, "DEPLOYMENT_MODE", "cloud", raising=False)
    monkeypatch.setattr(local_mode.settings, "LOCAL_FIRST_MODE", True, raising=False)

    assert local_mode.is_edge_deployment() is False
    assert local_mode.is_local_first_mode() is True
    assert local_mode.external_federation_enabled() is False


def test_cloud_mode_keeps_federation_enabled(monkeypatch):
    monkeypatch.setattr(local_mode.settings, "DEPLOYMENT_MODE", "cloud", raising=False)
    monkeypatch.setattr(local_mode.settings, "LOCAL_FIRST_MODE", False, raising=False)
    monkeypatch.setattr(local_mode.settings, "OLLAMA_BASE_URL", "", raising=False)

    assert local_mode.is_edge_deployment() is False
    assert local_mode.is_local_first_mode() is False
    assert local_mode.external_federation_enabled() is True
    assert local_mode.ollama_base_url_for_runtime() == "http://localhost:11434"


def test_edge_mode_ollama_default(monkeypatch):
    monkeypatch.setattr(local_mode.settings, "DEPLOYMENT_MODE", "edge", raising=False)
    monkeypatch.setattr(local_mode.settings, "LOCAL_FIRST_MODE", False, raising=False)
    monkeypatch.setattr(local_mode.settings, "OLLAMA_BASE_URL", "", raising=False)

    assert local_mode.ollama_base_url_for_runtime() == "http://ollama:11434"
