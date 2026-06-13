from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.middleware.tenant_context import TenantContext
from app.models.base import TenantMixin
from app.models.org_llm_config import OrgLlmConfig, VALID_PROVIDERS
from app.services.llm_router_service import LlmRouterService, LlmCallResult


@pytest.fixture
def tenant_ctx() -> TenantContext:
    return TenantContext(
        user_id="user-1",
        org_id="org-1",
        roles=["org_admin"],
        clearance_level=1,
    )


def _session_with_scalar(value) -> AsyncMock:
    session = AsyncMock()
    session.add = MagicMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    session.execute.return_value = result
    return session


def _mk_config(**kwargs):
    base = {
        "id": "cfg-1",
        "organization_id": "org-1",
        "provider": "vllm",
        "model": "qwen2.5:7b",
        "api_key_ref": None,
        "base_url": None,
        "is_active": True,
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


@pytest.mark.asyncio
async def test_get_config_returns_none_when_missing():
    db = _session_with_scalar(None)
    svc = LlmRouterService(db, "org-1")
    result = await svc.get_config()
    assert result is None


@pytest.mark.asyncio
async def test_get_config_returns_active_config():
    cfg = _mk_config(provider="openai", model="gpt-4")
    db = _session_with_scalar(cfg)
    svc = LlmRouterService(db, "org-1")
    result = await svc.get_config()
    assert result is cfg


@pytest.mark.asyncio
async def test_complete_falls_back_to_vllm_when_no_config():
    db = _session_with_scalar(None)
    svc = LlmRouterService(db, "org-1")
    with patch.object(svc, "_call_vllm", new=AsyncMock(return_value=LlmCallResult(
        provider="vllm", model="qwen2.5:7b", content="test response", tokens_used=10
    ))):
        result = await svc.complete(prompt="test prompt")
    assert result.provider == "vllm"


@pytest.mark.asyncio
async def test_complete_routes_to_openai():
    cfg = _mk_config(provider="openai")
    db = _session_with_scalar(cfg)
    svc = LlmRouterService(db, "org-1")
    with patch.object(svc, "_call_openai", new=AsyncMock(return_value=LlmCallResult(
        provider="openai", model="gpt-4", content="response", tokens_used=50
    ))):
        result = await svc.complete(prompt="test")
    assert result.provider == "openai"


@pytest.mark.asyncio
async def test_complete_routes_to_anthropic():
    cfg = _mk_config(provider="anthropic", model="claude-3")
    db = _session_with_scalar(cfg)
    svc = LlmRouterService(db, "org-1")
    with patch.object(svc, "_call_anthropic", new=AsyncMock(return_value=LlmCallResult(
        provider="anthropic", model="claude-3", content="response", tokens_used=30
    ))):
        result = await svc.complete(prompt="test")
    assert result.provider == "anthropic"


@pytest.mark.asyncio
async def test_call_vllm_handles_timeout():
    svc = LlmRouterService(_session_with_scalar(None), "org-1")
    result = await svc._call_vllm("test-model", "prompt", "system", 512)
    assert result.provider == "vllm"
    assert result.content == ""
    assert result.tokens_used == 0


@pytest.mark.asyncio
async def test_call_openai_returns_no_key_message_when_missing():
    cfg = _mk_config(provider="openai", api_key_ref=None)
    db = _session_with_scalar(cfg)
    svc = LlmRouterService(db, "org-1")
    result = await svc._call_openai(cfg, "prompt", "system", 512)
    assert "[no api key]" in result.content


@pytest.mark.asyncio
async def test_call_anthropic_returns_no_key_message_when_missing():
    cfg = _mk_config(provider="anthropic", api_key_ref=None)
    db = _session_with_scalar(cfg)
    svc = LlmRouterService(db, "org-1")
    result = await svc._call_anthropic(cfg, "prompt", "system", 512)
    assert "[no api key]" in result.content


def test_model_contract():
    assert OrgLlmConfig.__tablename__ == "org_llm_configs"
    assert issubclass(OrgLlmConfig, TenantMixin)
    assert hasattr(OrgLlmConfig, "provider")
    assert hasattr(OrgLlmConfig, "model")
    assert hasattr(OrgLlmConfig, "api_key_ref")
    assert hasattr(OrgLlmConfig, "is_active")


def test_valid_providers():
    assert VALID_PROVIDERS == frozenset({"vllm", "openai", "anthropic"})


def test_router_registration():
    from app.main import app
    paths = {getattr(r, "path", None) for r in app.router.routes}
    assert "/api/v1/admin/llm-config" in paths
