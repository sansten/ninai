import pytest
import httpx

from app.agents.llm.vllm import vLLMClient
from app.agents.llm.vllm_breaker import create_llm_client


def test_create_llm_client_keeps_cpu_fallback_when_primary_is_gpu(monkeypatch):
    monkeypatch.setattr("app.agents.llm.vllm_breaker.settings.VLLM_BASE_URL_CPU", "http://vllm-cpu:11434")
    monkeypatch.setattr("app.agents.llm.vllm_breaker.settings.VLLM_BASE_URL_GPU", "http://vllm-gpu:11434")
    monkeypatch.setattr("app.agents.llm.vllm_breaker.settings.VLLM_OVERFLOW_ENABLED", False)
    monkeypatch.setattr("app.agents.llm.vllm_breaker.settings.VLLM_OVERFLOW_PRIMARY_MAX_INFLIGHT", 0)

    client = create_llm_client(
        base_url="http://vllm-gpu:11434",
        model="qwen2.5:7b",
        use_circuit_breaker=False,
    )

    assert client._base_url == "http://vllm-gpu:11434"
    assert client._secondary_base_url == "http://vllm-cpu:11434"
    assert client._request_endpoints() == ["http://vllm-gpu:11434", "http://vllm-cpu:11434"]


def test_create_llm_client_keeps_gpu_fallback_when_primary_is_cpu(monkeypatch):
    monkeypatch.setattr("app.agents.llm.vllm_breaker.settings.VLLM_BASE_URL_CPU", "http://vllm-cpu:11434")
    monkeypatch.setattr("app.agents.llm.vllm_breaker.settings.VLLM_BASE_URL_GPU", "http://vllm-gpu:11434")
    monkeypatch.setattr("app.agents.llm.vllm_breaker.settings.VLLM_OVERFLOW_ENABLED", True)
    monkeypatch.setattr("app.agents.llm.vllm_breaker.settings.VLLM_OVERFLOW_PRIMARY_MAX_INFLIGHT", 3)

    client = create_llm_client(
        base_url="http://vllm-cpu:11434",
        model="qwen2.5:7b",
        use_circuit_breaker=False,
    )

    assert client._base_url == "http://vllm-cpu:11434"
    assert client._secondary_base_url == "http://vllm-gpu:11434"


@pytest.mark.asyncio
async def test_vllm_client_prefers_installed_fallback_model(monkeypatch):
    client = vLLMClient(
        base_url="http://vllm-cpu:11434",
        model="qwen2.5:14b",
        fallback_models=["qwen2.5:7b", "qwen2.5:0.5b"],
    )

    async def _available_models(endpoint, headers):
        return ["qwen2.5:7b", "gemma4:e4b"]

    monkeypatch.setattr(client, "_available_models", _available_models)

    resolved = await client._resolve_model_for_endpoint("http://vllm-cpu:11434", {})
    assert resolved == "qwen2.5:7b"


@pytest.mark.asyncio
async def test_vllm_client_uses_first_available_model_when_requested_missing(monkeypatch):
    client = vLLMClient(
        base_url="http://vllm-cpu:11434",
        model="qwen2.5:14b",
        fallback_models=["qwen2.5:7b"],
    )

    async def _available_models(endpoint, headers):
        return ["gemma4:e4b"]

    monkeypatch.setattr(client, "_available_models", _available_models)

    resolved = await client._resolve_model_for_endpoint("http://vllm-cpu:11434", {})
    assert resolved == "gemma4:e4b"


@pytest.mark.asyncio
async def test_vllm_client_falls_back_to_smaller_model_after_timeout(monkeypatch):
    client = vLLMClient(
        base_url="http://vllm-cpu:11434",
        model="qwen2.5:32b",
        fallback_models=["qwen2.5:7b"],
        timeout_seconds=30.0,
    )

    async def _available_models(endpoint, headers):
        return ["qwen2.5:32b", "qwen2.5:7b"]

    attempts: list[str] = []

    async def _generate_once(*, endpoint, payload, headers, sem):
        attempts.append(str(payload.get("model")))
        if payload.get("model") == "qwen2.5:32b":
            raise httpx.ReadTimeout("timed out")
        return {"response": "2023"}

    monkeypatch.setattr(client, "_available_models", _available_models)
    monkeypatch.setattr(client, "_generate_once", _generate_once)

    result = await client.complete_text(prompt="What year is it?", num_ctx=1024)

    assert result == "2023"
    assert attempts == ["qwen2.5:32b", "qwen2.5:32b", "qwen2.5:7b"]
    assert client.last_model_used == "qwen2.5:7b"


@pytest.mark.asyncio
async def test_vllm_client_falls_back_to_secondary_endpoint_after_primary_timeout(monkeypatch):
    client = vLLMClient(
        base_url="http://vllm-cpu:11434",
        secondary_base_url="http://vllm-gpu:11434",
        model="qwen2.5:7b",
        timeout_seconds=30.0,
    )

    async def _available_models(endpoint, headers):
        if endpoint == "http://vllm-cpu:11434":
            return ["qwen2.5:7b"]
        return ["qwen2.5:14b", "qwen2.5:7b"]

    attempts: list[tuple[str, str]] = []

    async def _generate_once(*, endpoint, payload, headers, sem):
        attempts.append((endpoint, str(payload.get("model"))))
        if endpoint == "http://vllm-cpu:11434":
            raise httpx.ReadTimeout("timed out")
        return {"response": "Stockholm"}

    monkeypatch.setattr(client, "_available_models", _available_models)
    monkeypatch.setattr(client, "_generate_once", _generate_once)

    result = await client.complete_text(prompt="Where did they move?", num_ctx=1024)

    assert result == "Stockholm"
    assert attempts == [
        ("http://vllm-cpu:11434", "qwen2.5:7b"),
        ("http://vllm-cpu:11434", "qwen2.5:7b"),
        ("http://vllm-gpu:11434", "qwen2.5:7b"),
    ]
    assert client.last_endpoint_used == "http://vllm-gpu:11434"
