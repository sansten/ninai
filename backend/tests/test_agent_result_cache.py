import contextlib
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.agents.types import AgentResult
from app.services.agent_runner import AgentRunner, PipelineContext


@contextlib.asynccontextmanager
async def _fake_tenant_session(**kwargs):
    # Minimal fake session; AgentRunner only passes it around.
    yield SimpleNamespace()


class _FakeAgent:
    name = "MetadataExtractionAgent"
    version = "v1"

    def __init__(self, *, run_impl):
        self._run_impl = run_impl

    def validate_outputs(self, result: AgentResult) -> None:
        return

    async def run(self, memory_id: str, context):
        return await self._run_impl(memory_id, context)


@pytest.mark.asyncio
async def test_runner_uses_cache_hit_and_skips_agent_run(monkeypatch):
    import app.services.agent_runner as agent_runner_module

    org_id = str(uuid4())
    user_id = str(uuid4())
    memory_id = str(uuid4())

    runner = AgentRunner(service_user_id=user_id)
    ctx = PipelineContext(org_id=org_id, memory_id=memory_id, initiator_user_id=user_id, storage="long_term")

    async def _run_should_not_be_called(memory_id: str, context):
        raise AssertionError("agent.run should not be called on cache hit")

    # Patch tenant session + agent registry
    monkeypatch.setattr(agent_runner_module, "get_tenant_session", _fake_tenant_session)
    monkeypatch.setattr(agent_runner_module, "get_agent", lambda name: _FakeAgent(run_impl=_run_should_not_be_called) if name == "MetadataExtractionAgent" else None)

    # Patch runner DB-facing internals
    async def _load_memory_inputs(self, session, _ctx):
        return "Need to return order 123", "internal", "personal", None

    async def _load_prior_enrichment(self, session, org_id, memory_id):
        return {}

    async def _get_or_create_run_row(self, **kwargs):
        return SimpleNamespace(status="retry", inputs_hash="", agent_name="MetadataExtractionAgent", agent_version="v1", memory_id=memory_id)

    persisted = {}

    async def _persist_result(self, session, row, result, inputs_hash):
        persisted["result"] = result

    async def _materialize_side_effects(self, **kwargs):
        return

    tool_events: list[dict] = []

    async def _fake_sink(e: dict) -> None:
        tool_events.append(e)

    def _create_tool_event_sink(self, *, session, run_row):
        return _fake_sink

    monkeypatch.setattr(AgentRunner, "_load_memory_inputs", _load_memory_inputs)
    monkeypatch.setattr(AgentRunner, "_load_prior_enrichment", _load_prior_enrichment)
    monkeypatch.setattr(AgentRunner, "_get_or_create_run_row", _get_or_create_run_row)
    monkeypatch.setattr(AgentRunner, "_persist_result", _persist_result)
    monkeypatch.setattr(AgentRunner, "_materialize_side_effects", _materialize_side_effects)
    monkeypatch.setattr(AgentRunner, "_create_tool_event_sink", _create_tool_event_sink)

    # Patch cache service to return a hit
    class _CacheSvc:
        def __init__(self, session):
            self.session = session

        async def get(self, **kwargs):
            return SimpleNamespace(outputs={"entities": ["order"], "confidence": 0.9}, confidence=0.9)

        async def upsert(self, **kwargs):
            raise AssertionError("cache upsert should not be called on cache hit")

    monkeypatch.setattr(agent_runner_module, "AgentResultCacheService", _CacheSvc)

    result = await runner.run_agent(ctx=ctx, agent_name="MetadataExtractionAgent")
    assert result.status == "success"
    assert result.outputs.get("entities") == ["order"]
    assert "cache_hit" in (result.warnings or [])
    assert persisted.get("result") is result

    assert any(e.get("payload", {}).get("tool") == "AgentResultCacheService.get" and e.get("event_type") == "tool_call" for e in tool_events)
    assert any(e.get("payload", {}).get("tool") == "AgentResultCacheService.get" and e.get("event_type") == "tool_result" and e.get("payload", {}).get("cache_hit") is True for e in tool_events)
    assert not any(e.get("payload", {}).get("tool") == "AgentResultCacheService.upsert" for e in tool_events)


@pytest.mark.asyncio
async def test_runner_writes_cache_on_miss_then_hits_for_second_memory(monkeypatch):
    import app.services.agent_runner as agent_runner_module

    org_id = str(uuid4())
    user_id = str(uuid4())
    memory_id1 = str(uuid4())
    memory_id2 = str(uuid4())

    runner = AgentRunner(service_user_id=user_id)

    async def _run(memory_id: str, context):
        calls["count"] += 1
        return AgentResult(
            agent_name="MetadataExtractionAgent",
            agent_version="v1",
            memory_id=memory_id,
            status="success",
            confidence=0.8,
            outputs={"entities": ["refund"], "confidence": 0.8},
            warnings=[],
            errors=[],
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
            trace_id=None,
        )

    monkeypatch.setattr(agent_runner_module, "get_tenant_session", _fake_tenant_session)
    monkeypatch.setattr(agent_runner_module, "get_agent", lambda name: _FakeAgent(run_impl=_run) if name == "MetadataExtractionAgent" else None)

    # Memory inputs depend on ctx.memory_id; both have identical content.
    async def _load_memory_inputs(self, session, _ctx):
        return "Customer called about refund", "internal", "personal", None

    async def _load_prior_enrichment(self, session, org_id, memory_id):
        return {}

    async def _get_or_create_run_row(self, **kwargs):
        return SimpleNamespace(status="retry", inputs_hash="", agent_name="MetadataExtractionAgent", agent_version="v1", memory_id=kwargs.get("memory_id"))

    async def _persist_result(self, session, row, result, inputs_hash):
        return

    async def _materialize_side_effects(self, **kwargs):
        return

    tool_events: list[dict] = []

    async def _fake_sink(e: dict) -> None:
        tool_events.append(e)

    def _create_tool_event_sink(self, *, session, run_row):
        return _fake_sink

    monkeypatch.setattr(AgentRunner, "_load_memory_inputs", _load_memory_inputs)
    monkeypatch.setattr(AgentRunner, "_load_prior_enrichment", _load_prior_enrichment)
    monkeypatch.setattr(AgentRunner, "_get_or_create_run_row", _get_or_create_run_row)
    monkeypatch.setattr(AgentRunner, "_persist_result", _persist_result)
    monkeypatch.setattr(AgentRunner, "_materialize_side_effects", _materialize_side_effects)
    monkeypatch.setattr(AgentRunner, "_create_tool_event_sink", _create_tool_event_sink)

    # Cache implementation: store by key, but the runner computes the key.
    stored = {"hit": False}

    calls = {"count": 0}

    class _CacheSvc:
        def __init__(self, session):
            self.session = session

        async def get(self, **kwargs):
            if stored.get("hit"):
                return SimpleNamespace(outputs={"entities": ["refund"], "confidence": 0.8}, confidence=0.8)
            return None

        async def upsert(self, **kwargs):
            stored["hit"] = True

    monkeypatch.setattr(agent_runner_module, "AgentResultCacheService", _CacheSvc)

    ctx1 = PipelineContext(org_id=org_id, memory_id=memory_id1, initiator_user_id=user_id, storage="long_term")
    ctx2 = PipelineContext(org_id=org_id, memory_id=memory_id2, initiator_user_id=user_id, storage="long_term")

    r1 = await runner.run_agent(ctx=ctx1, agent_name="MetadataExtractionAgent")
    assert r1.status == "success"
    assert calls["count"] == 1
    assert stored["hit"] is True

    assert any(e.get("payload", {}).get("tool") == "AgentResultCacheService.get" and e.get("event_type") == "tool_result" and e.get("payload", {}).get("cache_hit") is False for e in tool_events)
    assert any(e.get("payload", {}).get("tool") == "AgentResultCacheService.upsert" and e.get("event_type") == "tool_call" for e in tool_events)
    assert any(e.get("payload", {}).get("tool") == "AgentResultCacheService.upsert" and e.get("event_type") == "tool_result" and e.get("payload", {}).get("ok") is True for e in tool_events)

    r2 = await runner.run_agent(ctx=ctx2, agent_name="MetadataExtractionAgent")
    assert r2.status == "success"
    assert calls["count"] == 1
    assert "cache_hit" in (r2.warnings or [])

    assert any(e.get("payload", {}).get("tool") == "AgentResultCacheService.get" and e.get("event_type") == "tool_result" and e.get("payload", {}).get("cache_hit") is True for e in tool_events)
