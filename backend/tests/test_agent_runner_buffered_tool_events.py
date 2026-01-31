from __future__ import annotations

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

    def validate_outputs(self, result: AgentResult) -> None:
        return

    async def run(self, memory_id: str, context):
        now = datetime(2026, 1, 23, tzinfo=timezone.utc)
        return AgentResult(
            agent_name=self.name,
            agent_version=self.version,
            memory_id=memory_id,
            status="success",
            confidence=0.5,
            outputs={"ok": True},
            warnings=[],
            errors=[],
            started_at=now,
            finished_at=now,
            trace_id=context.get("runtime", {}).get("job_id") or None,
        )


@pytest.mark.asyncio
async def test_buffered_tool_events_flushed_into_run_sink(monkeypatch):
    import app.services.agent_runner as agent_runner_module

    org_id = str(uuid4())
    user_id = str(uuid4())
    memory_id = str(uuid4())

    runner = AgentRunner(service_user_id=user_id)

    # Patch tenant session + agent registry
    monkeypatch.setattr(agent_runner_module, "get_tenant_session", _fake_tenant_session)
    monkeypatch.setattr(agent_runner_module, "get_agent", lambda name: _FakeAgent() if name == "MetadataExtractionAgent" else None)

    # Use short-term path so we can instrument ShortTermMemoryService.get
    class _FakeSTM:
        def __init__(self, user_id: str, org_id: str):
            self.user_id = user_id
            self.org_id = org_id

        async def get(self, _memory_id: str):
            return SimpleNamespace(content="hello", scope="personal")

    monkeypatch.setattr(agent_runner_module, "ShortTermMemoryService", _FakeSTM)

    # Patch DB-facing internals
    async def _load_prior_enrichment(self, session, org_id, memory_id):
        return {}

    async def _get_or_create_run_row(self, **kwargs):
        now = datetime(2026, 1, 23, tzinfo=timezone.utc)
        return SimpleNamespace(
            id="ar1",
            organization_id=kwargs.get("org_id"),
            memory_id=kwargs.get("memory_id"),
            agent_name=kwargs.get("agent_name"),
            agent_version=kwargs.get("agent_version"),
            inputs_hash=kwargs.get("inputs_hash"),
            status="retry",
            confidence=0.0,
            outputs={},
            warnings=[],
            errors=[],
            started_at=now,
            finished_at=now,
            trace_id=kwargs.get("trace_id"),
        )

    async def _persist_result(self, session, row, result, inputs_hash):
        return

    async def _materialize_side_effects(self, **kwargs):
        return

    events: list[dict] = []

    async def _sink(e: dict) -> None:
        events.append(e)

    def _create_tool_event_sink(self, *, session, run_row):
        return _sink

    monkeypatch.setattr(AgentRunner, "_load_prior_enrichment", _load_prior_enrichment)
    monkeypatch.setattr(AgentRunner, "_get_or_create_run_row", _get_or_create_run_row)
    monkeypatch.setattr(AgentRunner, "_persist_result", _persist_result)
    monkeypatch.setattr(AgentRunner, "_materialize_side_effects", _materialize_side_effects)
    monkeypatch.setattr(AgentRunner, "_create_tool_event_sink", _create_tool_event_sink)

    ctx = PipelineContext(org_id=org_id, memory_id=memory_id, initiator_user_id=user_id, trace_id="t1", storage="short_term")

    result = await runner.run_agent(ctx=ctx, agent_name="MetadataExtractionAgent")
    assert result.status == "success"

    assert any(e.get("payload", {}).get("tool") == "get_tenant_session" and e.get("event_type") == "tool_call" for e in events)
    assert any(
        e.get("payload", {}).get("tool") == "get_tenant_session"
        and e.get("event_type") == "tool_result"
        and e.get("payload", {}).get("ok") is True
        for e in events
    )
    assert any(
        e.get("payload", {}).get("tool") == "compute_inputs_hash"
        and e.get("event_type") == "tool_call"
        and isinstance(e.get("payload", {}).get("inputs_hash_len"), int)
        for e in events
    )

    assert any(e.get("payload", {}).get("tool") == "ShortTermMemoryService.get" and e.get("event_type") == "tool_call" for e in events)
    assert any(
        e.get("payload", {}).get("tool") == "ShortTermMemoryService.get"
        and e.get("event_type") == "tool_result"
        and e.get("payload", {}).get("ok") is True
        and e.get("payload", {}).get("found") is True
        for e in events
    )


class _FakeCacheSvc:
    def __init__(self, session):
        self.session = session

    async def get(self, **kwargs):
        return None

    async def upsert(self, **kwargs):
        return None


@pytest.mark.asyncio
async def test_internal_cache_key_computation_emits_lengths(monkeypatch):
    import app.services.agent_runner as agent_runner_module

    org_id = str(uuid4())
    user_id = str(uuid4())
    memory_id = str(uuid4())

    runner = AgentRunner(service_user_id=user_id)

    monkeypatch.setattr(agent_runner_module, "get_tenant_session", _fake_tenant_session)
    monkeypatch.setattr(agent_runner_module, "get_agent", lambda name: _FakeAgent() if name == "MetadataExtractionAgent" else None)
    monkeypatch.setattr(agent_runner_module, "AgentResultCacheService", _FakeCacheSvc)

    monkeypatch.setattr(AgentRunner, "_cache_enabled", lambda self: True)
    monkeypatch.setattr(AgentRunner, "_cache_strategy", lambda self, _agent_name: "llm")
    monkeypatch.setattr(AgentRunner, "_cache_model", lambda self: "test-model")
    monkeypatch.setattr(AgentRunner, "_should_cache_agent", lambda self, _agent_name: True)

    class _FakeSTM:
        def __init__(self, user_id: str, org_id: str):
            self.user_id = user_id
            self.org_id = org_id

        async def get(self, _memory_id: str):
            return SimpleNamespace(content="hello", scope="personal")

    monkeypatch.setattr(agent_runner_module, "ShortTermMemoryService", _FakeSTM)

    async def _load_prior_enrichment(self, session, org_id, memory_id):
        return {}

    async def _get_or_create_run_row(self, **kwargs):
        now = datetime(2026, 1, 23, tzinfo=timezone.utc)
        return SimpleNamespace(
            id="ar1",
            organization_id=kwargs.get("org_id"),
            memory_id=kwargs.get("memory_id"),
            agent_name=kwargs.get("agent_name"),
            agent_version=kwargs.get("agent_version"),
            inputs_hash=kwargs.get("inputs_hash"),
            status="retry",
            confidence=0.0,
            outputs={},
            warnings=[],
            errors=[],
            started_at=now,
            finished_at=now,
            trace_id=kwargs.get("trace_id"),
        )

    async def _persist_result(self, session, row, result, inputs_hash):
        return

    async def _materialize_side_effects(self, **kwargs):
        return

    events: list[dict] = []

    async def _sink(e: dict) -> None:
        events.append(e)

    def _create_tool_event_sink(self, *, session, run_row):
        return _sink

    monkeypatch.setattr(AgentRunner, "_load_prior_enrichment", _load_prior_enrichment)
    monkeypatch.setattr(AgentRunner, "_get_or_create_run_row", _get_or_create_run_row)
    monkeypatch.setattr(AgentRunner, "_persist_result", _persist_result)
    monkeypatch.setattr(AgentRunner, "_materialize_side_effects", _materialize_side_effects)
    monkeypatch.setattr(AgentRunner, "_create_tool_event_sink", _create_tool_event_sink)

    ctx = PipelineContext(org_id=org_id, memory_id=memory_id, initiator_user_id=user_id, trace_id="t1", storage="short_term")
    result = await runner.run_agent(ctx=ctx, agent_name="MetadataExtractionAgent")
    assert result.status == "success"

    assert any(
        e.get("payload", {}).get("tool") == "AgentRunner._compute_cache_key"
        and e.get("event_type") == "tool_call"
        and isinstance(e.get("payload", {}).get("inputs_hash_len"), int)
        and isinstance(e.get("payload", {}).get("cache_key_len"), int)
        for e in events
    )


class _FailingAgent:
    name = "MetadataExtractionAgent"
    version = "v1"

    def validate_outputs(self, result: AgentResult) -> None:
        return

    async def run(self, memory_id: str, context):
        raise RuntimeError("agent boom")


@pytest.mark.asyncio
async def test_audit_tool_events_emitted_when_agent_raises_and_audit_fails(monkeypatch):
    import app.services.agent_runner as agent_runner_module

    org_id = str(uuid4())
    user_id = str(uuid4())
    memory_id = str(uuid4())

    runner = AgentRunner(service_user_id=user_id)

    monkeypatch.setattr(agent_runner_module, "get_tenant_session", _fake_tenant_session)
    monkeypatch.setattr(agent_runner_module, "get_agent", lambda name: _FailingAgent() if name == "MetadataExtractionAgent" else None)

    async def _load_memory_inputs(self, session, ctx):
        return "hello", None, "personal", None

    async def _load_prior_enrichment(self, session, org_id, memory_id):
        return {}

    async def _get_or_create_run_row(self, **kwargs):
        now = datetime(2026, 1, 23, tzinfo=timezone.utc)
        return SimpleNamespace(
            id="ar1",
            organization_id=kwargs.get("org_id"),
            memory_id=kwargs.get("memory_id"),
            agent_name=kwargs.get("agent_name"),
            agent_version=kwargs.get("agent_version"),
            inputs_hash=kwargs.get("inputs_hash"),
            status="retry",
            confidence=0.0,
            outputs={},
            warnings=[],
            errors=[],
            started_at=now,
            finished_at=now,
            trace_id=kwargs.get("trace_id"),
        )

    async def _persist_result(self, session, row, result, inputs_hash):
        return

    async def _materialize_side_effects(self, **kwargs):
        return

    async def _audit_log_event_raise(self, **kwargs):
        raise RuntimeError("audit boom")

    monkeypatch.setattr(agent_runner_module.AuditService, "log_event", _audit_log_event_raise)

    events: list[dict] = []

    async def _sink(e: dict) -> None:
        events.append(e)

    def _create_tool_event_sink(self, *, session, run_row):
        return _sink

    monkeypatch.setattr(AgentRunner, "_load_memory_inputs", _load_memory_inputs)
    monkeypatch.setattr(AgentRunner, "_load_prior_enrichment", _load_prior_enrichment)
    monkeypatch.setattr(AgentRunner, "_get_or_create_run_row", _get_or_create_run_row)
    monkeypatch.setattr(AgentRunner, "_persist_result", _persist_result)
    monkeypatch.setattr(AgentRunner, "_materialize_side_effects", _materialize_side_effects)
    monkeypatch.setattr(AgentRunner, "_create_tool_event_sink", _create_tool_event_sink)

    ctx = PipelineContext(org_id=org_id, memory_id=memory_id, initiator_user_id=user_id, trace_id="t1", storage="long_term")

    with pytest.raises(RuntimeError, match="agent boom"):
        await runner.run_agent(ctx=ctx, agent_name="MetadataExtractionAgent")

    assert any(
        e.get("payload", {}).get("tool") == "AuditService.log_event" and e.get("event_type") == "tool_call" for e in events
    )
    assert any(
        e.get("payload", {}).get("tool") == "AuditService.log_event"
        and e.get("event_type") == "tool_result"
        and e.get("payload", {}).get("ok") is False
        for e in events
    )
