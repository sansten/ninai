from __future__ import annotations

from unittest.mock import AsyncMock


class _AsyncSessionCtx:
    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _AsyncBeginCtx:
    async def __aenter__(self):
        return None

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _DB:
    def begin(self):
        return _AsyncBeginCtx()

    def begin_nested(self):
        return _AsyncBeginCtx()

    async def flush(self):
        return None

    async def execute(self, *args, **kwargs):
        class _Result:
            def scalar_one_or_none(self):
                return None

        return _Result()


def test_cognitive_loop_task_wires_learning_and_context_services(monkeypatch):
    """The live task should pass the built strategy/context services into the orchestrator."""
    from app.tasks import cognitive_loop as mod

    captured: dict[str, object] = {}

    monkeypatch.setattr(mod, "_broker_enabled", lambda: False)

    async def _noop_set_tenant_context(*args, **kwargs):
        return None

    monkeypatch.setattr(mod, "set_tenant_context", _noop_set_tenant_context)
    monkeypatch.setattr(mod, "async_session_factory", lambda: _AsyncSessionCtx(_DB()))

    class _FakeSummary:
        unreliable_tools = []
        low_confidence_domains = []
        recommended_evidence_multiplier = 1

    monkeypatch.setattr(
        mod.SelfModelService,
        "get_planner_summary",
        AsyncMock(return_value=_FakeSummary()),
    )
    monkeypatch.setattr(
        mod.EvaluationReportService,
        "generate_for_session",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        mod.SelfModelService,
        "ingest_from_session",
        AsyncMock(return_value=0),
    )

    class _FakeOrchestrator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def run(self, *, session_id: str, tool_ctx):
            return "succeeded"

    monkeypatch.setattr(mod, "LoopOrchestrator", _FakeOrchestrator)

    status = mod.cognitive_loop_task(
        org_id="org1",
        session_id="session-1",
        initiator_user_id="user-1",
        roles="member",
        clearance_level=0,
        justification="test",
        max_iterations=1,
    )

    assert status == "succeeded"
    assert isinstance(captured["adaptive_strategy"], mod.AdaptiveStrategyService)
    assert isinstance(captured["strategy_learning"], mod.StrategyLearningService)
    assert isinstance(captured["context_aggregator"], mod.CognitiveContextAggregator)
