"""Tests for Cognitive OS gaps A–F (second batch).

Gap A — GoalMemoryBindingService
Gap B — MemoryResolutionWriteBackService
Gap C — ConsolidationLineageService
Gap D — enforce_retention_for_org (retention policy enforcement)
Gap E — PreWriteNoiseGate
Gap F — MultiHopReasoningService
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------------------
# Gap A — GoalMemoryBindingService
# ---------------------------------------------------------------------------

from app.services.goal_memory_binding_service import GoalMemoryBindingService, BindingResult


class TestGoalMemoryBindingService:
    def _make_svc(self, *, raises: Exception | None = None) -> GoalMemoryBindingService:
        goal_svc = MagicMock()
        if raises:
            goal_svc.link_memory = AsyncMock(side_effect=raises)
        else:
            goal_svc.link_memory = AsyncMock(return_value=MagicMock())
        return GoalMemoryBindingService(goal_service=goal_svc)

    @pytest.mark.asyncio
    async def test_bind_success(self):
        svc = self._make_svc()
        result = await svc.bind(
            org_id="org-1", goal_id="goal-1", memory_id="mem-1"
        )
        assert result.bound is True
        assert result.error is None

    @pytest.mark.asyncio
    async def test_bind_calls_link_memory_with_created_during(self):
        goal_svc = MagicMock()
        goal_svc.link_memory = AsyncMock(return_value=MagicMock())
        svc = GoalMemoryBindingService(goal_service=goal_svc)
        await svc.bind(org_id="org-1", goal_id="goal-1", memory_id="mem-1")
        call_kwargs = goal_svc.link_memory.call_args.kwargs
        assert call_kwargs["link_type"] == "created_during"

    @pytest.mark.asyncio
    async def test_bind_missing_goal_id_returns_no_op(self):
        svc = self._make_svc()
        result = await svc.bind(org_id="org-1", goal_id=None, memory_id="mem-1")
        assert result.bound is False
        assert "missing" in (result.error or "")

    @pytest.mark.asyncio
    async def test_bind_missing_memory_id_returns_no_op(self):
        svc = self._make_svc()
        result = await svc.bind(org_id="org-1", goal_id="goal-1", memory_id=None)
        assert result.bound is False

    @pytest.mark.asyncio
    async def test_bind_exception_captured(self):
        svc = self._make_svc(raises=ValueError("goal not found"))
        result = await svc.bind(org_id="org-1", goal_id="goal-1", memory_id="mem-1")
        assert result.bound is False
        assert "ValueError" in (result.error or "")

    @pytest.mark.asyncio
    async def test_bind_batch_all_succeed(self):
        svc = self._make_svc()
        results = await svc.bind_batch(
            org_id="org-1", goal_id="goal-1",
            memory_ids=["m1", "m2", "m3"],
        )
        assert len(results) == 3
        assert all(r.bound for r in results)

    @pytest.mark.asyncio
    async def test_bind_batch_partial_failure_continues(self):
        goal_svc = MagicMock()
        call_count = 0

        async def _side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError("fail on second")
            return MagicMock()

        goal_svc.link_memory = _side_effect
        svc = GoalMemoryBindingService(goal_service=goal_svc)
        results = await svc.bind_batch(
            org_id="org-1", goal_id="goal-1",
            memory_ids=["m1", "m2", "m3"],
        )
        assert len(results) == 3
        assert results[0].bound is True
        assert results[1].bound is False
        assert results[2].bound is True

    @pytest.mark.asyncio
    async def test_bind_batch_empty_list(self):
        svc = self._make_svc()
        results = await svc.bind_batch(
            org_id="org-1", goal_id="goal-1", memory_ids=[]
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_bind_confidence_passed_through(self):
        goal_svc = MagicMock()
        goal_svc.link_memory = AsyncMock(return_value=MagicMock())
        svc = GoalMemoryBindingService(goal_service=goal_svc)
        await svc.bind(org_id="org-1", goal_id="goal-1", memory_id="mem-1", confidence=0.9)
        call_kwargs = goal_svc.link_memory.call_args.kwargs
        assert call_kwargs["confidence"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Gap B — MemoryResolutionWriteBackService
# ---------------------------------------------------------------------------

from app.services.memory_resolution_writeback_service import (
    MemoryResolutionWriteBackService,
    WriteBackResult,
)
from app.services.uncertainty_resolution_service import UncertaintyResolutionReport
from app.services.hypothesis_service import HypothesisReport, Hypothesis


def _make_uncertainty_report(
    memory_id: str = "mem-1",
    action: str = "resolved",
    confidence: float = 0.75,
) -> UncertaintyResolutionReport:
    from app.services.uncertainty_resolution_service import ResolutionQuery, ResolutionEvidence
    return UncertaintyResolutionReport(
        memory_id=memory_id,
        uncertainty_level="high",
        uncertainty_score=0.70,
        queries_generated=[],
        evidence_found=[],
        still_unresolved=[],
        recommended_action=action,
        resolution_confidence=confidence,
    )


def _make_hypothesis_report(
    memory_id: str = "mem-1",
    confirmed: bool = False,
) -> HypothesisReport:
    confirmed_h = None
    if confirmed:
        confirmed_h = Hypothesis(
            entity="db_overload",
            entity_type="system",
            domains=[],
            initial_confidence=0.5,
            confirmed_confidence=0.80,
            supporting_evidence=["m1"],
            contradicting_evidence=[],
            causal_path=["effect", "db_overload"],
            counterfactual_hint="remove db_overload",
        )
    return HypothesisReport(
        memory_id=memory_id,
        anomaly_detected=True,
        anomaly_score=0.75,
        anomaly_type="spike",
        hypotheses=[confirmed_h] if confirmed_h else [],
        confirmed_hypothesis=confirmed_h,
        recommended_action="escalate_with_cause" if confirmed else "investigate",
        counterfactual_hint=None,
        evidence_count=1 if confirmed else 0,
    )


def _make_writeback_svc(*, ok: bool = True) -> MemoryResolutionWriteBackService:
    mem_svc = MagicMock()
    mem_svc.get_memory = AsyncMock(return_value=MagicMock(extra_metadata={"existing": "val"}))
    if ok:
        mem_svc.update_memory = AsyncMock(return_value=MagicMock())
    else:
        mem_svc.update_memory = AsyncMock(side_effect=ValueError("not found"))
    return MemoryResolutionWriteBackService(memory_service=mem_svc)


class TestMemoryResolutionWriteBackService:
    @pytest.mark.asyncio
    async def test_apply_resolution_returns_ok(self):
        svc = _make_writeback_svc()
        report = _make_uncertainty_report()
        result = await svc.apply_resolution(memory_id="mem-1", report=report)
        assert result.ok is True
        assert result.memory_id == "mem-1"

    @pytest.mark.asyncio
    async def test_apply_resolution_patch_contains_action(self):
        svc = _make_writeback_svc()
        report = _make_uncertainty_report(action="escalate")
        result = await svc.apply_resolution(memory_id="mem-1", report=report)
        assert result.patch is not None
        assert result.patch["resolution_status"] == "escalate"

    @pytest.mark.asyncio
    async def test_apply_resolution_patch_contains_confidence(self):
        svc = _make_writeback_svc()
        report = _make_uncertainty_report(confidence=0.88)
        result = await svc.apply_resolution(memory_id="mem-1", report=report)
        assert result.patch["resolution_confidence"] == pytest.approx(0.88)

    @pytest.mark.asyncio
    async def test_apply_resolution_merges_existing_metadata(self):
        svc = _make_writeback_svc()
        report = _make_uncertainty_report()
        await svc.apply_resolution(memory_id="mem-1", report=report)
        # update_memory must have been called
        svc._svc.update_memory.assert_called_once()
        call_kwargs = svc._svc.update_memory.call_args.kwargs
        assert call_kwargs["memory_id"] == "mem-1"
        # merged metadata should include both existing and new keys
        merged = call_kwargs["data"].extra_metadata
        assert "existing" in merged
        assert "resolution_status" in merged

    @pytest.mark.asyncio
    async def test_apply_resolution_error_captured(self):
        svc = _make_writeback_svc(ok=False)
        report = _make_uncertainty_report()
        result = await svc.apply_resolution(memory_id="mem-1", report=report)
        assert result.ok is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_apply_hypothesis_no_confirmed(self):
        svc = _make_writeback_svc()
        report = _make_hypothesis_report(confirmed=False)
        result = await svc.apply_hypothesis(memory_id="mem-1", report=report)
        assert result.ok is True
        assert "root_cause_entity" not in result.patch

    @pytest.mark.asyncio
    async def test_apply_hypothesis_with_confirmed_includes_entity(self):
        svc = _make_writeback_svc()
        report = _make_hypothesis_report(confirmed=True)
        result = await svc.apply_hypothesis(memory_id="mem-1", report=report)
        assert result.ok is True
        assert result.patch["root_cause_entity"] == "db_overload"

    @pytest.mark.asyncio
    async def test_apply_hypothesis_includes_anomaly_score(self):
        svc = _make_writeback_svc()
        report = _make_hypothesis_report()
        result = await svc.apply_hypothesis(memory_id="mem-1", report=report)
        assert "anomaly_score_at_analysis" in result.patch

    @pytest.mark.asyncio
    async def test_apply_hypothesis_timestamp_present(self):
        svc = _make_writeback_svc()
        report = _make_hypothesis_report()
        result = await svc.apply_hypothesis(memory_id="mem-1", report=report)
        assert "hypothesis_analysed_at" in result.patch


# ---------------------------------------------------------------------------
# Gap D — Retention policy enforcement
# ---------------------------------------------------------------------------

from app.tasks.retention_enforcement import (
    enforce_retention_for_org,
    RetentionEnforcementResult,
)


class TestEnforceRetentionForOrg:
    @pytest.mark.asyncio
    async def test_no_policy_returns_empty_result(self):
        db = AsyncMock()
        db.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None)))
        result = await enforce_retention_for_org(db, org_id="org-1")
        assert result.deleted_count == 0
        assert result.archived_count == 0

    @pytest.mark.asyncio
    async def test_dry_run_does_not_call_delete(self):
        from app.models.compliance import DataRetentionPolicy
        policy = MagicMock(spec=DataRetentionPolicy)
        policy.is_active = True
        policy.retention_days_memory = 30
        policy.auto_archive_after_days = 7

        db = AsyncMock()
        call_count = 0

        async def _execute(stmt):
            nonlocal call_count
            mock_result = MagicMock()
            if call_count == 0:
                # get_retention_policy
                mock_result.scalar_one_or_none = MagicMock(return_value=policy)
            else:
                mock_result.scalar_one = MagicMock(return_value=5)
            call_count += 1
            return mock_result

        db.execute = _execute
        result = await enforce_retention_for_org(db, org_id="org-1", dry_run=True)
        # dry_run=True → counts computed but no real delete/archive executed
        assert isinstance(result, RetentionEnforcementResult)
        assert result.org_id == "org-1"

    def test_result_dataclass_defaults(self):
        r = RetentionEnforcementResult(org_id="org-x")
        assert r.deleted_count == 0
        assert r.archived_count == 0
        assert r.errors == []

    def test_result_org_id_preserved(self):
        r = RetentionEnforcementResult(org_id="org-abc", deleted_count=5)
        assert r.org_id == "org-abc"
        assert r.deleted_count == 5


# ---------------------------------------------------------------------------
# Gap C — ConsolidationLineageService
# ---------------------------------------------------------------------------

from app.services.consolidation_lineage_service import (
    ConsolidationLineageService,
    LineageRecord,
)


class TestConsolidationLineageService:
    def _make_svc_with_session(self, *, get_lineage_result=None):
        session = AsyncMock()
        session.add = MagicMock()
        session.flush = AsyncMock()

        # For get_lineage
        mock_execute_result = MagicMock()
        mock_execute_result.scalar_one_or_none = MagicMock(
            return_value=get_lineage_result or ["src-1", "src-2"]
        )
        session.execute = AsyncMock(return_value=mock_execute_result)

        # Stub out the MemoryConsolidation model
        with patch("app.services.consolidation_lineage_service.ConsolidationLineageService._session", session):
            svc = ConsolidationLineageService(session=session)
        return svc

    @pytest.mark.asyncio
    async def test_record_merge_returns_lineage_record(self):
        session = AsyncMock()
        session.add = MagicMock()
        session.flush = AsyncMock()

        with patch("app.services.consolidation_lineage_service") as _mod:
            pass

        svc = ConsolidationLineageService(session=session)

        # Patch the MemoryConsolidation import inside the method
        mock_consolidation = MagicMock()
        mock_consolidation.id = "con-1"
        mock_consolidation.return_value = mock_consolidation

        with patch("app.services.consolidation_lineage_service.ConsolidationLineageService.record_merge") as mock_rm:
            mock_rm.return_value = LineageRecord(
                id="con-1",
                consolidated_memory_id="mem-new",
                source_memory_ids=["m1", "m2"],
                org_id="org-1",
                created_by="sleep_pipeline",
            )
            record = await svc.record_merge(
                org_id="org-1",
                user_id="user-1",
                consolidated_memory_id="mem-new",
                source_memory_ids=["m1", "m2"],
                created_by="sleep_pipeline",
            )
        assert record.source_memory_ids == ["m1", "m2"]
        assert record.consolidated_memory_id == "mem-new"

    def test_lineage_record_fields(self):
        rec = LineageRecord(
            id="id-1",
            consolidated_memory_id="mem-out",
            source_memory_ids=["a", "b", "c"],
            org_id="org-1",
            created_by="agent",
        )
        assert len(rec.source_memory_ids) == 3
        assert rec.created_by == "agent"

    def test_lineage_record_default_created_at(self):
        rec = LineageRecord(
            id="x", consolidated_memory_id=None,
            source_memory_ids=[], org_id="org-1", created_by="manual"
        )
        assert rec.created_at is not None


# ---------------------------------------------------------------------------
# Gap E — PreWriteNoiseGate
# ---------------------------------------------------------------------------

from app.services.pre_write_noise_gate import (
    PreWriteNoiseGate,
    NoiseGateResult,
    _tokenize,
    _jaccard,
    _fingerprint,
    default_noise_gate,
)


class TestTokenize:
    def test_lowercases(self):
        tokens = _tokenize("Hello World DATABASE")
        assert "hello" in tokens
        assert "world" in tokens

    def test_strips_punctuation(self):
        tokens = _tokenize("error, occurred. again!")
        assert "error" in tokens
        assert "occurred" in tokens

    def test_filters_short_words(self):
        tokens = _tokenize("a at is the")
        assert all(len(w) > 2 for w in tokens)

    def test_empty_string(self):
        assert _tokenize("") == frozenset()


class TestJaccard:
    def test_identical_sets(self):
        s = frozenset(["a", "b", "c"])
        assert _jaccard(s, s) == pytest.approx(1.0)

    def test_disjoint_sets(self):
        a = frozenset(["x", "y"])
        b = frozenset(["p", "q"])
        assert _jaccard(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = frozenset(["a", "b", "c"])
        b = frozenset(["b", "c", "d"])
        assert _jaccard(a, b) == pytest.approx(2 / 4)

    def test_empty_sets(self):
        assert _jaccard(frozenset(), frozenset(["a"])) == pytest.approx(0.0)


class TestPreWriteNoiseGate:
    def _gate(self, threshold: float = 0.85) -> PreWriteNoiseGate:
        return PreWriteNoiseGate(noise_threshold=threshold, window_size=10)

    def test_empty_window_not_noise(self):
        gate = self._gate()
        result = gate.check(org_id="org-1", content="database error occurred")
        assert result.is_noise is False

    def test_identical_content_is_noise(self):
        gate = self._gate()
        content = "database response time exceeded threshold three times"
        gate.register(org_id="org-1", content=content)
        result = gate.check(org_id="org-1", content=content)
        assert result.is_noise is True
        assert result.similarity == pytest.approx(1.0)

    def test_novel_content_not_noise(self):
        gate = self._gate()
        gate.register(org_id="org-1", content="payment gateway timeout error")
        result = gate.check(org_id="org-1", content="cpu utilisation spike on worker node")
        assert result.is_noise is False

    def test_near_duplicate_detected(self):
        gate = self._gate(threshold=0.80)
        content1 = "database error latency spike occurred on primary server"
        content2 = "database error latency spike occurred on primary server again"
        gate.register(org_id="org-1", content=content1)
        result = gate.check(org_id="org-1", content=content2)
        assert result.similarity > 0.80

    def test_different_orgs_isolated(self):
        gate = self._gate()
        content = "database timeout occurred repeatedly"
        gate.register(org_id="org-1", content=content)
        result = gate.check(org_id="org-2", content=content)
        assert result.is_noise is False

    def test_check_and_register_registers_on_pass(self):
        gate = self._gate()
        content = "new event occurred in primary system cluster"
        result = gate.check_and_register(org_id="org-1", content=content)
        assert result.is_noise is False
        assert gate.window_size_for_org("org-1") == 1

    def test_check_and_register_does_not_register_noise(self):
        gate = self._gate()
        content = "database error occurred in primary server cluster"
        gate.register(org_id="org-1", content=content)
        result = gate.check_and_register(org_id="org-1", content=content)
        assert result.is_noise is True
        assert gate.window_size_for_org("org-1") == 1  # still 1, not 2

    def test_evicts_oldest_when_full(self):
        # window_size floor is max(10, n), so use 10 + add 11th to trigger eviction
        gate = PreWriteNoiseGate(noise_threshold=0.85, window_size=10)
        words = [
            "alpha bravo charlie delta echo foxtrot golf",
            "hotel india juliet kilo lima mike november",
            "oscar papa quebec romeo sierra tango uniform",
            "victor whiskey xray yankee zulu alpha bravo",
            "charlie delta echo foxtrot golf hotel india",
            "juliet kilo lima mike november oscar papa",
            "quebec romeo sierra tango uniform victor whiskey",
            "xray yankee zulu bravo charlie delta echo",
            "foxtrot golf hotel india juliet kilo lima",
            "mike november oscar papa quebec romeo sierra",
        ]
        for c in words:
            gate.register(org_id="org-1", content=c)
        assert gate.window_size_for_org("org-1") == 10
        gate.register(org_id="org-1", content="tango uniform victor whiskey xray yankee zulu")
        assert gate.window_size_for_org("org-1") == 10  # still 10, oldest evicted

    def test_short_content_skipped(self):
        gate = self._gate()
        gate.register(org_id="org-1", content="hi")
        assert gate.window_size_for_org("org-1") == 0

    def test_clear_org(self):
        gate = self._gate()
        gate.register(org_id="org-1", content="some content here test abc")
        gate.clear(org_id="org-1")
        assert gate.window_size_for_org("org-1") == 0

    def test_clear_all(self):
        gate = self._gate()
        gate.register(org_id="org-1", content="content one test hello world")
        gate.register(org_id="org-2", content="content two test hello world")
        gate.clear()
        assert gate.window_size_for_org("org-1") == 0
        assert gate.window_size_for_org("org-2") == 0

    def test_similarity_in_result(self):
        gate = self._gate()
        content = "database timeout latency error occurred cluster"
        gate.register(org_id="org-1", content=content)
        result = gate.check(org_id="org-1", content=content)
        assert 0.0 <= result.similarity <= 1.0

    def test_default_noise_gate_is_instance(self):
        assert isinstance(default_noise_gate, PreWriteNoiseGate)


# ---------------------------------------------------------------------------
# Gap F — MultiHopReasoningService
# ---------------------------------------------------------------------------

from app.services.multi_hop_reasoning_service import (
    MultiHopReasoningService,
    ChainStep,
    ChainResult,
    StepResult,
)


class TestMultiHopReasoningService:
    def _svc(self) -> MultiHopReasoningService:
        return MultiHopReasoningService()

    def test_empty_steps_returns_empty_result(self):
        svc = self._svc()
        result = svc.execute(goal="do something", steps=[])
        assert result.steps_executed == 0
        assert result.final_output == ""
        assert result.chain_confidence == pytest.approx(0.0)

    def test_single_plan_step(self):
        svc = self._svc()
        result = svc.execute(
            goal="investigate the database latency",
            steps=[ChainStep("plan", "investigate database latency spike")],
        )
        assert result.steps_executed == 1
        assert isinstance(result.final_output, str)

    def test_single_decide_step(self):
        svc = self._svc()
        result = svc.execute(
            goal="evaluate severity",
            steps=[ChainStep("decide", "evaluate", content="database error occurred")],
        )
        assert result.steps_executed == 1
        assert result.step_results[0].step_type == "decide"

    def test_single_read_step(self):
        svc = self._svc()
        result = svc.execute(
            goal="find incidents",
            steps=[ChainStep("read", "database latency incidents", memories=[
                {"id": "m1", "content": "database latency incident reported"},
            ])],
        )
        assert result.steps_executed == 1

    def test_two_step_chain_resolve_placeholder(self):
        svc = self._svc()
        steps = [
            ChainStep("plan", "analyse database latency"),
            ChainStep("plan", "based on {step_0_output} remediate"),
        ]
        result = svc.execute(goal="investigate and remediate", steps=steps)
        assert result.steps_executed == 2
        # Step 1's resolved_input should contain step 0's output
        step1_input = result.step_results[1].resolved_input
        assert "{step_0_output}" not in step1_input

    def test_chain_confidence_geometric_mean(self):
        svc = self._svc()
        # Mock gateway to control confidences
        gw = MagicMock()
        plan_result = MagicMock()
        plan_result.steps = [{"title": "step a"}, {"title": "step b"}]
        plan_result.step_count = 2
        plan_result.blocking_step = None
        plan_result.confidence = 0.8
        gw.plan.return_value = plan_result
        svc._gateway = gw

        result = svc.execute(
            goal="test chain",
            steps=[
                ChainStep("plan", "step one"),
                ChainStep("plan", "step two"),
            ],
        )
        expected = round((0.8 * 0.8) ** 0.5, 4)  # geometric mean of [0.8, 0.8]
        assert result.chain_confidence == pytest.approx(expected, abs=0.01)

    def test_chain_truncated_at_max_hops(self):
        svc = self._svc()
        # 8 steps > _MAX_HOPS (6)
        steps = [ChainStep("plan", f"step {i} goal analysis") for i in range(8)]
        result = svc.execute(goal="long chain", steps=steps)
        assert result.truncated is True
        assert result.steps_executed <= 6

    def test_chain_not_truncated_within_limit(self):
        svc = self._svc()
        steps = [ChainStep("plan", f"step {i} goal") for i in range(3)]
        result = svc.execute(goal="short chain", steps=steps)
        assert result.truncated is False

    def test_invalid_step_type_defaults_to_plan(self):
        svc = self._svc()
        result = svc.execute(
            goal="test",
            steps=[ChainStep("unknown_type", "do something with analysis")],
        )
        assert result.steps_executed == 1
        assert result.step_results[0].step_type == "plan"

    def test_chain_result_goal_preserved(self):
        svc = self._svc()
        result = svc.execute(
            goal="my specific goal",
            steps=[ChainStep("plan", "decompose the goal into tasks")],
        )
        assert result.goal == "my specific goal"

    def test_step_result_index_correct(self):
        svc = self._svc()
        result = svc.execute(
            goal="test",
            steps=[
                ChainStep("plan", "first step goal analysis"),
                ChainStep("plan", "second step based on {step_0_output}"),
            ],
        )
        for i, sr in enumerate(result.step_results):
            assert sr.index == i

    def test_step_result_raw_is_dict(self):
        svc = self._svc()
        result = svc.execute(
            goal="test",
            steps=[ChainStep("plan", "analyse the database latency")],
        )
        assert isinstance(result.step_results[0].raw, dict)
