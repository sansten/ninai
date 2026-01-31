"""AutoEvalBench-style drift regression tests for activation scoring.

These tests intentionally pin a few activation numbers for a fixed synthetic
scenario. If you change the scoring algorithm/config, update the expected
snapshots here as part of the change.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from app.services.memory_activation.scoring import ActivationScorer


@pytest.mark.asyncio
async def test_activation_scorer_snapshot_default_config() -> None:
    scorer = ActivationScorer()

    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    last_accessed_at = now - timedelta(days=2)

    activation, components = scorer.score_memory(
        similarity=0.82,
        base_importance=0.7,
        confidence=0.9,
        contradicted=False,
        risk_factor=0.2,
        access_count=5,
        last_accessed_at=last_accessed_at,
        evidence_link_count=3,
        scope_match=0.7,
        episode_match=0.3,
        goal_match=0.5,
        neighbor_activation=None,
        current_time=now,
        age_days=10.0,
    )

    assert activation == pytest.approx(0.6653105988102114, abs=1e-9)
    assert components.to_dict() == {
        "rel": pytest.approx(0.82, abs=1e-12),
        "rec": pytest.approx(0.8187307530779818, abs=1e-12),
        "freq": pytest.approx(0.9179150013761012, abs=1e-12),
        "imp": pytest.approx(0.42457146179884336, abs=1e-12),
        "conf": pytest.approx(0.9, abs=1e-12),
        "ctx": pytest.approx(0.5, abs=1e-12),
        "prov": pytest.approx(0.4511883639059736, abs=1e-12),
        "risk": pytest.approx(0.8, abs=1e-12),
    }


@pytest.mark.asyncio
async def test_activation_scorer_neighbor_boost_is_additive() -> None:
    scorer = ActivationScorer()

    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    last_accessed_at = now - timedelta(days=2)

    base_activation, _ = scorer.score_memory(
        similarity=0.82,
        base_importance=0.7,
        confidence=0.9,
        contradicted=False,
        risk_factor=0.2,
        access_count=5,
        last_accessed_at=last_accessed_at,
        evidence_link_count=3,
        scope_match=0.7,
        episode_match=0.3,
        goal_match=0.5,
        neighbor_activation=None,
        current_time=now,
        age_days=10.0,
    )

    boosted_activation, _ = scorer.score_memory(
        similarity=0.82,
        base_importance=0.7,
        confidence=0.9,
        contradicted=False,
        risk_factor=0.2,
        access_count=5,
        last_accessed_at=last_accessed_at,
        evidence_link_count=3,
        scope_match=0.7,
        episode_match=0.3,
        goal_match=0.5,
        neighbor_activation=0.65,
        current_time=now,
        age_days=10.0,
    )

    assert boosted_activation == pytest.approx(0.7303105988102114, abs=1e-9)
    assert (boosted_activation - base_activation) == pytest.approx(0.065, abs=1e-12)


@pytest.mark.asyncio
async def test_activation_scorer_contradiction_penalizes_confidence() -> None:
    scorer = ActivationScorer()

    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    last_accessed_at = now - timedelta(days=2)

    activation_ok, components_ok = scorer.score_memory(
        similarity=0.82,
        base_importance=0.7,
        confidence=0.9,
        contradicted=False,
        risk_factor=0.2,
        access_count=5,
        last_accessed_at=last_accessed_at,
        evidence_link_count=3,
        scope_match=0.7,
        episode_match=0.3,
        goal_match=0.5,
        neighbor_activation=None,
        current_time=now,
        age_days=10.0,
    )

    activation_bad, components_bad = scorer.score_memory(
        similarity=0.82,
        base_importance=0.7,
        confidence=0.9,
        contradicted=True,
        risk_factor=0.2,
        access_count=5,
        last_accessed_at=last_accessed_at,
        evidence_link_count=3,
        scope_match=0.7,
        episode_match=0.3,
        goal_match=0.5,
        neighbor_activation=None,
        current_time=now,
        age_days=10.0,
    )

    assert activation_ok == pytest.approx(0.6653105988102114, abs=1e-9)
    assert activation_bad == pytest.approx(0.6501164364709745, abs=1e-9)

    assert components_ok.conf == pytest.approx(0.9, abs=1e-12)
    assert components_bad.conf == pytest.approx(0.45, abs=1e-12)
    assert activation_bad < activation_ok
