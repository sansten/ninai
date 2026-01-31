from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.feedback_learning_config_service import (
    FeedbackLearningConfigService,
    extract_calibration,
    normalize_stopwords,
)


def test_normalize_stopwords_dedupes_and_sorts():
    assert normalize_stopwords(["  The ", "the", "API!", "api", ""]) == ["api", "the"]


def test_extract_calibration_from_applied_feedback_updates_delta_counts():
    extracted = extract_calibration(
        {
            "applied": True,
            "applied_count": 3,
            "updates": [{"type": "tag_add"}, {"type": "tag_add"}, {"type": "note"}],
            "rationale": "applied_pending_feedback",
        }
    )

    assert extracted.should_update is True
    assert extracted.updated_thresholds == {}
    assert extracted.new_stopwords == []
    assert extracted.heuristic_weights == {}

    assert extracted.calibration_delta.get("applied_count") == 3
    assert extracted.calibration_delta.get("feedback_update_counts") == {"tag_add": 2, "note": 1}
    assert extracted.calibration_delta.get("rationale") == "applied_pending_feedback"


@pytest.mark.asyncio
async def test_feedback_learning_config_service_noop_when_no_updates():
    session = AsyncMock()
    session.execute = AsyncMock()

    svc = FeedbackLearningConfigService(session)
    res = await svc.apply_from_agent_outputs(
        organization_id="org",
        source_memory_id="mem",
        outputs={"applied": False, "applied_count": 0, "updates": []},
        updated_by_user_id=None,
        agent_version="v1",
        trace_id="t",
    )

    assert res["config_updated"] is False
    assert session.execute.call_count == 0


@pytest.mark.asyncio
async def test_feedback_learning_config_service_upsert_executes_and_flushes():
    session = AsyncMock()
    session.execute = AsyncMock(
        side_effect=[
            # select existing
            SimpleNamespace(
                scalar_one_or_none=lambda: SimpleNamespace(
                    updated_thresholds={"a": 1},
                    stopwords=["foo"],
                    heuristic_weights={"w": 1},
                    calibration_delta={},
                )
            ),
            # upsert
            SimpleNamespace(),
        ]
    )
    session.flush = AsyncMock()

    svc = FeedbackLearningConfigService(session)
    res = await svc.apply_from_agent_outputs(
        organization_id="org",
        source_memory_id="mem",
        outputs={
            "applied": True,
            "applied_count": 1,
            "updates": [{"type": "tag_add"}],
            "updated_thresholds": {"a": 2, "b": 3},
            "new_stopwords": ["Foo", "Bar"],
            "heuristic_weights": {"w": 2},
            "calibration_delta": {"source": "unit_test"},
        },
        updated_by_user_id="u",
        agent_version="v1",
        trace_id="t",
    )

    assert res["config_updated"] is True
    assert res["stopwords_added"] >= 1
    assert res["thresholds_updated"] == 2
    assert res["weights_updated"] == 1
    assert session.execute.call_count == 2
    assert session.flush.call_count == 1
