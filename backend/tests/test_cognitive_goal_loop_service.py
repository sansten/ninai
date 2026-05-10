from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.cognitive_goal_loop_service import CognitiveGoalLoopService


class _ScalarResult:
    def __init__(self, rows):
        self._rows = list(rows)

    def scalars(self):
        return self

    def all(self):
        return list(self._rows)


@pytest.mark.asyncio
async def test_goal_loop_build_context_rolls_up_counts(monkeypatch):
    svc = CognitiveGoalLoopService(AsyncMock(), org_id="org-1")

    monkeypatch.setattr(
        svc,
        "_load_active_goals",
        AsyncMock(return_value=[{"goal_id": "g1"}, {"goal_id": "g2"}]),
    )
    monkeypatch.setattr(
        svc,
        "_load_open_gaps",
        AsyncMock(return_value=[{"gap_id": "kg1"}]),
    )
    monkeypatch.setattr(
        svc,
        "_suggest_goals",
        AsyncMock(return_value=[{"goal_id": "sg1"}]),
    )
    monkeypatch.setattr(
        svc,
        "_load_world_state",
        AsyncMock(
            return_value={
                "recent_changes": [{"entity": "deployment"}],
                "highlighted_entities": [{"entity": "release train"}, {"entity": "database"}],
                "last_world_model_run_at": None,
            }
        ),
    )

    result = await svc.build_context()

    assert result["loop_health"] == {
        "active_goal_count": 2,
        "knowledge_gap_count": 1,
        "suggested_goal_count": 1,
        "world_change_count": 1,
        "highlighted_entity_count": 2,
    }


@pytest.mark.asyncio
async def test_goal_loop_world_state_summarizes_recent_changes_and_entities():
    finished_at = datetime(2026, 5, 8, 18, 0, tzinfo=timezone.utc)
    older = datetime(2026, 5, 7, 16, 30, tzinfo=timezone.utc)
    session = AsyncMock()
    session.execute = AsyncMock(
        return_value=_ScalarResult(
            [
                SimpleNamespace(
                    memory_id="m1",
                    finished_at=finished_at,
                    outputs={
                        "state_changes": [
                            {
                                "entity": "deployment",
                                "change_type": "delayed",
                                "description": "Migration blocked rollout",
                            }
                        ],
                        "world_nodes": [
                            {
                                "entity": "release train",
                                "entity_type": "process",
                                "domains": ["deployments"],
                                "node_confidence": 0.9,
                            },
                            {
                                "entity": "database",
                                "entity_type": "system",
                                "domains": ["platform"],
                                "node_confidence": 0.6,
                            },
                        ],
                    },
                ),
                SimpleNamespace(
                    memory_id="m2",
                    finished_at=older,
                    outputs={
                        "state_changes": [
                            {
                                "entity": "deployment",
                                "change_type": "resumed",
                                "description": "Rollback path ready",
                            }
                        ],
                        "world_nodes": [
                            {
                                "entity": "release train",
                                "entity_type": "process",
                                "domains": ["deployments"],
                                "node_confidence": 0.4,
                            }
                        ],
                    },
                ),
            ]
        )
    )

    world_state = await CognitiveGoalLoopService(session, org_id="org-1")._load_world_state(
        world_change_limit=2,
        entity_limit=2,
    )

    assert world_state["last_world_model_run_at"] == finished_at.isoformat()
    assert [item["change_type"] for item in world_state["recent_changes"]] == ["delayed", "resumed"]
    assert [item["entity"] for item in world_state["highlighted_entities"]] == ["release train", "database"]
    assert world_state["highlighted_entities"][0]["node_confidence"] == 0.9
