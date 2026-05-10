from __future__ import annotations

import pytest

from app.services.temporal_reasoning_service import TemporalReasoningService


@pytest.mark.asyncio
async def test_temporal_reasoning_service_builds_chronological_timeline():
    svc = TemporalReasoningService()

    result = await svc.temporal_query(
        org_id="org-1",
        query_type="timeline_of_memories",
        query="When did the outage resolve?",
        extracted_entities=["outage"],
        candidate_memories=[
            {
                "id": "m3",
                "title": "Postmortem",
                "content_preview": "The outage review happened after recovery.",
                "occurred_at": "2026-05-03T10:00:00+00:00",
                "score": 0.7,
            },
            {
                "id": "m1",
                "title": "Outage detected",
                "content_preview": "The outage started in the morning.",
                "occurred_at": "2026-05-01T09:00:00+00:00",
                "score": 0.8,
            },
            {
                "id": "m2",
                "title": "Outage resolved",
                "content_preview": "The outage was resolved by noon.",
                "occurred_at": "2026-05-02T12:00:00+00:00",
                "score": 0.9,
            },
        ],
    )

    assert result["ordering"] == "chronological"
    assert result["anchor_count"] == 3
    assert [item["memory_id"] for item in result["timeline"]] == ["m1", "m2", "m3"]
    assert result["earliest"] == "2026-05-01T09:00:00+00:00"
    assert result["latest"] == "2026-05-03T10:00:00+00:00"
