from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.topic_service import (
    TopicService,
    extract_topics,
    normalize_topic_label,
)


def test_normalize_topic_label_basic():
    assert normalize_topic_label(" Billing ") == "billing"
    assert normalize_topic_label("Security & Auth") == "security_auth"
    assert normalize_topic_label("__weird__") == "weird"


def test_extract_topics_includes_primary_and_dedupes():
    extracted = extract_topics({"topics": ["Billing", "billing", ""], "primary_topic": "Billing", "confidence": 0.9})
    assert extracted is not None
    assert extracted.primary_topic == "billing"
    assert extracted.topics[0] == "billing"
    assert len(extracted.topics) == 1


@pytest.mark.asyncio
async def test_topic_service_upsert_executes_upserts_and_flushes():
    session = AsyncMock()
    # For each topic: one execute returning topic id, then one execute for membership.
    session.execute = AsyncMock(
        side_effect=[
            SimpleNamespace(scalar_one=lambda: "topic-1"),
            SimpleNamespace(),
            SimpleNamespace(scalar_one=lambda: "topic-2"),
            SimpleNamespace(),
        ]
    )
    session.flush = AsyncMock()

    svc = TopicService(session)
    res = await svc.upsert_topics_for_memory(
        organization_id="org",
        memory_id="mem",
        scope="personal",
        scope_id=None,
        outputs={"topics": ["billing", "security"], "primary_topic": "billing", "confidence": 0.7},
        created_by="agent",
    )

    assert res["topics_upserted"] == 2
    assert res["memberships_upserted"] == 2
    assert session.execute.call_count == 4
    assert session.flush.call_count == 1
