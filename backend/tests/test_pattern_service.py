from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.pattern_service import (
    PatternService,
    extract_patterns,
    normalize_pattern_key,
)


def test_normalize_pattern_key():
    assert normalize_pattern_key("Issue Resolution") == "issue_resolution"
    assert normalize_pattern_key("  auth/credential issue ") == "auth_credential_issue"
    assert normalize_pattern_key("__") == ""


def test_extract_patterns_dedupes_and_clamps_confidence():
    out = extract_patterns(
        {
            "patterns": [
                {"pattern": "issue_resolution", "type": "support", "confidence": 2, "evidence": ["x"]},
                {"pattern": "issue_resolution", "type": "support", "confidence": 0.1, "evidence": []},
                {"pattern": "", "type": "x", "confidence": 0.3},
            ]
        }
    )
    assert len(out) == 1
    assert out[0].pattern_key == "issue_resolution"
    assert out[0].pattern_type == "support"
    assert out[0].confidence == 1.0


@pytest.mark.asyncio
async def test_pattern_service_upsert_executes_upserts_and_flushes():
    session = AsyncMock()
    session.execute = AsyncMock(
        side_effect=[
            SimpleNamespace(scalar_one=lambda: "pattern-1"),
            SimpleNamespace(),
        ]
    )
    session.flush = AsyncMock()

    svc = PatternService(session)
    res = await svc.upsert_patterns_for_memory(
        organization_id="org",
        memory_id="mem",
        scope="personal",
        scope_id=None,
        outputs={
            "patterns": [
                {"pattern": "issue_resolution", "type": "support", "confidence": 0.72, "evidence": ["a"]}
            ]
        },
    )

    assert res["patterns_upserted"] == 1
    assert res["evidence_upserted"] == 1
    assert session.execute.call_count == 2
    assert session.flush.call_count == 1
