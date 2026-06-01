"""Tests for LoCoMo retrieval quality fixes F2 (date header), F3 (speaker attribution), F4 (unanswerable)."""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import app.services.memory_service as memory_service_module
from app.services.cognitive_gateway_service import _detect_speaker_mismatch, _prepare_read_candidates
from app.schemas.memory import MemorySearchRequest
from app.services.memory_service import MemoryService


def _make_svc(memories: list, memory_ids: list[str] | None = None) -> MemoryService:
    ids = memory_ids if memory_ids is not None else [m.id for m in memories]

    def _execute_result_with_scalars(rows):
        result = MagicMock()
        scalars = MagicMock()
        scalars.all.return_value = rows
        result.scalars.return_value = scalars
        return result

    async def _execute(stmt):
        return _execute_result_with_scalars(memories)

    session = AsyncMock()
    session.execute = _execute

    svc = MemoryService(session=session, user_id="u1", org_id="org1", clearance_level=0)
    svc.permission_checker.filter_memory_ids_with_access = AsyncMock(return_value=ids)
    svc.audit_service.log_memory_access = AsyncMock()
    return svc


def _qdrant_hit(memory_id: str, score: float = 1.0):
    return {"id": f"v_{memory_id}", "score": score, "payload": {"memory_id": memory_id}}


# ---------------------------------------------------------------------------
# F2 — Date header injection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_date_header_prepended_when_occurred_at_set(monkeypatch):
    """occurred_at date is prepended to content_preview in search results."""
    occ = datetime(2023, 5, 8, tzinfo=timezone.utc)
    mem = SimpleNamespace(
        id="m1",
        is_active=True,
        content_preview="[Caroline] I went to a LGBTQ support group yesterday.",
        occurred_at=occ,
        source_type=None,
        source_id=None,
        title=None,
        content_hash="h1",
        vector_id="v1",
        embedding_model="e",
        scope="personal",
        scope_id=None,
        classification="internal",
        created_at=occ,
        updated_at=occ,
    )

    monkeypatch.setattr(
        memory_service_module.QdrantService, "search", AsyncMock(return_value=[_qdrant_hit("m1")])
    )

    svc = _make_svc([mem])
    req = MemorySearchRequest(query="When did Caroline visit the support group?", limit=5)
    results = await svc.search_memories(query_embedding=[0.0] * 3, request=req)

    assert len(results) == 1
    assert results[0].content_preview.startswith("[2023-05-08]")
    assert "[Caroline]" in results[0].content_preview


@pytest.mark.asyncio
async def test_date_header_not_duplicated(monkeypatch):
    """If content_preview already starts with the date header, it is not prepended again."""
    occ = datetime(2023, 5, 8, tzinfo=timezone.utc)
    mem = SimpleNamespace(
        id="m1",
        is_active=True,
        content_preview="[2023-05-08] [Caroline] Already has a date.",
        occurred_at=occ,
        source_type=None,
        source_id=None,
        title=None,
        content_hash="h1",
        vector_id="v1",
        embedding_model="e",
        scope="personal",
        scope_id=None,
        classification="internal",
        created_at=occ,
        updated_at=occ,
    )

    monkeypatch.setattr(
        memory_service_module.QdrantService, "search", AsyncMock(return_value=[_qdrant_hit("m1")])
    )

    svc = _make_svc([mem])
    req = MemorySearchRequest(query="Caroline", limit=5)
    results = await svc.search_memories(query_embedding=[0.0] * 3, request=req)

    assert results[0].content_preview.count("[2023-05-08]") == 1


@pytest.mark.asyncio
async def test_no_date_header_when_occurred_at_is_none(monkeypatch):
    """Memories without occurred_at are returned without a date prefix."""
    mem = SimpleNamespace(
        id="m1",
        is_active=True,
        content_preview="[Caroline] Some content.",
        occurred_at=None,
        source_type=None,
        source_id=None,
        title=None,
        content_hash="h1",
        vector_id="v1",
        embedding_model="e",
        scope="personal",
        scope_id=None,
        classification="internal",
        created_at=None,
        updated_at=None,
    )

    monkeypatch.setattr(
        memory_service_module.QdrantService, "search", AsyncMock(return_value=[_qdrant_hit("m1")])
    )

    svc = _make_svc([mem])
    req = MemorySearchRequest(query="Caroline", limit=5)
    results = await svc.search_memories(query_embedding=[0.0] * 3, request=req)

    assert results[0].content_preview == "[Caroline] Some content."


# ---------------------------------------------------------------------------
# F3 — Speaker attribution re-ranking helpers
# ---------------------------------------------------------------------------


def test_extract_query_subject_name_finds_proper_name():
    assert MemoryService._extract_query_subject_name("What is Caroline's job?") == "Caroline"


def test_extract_query_subject_name_skips_question_words():
    assert MemoryService._extract_query_subject_name("What did she do?") is None


def test_extract_query_subject_name_returns_none_for_lowercase_query():
    assert MemoryService._extract_query_subject_name("hello world") is None


def test_speaker_attribution_multiplier_boosts_matching_speaker():
    mult = MemoryService._speaker_attribution_multiplier("[Caroline] I am single.", "Caroline")
    assert mult > 1.0


def test_speaker_attribution_multiplier_discounts_different_speaker():
    mult = MemoryService._speaker_attribution_multiplier("[Melanie] I ran a charity race.", "Caroline")
    assert mult < 1.0


def test_speaker_attribution_multiplier_neutral_for_derived_prefix():
    mult = MemoryService._speaker_attribution_multiplier("[Fact] Caroline origin: Sweden", "Caroline")
    assert mult == 1.0


def test_speaker_attribution_multiplier_neutral_for_no_bracket():
    mult = MemoryService._speaker_attribution_multiplier("Some plain text.", "Caroline")
    assert mult == 1.0


def test_speaker_attribution_multiplier_case_insensitive():
    mult = MemoryService._speaker_attribution_multiplier("[caroline] I am single.", "Caroline")
    assert mult > 1.0


# ---------------------------------------------------------------------------
# F3 — End-to-end speaker re-ranking in search_memories
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_speaker_attribution_boosts_matching_speaker_in_results(monkeypatch):
    """Memory from the query subject speaker is ranked above an equal-score other-speaker memory."""
    monkeypatch.setattr(memory_service_module.settings, "SEARCH_HEURISTICS_ENABLED", True, raising=False)

    caroline_mem = SimpleNamespace(
        id="caroline",
        is_active=True,
        content_preview="[Caroline] It'll be tough as a single parent.",
        occurred_at=None,
        source_type=None,
        source_id=None,
        title=None,
        content_hash="h1",
        vector_id="vc",
        embedding_model="e",
        scope="personal",
        scope_id=None,
        classification="internal",
        created_at=None,
        updated_at=None,
    )
    melanie_mem = SimpleNamespace(
        id="melanie",
        is_active=True,
        content_preview="[Melanie] I ran a charity race last Saturday.",
        occurred_at=None,
        source_type=None,
        source_id=None,
        title=None,
        content_hash="h2",
        vector_id="vm",
        embedding_model="e",
        scope="personal",
        scope_id=None,
        classification="internal",
        created_at=None,
        updated_at=None,
    )

    # Both get equal vector score — speaker boost decides the ranking.
    monkeypatch.setattr(
        memory_service_module.QdrantService,
        "search",
        AsyncMock(return_value=[_qdrant_hit("caroline", 1.0), _qdrant_hit("melanie", 1.0)]),
    )

    svc = _make_svc([caroline_mem, melanie_mem])
    req = MemorySearchRequest(query="What is Caroline's relationship status?", limit=10)
    results = await svc.search_memories(query_embedding=[0.0] * 3, request=req)

    ids = [r.id for r in results]
    assert ids[0] == "caroline", f"Expected caroline first, got {ids}"
    assert ids[1] == "melanie"


@pytest.mark.asyncio
async def test_speaker_attribution_no_subject_leaves_order_unchanged(monkeypatch):
    """When no proper name is in the query, speaker boost is not applied."""
    monkeypatch.setattr(memory_service_module.settings, "SEARCH_HEURISTICS_ENABLED", True, raising=False)

    mem_a = SimpleNamespace(
        id="a",
        is_active=True,
        content_preview="[Caroline] Something happened.",
        occurred_at=None,
        source_type=None,
        source_id=None,
        title=None,
        content_hash="ha",
        vector_id="va",
        embedding_model="e",
        scope="personal",
        scope_id=None,
        classification="internal",
        created_at=None,
        updated_at=None,
    )
    mem_b = SimpleNamespace(
        id="b",
        is_active=True,
        content_preview="[Melanie] Something else.",
        occurred_at=None,
        source_type=None,
        source_id=None,
        title=None,
        content_hash="hb",
        vector_id="vb",
        embedding_model="e",
        scope="personal",
        scope_id=None,
        classification="internal",
        created_at=None,
        updated_at=None,
    )

    monkeypatch.setattr(
        memory_service_module.QdrantService,
        "search",
        AsyncMock(return_value=[_qdrant_hit("a", 1.0), _qdrant_hit("b", 0.8)]),
    )

    svc = _make_svc([mem_a, mem_b])
    req = MemorySearchRequest(query="what happened yesterday?", limit=10)
    results = await svc.search_memories(query_embedding=[0.0] * 3, request=req)

    assert results[0].id == "a"


# ---------------------------------------------------------------------------
# F4 — Speaker mismatch detection (unanswerable questions)
# ---------------------------------------------------------------------------


def test_detect_speaker_mismatch_returns_true_when_all_from_wrong_speaker():
    mems = [
        {"content_preview": "[Melanie] Thanks, Caroline! That was really thought-provoking."},
        {"content_preview": "[Melanie] I ran a charity race last Saturday."},
    ]
    assert _detect_speaker_mismatch("What did Caroline realize after her charity race?", mems) is True


def test_detect_speaker_mismatch_returns_false_when_subject_is_speaker():
    mems = [
        {"content_preview": "[Caroline] It's tough being a single parent."},
    ]
    assert _detect_speaker_mismatch("What is Caroline's relationship status?", mems) is False


def test_detect_speaker_mismatch_returns_false_with_no_proper_name_in_query():
    mems = [
        {"content_preview": "[Melanie] I ran a charity race."},
    ]
    assert _detect_speaker_mismatch("what did she do last weekend?", mems) is False


def test_detect_speaker_mismatch_neutral_for_derived_prefixes():
    mems = [
        {"content_preview": "[Fact] Caroline relationship_status: single"},
        {"content_preview": "[Temporal] Caroline ran on 2023-05-07"},
    ]
    # All candidates are derived prefixes (not person speaker turns) → no mismatch signal
    assert _detect_speaker_mismatch("What is Caroline's relationship status?", mems) is False


def test_detect_speaker_mismatch_empty_memories_returns_false():
    assert _detect_speaker_mismatch("What did Caroline do?", []) is False


def test_detect_speaker_mismatch_mixed_speakers_not_all_mismatch():
    mems = [
        {"content_preview": "[Caroline] I'll be a single parent."},
        {"content_preview": "[Melanie] I ran a charity race."},
    ]
    # One matches → not a full mismatch
    assert _detect_speaker_mismatch("What is Caroline's status?", mems) is False


def test_detect_speaker_mismatch_false_when_other_speaker_memory_is_about_subject():
    mems = [
        {"content_preview": "[Melanie] Caroline realized that self-care is important after the charity race."},
        {"content_preview": "[Melanie] Caroline said she needed more me-time."},
    ]
    assert _detect_speaker_mismatch("What did Caroline realize after her charity race?", mems) is False


def test_prepare_read_candidates_prefers_subject_focused_memory_when_scores_tie():
    ranked = _prepare_read_candidates(
        [
            {
                "id": "m1",
                "score": 0.91,
                "content_preview": "[Caroline] I'm looking into summer ideas.",
            },
            {
                "id": "m2",
                "score": 0.91,
                "content_preview": "[Melanie] I'm researching adoption agencies this summer.",
            },
        ],
        "What are Melanie's plans for the summer with respect to adoption?",
        requester=None,
        limit=5,
    )

    assert [item["id"] for item in ranked[:2]] == ["m2", "m1"]
