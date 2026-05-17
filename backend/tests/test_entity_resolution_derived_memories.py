"""Tests for derived [Fact] memory writing when EntityResolutionAgent finds personal attributes."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.memory import MemoryMetadata
from app.services.agent_runner import AgentRunner


def _source_memory(**kwargs) -> MemoryMetadata:
    m = MagicMock(spec=MemoryMetadata)
    m.id = kwargs.get("id", "src-mem-001")
    m.organization_id = kwargs.get("organization_id", "org-001")
    m.owner_id = kwargs.get("owner_id", "user-001")
    m.scope = kwargs.get("scope", "personal")
    m.scope_id = kwargs.get("scope_id", None)
    m.classification = kwargs.get("classification", "internal")
    m.required_clearance = kwargs.get("required_clearance", 0)
    m.entities = {}
    return m


def _no_dup_session(source_mem):
    """Session where dedup query always returns None (no existing derived memory)."""
    no_dup_result = MagicMock()
    no_dup_result.scalar_one_or_none.return_value = None

    session = SimpleNamespace(
        get=AsyncMock(return_value=source_mem),
        execute=AsyncMock(return_value=no_dup_result),
        add=MagicMock(),
        flush=AsyncMock(),
    )
    return session


@pytest.mark.asyncio
async def test_personal_attribute_writes_derived_memory(monkeypatch):
    """personal_attribute entity from LLM path triggers a [Fact] derived memory row."""
    source = _source_memory()
    session = _no_dup_session(source)

    embed_calls: list[dict] = []
    monkeypatch.setattr(
        "app.tasks.embed_task.enqueue_embed_and_index",
        lambda **kw: embed_calls.append(kw),
    )

    runner = AgentRunner()
    outputs = {
        "resolved_entities": [
            {
                "entity_type": "personal_attribute",
                "canonical": "caroline_origin",
                "subject": "Caroline",
                "attribute": "origin",
                "value": "Sweden",
                "domains": ["personal"],
                "match_type": "extracted",
                "confidence": 0.8,
            }
        ]
    }

    result = await runner._apply_entity_resolution(
        session=session,
        memory_id="src-mem-001",
        outputs=outputs,
    )

    assert result["derived_facts"] == 1
    session.add.assert_called_once()

    # Verify the derived memory content
    added = session.add.call_args[0][0]
    assert "[Fact] Caroline origin: Sweden" in added.content_preview
    assert added.source_type == "derived"
    assert added.source_id == "src-mem-001"
    assert "personal_attribute" in added.tags
    assert "origin" in added.tags


@pytest.mark.asyncio
async def test_no_personal_attribute_no_derived_memory(monkeypatch):
    """Enterprise entity without personal_attribute type → no derived memory written."""
    source = _source_memory()

    dup_result = MagicMock()
    dup_result.scalar_one_or_none.return_value = None
    session = SimpleNamespace(
        get=AsyncMock(return_value=source),
        execute=AsyncMock(return_value=dup_result),
        add=MagicMock(),
        flush=AsyncMock(),
    )

    runner = AgentRunner()
    outputs = {
        "resolved_entities": [
            {
                "entity_type": "role",
                "canonical": "customer",
                "domains": ["sales"],
                "match_type": "alias",
                "confidence": 0.85,
            }
        ]
    }

    result = await runner._apply_entity_resolution(
        session=session,
        memory_id="src-mem-001",
        outputs=outputs,
    )

    assert result["derived_facts"] == 0
    session.add.assert_not_called()


@pytest.mark.asyncio
async def test_derived_memory_dedup_skips_existing(monkeypatch):
    """If a derived memory with the same content_hash already exists, skip insertion."""
    source = _source_memory()

    existing = MagicMock(spec=MemoryMetadata)
    dup_result = MagicMock()
    dup_result.scalar_one_or_none.return_value = existing

    session = SimpleNamespace(
        get=AsyncMock(return_value=source),
        execute=AsyncMock(return_value=dup_result),
        add=MagicMock(),
        flush=AsyncMock(),
    )

    embed_calls: list[dict] = []
    monkeypatch.setattr(
        "app.tasks.embed_task.enqueue_embed_and_index",
        lambda **kw: embed_calls.append(kw),
    )

    runner = AgentRunner()
    outputs = {
        "resolved_entities": [
            {
                "entity_type": "personal_attribute",
                "canonical": "caroline_origin",
                "subject": "Caroline",
                "attribute": "origin",
                "value": "Sweden",
                "domains": ["personal"],
                "match_type": "extracted",
                "confidence": 0.8,
            }
        ]
    }

    result = await runner._apply_entity_resolution(
        session=session,
        memory_id="src-mem-001",
        outputs=outputs,
    )

    assert result["derived_facts"] == 0
    session.add.assert_not_called()
    assert embed_calls == []


@pytest.mark.asyncio
async def test_multiple_attributes_write_multiple_derived_memories(monkeypatch):
    """Each distinct personal attribute produces one derived memory row."""
    source = _source_memory()
    session = _no_dup_session(source)

    embed_calls: list[dict] = []
    monkeypatch.setattr(
        "app.tasks.embed_task.enqueue_embed_and_index",
        lambda **kw: embed_calls.append(kw),
    )

    runner = AgentRunner()
    outputs = {
        "resolved_entities": [
            {
                "entity_type": "personal_attribute",
                "canonical": "caroline_origin",
                "subject": "Caroline",
                "attribute": "origin",
                "value": "Sweden",
                "domains": ["personal"],
                "match_type": "extracted",
                "confidence": 0.8,
            },
            {
                "entity_type": "personal_attribute",
                "canonical": "caroline_relationship_status",
                "subject": "Caroline",
                "attribute": "relationship_status",
                "value": "single",
                "domains": ["personal"],
                "match_type": "extracted",
                "confidence": 0.8,
            },
        ]
    }

    result = await runner._apply_entity_resolution(
        session=session,
        memory_id="src-mem-001",
        outputs=outputs,
    )

    assert result["derived_facts"] == 2
    assert session.add.call_count == 2
    assert len(embed_calls) == 2

    contents = [
        session.add.call_args_list[i][0][0].content_preview
        for i in range(2)
    ]
    assert any("origin: Sweden" in c for c in contents)
    assert any("relationship_status: single" in c for c in contents)


@pytest.mark.asyncio
async def test_missing_subject_or_value_skipped(monkeypatch):
    """Incomplete personal attribute (no subject or value) is silently skipped."""
    source = _source_memory()
    session = _no_dup_session(source)

    runner = AgentRunner()
    outputs = {
        "resolved_entities": [
            {
                "entity_type": "personal_attribute",
                "canonical": "caroline_origin",
                "subject": "Caroline",
                "attribute": "origin",
                "value": "",  # empty value
                "domains": ["personal"],
                "match_type": "extracted",
                "confidence": 0.8,
            }
        ]
    }

    result = await runner._apply_entity_resolution(
        session=session,
        memory_id="src-mem-001",
        outputs=outputs,
    )

    assert result["derived_facts"] == 0
    session.add.assert_not_called()
