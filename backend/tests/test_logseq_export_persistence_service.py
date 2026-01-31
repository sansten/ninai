from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.logseq_export_persistence_service import (
    LogseqExportPersistenceService,
    extract_logseq_export,
)


def test_extract_logseq_export_validates_types():
    assert extract_logseq_export({"markdown": "# x", "graph": {}, "item_count": 1}) is not None
    assert extract_logseq_export({"markdown": 123, "graph": {}}) is None
    assert extract_logseq_export({"markdown": "x", "graph": "nope"}) is None


@pytest.mark.asyncio
async def test_logseq_export_persistence_skips_without_actor():
    session = AsyncMock()
    svc = LogseqExportPersistenceService(session)

    res = await svc.upsert_export_for_memory(
        organization_id="org",
        memory_id="mem",
        outputs={"markdown": "# x", "graph": {}, "item_count": 1},
        updated_by_user_id=None,
        agent_version="v1",
        trace_id="t",
    )

    assert res["persisted"] is False
    assert res["reason"] == "no_actor"
    assert session.execute.call_count == 0


@pytest.mark.asyncio
async def test_logseq_export_persistence_skips_when_not_admin(monkeypatch):
    session = AsyncMock()
    session.get = AsyncMock(return_value=SimpleNamespace(is_superuser=False))

    svc = LogseqExportPersistenceService(session)

    async def _no_admin(**_kwargs):
        return False

    monkeypatch.setattr(svc, "_is_org_admin_or_system_admin", _no_admin)

    res = await svc.upsert_export_for_memory(
        organization_id="org",
        memory_id="mem",
        outputs={"markdown": "# x", "graph": {}, "item_count": 1},
        updated_by_user_id="u",
        agent_version="v1",
        trace_id="t",
    )

    assert res["persisted"] is False
    assert res["reason"] == "not_admin"
    assert session.execute.call_count == 0


@pytest.mark.asyncio
async def test_logseq_export_persistence_upserts_when_admin(monkeypatch):
    session = AsyncMock()
    session.execute = AsyncMock(return_value=SimpleNamespace())
    session.flush = AsyncMock()
    session.get = AsyncMock(return_value=SimpleNamespace(is_superuser=False))

    svc = LogseqExportPersistenceService(session)

    async def _yes_admin(**_kwargs):
        return True

    monkeypatch.setattr(svc, "_is_org_admin_or_system_admin", _yes_admin)

    res = await svc.upsert_export_for_memory(
        organization_id="org",
        memory_id="mem",
        outputs={"markdown": "# x", "graph": {"nodes": [], "edges": []}, "item_count": 1},
        updated_by_user_id="u",
        agent_version="v1",
        trace_id="t",
    )

    assert res["persisted"] is True
    assert session.execute.call_count == 1
    assert session.flush.call_count == 1
