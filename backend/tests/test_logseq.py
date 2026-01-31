from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import create_access_token
from app.core.database import get_db
from app.main import app
from app.models.logseq_export_file import LogseqExportFile
from app.services.logseq_service import (
    ExportableMemory,
    build_logseq_graph,
    render_logseq_markdown,
    write_logseq_vault_pages,
)


def test_render_logseq_markdown_basic():
    memories = [
        ExportableMemory(
            id="m1",
            title="Hello",
            content_preview="Preview text",
            created_at=None,
            scope="personal",
            classification="internal",
            tags=["Support", "Billing"],
            entities={"person": ["Alice"], "order_id": "123"},
        )
    ]

    md, count = render_logseq_markdown(memories, title="Test Export")
    assert count == 1
    assert "# Test Export" in md
    assert "id:: m1" in md
    assert "tags:: Support, Billing" in md
    assert "entities::" in md
    assert "- Preview text" in md


def test_build_logseq_graph_nodes_and_edges():
    memories = [
        ExportableMemory(
            id="m1",
            title="Hello",
            content_preview="x",
            created_at=None,
            scope="personal",
            classification="internal",
            tags=["Support"],
            entities={"person": ["Alice"]},
        )
    ]

    g = build_logseq_graph(memories)
    node_ids = {n["id"] for n in g["nodes"]}
    assert "memory:m1" in node_ids
    assert "tag:support" in node_ids
    assert "entity:person:alice" in node_ids

    edge_types = {e["type"] for e in g["edges"]}
    assert "has_tag" in edge_types
    assert "mentions" in edge_types


def test_write_logseq_vault_pages_writes_pages_and_run_meta(tmp_path):
    memories = [
        ExportableMemory(
            id="m1",
            title="Hello",
            content_preview="Preview text",
            created_at=None,
            scope="personal",
            classification="internal",
            tags=["Support"],
            entities={"person": ["Alice"]},
        ),
        ExportableMemory(
            id="m2",
            title="World",
            content_preview="More",
            created_at=None,
            scope="personal",
            classification="internal",
            tags=[],
            entities={},
        ),
    ]

    run_path, total_bytes = write_logseq_vault_pages(
        export_dir=tmp_path,
        organization_id="o1",
        memories=memories,
        outgoing_links={"m1": {"m2"}},
        backlinks={"m2": {"m1"}},
        stamp="20260122_000000",
    )

    assert total_bytes > 0
    assert (tmp_path / "pages" / "mem_m1.md").exists()
    assert (tmp_path / "pages" / "mem_m2.md").exists()
    assert (tmp_path / "export_meta.json").exists()
    assert (tmp_path / "export_runs" / "export_run_20260122_000000.json").exists()
    assert run_path.endswith("export_run_20260122_000000.json")


@pytest.mark.asyncio
async def test_logseq_write_export_requires_org_admin(client):
    member_token = create_access_token(
        user_id="u1",
        org_id="o1",
        roles=["member"],
    )
    headers = {"Authorization": f"Bearer {member_token}"}

    resp = await client.post(
        "/api/v1/logseq/export/write",
        headers=headers,
        json={"memory_ids": [], "limit": 0},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_logseq_write_export_records_file_without_fs(monkeypatch):
    session = AsyncMock(spec=AsyncSession)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    import app.api.v1.endpoints.logseq as logseq_endpoints

    def fake_write_markdown_export(markdown: str, export_dir, filename: str):
        return ("C:/tmp/export.md", 123)

    monkeypatch.setattr(logseq_endpoints, "write_markdown_export", fake_write_markdown_export)

    admin_token = create_access_token(
        user_id="u_admin",
        org_id="o1",
        roles=["org_admin"],
    )
    headers = {"Authorization": f"Bearer {admin_token}"}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post(
            "/api/v1/logseq/export/write",
            headers=headers,
            json={"memory_ids": [], "limit": 0},
        )

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["bytes_written"] == 123
    assert payload["relative_path"] == "o1/export.md"

    # Ensure the export file record insert ran.
    assert any(
        "logseq_export_files" in str(call.args[0])
        for call in session.execute.await_args_list
        if call.args
    )

    # Audit event is recorded via session.add
    assert session.add.called


@pytest.mark.asyncio
async def test_logseq_export_logs_admin_happy_path():
    session = AsyncMock(spec=AsyncSession)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    class _CountResult:
        def scalar(self):
            return 1

    class _RowsResult:
        def scalars(self):
            return self

        def all(self):
            row = LogseqExportFile(
                organization_id="o1",
                relative_path="o1/export.md",
                bytes_written=10,
                requested_by_user_id="u_admin",
                trace_id="rid",
                options={"limit": 0},
            )
            row.id = "00000000-0000-0000-0000-000000000001"
            # Make timestamps deterministic for response.
            row.created_at = datetime(2026, 1, 21, tzinfo=timezone.utc)
            row.updated_at = datetime(2026, 1, 21, tzinfo=timezone.utc)
            return [row]

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "count" in sql and "logseq_export_files" in sql:
            return _CountResult()
        if "logseq_export_files" in sql:
            return _RowsResult()
        return AsyncMock()

    session.execute = AsyncMock(side_effect=_execute)

    admin_token = create_access_token(
        user_id="u_admin",
        org_id="o1",
        roles=["org_admin"],
    )
    headers = {"Authorization": f"Bearer {admin_token}"}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/v1/logseq/export/logs", headers=headers)

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["total"] == 1
    assert len(payload["items"]) == 1
    assert payload["items"][0]["relative_path"] == "o1/export.md"


@pytest.mark.asyncio
async def test_logseq_export_config_admin_get():
    session = AsyncMock(spec=AsyncSession)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    dt = datetime(2026, 1, 21, tzinfo=timezone.utc)

    class _SelectResult:
        def scalar_one_or_none(self):
            return type(
                "Cfg",
                (),
                {
                    "export_base_dir": "C:/exports/logseq",
                    "last_nightly_export_at": dt,
                },
            )()

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "org_logseq_export_config" in sql:
            return _SelectResult()
        return AsyncMock()

    session.execute = AsyncMock(side_effect=_execute)

    admin_token = create_access_token(
        user_id="u_admin",
        org_id="o1",
        roles=["org_admin"],
    )
    headers = {"Authorization": f"Bearer {admin_token}"}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/v1/logseq/export/config", headers=headers)

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["effective"]["export_base_dir"] == "C:/exports/logseq"
    assert payload["effective"]["org_export_dir"].endswith("/o1")
    assert payload["effective"]["last_nightly_export_at"]


@pytest.mark.asyncio
async def test_logseq_export_config_admin_put_sets_override():
    session = AsyncMock(spec=AsyncSession)
    session.commit = AsyncMock()

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    class _UpsertResult:
        def scalar_one(self):
            return type("Cfg", (), {"export_base_dir": "D:/logseq_vaults"})()

    class _SelectResult:
        def scalar_one_or_none(self):
            return type(
                "Cfg",
                (),
                {
                    "export_base_dir": "D:/logseq_vaults",
                    "last_nightly_export_at": None,
                },
            )()

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "INSERT INTO org_logseq_export_config" in sql:
            return _UpsertResult()
        if "org_logseq_export_config" in sql:
            return _SelectResult()
        return AsyncMock()

    session.execute = AsyncMock(side_effect=_execute)

    admin_token = create_access_token(
        user_id="u_admin",
        org_id="o1",
        roles=["org_admin"],
    )
    headers = {"Authorization": f"Bearer {admin_token}"}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.put(
            "/api/v1/logseq/export/config",
            headers=headers,
            json={"export_base_dir": "D:/logseq_vaults"},
        )

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["effective"]["export_base_dir"] == "D:/logseq_vaults"
    assert payload["effective"]["org_export_dir"].endswith("/o1")
    assert session.commit.called


@pytest.mark.asyncio
async def test_logseq_export_zip_admin_happy_path(monkeypatch, tmp_path):
    session = AsyncMock(spec=AsyncSession)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    import app.api.v1.endpoints.logseq as logseq_endpoints

    export_dir = tmp_path / "o1"
    export_dir.mkdir(parents=True, exist_ok=True)

    async def _fake_effective_org_export_dir(db, *, organization_id: str):
        return export_dir, {}

    monkeypatch.setattr(logseq_endpoints, "_effective_org_export_dir", _fake_effective_org_export_dir)
    monkeypatch.setattr(logseq_endpoints, "_zip_bytes_from_dir", lambda p: b"zipdata")

    admin_token = create_access_token(
        user_id="u_admin",
        org_id="o1",
        roles=["org_admin"],
    )
    headers = {"Authorization": f"Bearer {admin_token}"}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/v1/logseq/export/zip", headers=headers)

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert resp.headers.get("content-type") == "application/zip"
    assert resp.content == b"zipdata"
    assert session.add.called
