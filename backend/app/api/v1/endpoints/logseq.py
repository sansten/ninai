"""Logseq Integration Endpoints.

Implements:
- Markdown export (read-only for authenticated users)
- Graph visualization data (read-only for authenticated users)
- Admin-only write-to-disk export
"""

from __future__ import annotations

from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path
import io
import zipfile
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from starlette.requests import Request
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import (
    TenantContext,
    get_tenant_context,
    require_org_admin,
)
from app.schemas.logseq import (
    LogseqExportRequest,
    LogseqExportResponse,
    LogseqExportConfig,
    LogseqExportConfigResponse,
    LogseqExportConfigUpdate,
    LogseqExportFileLogEntry,
    LogseqGraphResponse,
    LogseqWriteExportResponse,
)
from app.schemas.base import PaginatedResponse
from app.models.logseq_export_file import LogseqExportFile
from app.models.memory_edge import MemoryEdge
from app.services.memory_service import MemoryService
from app.services.short_term_memory import ShortTermMemoryService
from app.services.logseq_service import (
    ExportableMemory,
    build_logseq_graph,
    render_logseq_markdown,
    write_markdown_export,
    write_logseq_vault_pages,
)
from app.services.logseq_export_file_service import (
    LogseqExportFileRecord,
    LogseqExportFileService,
)
from app.services.org_logseq_export_config_service import OrgLogseqExportConfigService
from app.services.audit_service import AuditService


router = APIRouter()


async def _effective_export_base_dir(db: AsyncSession, *, organization_id: str) -> tuple[str, dict]:
    svc = OrgLogseqExportConfigService(db)
    cfg = await svc.get_config(organization_id=organization_id)
    override = getattr(cfg, "export_base_dir", None) if cfg is not None else None
    if isinstance(override, str) and override.strip():
        v = override.strip()
        return v, {"export_base_dir": v}
    return str(getattr(settings, "LOGSEQ_EXPORT_DIR", None) or "exports/logseq"), {}


async def _effective_org_export_dir(db: AsyncSession, *, organization_id: str) -> tuple[Path, dict]:
    base, overrides = await _effective_export_base_dir(db, organization_id=organization_id)
    return Path(base) / organization_id, overrides


def _zip_bytes_from_dir(root: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        if root.exists():
            for p in root.rglob("*"):
                if p.is_dir():
                    continue
                arcname = str(p.relative_to(root).as_posix())
                zf.write(p, arcname=arcname)
    return buf.getvalue()


@asynccontextmanager
async def _maybe_begin(session: AsyncSession):
    """Best-effort transaction wrapper.

    `set_tenant_context` uses `SET LOCAL`, which requires a transaction.
    In unit tests, `session` is often an AsyncMock, so we tolerate missing
    async context manager support.
    """

    try:
        async with session.begin():
            yield
    except TypeError:
        yield


def _to_exportable_from_ltm(m) -> ExportableMemory:
    return ExportableMemory(
        id=m.id,
        title=getattr(m, "title", None),
        content_preview=getattr(m, "content_preview", "") or "",
        created_at=getattr(m, "created_at", None),
        scope=getattr(m, "scope", None),
        classification=getattr(m, "classification", None),
        tags=list(getattr(m, "tags", []) or []),
        entities=dict(getattr(m, "entities", {}) or {}),
    )


def _to_exportable_from_stm(m) -> ExportableMemory:
    return ExportableMemory(
        id=m.id,
        title=getattr(m, "title", None),
        content_preview=getattr(m, "content", "") or "",
        created_at=None,
        scope=getattr(m, "scope", None),
        classification=None,
        tags=list(getattr(m, "tags", []) or []),
        entities=dict(getattr(m, "entities", {}) or {}),
    )


def get_memory_service(
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> MemoryService:
    return MemoryService(
        session=db,
        user_id=tenant.user_id,
        org_id=tenant.org_id,
        clearance_level=tenant.clearance_level,
    )


@router.post("/export/markdown", response_model=LogseqExportResponse)
async def export_markdown(
    body: LogseqExportRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
    memory_service: MemoryService = Depends(get_memory_service),
):
    """Render a Logseq-friendly Markdown export.

    Read-only: any authenticated user can generate the Markdown.

    Notes:
    - If `memory_ids` is provided, we fetch those memories (and optionally STM).
    - If `memory_ids` is empty, we export up to `limit` LTM memories visible to the caller.
    """

    async with _maybe_begin(db):
        await set_tenant_context(
            db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
        )

    export_items: list[ExportableMemory] = []

    if body.memory_ids:
        # Fetch specific LTM memories with full permission checks.
        for memory_id in body.memory_ids:
            try:
                m = await memory_service.get_memory(memory_id)
            except PermissionError as e:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
            if m is not None:
                export_items.append(_to_exportable_from_ltm(m))

        if body.include_short_term:
            stm_service = ShortTermMemoryService(tenant.user_id, tenant.org_id)
            for stm_id in body.memory_ids:
                stm = await stm_service.get(stm_id)
                if stm is not None:
                    export_items.append(_to_exportable_from_stm(stm))
    else:
        # Fallback: export a slice of LTM visible to caller.
        page_size = max(0, body.limit)
        if page_size == 0:
            export_items = []
        else:
            items, _total, _has_more = await memory_service.list_memories(
                scope=body.scope,
                tags=None,
                memory_type=None,
                page=1,
                page_size=page_size,
            )
            export_items = [_to_exportable_from_ltm(m) for m in items]

    title = f"NinaivOS Logseq Export ({tenant.org_id})"
    markdown, count = render_logseq_markdown(export_items, title=title)
    return LogseqExportResponse(markdown=markdown, item_count=count)


@router.post("/graph", response_model=LogseqGraphResponse)
async def graph_data(
    body: LogseqExportRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
    memory_service: MemoryService = Depends(get_memory_service),
):
    """Return nodes/edges representing memories, tags, and entities.

    Read-only: any authenticated user can retrieve the graph for their visible memories.
    """

    async with _maybe_begin(db):
        await set_tenant_context(
            db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
        )

    export_items: list[ExportableMemory] = []

    if body.memory_ids:
        for memory_id in body.memory_ids:
            try:
                m = await memory_service.get_memory(memory_id)
            except PermissionError as e:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
            if m is not None:
                export_items.append(_to_exportable_from_ltm(m))
    else:
        page_size = max(0, body.limit)
        if page_size:
            items, _total, _has_more = await memory_service.list_memories(
                scope=body.scope,
                tags=None,
                memory_type=None,
                page=1,
                page_size=page_size,
            )
            export_items = [_to_exportable_from_ltm(m) for m in items]

    graph = build_logseq_graph(export_items)
    return LogseqGraphResponse(nodes=graph["nodes"], edges=graph["edges"])


@router.post("/export/write", response_model=LogseqWriteExportResponse)
async def write_export_to_disk(
    body: LogseqExportRequest,
    request: Request,
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
    memory_service: MemoryService = Depends(get_memory_service),
):
    """Admin-only: write a Markdown export file to disk.

    This is restricted to org_admin/system_admin roles.
    """

    request_id = getattr(getattr(request, "state", None), "request_id", None)

    async with _maybe_begin(db):
        await set_tenant_context(
            db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
        )

    # Reuse the same export logic by calling export_markdown-like behavior.
    export_items: list[ExportableMemory] = []

    if body.memory_ids:
        for memory_id in body.memory_ids:
            try:
                m = await memory_service.get_memory(memory_id)
            except PermissionError as e:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
            if m is not None:
                export_items.append(_to_exportable_from_ltm(m))
    else:
        page_size = max(0, body.limit)
        if page_size:
            items, _total, _has_more = await memory_service.list_memories(
                scope=body.scope,
                tags=None,
                memory_type=None,
                page=1,
                page_size=page_size,
            )
            export_items = [_to_exportable_from_ltm(m) for m in items]

    title = f"NinaivOS Logseq Export ({tenant.org_id})"
    export_dir, _overrides = await _effective_org_export_dir(db, organization_id=tenant.org_id)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Mode selection: request override > env/default.
    mode = body.export_mode or str(getattr(settings, "LOGSEQ_EXPORT_MODE", "") or "").strip() or "single_file"

    if mode == "vault_pages":
        # Build memory-to-memory links from MemoryEdge for exported memories.
        exported_ids = {m.id for m in export_items}
        outgoing: dict[str, set[str]] = {mid: set() for mid in exported_ids}
        backlinks: dict[str, set[str]] = {mid: set() for mid in exported_ids}

        if exported_ids:
            q = (
                select(MemoryEdge)
                .where(MemoryEdge.organization_id == tenant.org_id)
                .where(MemoryEdge.memory_id.in_(list(exported_ids)))
            )
            res = await db.execute(q)
            rows = list(res.scalars().all())
            for e in rows:
                to_node = getattr(e, "to_node", "") or ""
                if not to_node.startswith("memory:"):
                    continue
                target_id = to_node.split(":", 1)[1].strip()
                if not target_id or target_id not in exported_ids:
                    continue
                outgoing[str(e.memory_id)].add(target_id)
                backlinks[target_id].add(str(e.memory_id))

        absolute_path, bytes_written = write_logseq_vault_pages(
            export_dir=export_dir,
            organization_id=tenant.org_id,
            memories=export_items,
            outgoing_links=outgoing,
            backlinks=backlinks,
            stamp=stamp,
        )
        # For vault mode, return the run marker file path.
        rel = str((Path(tenant.org_id) / "export_runs" / Path(absolute_path).name).as_posix())
    else:
        markdown, _count = render_logseq_markdown(export_items, title=title)
        filename = f"logseq_export_{stamp}.md"

        absolute_path, bytes_written = write_markdown_export(
            markdown,
            export_dir=export_dir,
            filename=filename,
        )

        # Return a relative-ish path for UI display. Avoid leaking full machine paths.
        rel = str((Path(tenant.org_id) / Path(absolute_path).name).as_posix())

    record_svc = LogseqExportFileService(db)
    await record_svc.record_export(
        organization_id=tenant.org_id,
        record=LogseqExportFileRecord(
            relative_path=rel,
            bytes_written=bytes_written,
            options={
                "memory_ids": body.memory_ids,
                "include_short_term": body.include_short_term,
                "scope": body.scope,
                "limit": body.limit,
                "export_mode": mode,
            },
        ),
        requested_by_user_id=tenant.user_id,
        trace_id=request_id,
    )

    audit = AuditService(db)
    await audit.log_event(
        event_type="logseq.export_written",
        actor_id=tenant.user_id,
        organization_id=tenant.org_id,
        resource_type="logseq_export_file",
        success=True,
        request_id=request_id,
        details={
            "relative_path": rel,
            "bytes_written": bytes_written,
            "options": {
                "memory_ids": body.memory_ids,
                "include_short_term": body.include_short_term,
                "scope": body.scope,
                "limit": body.limit,
                "export_mode": mode,
            },
        },
    )

    return LogseqWriteExportResponse(relative_path=rel, bytes_written=bytes_written)


@router.get("/export/config", response_model=LogseqExportConfigResponse)
async def get_export_config(
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    """Admin-only: view effective Logseq export path configuration."""

    async with _maybe_begin(db):
        await set_tenant_context(
            db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
        )

        svc = OrgLogseqExportConfigService(db)
        row = await svc.get_config(organization_id=tenant.org_id)
        base, overrides = await _effective_export_base_dir(db, organization_id=tenant.org_id)

    effective = LogseqExportConfig(
        export_base_dir=str(base),
        org_export_dir=str((Path(base) / tenant.org_id).as_posix()),
        last_nightly_export_at=getattr(row, "last_nightly_export_at", None) if row is not None else None,
    )
    return LogseqExportConfigResponse(effective=effective, overrides=overrides)


@router.put("/export/config", response_model=LogseqExportConfigResponse)
async def put_export_config(
    body: LogseqExportConfigUpdate,
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    """Admin-only: update Logseq export base directory override.

    Send {"export_base_dir": null} to revert to env/default.
    """

    async with _maybe_begin(db):
        await set_tenant_context(
            db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
        )

        patch = body.model_dump(exclude_unset=True)
        export_base_dir = patch.get("export_base_dir") if "export_base_dir" in patch else None

        svc = OrgLogseqExportConfigService(db)
        await svc.upsert_config(
            organization_id=tenant.org_id,
            export_base_dir=export_base_dir,
            updated_by_user_id=tenant.user_id,
        )
        await db.commit()

    # Return effective after commit
    async with _maybe_begin(db):
        await set_tenant_context(
            db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
        )
        row = await OrgLogseqExportConfigService(db).get_config(organization_id=tenant.org_id)
        base, overrides = await _effective_export_base_dir(db, organization_id=tenant.org_id)

    effective = LogseqExportConfig(
        export_base_dir=str(base),
        org_export_dir=str((Path(base) / tenant.org_id).as_posix()),
        last_nightly_export_at=getattr(row, "last_nightly_export_at", None) if row is not None else None,
    )
    return LogseqExportConfigResponse(effective=effective, overrides=overrides)


@router.get("/export/zip")
async def download_export_zip(
    request: Request,
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    """Admin-only: download a zip of the org's Logseq export directory."""

    request_id = getattr(getattr(request, "state", None), "request_id", None)

    async with _maybe_begin(db):
        await set_tenant_context(
            db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
        )

    export_dir, _overrides = await _effective_org_export_dir(db, organization_id=tenant.org_id)
    if not export_dir.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No export directory found")

    data = _zip_bytes_from_dir(export_dir)

    audit = AuditService(db)
    await audit.log_event(
        event_type="logseq.export_zip_downloaded",
        actor_id=tenant.user_id,
        organization_id=tenant.org_id,
        resource_type="logseq_export_zip",
        success=True,
        request_id=request_id,
        details={"org_export_dir": str(export_dir.as_posix()), "bytes": len(data)},
    )

    from fastapi.responses import Response

    filename = f"logseq_exports_{tenant.org_id}.zip"
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=\"{filename}\""},
    )


@router.get("/export/logs", response_model=PaginatedResponse)
async def list_export_logs(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    """Admin-only: list Logseq write-to-disk export logs."""

    async with _maybe_begin(db):
        await set_tenant_context(
            db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
        )

        count_query = select(func.count()).select_from(LogseqExportFile).where(
            LogseqExportFile.organization_id == tenant.org_id
        )
        total_result = await db.execute(count_query)
        total = int(total_result.scalar() or 0)

        offset = (page - 1) * page_size
        query = (
            select(LogseqExportFile)
            .where(LogseqExportFile.organization_id == tenant.org_id)
            .order_by(desc(LogseqExportFile.created_at))
            .offset(offset)
            .limit(page_size)
        )

        result = await db.execute(query)
        rows = result.scalars().all()

    return PaginatedResponse(
        items=[LogseqExportFileLogEntry.model_validate(r) for r in rows],
        total=total,
        page=page,
        page_size=page_size,
        pages=(total + page_size - 1) // page_size if total > 0 else 0,
    )
