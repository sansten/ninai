"""Export job tasks."""

from __future__ import annotations

import asyncio
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from celery.utils.log import get_task_logger
from sqlalchemy import select

from app.core.celery_app import celery_app
from app.core.config import settings
from app.core.database import async_session_factory, set_tenant_context
from app.models.export_job import ExportJob
from app.models.memory import MemoryMetadata
from app.services.export_job_service import ExportJobService, sha256_file


logger = get_task_logger(__name__)


def _json_default(obj):
    if isinstance(obj, datetime):
        if obj.tzinfo is None:
            return obj.replace(tzinfo=timezone.utc).isoformat()
        return obj.astimezone(timezone.utc).isoformat()
    return str(obj)


@celery_app.task(name="app.tasks.export_jobs.run_snapshot_export_job_task")
def run_snapshot_export_job_task(*, org_id: str, job_id: str, initiator_user_id: str | None = None) -> bool:
    """Generate a snapshot export bundle (metadata-only) for an organization."""

    async def _run() -> bool:
        async with async_session_factory() as session:
            async with session.begin():
                service_user_id = str(getattr(settings, "SYSTEM_TASK_USER_ID", None) or "")
                effective_user_id = str(initiator_user_id or service_user_id or "")
                effective_roles = "org_admin" if initiator_user_id else ("system_admin" if service_user_id else "")

                # Ensure RLS is scoped correctly for org reads.
                await set_tenant_context(
                    session,
                    effective_user_id,
                    org_id,
                    roles=effective_roles,
                    clearance_level=4,
                    justification="snapshot_export",
                )

                svc = ExportJobService(session)
                job = await svc.get_job(organization_id=org_id, job_id=job_id)
                if job is None:
                    logger.warning("Export job not found: %s", job_id)
                    return False

                try:
                    await svc.mark_running(job=job)

                    job_dir = ExportJobService.job_dir(organization_id=org_id, job_id=job_id)
                    job_dir.mkdir(parents=True, exist_ok=True)

                    memories_path = job_dir / "memories.jsonl"
                    meta_path = job_dir / "export_meta.json"

                    stmt = (
                        select(MemoryMetadata)
                        .where(MemoryMetadata.organization_id == org_id)
                        .order_by(MemoryMetadata.created_at.asc())
                    )
                    res = await session.execute(stmt)
                    rows = res.scalars().all()

                    with memories_path.open("w", encoding="utf-8") as f:
                        for m in rows:
                            f.write(json.dumps(m.to_dict(), ensure_ascii=False, default=_json_default))
                            f.write("\n")

                    meta = {
                        "organization_id": org_id,
                        "job_id": job_id,
                        "exported_at": datetime.now(timezone.utc).isoformat(),
                        "item_count": len(rows),
                        "format": "jsonl",
                        "notes": "Metadata-only snapshot; full memory content is not persisted in Postgres.",
                    }
                    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")

                    zip_path = ExportJobService.job_zip_path(organization_id=org_id, job_id=job_id)
                    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                        zf.write(memories_path, arcname="memories.jsonl")
                        zf.write(meta_path, arcname="export_meta.json")

                    digest = sha256_file(zip_path)
                    await svc.mark_succeeded(
                        job=job,
                        file_path=str(zip_path.as_posix()),
                        file_bytes=zip_path.stat().st_size,
                        file_sha256=digest,
                    )
                    return True
                except Exception as e:
                    logger.exception("Snapshot export failed")
                    await svc.mark_failed(job=job, error_message=str(e))
                    return False

    try:
        return asyncio.run(_run())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_run())
