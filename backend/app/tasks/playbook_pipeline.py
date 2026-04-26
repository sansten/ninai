"""Playbook extraction pipeline (PR4)."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from uuid import uuid4

from celery.utils.log import get_task_logger
from sqlalchemy import and_, select

from app.core.celery_app import celery_app
from app.core.database import async_session_factory, set_tenant_context
from app.models.agent_run import AgentRun
from app.models.playbook import Playbook, PlaybookScopeType
from app.tasks.async_runtime import run_async


logger = get_task_logger(__name__)


def _stable_hash(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _tokenize(*parts) -> list[str]:
    text = " ".join(str(p or "") for p in parts).lower()
    return sorted(set(re.findall(r"[a-z0-9_\-]+", text)))


def _extract_steps(outputs: dict, provenance: list[dict]) -> list[dict]:
    if isinstance(outputs.get("steps"), list) and outputs.get("steps"):
        return outputs["steps"]

    if isinstance(outputs.get("tool_calls"), list) and outputs.get("tool_calls"):
        return [
            {"action": str(step.get("tool") or step.get("name") or "tool_call"), "details": step}
            for step in outputs["tool_calls"]
            if isinstance(step, dict)
        ]

    if provenance:
        return [
            {
                "action": str(item.get("source_type") or "provenance"),
                "details": {
                    "source_id": item.get("source_id"),
                    "snippet": item.get("snippet"),
                },
            }
            for item in provenance
            if isinstance(item, dict)
        ]

    return []


async def extract_playbook_from_run(*, org_id: str, agent_run_id: str, actor_user_id: str) -> dict:
    async with async_session_factory() as db:
        async with db.begin():
            await set_tenant_context(db, actor_user_id, org_id, roles="system,org_admin", clearance_level=4)

            run = await db.get(AgentRun, agent_run_id)
            if not run or run.organization_id != org_id:
                raise ValueError(f"AgentRun {agent_run_id} not found for organization {org_id}")

            if run.status != "success":
                return {"status": "skipped", "reason": "run_not_success", "agent_run_id": agent_run_id}

            outputs = run.outputs or {}
            provenance = list(run.provenance or [])
            steps = _extract_steps(outputs, provenance)
            if not steps:
                return {"status": "skipped", "reason": "no_extractable_steps", "agent_run_id": agent_run_id}

            title = f"{run.agent_name} playbook"
            signature_payload = {
                "agent_name": run.agent_name,
                "output_keys": sorted(list(outputs.keys())),
                "tokens": _tokenize(run.agent_name, run.memory_id, outputs),
            }
            signature_hash = _stable_hash(signature_payload)

            existing_stmt = select(Playbook).where(
                and_(
                    Playbook.organization_id == org_id,
                    Playbook.signature_hash == signature_hash,
                    Playbook.scope_type == PlaybookScopeType.ORGANIZATION,
                )
            )
            existing = (await db.execute(existing_stmt)).scalar_one_or_none()

            if existing:
                evidence = dict(existing.evidence or {})
                run_ids = list(evidence.get("agent_run_ids", []))
                if agent_run_id not in run_ids:
                    run_ids.append(agent_run_id)
                evidence["agent_run_ids"] = run_ids
                evidence["memory_ids"] = sorted(set(list(evidence.get("memory_ids", [])) + [run.memory_id]))
                evidence_count = max(1, len(run_ids))
                existing.success_rate = ((existing.success_rate * (evidence_count - 1)) + 1.0) / evidence_count
                existing.steps = steps
                existing.constraints = {
                    "warnings": run.warnings or [],
                    "agent_version": run.agent_version,
                }
                existing.evidence = evidence
                await db.flush()
                return {"status": "updated", "playbook_id": existing.id, "agent_run_id": agent_run_id}

            playbook = Playbook(
                id=str(uuid4()),
                organization_id=org_id,
                scope_type=PlaybookScopeType.ORGANIZATION,
                scope_id=None,
                title=title,
                problem_signature=signature_payload,
                signature_hash=signature_hash,
                steps=steps,
                constraints={"warnings": run.warnings or [], "agent_version": run.agent_version},
                success_rate=1.0,
                evidence={"agent_run_ids": [agent_run_id], "memory_ids": [run.memory_id]},
            )
            db.add(playbook)
            await db.flush()
            return {"status": "created", "playbook_id": playbook.id, "agent_run_id": agent_run_id}


@celery_app.task(bind=True, max_retries=5, autoretry_for=(Exception,), dont_autoretry_for=(ValueError,), retry_backoff=True)
def playbook_extractor_task(
    self,
    org_id: str,
    agent_run_id: str,
    initiator_user_id: str | None = None,
):
    actor_user_id = initiator_user_id or "00000000-0000-0000-0000-000000000001"
    result = run_async(extract_playbook_from_run(org_id=org_id, agent_run_id=agent_run_id, actor_user_id=actor_user_id))
    return {"status": "ok", "org_id": org_id, "agent_run_id": agent_run_id, "result": result}


def enqueue_playbook_extraction(*, org_id: str, agent_run_id: str, initiator_user_id: str | None = None):
    broker = celery_app.conf.broker_url
    if not broker or str(broker).startswith("memory://"):
        return None

    return playbook_extractor_task.si(
        org_id=org_id,
        agent_run_id=agent_run_id,
        initiator_user_id=initiator_user_id,
    ).apply_async()
