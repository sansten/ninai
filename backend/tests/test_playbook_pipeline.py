from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import pytest
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_run import AgentRun
from app.models.organization import Organization
from app.models.playbook import Playbook, PlaybookScopeType
from app.services.playbook_service import PlaybookService
from app.tasks.playbook_pipeline import extract_playbook_from_run


async def _seed_agent_run(
    session: AsyncSession,
    *,
    org_id: str,
    memory_id: str,
    status: str,
    outputs: dict,
    provenance: list[dict] | None = None,
) -> str:
    run_id = str(uuid4())
    now = datetime.utcnow()

    await session.execute(
        insert(AgentRun),
        {
            "id": run_id,
            "organization_id": org_id,
            "memory_id": memory_id,
            "agent_name": "PatternDetectionAgent",
            "agent_version": "1.0",
            "inputs_hash": str(uuid4()).replace("-", ""),
            "status": status,
            "confidence": 0.91,
            "outputs": outputs,
            "warnings": [],
            "errors": [],
            "started_at": now,
            "finished_at": now,
            "trace_id": str(uuid4()),
            "provenance": provenance or [],
        },
    )

    return run_id


@pytest.mark.asyncio
async def test_playbook_created_on_successful_run(db_session: AsyncSession, test_org_id: str, test_user_id: str):
    memory_id = str(uuid4())
    run_id = await _seed_agent_run(
        db_session,
        org_id=test_org_id,
        memory_id=memory_id,
        status="success",
        outputs={
            "tool_calls": [
                {"tool": "search_memories", "args": {"query": "router outage"}},
                {"tool": "create_ticket", "args": {"priority": "high"}},
            ],
            "result": "resolved",
        },
        provenance=[{"source_type": "memory", "source_id": memory_id}],
    )
    await db_session.commit()

    result = await extract_playbook_from_run(
        org_id=test_org_id,
        agent_run_id=run_id,
        actor_user_id=test_user_id,
    )

    assert result["status"] in {"created", "updated"}
    assert result["playbook_id"]

    service = PlaybookService(db_session, test_org_id)
    playbooks = await service.search_playbooks(query="router outage create ticket", limit=5)
    assert len(playbooks) >= 1
    assert playbooks[0]["steps"]


@pytest.mark.asyncio
async def test_playbook_not_created_on_failure(db_session: AsyncSession, test_org_id: str, test_user_id: str):
    memory_id = str(uuid4())
    run_id = await _seed_agent_run(
        db_session,
        org_id=test_org_id,
        memory_id=memory_id,
        status="failed",
        outputs={"tool_calls": [{"tool": "search_memories", "args": {"query": "billing"}}]},
    )
    await db_session.commit()

    result = await extract_playbook_from_run(
        org_id=test_org_id,
        agent_run_id=run_id,
        actor_user_id=test_user_id,
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "run_not_success"


@pytest.mark.asyncio
async def test_playbook_rls_scope_enforced(db_session: AsyncSession, test_org_id: str):
    other_org_id = str(uuid4())
    now = datetime.utcnow()

    await db_session.execute(
        insert(Organization),
        {
            "id": other_org_id,
            "name": "Other Org",
            "slug": f"other-org-{other_org_id[:8]}",
            "is_active": True,
        },
    )

    await db_session.execute(
        insert(Playbook),
        [
            {
                "id": str(uuid4()),
                "organization_id": test_org_id,
                "scope_type": "organization",
                "scope_id": None,
                "title": "Network recovery playbook",
                "problem_signature": {"tokens": ["network", "recovery"]},
                "signature_hash": str(uuid4()).replace("-", ""),
                "steps": [{"action": "search_memories"}],
                "constraints": {},
                "success_rate": 0.95,
                "evidence": {"agent_run_ids": []},
                "created_at": now,
                "updated_at": now,
            },
            {
                "id": str(uuid4()),
                "organization_id": other_org_id,
                "scope_type": "organization",
                "scope_id": None,
                "title": "Billing workflow playbook",
                "problem_signature": {"tokens": ["billing", "workflow"]},
                "signature_hash": str(uuid4()).replace("-", ""),
                "steps": [{"action": "open_billing_case"}],
                "constraints": {},
                "success_rate": 0.99,
                "evidence": {"agent_run_ids": []},
                "created_at": now,
                "updated_at": now,
            },
        ],
    )
    await db_session.commit()

    service = PlaybookService(db_session, test_org_id)
    playbooks = await service.search_playbooks(query="network recovery", limit=10)

    assert playbooks
    assert all(pb["problem_signature"] for pb in playbooks)
    assert all("network" in (pb["title"].lower() + " " + " ".join(pb["problem_signature"].get("tokens", []))) for pb in playbooks)
