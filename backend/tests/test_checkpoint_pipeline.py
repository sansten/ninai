"""Checkpoint tests (PR5: Replayability)."""

import pytest
from datetime import datetime
from uuid import uuid4
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_run import AgentRun
from app.models.run_checkpoint import RunCheckpoint
from app.core.database import async_session_factory, set_tenant_context
from app.services.checkpoint_service import CheckpointService
from app.tasks.checkpoint_pipeline import persist_checkpoint, enqueue_checkpoint_persistence


@pytest.mark.asyncio
async def test_checkpoint_created_and_persisted(db_session: AsyncSession, test_org_id: str, test_user_id: str):
    """Test that checkpoints are created and stored with full snapshots."""
    await set_tenant_context(db_session, test_user_id, test_org_id, roles="system,org_admin", clearance_level=4)

    # Setup - Organization is already created by db_session fixture
    run_id = str(uuid4())
    mem_id = str(uuid4())
    run = AgentRun(
        id=run_id,
        organization_id=test_org_id,
        memory_id=mem_id,
        agent_name="agent1",
        agent_version="1.0",
        inputs_hash="hash1",
        status="success",
        confidence=0.9,
        outputs={},
        started_at=datetime.utcnow(),
        finished_at=datetime.utcnow(),
    )
    db_session.add(run)
    await db_session.commit()

    # Create checkpoint
    svc = CheckpointService(db_session, test_org_id)
    checkpoint_id = await svc.create_checkpoint(
        agent_run_id=run_id,
        step_index=0,
        input_snapshot={"query": "test query"},
        retrieval_snapshot={"ids": ["mem-1"], "scores": [0.95]},
        model_snapshot={"temperature": 0.7},
        output_snapshot={"response": "test response"},
    )

    # Verify
    result = await db_session.execute(
        select(RunCheckpoint).where(RunCheckpoint.id == checkpoint_id)
    )
    checkpoint = result.scalar_one_or_none()
    assert checkpoint is not None
    assert checkpoint.step_index == 0
    assert checkpoint.input_snapshot["query"] == "test query"
    assert checkpoint.retrieval_snapshot["ids"] == ["mem-1"]
    assert checkpoint.model_snapshot["temperature"] == 0.7
    assert checkpoint.output_snapshot["response"] == "test response"


@pytest.mark.asyncio
async def test_replay_returns_checkpoints_in_order(db_session: AsyncSession, test_org_id: str, test_user_id: str):
    """Test that replay returns correct checkpoints ordered by step."""
    await set_tenant_context(db_session, test_user_id, test_org_id, roles="system,org_admin", clearance_level=4)

    # Setup - Organization is already created by db_session fixture
    run_id = str(uuid4())
    mem_id = str(uuid4())
    run = AgentRun(
        id=run_id,
        organization_id=test_org_id,
        memory_id=mem_id,
        agent_name="agent2",
        agent_version="1.0",
        inputs_hash="hash2",
        status="success",
        confidence=0.85,
        outputs={},
        started_at=datetime.utcnow(),
        finished_at=datetime.utcnow(),
    )
    db_session.add(run)
    await db_session.commit()

    # Create multiple checkpoints
    svc = CheckpointService(db_session, test_org_id)
    for step in range(3):
        await svc.create_checkpoint(
            agent_run_id=run_id,
            step_index=step,
            input_snapshot={"step": step},
            retrieval_snapshot={},
            model_snapshot={},
            output_snapshot={"step_result": step},
        )

    # Retrieve with limit
    snapshots = await svc.get_checkpoints_up_to_step(agent_run_id=run_id, to_step=1)
    assert len(snapshots) == 2
    assert snapshots[0]["step_index"] == 0
    assert snapshots[1]["step_index"] == 1
    assert snapshots[0]["output_snapshot"]["step_result"] == 0
    assert snapshots[1]["output_snapshot"]["step_result"] == 1


@pytest.mark.asyncio
async def test_explain_retrieval_details(db_session: AsyncSession, test_org_id: str, test_user_id: str):
    """Test that retrieval explanation provides all relevant signals."""
    await set_tenant_context(db_session, test_user_id, test_org_id, roles="system,org_admin", clearance_level=4)

    # Setup - Organization is already created by db_session fixture
    run_id = str(uuid4())
    mem_id = str(uuid4())
    run = AgentRun(
        id=run_id,
        organization_id=test_org_id,
        memory_id=mem_id,
        agent_name="agent3",
        agent_version="1.0",
        inputs_hash="hash3",
        status="success",
        confidence=0.8,
        outputs={},
        started_at=datetime.utcnow(),
        finished_at=datetime.utcnow(),
    )
    db_session.add(run)
    await db_session.commit()

    # Create checkpoint with detailed retrieval
    svc = CheckpointService(db_session, test_org_id)
    await svc.create_checkpoint(
        agent_run_id=run_id,
        step_index=0,
        input_snapshot={"query": "customer issue"},
        retrieval_snapshot={
            "ids": ["mem-a", "mem-b", "mem-c"],
            "scores": [0.98, 0.85, 0.72],
            "filters": {"status": "open"},
            "cutoff": 0.7,
        },
        model_snapshot={"temperature": 0.5, "top_k": 5, "top_p": 0.9},
        output_snapshot={"reasoning": "chose top 3 results"},
    )

    # Explain
    explanation = await svc.explain_retrieval_at_step(agent_run_id=run_id, step_index=0)
    assert "error" not in explanation
    assert explanation["input_query"] == "customer issue"
    assert explanation["retrieved_ids"] == ["mem-a", "mem-b", "mem-c"]
    assert explanation["retrieved_scores"] == [0.98, 0.85, 0.72]
    assert explanation["retrieval_filters"]["status"] == "open"
    assert explanation["model_config"]["temperature"] == 0.5
    assert "reasoning" in explanation["step_output_keys"]
