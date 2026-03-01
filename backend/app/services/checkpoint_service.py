"""Checkpoint persistence and retrieval service (PR5: Replayability)."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.run_checkpoint import RunCheckpoint


class CheckpointService:
    def __init__(self, session: AsyncSession, org_id: str):
        self.session = session
        self.org_id = org_id

    async def create_checkpoint(
        self,
        *,
        agent_run_id: str,
        step_index: int,
        input_snapshot: dict,
        retrieval_snapshot: dict,
        model_snapshot: dict,
        output_snapshot: dict,
    ) -> str:
        """Create a checkpoint for an agent run step.
        
        Args:
            agent_run_id: ID of the parent agent run
            step_index: Sequential step number (0-based)
            input_snapshot: Input state at this step
            retrieval_snapshot: Retrieved memories with ids/scores/filters
            model_snapshot: Model configuration and state
            output_snapshot: Step output/response
            
        Returns:
            Checkpoint ID
        """
        checkpoint = RunCheckpoint(
            organization_id=self.org_id,
            agent_run_id=agent_run_id,
            step_index=step_index,
            input_snapshot=input_snapshot or {},
            retrieval_snapshot=retrieval_snapshot or {},
            model_snapshot=model_snapshot or {},
            output_snapshot=output_snapshot or {},
            created_at=datetime.utcnow(),
        )
        self.session.add(checkpoint)
        await self.session.flush()
        return checkpoint.id

    async def get_checkpoints_up_to_step(
        self,
        *,
        agent_run_id: str,
        to_step: Optional[int] = None,
    ) -> list[dict]:
        """Get all checkpoints for a run up to specified step (inclusive).
        
        Args:
            agent_run_id: ID of the agent run
            to_step: Maximum step index (None = all steps)
            
        Returns:
            List of checkpoint snapshots in step order
        """
        query = select(RunCheckpoint).where(
            and_(
                RunCheckpoint.organization_id == self.org_id,
                RunCheckpoint.agent_run_id == agent_run_id,
            )
        )
        if to_step is not None:
            query = query.where(RunCheckpoint.step_index <= to_step)
        
        query = query.order_by(RunCheckpoint.step_index)
        result = await self.session.execute(query)
        checkpoints = result.scalars().all()
        
        return [
            {
                "id": cp.id,
                "step_index": cp.step_index,
                "input_snapshot": cp.input_snapshot,
                "retrieval_snapshot": cp.retrieval_snapshot,
                "model_snapshot": cp.model_snapshot,
                "output_snapshot": cp.output_snapshot,
                "created_at": cp.created_at.isoformat(),
            }
            for cp in checkpoints
        ]

    async def explain_retrieval_at_step(
        self,
        *,
        agent_run_id: str,
        step_index: int,
    ) -> dict:
        """Get detailed retrieval information for debugging "why was this recalled?"
        
        Args:
            agent_run_id: ID of the agent run
            step_index: Step index to explain
            
        Returns:
            Detailed retrieval explanation with scores, filters, etc.
        """
        result = await self.session.execute(
            select(RunCheckpoint).where(
                and_(
                    RunCheckpoint.organization_id == self.org_id,
                    RunCheckpoint.agent_run_id == agent_run_id,
                    RunCheckpoint.step_index == step_index,
                )
            )
        )
        checkpoint = result.scalars().first()
        
        if not checkpoint:
            return {"error": "Checkpoint not found", "step_index": step_index}
        
        retrieval = checkpoint.retrieval_snapshot or {}
        model = checkpoint.model_snapshot or {}
        input_data = checkpoint.input_snapshot or {}
        
        return {
            "step_index": step_index,
            "input_query": input_data.get("query", ""),
            "input_filters": input_data.get("filters", {}),
            "retrieved_ids": retrieval.get("ids", []),
            "retrieved_scores": retrieval.get("scores", []),
            "retrieval_filters": retrieval.get("filters", {}),
            "retrieval_cutoff": retrieval.get("cutoff", None),
            "model_config": {
                "temperature": model.get("temperature"),
                "top_k": model.get("top_k"),
                "top_p": model.get("top_p"),
            },
            "step_output_keys": list((checkpoint.output_snapshot or {}).keys()),
        }

    async def get_checkpoint_for_reproduce(
        self,
        *,
        agent_run_id: str,
        step_index: int,
    ) -> Optional[dict]:
        """Get checkpoint data for reproducing a specific step.
        
        Args:
            agent_run_id: ID of the agent run
            step_index: Step to reproduce
            
        Returns:
            Full checkpoint snapshots (input, retrieval, model, output) or None
        """
        result = await self.session.execute(
            select(RunCheckpoint).where(
                and_(
                    RunCheckpoint.organization_id == self.org_id,
                    RunCheckpoint.agent_run_id == agent_run_id,
                    RunCheckpoint.step_index == step_index,
                )
            )
        )
        checkpoint = result.scalars().first()
        
        if not checkpoint:
            return None
        
        return {
            "step_index": checkpoint.step_index,
            "input_snapshot": checkpoint.input_snapshot,
            "retrieval_snapshot": checkpoint.retrieval_snapshot,
            "model_snapshot": checkpoint.model_snapshot,
            "output_snapshot": checkpoint.output_snapshot,
        }
