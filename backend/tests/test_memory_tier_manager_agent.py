from __future__ import annotations

import pytest

from app.agents.memory_tier_manager_agent import MemoryTierManagerAgent


class TestMemoryTierManagerAgent:
    def test_reconcile_loads_incoming_into_working_set(self):
        agent = MemoryTierManagerAgent()
        summary = agent.reconcile(
            working_set=[],
            archival=[],
            incoming=[{"id": "m1", "content": "auth failure report"}],
            max_working_set=8,
        )

        assert summary["working_set_size"] == 1
        assert summary["loaded_ids"] == ["m1"]
        assert summary["archival_size"] == 0

    def test_reconcile_offloads_when_capacity_exceeded(self):
        agent = MemoryTierManagerAgent()
        incoming = [{"id": f"m{i}", "content": f"content {i}"} for i in range(5)]
        summary = agent.reconcile(
            working_set=[],
            archival=[],
            incoming=incoming,
            max_working_set=3,
        )

        assert summary["working_set_size"] == 3
        assert summary["archival_size"] == 2
        assert len(summary["offloaded_ids"]) == 2

    def test_reconcile_deduplicates_same_memory(self):
        agent = MemoryTierManagerAgent()
        summary = agent.reconcile(
            working_set=[{"memory_id": "m1", "content_preview": "x"}],
            archival=[],
            incoming=[{"id": "m1", "content": "x"}],
            max_working_set=8,
        )

        assert summary["working_set_size"] == 1
        assert summary["loaded_ids"] == ["m1"]

    @pytest.mark.asyncio
    async def test_run_returns_working_set_summary(self):
        agent = MemoryTierManagerAgent()
        result = await agent.run(
            memory_id="m-run",
            context={
                "memory_tiers": {
                    "working_set": [],
                    "archival": [],
                    "incoming": [{"id": "m-run", "content": "incident summary"}],
                    "max_working_set": 8,
                }
            },
        )

        assert result.status == "success"
        assert "working_set_summary" in result.outputs
        assert result.outputs["working_set_summary"]["working_set_size"] == 1
