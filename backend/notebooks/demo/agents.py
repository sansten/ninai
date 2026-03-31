"""Agent pipeline runner for the demo notebook.

Provides run_pipeline() which runs a sequence of Ninai agents on a batch
of memory records, accumulating enrichment across agents.

Each case calls run_pipeline() with the list of records it created in
data_prep.py and the agent classes it needs.  The function returns:

{
    "hero_id":   str,            # local id of the primary record
    "hero":      dict,           # raw hero record
    "memory_ids": {str: str},    # local_id -> ninai_client memory id
    "enrichments": {             # local_id -> accumulated enrichment dict
        "C1-T3": {
            "canonical_entity": ...,
            "episode_id": ...,
            ...
        },
        ...
    },
    "agent_outputs": {           # local_id -> {agent_name -> outputs dict}
        "C1-T3": {
            "EntityResolutionAgent":    {...},
            "TemporalReasoningAgent":   {...},
            ...
        }
    }
}
"""

from __future__ import annotations

import asyncio
from typing import Any, Type

from app.agents.base import BaseAgent
from app.agents.types import AgentContext


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _build_context(record: dict, enrichment: dict | None = None) -> AgentContext:
    """Build an AgentContext from a demo memory record dict."""
    meta = record.get("extra_metadata", {})
    return {
        "memory": {
            "id": record["id"],
            "content": record["content"],
            "title": record.get("title", ""),
            "memory_type": record.get("memory_type", "semantic"),
            "source_type": record.get("source_type", "api"),
            "author_role": record.get("author_role", "engineer"),
            "tags": record.get("tags", []),
            "organization_id": meta.get("org_id", ""),
            "team_id": meta.get("team_id", ""),
            "team_slug": meta.get("team_slug", ""),
            "user_id": meta.get("user_id", ""),
            "user_email": meta.get("user_email", ""),
            "created_at": meta.get("created_at", ""),
            "enrichment": enrichment or {},
            "metadata": meta,
        },
        "runtime": {"job_id": f"demo-{record['id']}"},
    }


# ---------------------------------------------------------------------------
# Ingest helper (uses NinaiClient)
# ---------------------------------------------------------------------------

def ingest_records(ninai_client: Any, records: list[dict]) -> dict[str, str]:
    """Ingest records via NinaiClient and return {local_id: ninai_memory_id}."""
    memory_ids: dict[str, str] = {}
    for rec in records:
        meta = {
            **rec.get("extra_metadata", {}),
            "source_type": rec.get("source_type", "api"),
            "author_role": rec.get("author_role", "engineer"),
            "days_ago": rec.get("days_ago", 0),
        }
        r = ninai_client.memories.create(
            content=rec["content"],
            title=rec.get("title", ""),
            memory_type=rec.get("memory_type", "semantic"),
            scope=rec.get("scope", "team"),
            source_type=rec.get("source_type", "api"),
            tags=rec.get("tags", []),
            metadata=meta,
        )
        ninai_id = r.id if hasattr(r, "id") else r["id"]
        memory_ids[rec["id"]] = ninai_id
    return memory_ids


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    records: list[dict],
    agent_classes: list[Type[BaseAgent]],
    hero_local_id: str,
    *,
    cross_record_enrichment: dict | None = None,
) -> dict[str, Any]:
    """Run a sequence of agents on each record, accumulate enrichment.

    Parameters
    ----------
    records:
        List of demo memory record dicts (from data_prep.py).
    agent_classes:
        Ordered list of agent classes to run on each record.
    hero_local_id:
        The local id of the primary record (the one to surface in the
        CSR briefing card).
    cross_record_enrichment:
        Extra enrichment keys to inject before running agents on the hero
        record (e.g. episode info discovered from another record).

    Returns
    -------
    dict with keys: hero_id, hero, enrichments, agent_outputs.
    """
    enrichments: dict[str, dict] = {r["id"]: {} for r in records}
    agent_outputs: dict[str, dict] = {r["id"]: {} for r in records}

    async def _run_all() -> None:
        for rec in records:
            local_id = rec["id"]
            current_enrichment = enrichments[local_id]

            # inject cross-record context into the hero record
            if local_id == hero_local_id and cross_record_enrichment:
                current_enrichment.update(cross_record_enrichment)

            for agent_cls in agent_classes:
                agent = agent_cls()
                ctx = _build_context(rec, enrichment=current_enrichment)
                try:
                    result = await agent.run(local_id, ctx)
                    if result.status == "success" and result.outputs:
                        current_enrichment.update(result.outputs)
                        agent_outputs[local_id][agent.name] = result.outputs
                except Exception as exc:
                    agent_outputs[local_id][agent_cls.__name__] = {
                        "error": str(exc)
                    }

            # Seeded cross-record facts must survive agent processing.
            # Agents may overwrite keys like episode_label or tone — restore them.
            if local_id == hero_local_id and cross_record_enrichment:
                current_enrichment.update(cross_record_enrichment)

    asyncio.run(_run_all())

    hero_record = next(r for r in records if r["id"] == hero_local_id)
    return {
        "hero_id": hero_local_id,
        "hero": hero_record,
        "enrichments": enrichments,
        "agent_outputs": agent_outputs,
    }


# ---------------------------------------------------------------------------
# Convenience: run only the hero record through a single agent
# ---------------------------------------------------------------------------

def run_one(
    record: dict,
    agent_cls: Type[BaseAgent],
    enrichment: dict | None = None,
) -> dict[str, Any]:
    """Run a single agent on a single record.  Returns agent outputs dict."""
    async def _run() -> dict:
        agent = agent_cls()
        ctx = _build_context(record, enrichment=enrichment or {})
        result = await agent.run(record["id"], ctx)
        return result.outputs or {}

    return asyncio.run(_run())
