"""Semantic Distillation Service (GAP-1: Four-Level Hierarchy – Level 3).

Extracts **reusable knowledge units** (MemorySemanticNode) from closed
episodes by applying the four xMemory distillation filters:

    1. Persistence  – will it remain useful across sessions?
    2. Specificity  – is it specific enough to drive future actions?
    3. Utility      – can an agent use it to improve responses?
    4. Independence  – does it stand alone without conversational context?

Each candidate fact is scored on all four axes.  The **composite quality**
is the geometric mean:

    Q = (persistence × specificity × utility × independence) ^ (1/4)

Facts with Q ≥ τ_quality (default 0.55) are promoted to MemorySemanticNode.

The distiller can be invoked:
    • On episode close (real-time pipeline)
    • In batch mode (Celery beat, nightly)
"""

from __future__ import annotations

import hashlib
import json as _json
import logging
import math
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

import httpx
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.memory import MemoryMetadata
from app.models.memory_episode import MemoryEpisode
from app.models.memory_episode_membership import MemoryEpisodeMembership
from app.models.memory_semantic_node import MemorySemanticNode
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

# ── Tunables ────────────────────────────────────────────────────────────
TAU_QUALITY: float = 0.55   # minimum composite quality to promote
OLLAMA_URL: str = getattr(settings, "OLLAMA_URL", None) or "http://ollama:11434"

# ── Prompt template ─────────────────────────────────────────────────────
DISTILLATION_PROMPT = """\
You are a knowledge distillation engine.  Given a set of conversational
messages from a single episode, extract **reusable facts** that an AI agent
should remember long-term.

For EACH extracted fact, score it on four axes (0.0-1.0):
  - persistence:  Will this remain useful across future sessions?
  - specificity:  Is this specific enough to inform future decisions?
  - utility:      Can an agent use this to improve its responses?
  - independence:  Does it make sense without conversational context?

Reply with a JSON array of objects:
[
  {{
    "fact": "<1-3 sentence distilled knowledge>",
    "persistence": 0.0-1.0,
    "specificity": 0.0-1.0,
    "utility": 0.0-1.0,
    "independence": 0.0-1.0,
    "entities": ["entity1", "entity2"],
    "tags": ["tag1", "tag2"]
  }}
]

If no reusable facts can be extracted, return an empty array: []

Messages:
{messages}
"""


def _geometric_mean_4(a: float, b: float, c: float, d: float) -> float:
    """Geometric mean of four positive values.  Clamp to [0, 1]."""
    product = max(0.0, a) * max(0.0, b) * max(0.0, c) * max(0.0, d)
    return min(1.0, product ** 0.25)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SemanticDistillationService:
    """Distils closed episodes into semantic nodes (Level 3)."""

    def __init__(self, session: AsyncSession):
        self.session = session

    # ── Public API ──────────────────────────────────────────────────

    async def distill_episode(
        self,
        episode_id: str,
        *,
        organization_id: str,
        tau_quality: float = TAU_QUALITY,
    ) -> List[Dict[str, Any]]:
        """Distil a single episode into semantic nodes.

        Returns a list of ``{"semantic_node_id": ..., "content": ..., "quality": ...}``.
        """
        episode = (
            await self.session.execute(
                select(MemoryEpisode).where(
                    MemoryEpisode.id == episode_id,
                    MemoryEpisode.organization_id == organization_id,
                )
            )
        ).scalar_one_or_none()
        if not episode:
            raise ValueError(f"Episode {episode_id} not found")

        # Fetch messages in order
        messages = await self._get_episode_messages(episode_id, organization_id)
        if not messages:
            return []

        # LLM extraction
        candidates = await self._extract_facts(messages)

        # Score, filter, and persist
        created = []
        for cand in candidates:
            q = _geometric_mean_4(
                cand.get("persistence", 0),
                cand.get("specificity", 0),
                cand.get("utility", 0),
                cand.get("independence", 0),
            )
            if q < tau_quality:
                logger.debug(
                    "Fact rejected (Q=%.3f < τ=%.3f): %s",
                    q, tau_quality, cand.get("fact", "")[:80],
                )
                continue

            node = await self._create_semantic_node(
                episode=episode,
                candidate=cand,
                composite_quality=q,
                source_memory_ids=[m.id for m in messages],
            )
            created.append({
                "semantic_node_id": node.id,
                "content": node.content,
                "quality": q,
            })

        return created

    async def distill_batch(
        self,
        *,
        organization_id: str,
        limit: int = 50,
        tau_quality: float = TAU_QUALITY,
    ) -> Dict[str, Any]:
        """Batch-distil all closed, un-distilled episodes for an org.

        Returns stats: ``{"episodes_processed": N, "nodes_created": M}``.
        """
        # Find closed episodes that have no semantic nodes yet
        stmt = (
            select(MemoryEpisode)
            .where(
                MemoryEpisode.organization_id == organization_id,
                MemoryEpisode.status == "closed",
            )
            .order_by(MemoryEpisode.created_at.asc())
            .limit(limit)
        )
        episodes = (await self.session.execute(stmt)).scalars().all()

        total_nodes = 0
        processed = 0
        for ep in episodes:
            # Skip if already distilled (has semantic nodes)
            existing = (
                await self.session.execute(
                    select(MemorySemanticNode)
                    .where(MemorySemanticNode.source_episode_ids.contains([ep.id]))
                    .limit(1)
                )
            ).scalar_one_or_none()
            if existing:
                continue

            try:
                nodes = await self.distill_episode(
                    ep.id, organization_id=organization_id, tau_quality=tau_quality
                )
                total_nodes += len(nodes)
                processed += 1
            except Exception as exc:
                logger.warning("Distillation failed for episode %s: %s", ep.id, exc)

        return {"episodes_processed": processed, "nodes_created": total_nodes}

    # ── Internal helpers ────────────────────────────────────────────

    async def _get_episode_messages(
        self, episode_id: str, organization_id: str
    ) -> Sequence[MemoryMetadata]:
        """Fetch messages for an episode in order."""
        membership_stmt = (
            select(MemoryEpisodeMembership.memory_id)
            .where(
                MemoryEpisodeMembership.episode_id == episode_id,
                MemoryEpisodeMembership.organization_id == organization_id,
            )
            .order_by(MemoryEpisodeMembership.position.asc())
        )
        memberships = (await self.session.execute(membership_stmt)).scalars().all()
        if not memberships:
            return []

        # Fetch actual messages preserving order
        stmt = select(MemoryMetadata).where(MemoryMetadata.id.in_(memberships))
        rows = (await self.session.execute(stmt)).scalars().all()
        order = {mid: idx for idx, mid in enumerate(memberships)}
        return sorted(rows, key=lambda m: order.get(m.id, 999))

    async def _extract_facts(
        self, messages: Sequence[MemoryMetadata]
    ) -> List[Dict[str, Any]]:
        """Call LLM to extract scored facts from episode messages."""
        texts = "\n---\n".join(
            f"[{i+1}] {m.content_preview}" for i, m in enumerate(messages)
        )
        prompt = DISTILLATION_PROMPT.format(messages=texts[:4000])

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": "llama3", "prompt": prompt, "stream": False},
                )
                resp.raise_for_status()
                data = resp.json()
                raw = data.get("response", "")

                # Extract JSON array
                start = raw.find("[")
                end = raw.rfind("]") + 1
                if start >= 0 and end > start:
                    return _json.loads(raw[start:end])
        except Exception as exc:
            logger.warning("LLM fact extraction failed: %s", exc)

        return []

    async def _create_semantic_node(
        self,
        *,
        episode: MemoryEpisode,
        candidate: Dict[str, Any],
        composite_quality: float,
        source_memory_ids: List[str],
    ) -> MemorySemanticNode:
        """Persist a semantic node with its embedding."""
        content = str(candidate.get("fact", "")).strip()
        content_hash = _sha256(content)

        # Dedup: check if this exact content already exists
        existing = (
            await self.session.execute(
                select(MemorySemanticNode).where(
                    MemorySemanticNode.organization_id == episode.organization_id,
                    MemorySemanticNode.content_hash == content_hash,
                )
            )
        ).scalar_one_or_none()
        if existing:
            # Update source tracking
            existing_eps = existing.source_episode_ids or []
            if episode.id not in existing_eps:
                existing.source_episode_ids = existing_eps + [episode.id]
            existing.reference_count = (existing.reference_count or 0) + 1
            await self.session.flush()
            return existing

        node = MemorySemanticNode(
            id=str(uuid4()),
            organization_id=episode.organization_id,
            owner_id=episode.owner_id,
            scope=episode.scope,
            scope_id=episode.scope_id,
            content=content,
            content_hash=content_hash,
            persistence_score=float(candidate.get("persistence", 0)),
            specificity_score=float(candidate.get("specificity", 0)),
            utility_score=float(candidate.get("utility", 0)),
            independence_score=float(candidate.get("independence", 0)),
            composite_quality=composite_quality,
            topic_id=episode.topic_id,
            source_episode_ids=[episode.id],
            source_memory_ids=source_memory_ids,
            entities=candidate.get("entities", []),
            tags=candidate.get("tags", []),
        )
        self.session.add(node)
        await self.session.flush()

        # Embed and store in Qdrant
        try:
            embedding = await EmbeddingService.embed(content)
            if embedding and any(v != 0 for v in embedding):
                from app.core.qdrant import QdrantService
                vector_id = f"semantic:{node.id}"
                await QdrantService.upsert_memory(
                    memory_id=vector_id,
                    org_id=episode.organization_id,
                    vector=embedding,
                    payload={
                        "type": "semantic_node",
                        "semantic_node_id": node.id,
                        "owner_id": episode.owner_id,
                        "scope": episode.scope,
                        "topic_id": episode.topic_id or "",
                    },
                )
                node.vector_id = vector_id
                await self.session.flush()
        except Exception as exc:
            logger.warning("Semantic node embedding upsert failed: %s", exc)

        return node
