"""Episode Boundary Detection Service (GAP-1: Four-Level Hierarchy).

Segments a chronological stream of messages into coherent **episodes**.

Three boundary signals are fused:

1. **Topic shift** – cosine distance between consecutive message embeddings
   exceeds θ_topic (default 0.45).
2. **Temporal gap** – time delta between consecutive messages exceeds θ_time
   (default 30 min).
3. **Intent transition** – an LLM detects a goal or topic change between the
   last message of the current episode and the new message.

Split threshold (Fano inequality):
    n_k = 2^B / (1 − H(P_e))  with B ≈ 2 bits, P_e ≈ 0.15 → n_k ≈ 12

If episode length exceeds n_k the service will attempt a sub-split.
"""

from __future__ import annotations

import hashlib
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

import httpx
from sqlalchemy import and_, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.memory import MemoryMetadata
from app.models.memory_episode import MemoryEpisode
from app.models.memory_episode_membership import MemoryEpisodeMembership
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

# ── Tunables ────────────────────────────────────────────────────────────
THETA_TOPIC: float = 0.45          # cosine *distance* threshold for topic shift
THETA_TIME: timedelta = timedelta(minutes=30)
FANO_NK: int = 12                  # max messages per episode before sub-split


def _cosine_distance(a: List[float], b: List[float]) -> float:
    """1 − cosine_similarity.  Returns 1.0 if either vector is zero."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - (dot / (na * nb))


async def _llm_intent_boundary(prev_text: str, curr_text: str) -> tuple[bool, float]:
    """Ask the LLM whether two consecutive messages represent an intent/topic change.

    Returns (is_boundary, confidence).
    Falls back to (False, 0.0) on any error.
    """
    prompt = (
        "You are an intent-boundary detector.  Given two consecutive user messages, "
        "decide whether the second message starts a NEW topic or goal, or continues "
        "the same thread.\n\n"
        f"Message A (prior): {prev_text[:500]}\n"
        f"Message B (current): {curr_text[:500]}\n\n"
        "Reply with EXACTLY one JSON object: {\"boundary\": true/false, \"confidence\": 0.0-1.0}\n"
        "No other text."
    )
    try:
        model_name = settings.get_ollama_model("boundary")
        logger.info(
            "llm.model_route provider=ollama purpose=boundary model=%s base_url=%s",
            model_name,
            settings.OLLAMA_BASE_URL,
        )
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            raw = data.get("response", "")
            # Robust JSON extraction
            import json as _json
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = _json.loads(raw[start:end])
                return bool(parsed.get("boundary", False)), float(parsed.get("confidence", 0.5))
    except Exception as exc:
        logger.debug("LLM intent boundary check failed: %s", exc)
    return False, 0.0


class EpisodeBoundaryService:
    """Segments messages into episodes for a given (org, owner, scope)."""

    def __init__(self, session: AsyncSession):
        self.session = session

    # ── Public API ──────────────────────────────────────────────────

    async def segment_messages(
        self,
        *,
        organization_id: str,
        owner_id: str,
        scope: str = "personal",
        scope_id: Optional[str] = None,
        memory_ids: Optional[List[str]] = None,
        theta_topic: float = THETA_TOPIC,
        theta_time: timedelta = THETA_TIME,
        use_llm: bool = True,
    ) -> List[Dict[str, Any]]:
        """Run episode boundary detection over a set of messages.

        If *memory_ids* is provided, only those messages are processed.
        Otherwise all un-episoded messages for the (org, owner, scope) are used.

        Returns a list of ``{"episode_id": ..., "memory_ids": [...], "reason": ...}``
        dicts describing newly created episodes.
        """
        messages = await self._fetch_messages(
            organization_id=organization_id,
            owner_id=owner_id,
            scope=scope,
            scope_id=scope_id,
            memory_ids=memory_ids,
        )
        if not messages:
            return []

        # Get embeddings for all messages (batch for efficiency)
        embeddings: Dict[str, List[float]] = {}
        for m in messages:
            emb = await EmbeddingService.embed(m.content_preview)
            embeddings[m.id] = emb

        # Detect boundaries
        episodes_plan = await self._detect_boundaries(
            messages=messages,
            embeddings=embeddings,
            theta_topic=theta_topic,
            theta_time=theta_time,
            use_llm=use_llm,
        )

        # Materialize episodes
        created = []
        for ep in episodes_plan:
            episode = await self._create_episode(
                organization_id=organization_id,
                owner_id=owner_id,
                scope=scope,
                scope_id=scope_id,
                messages=[m for m in messages if m.id in ep["memory_ids"]],
                boundary_reason=ep["reason"],
                boundary_confidence=ep["confidence"],
            )
            created.append({
                "episode_id": episode.id,
                "memory_ids": ep["memory_ids"],
                "reason": ep["reason"],
            })

        return created

    async def add_message_to_current_episode(
        self,
        *,
        organization_id: str,
        owner_id: str,
        memory_id: str,
        scope: str = "personal",
        scope_id: Optional[str] = None,
        theta_topic: float = THETA_TOPIC,
        theta_time: timedelta = THETA_TIME,
        use_llm: bool = True,
    ) -> Dict[str, Any]:
        """Streaming API: process a single new message and either append it to
        the current open episode or start a new one.

        Returns ``{"episode_id": ..., "action": "appended"|"new_episode", ...}``.
        """
        # Get the message
        msg = (
            await self.session.execute(
                select(MemoryMetadata).where(
                    MemoryMetadata.id == memory_id,
                    MemoryMetadata.organization_id == organization_id,
                )
            )
        ).scalar_one_or_none()
        if not msg:
            raise ValueError(f"Memory {memory_id} not found")

        # Find current open episode
        open_episode = (
            await self.session.execute(
                select(MemoryEpisode)
                .where(
                    MemoryEpisode.organization_id == organization_id,
                    MemoryEpisode.owner_id == owner_id,
                    MemoryEpisode.scope == scope,
                    MemoryEpisode.status == "open",
                )
                .order_by(MemoryEpisode.updated_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

        if open_episode is None:
            # No open episode → start fresh
            ep = await self._create_episode(
                organization_id=organization_id,
                owner_id=owner_id,
                scope=scope,
                scope_id=scope_id,
                messages=[msg],
                boundary_reason="initial",
                boundary_confidence=1.0,
            )
            return {"episode_id": ep.id, "action": "new_episode", "reason": "initial"}

        # Check boundary against last message in the open episode
        last_membership = (
            await self.session.execute(
                select(MemoryEpisodeMembership)
                .where(MemoryEpisodeMembership.episode_id == open_episode.id)
                .order_by(MemoryEpisodeMembership.position.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

        if last_membership is None:
            # Empty episode (shouldn't happen) → just append
            await self._append_to_episode(open_episode, msg, position=0)
            return {"episode_id": open_episode.id, "action": "appended"}

        last_msg = (
            await self.session.execute(
                select(MemoryMetadata).where(MemoryMetadata.id == last_membership.memory_id)
            )
        ).scalar_one_or_none()

        # Compute boundary signals
        is_boundary, reason, confidence = await self._check_boundary(
            prev_msg=last_msg,
            curr_msg=msg,
            prev_embedding=await EmbeddingService.embed(last_msg.content_preview),
            curr_embedding=await EmbeddingService.embed(msg.content_preview),
            theta_topic=theta_topic,
            theta_time=theta_time,
            use_llm=use_llm,
        )

        # Also check Fano split threshold
        if not is_boundary and open_episode.message_count >= FANO_NK:
            is_boundary = True
            reason = "fano_split"
            confidence = 0.85

        if is_boundary:
            # Close current episode
            open_episode.status = "closed"
            await self.session.flush()

            # Create new episode
            ep = await self._create_episode(
                organization_id=organization_id,
                owner_id=owner_id,
                scope=scope,
                scope_id=scope_id,
                messages=[msg],
                boundary_reason=reason,
                boundary_confidence=confidence,
            )
            return {"episode_id": ep.id, "action": "new_episode", "reason": reason}
        else:
            # Append to open episode
            await self._append_to_episode(
                open_episode, msg, position=open_episode.message_count
            )
            return {"episode_id": open_episode.id, "action": "appended"}

    # ── Internal helpers ────────────────────────────────────────────

    async def _fetch_messages(
        self,
        *,
        organization_id: str,
        owner_id: str,
        scope: str,
        scope_id: Optional[str],
        memory_ids: Optional[List[str]],
    ) -> Sequence[MemoryMetadata]:
        """Fetch messages ordered by created_at, optionally filtered."""
        stmt = (
            select(MemoryMetadata)
            .where(
                MemoryMetadata.organization_id == organization_id,
                MemoryMetadata.owner_id == owner_id,
                MemoryMetadata.scope == scope,
            )
            .order_by(MemoryMetadata.created_at.asc())
        )
        if scope_id:
            stmt = stmt.where(MemoryMetadata.scope_id == scope_id)
        if memory_ids:
            stmt = stmt.where(MemoryMetadata.id.in_(memory_ids))

        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def _detect_boundaries(
        self,
        *,
        messages: Sequence[MemoryMetadata],
        embeddings: Dict[str, List[float]],
        theta_topic: float,
        theta_time: timedelta,
        use_llm: bool,
    ) -> List[Dict[str, Any]]:
        """Split messages into episode groups. Each group records its boundary reason."""
        groups: List[Dict[str, Any]] = []
        current_ids: List[str] = []
        current_reason = "initial"
        current_confidence = 1.0

        for i, msg in enumerate(messages):
            if i == 0:
                current_ids.append(msg.id)
                continue

            prev_msg = messages[i - 1]
            is_boundary, reason, confidence = await self._check_boundary(
                prev_msg=prev_msg,
                curr_msg=msg,
                prev_embedding=embeddings.get(prev_msg.id, []),
                curr_embedding=embeddings.get(msg.id, []),
                theta_topic=theta_topic,
                theta_time=theta_time,
                use_llm=use_llm,
            )

            # Fano sub-split check
            if not is_boundary and len(current_ids) >= FANO_NK:
                is_boundary = True
                reason = "fano_split"
                confidence = 0.85

            if is_boundary:
                groups.append({
                    "memory_ids": list(current_ids),
                    "reason": current_reason,
                    "confidence": current_confidence,
                })
                current_ids = [msg.id]
                current_reason = reason
                current_confidence = confidence
            else:
                current_ids.append(msg.id)

        # Flush last group
        if current_ids:
            groups.append({
                "memory_ids": list(current_ids),
                "reason": current_reason,
                "confidence": current_confidence,
            })

        return groups

    async def _check_boundary(
        self,
        *,
        prev_msg: MemoryMetadata,
        curr_msg: MemoryMetadata,
        prev_embedding: List[float],
        curr_embedding: List[float],
        theta_topic: float,
        theta_time: timedelta,
        use_llm: bool,
    ) -> tuple[bool, str, float]:
        """Return (is_boundary, reason, confidence)."""
        # 1. Temporal gap
        if prev_msg.created_at and curr_msg.created_at:
            delta = curr_msg.created_at - prev_msg.created_at
            if delta > theta_time:
                return True, "temporal_gap", 0.95

        # 2. Topic shift (embedding cosine distance)
        if prev_embedding and curr_embedding:
            dist = _cosine_distance(prev_embedding, curr_embedding)
            if dist > theta_topic:
                return True, "topic_shift", min(1.0, dist / theta_topic * 0.8)

        # 3. LLM intent detection (optional, most expensive)
        if use_llm:
            is_b, conf = await _llm_intent_boundary(
                prev_msg.content_preview, curr_msg.content_preview
            )
            if is_b and conf >= 0.6:
                return True, "intent_change", conf

        return False, "", 0.0

    async def _create_episode(
        self,
        *,
        organization_id: str,
        owner_id: str,
        scope: str,
        scope_id: Optional[str],
        messages: Sequence[MemoryMetadata],
        boundary_reason: str,
        boundary_confidence: float,
    ) -> MemoryEpisode:
        """Persist a new MemoryEpisode + its memberships."""
        timestamps = [m.created_at for m in messages if m.created_at]
        b_start = min(timestamps) if timestamps else None
        b_end = max(timestamps) if timestamps else None

        episode = MemoryEpisode(
            id=str(uuid4()),
            organization_id=organization_id,
            owner_id=owner_id,
            scope=scope,
            scope_id=scope_id,
            boundary_start=b_start,
            boundary_end=b_end,
            message_count=len(messages),
            boundary_reason=boundary_reason,
            boundary_confidence=boundary_confidence,
            status="open",
        )
        self.session.add(episode)
        await self.session.flush()

        for pos, m in enumerate(messages):
            mem = MemoryEpisodeMembership(
                id=str(uuid4()),
                organization_id=organization_id,
                memory_id=m.id,
                episode_id=episode.id,
                position=pos,
            )
            self.session.add(mem)

        await self.session.flush()

        # Generate narrative summary asynchronously (best-effort)
        try:
            summary = await self._summarize_episode(messages)
            if summary:
                episode.title = summary[:500]
                episode.narrative_summary = summary
                await self.session.flush()
        except Exception as exc:
            logger.warning("Episode summary generation failed: %s", exc)

        # Generate centroid embedding and store in Qdrant
        try:
            centroid = await self._compute_centroid(messages)
            if centroid and any(v != 0 for v in centroid):
                from app.core.qdrant import QdrantService
                vector_id = f"episode:{episode.id}"
                await QdrantService.upsert_memory(
                    memory_id=vector_id,
                    org_id=organization_id,
                    vector=centroid,
                    payload={
                        "type": "episode",
                        "episode_id": episode.id,
                        "owner_id": owner_id,
                        "scope": scope,
                    },
                )
                episode.vector_id = vector_id
                await self.session.flush()
        except Exception as exc:
            logger.warning("Episode centroid upsert failed: %s", exc)

        return episode

    async def _append_to_episode(
        self,
        episode: MemoryEpisode,
        msg: MemoryMetadata,
        position: int,
    ) -> None:
        """Append a message to an existing open episode."""
        mem = MemoryEpisodeMembership(
            id=str(uuid4()),
            organization_id=episode.organization_id,
            memory_id=msg.id,
            episode_id=episode.id,
            position=position,
        )
        self.session.add(mem)
        episode.message_count += 1
        if msg.created_at:
            if episode.boundary_end is None or msg.created_at > episode.boundary_end:
                episode.boundary_end = msg.created_at
        await self.session.flush()

    async def _summarize_episode(
        self, messages: Sequence[MemoryMetadata]
    ) -> Optional[str]:
        """Generate a narrative summary of the episode via LLM."""
        if not messages:
            return None
        texts = [m.content_preview for m in messages if m.content_preview]
        if not texts:
            return None
        from app.services.summarization_service import summarize_short_term_memories
        return await summarize_short_term_memories(texts)

    async def _compute_centroid(
        self, messages: Sequence[MemoryMetadata]
    ) -> Optional[List[float]]:
        """Compute mean embedding (centroid) of message embeddings."""
        embeddings = []
        for m in messages:
            emb = await EmbeddingService.embed(m.content_preview)
            if emb and any(v != 0 for v in emb):
                embeddings.append(emb)
        if not embeddings:
            return None
        dim = len(embeddings[0])
        centroid = [0.0] * dim
        for emb in embeddings:
            for i in range(dim):
                centroid[i] += emb[i]
        n = len(embeddings)
        return [v / n for v in centroid]
