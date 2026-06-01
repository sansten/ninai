"""
V2 DNC-Inspired Memory Router (Component C)

Acts as the bridging layer between the Reasoning Engine (Ollama) and the
Chronological Knowledge Graph (FalkorDB).

Read Weighting:
  1. Embed query with Ollama (nomic-embed-text)
  2. Dense search in Qdrant → top-K chunk ids (episodic memory)
  3. FalkorDB subgraph traversal seeded from those ids → contextual nodes
  4. Rank by (weight * recency_score), return top-M nodes as prompt context

Write Weighting:
  1. Extract entities from new interaction via Ollama (structured JSON)
  2. MERGE each entity into FalkorDB (create-or-reinforce)
  3. Create Utterance node for the interaction
  4. Link utterance → entities via RESPONDED_TO edges
  5. Upsert interaction embedding into Qdrant

Purge:
  1. Decay neighborhood edges after write
  2. Delete edges below WEIGHT_PRUNE_THRESHOLD (called periodically)
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.v2.llm.reranker import rerank_context
from app.v2.memory.entity_extraction import parse_temporal_context

_Q_SKIP = frozenset({
    "what", "when", "where", "who", "why", "how", "which", "did", "does",
    "do", "has", "have", "had", "was", "were", "is", "are", "the", "a",
    "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "that", "this", "these", "those", "their", "his", "her",
    "its", "your", "our", "my", "not", "no", "any", "some", "about",
})

_CAP_SKIP = frozenset({
    "What", "When", "Where", "Who", "Why", "How", "Which", "The", "This",
    "That", "These", "Those", "Is", "Are", "Was", "Were", "Did", "Does",
    "Do", "Has", "Have", "Had", "In", "On", "At", "To", "For", "Of",
    "With", "By", "From", "And", "Or", "But", "If", "As", "An", "A",
    # Modal / auxiliary verbs that start sentences
    "Would", "Could", "Should", "Might", "Will", "Can", "Shall",
    # Other question-word starters
    "Based", "Considering", "Given", "Looking", "According",
    # Common non-name capitalized words
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
})

_MONTH_ABBR = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05",
    "jun": "06", "jul": "07", "aug": "08", "sep": "09", "oct": "10",
    "nov": "11", "dec": "12",
}


def _extract_query_names(query: str) -> list[str]:
    """Extract likely person names (title-case words) from a query string."""
    words = re.findall(r"\b([A-Z][a-z]{1,15})\b", query)
    return [w for w in words if w not in _CAP_SKIP]


# Nouns that signal a list-style question needing multi-topic coverage.
_LIST_NOUNS = frozenset({
    "activities", "activity", "events", "event", "things", "ways",
    "books", "movies", "films", "songs", "places", "hobbies", "hobby",
    "interests", "sports", "foods", "topics", "causes", "projects",
    "achievements", "experiences", "memories", "goals", "plans",
})

_LIST_Q_RE = re.compile(
    r"\bwhat\s+(?:\w+\s+)?(?:" + "|".join(_LIST_NOUNS) + r")\b"
    r"|\bin\s+what\s+ways\b"
    r"|\bhow\s+(?:many|often)\b",
    re.IGNORECASE,
)

# Possessive pattern: "Caroline's", "Melanie's"
_POSSESSIVE_RE = re.compile(r"\b([A-Z][a-z]{1,15})'s\b")


def _is_list_question(query: str) -> bool:
    return bool(_LIST_Q_RE.search(query))


def _extract_possessor(query: str) -> str | None:
    """Return the possessor name from 'X's attribute' phrasing, or None."""
    m = _POSSESSIVE_RE.search(query)
    if m and m.group(1) not in _CAP_SKIP:
        return m.group(1).lower()
    return None


def _content_nouns(query: str) -> list[str]:
    """Extract lowercase content words (not stopwords, len > 3) for keyword fallback."""
    words = re.findall(r"\b([a-zA-Z]{4,})\b", query.lower())
    return [w for w in words if w not in _Q_SKIP]


def _extract_query_dates(query: str) -> list[str]:
    """Extract year or year-month date prefixes for temporal retrieval."""
    prefixes: list[str] = []
    for m in re.finditer(r"\d{4}-\d{2}-\d{2}", query):
        prefixes.append(m.group()[:7])
    for m in re.finditer(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4})", query.lower()):
        abbr = m.group(1)[:3]
        year = m.group(2)
        prefixes.append(f"{year}-{_MONTH_ABBR[abbr]}")
    # Month name only without year (e.g. "in December", "last March") — add as month prefix for all recent years
    for m in re.finditer(r"\b(?:in\s+|last\s+|during\s+)?(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b", query.lower()):
        abbr = m.group(1)[:3]
        mo = _MONTH_ABBR[abbr]
        # Add prefixes for common recent years (covers 2021-2024 range)
        for yr in ("2021", "2022", "2023", "2024"):
            prefixes.append(f"{yr}-{mo}")
    for m in re.finditer(r"\b(20\d{2})\b", query):
        prefixes.append(m.group())
    return list(dict.fromkeys(prefixes))

logger = logging.getLogger(__name__)


@dataclass
class ReadResult:
    graph_nodes: list[dict[str, Any]] = field(default_factory=list)
    qdrant_chunks: list[dict[str, Any]] = field(default_factory=list)
    seed_entity_ids: list[str] = field(default_factory=list)


@dataclass
class WriteResult:
    utterance_id: str = ""
    entity_ids: list[str] = field(default_factory=list)
    qdrant_point_id: str = ""
    graph_writes: int = 0


class DNCMemoryRouter:
    """
    Stateless DNC memory router.  Instantiated once per request; does not
    hold any cross-request state — all state lives in FalkorDB and Qdrant.
    """

    def __init__(
        self,
        graph_client: Any,           # V2GraphClient
        qdrant_service: Any | None,  # QdrantService (optional — graceful degrade)
        embedding_fn: Any | None,    # async callable(text) -> list[float]
        entity_extractor: Any | None, # async callable(text) -> list[dict]
        top_k_qdrant: int = 10,
        top_m_graph: int = 20,
        graph_hops: int = 2,
    ) -> None:
        self._graph = graph_client
        self._qdrant = qdrant_service
        self._embed = embedding_fn
        self._extract_entities = entity_extractor
        self._top_k = top_k_qdrant
        self._top_m = top_m_graph
        self._hops = graph_hops

    # ------------------------------------------------------------------
    # Read weighting — Phase 1
    # ------------------------------------------------------------------

    async def read(self, tenant_id: str, session_id: str, query: str) -> ReadResult:
        result = ReadResult()

        # Pre-compute query signals used across multiple retrieval steps
        _possessor = _extract_possessor(query)
        _is_list = _is_list_question(query)

        # 1. Dense retrieval from Qdrant (episodic memory)
        qdrant_chunks: list[dict[str, Any]] = []
        seed_ids: list[str] = []
        if self._embed and self._qdrant:
            try:
                embedding = await self._embed(query)
                raw = await self._qdrant_search(tenant_id, embedding)
                qdrant_chunks = raw
                seen_chunk_ids = {c["id"] for c in qdrant_chunks}

                # Fix A — list questions: run a noun-focused sub-query to surface
                # semantically distant facts that a single embedding misses.
                # e.g. "what activities" → one embedding finds camping but not pottery.
                if _is_list:
                    nouns = _content_nouns(query)
                    names = _extract_query_names(query)
                    keyword_q = " ".join(names + nouns)
                    if keyword_q.strip():
                        kw_emb = await self._embed(keyword_q)
                        extra = await self._qdrant_search(tenant_id, kw_emb)
                        for ch in extra:
                            if ch["id"] not in seen_chunk_ids:
                                qdrant_chunks.append(ch)
                                seen_chunk_ids.add(ch["id"])

                # Fix B — low-hit fallback: when dense search returns almost nothing
                # (specific proper nouns like "Sweden", book titles), retry with
                # content-word keyword query which embeds closer to those nouns.
                if len(qdrant_chunks) < 8:
                    nouns = _content_nouns(query)
                    names = _extract_query_names(query)
                    fallback_q = " ".join(names + nouns)
                    if fallback_q.strip() and fallback_q.strip() != query.strip():
                        fb_emb = await self._embed(fallback_q)
                        extra = await self._qdrant_search(tenant_id, fb_emb)
                        for ch in extra:
                            if ch["id"] not in seen_chunk_ids:
                                qdrant_chunks.append(ch)
                                seen_chunk_ids.add(ch["id"])

                # Fix C — subject scoping: for possessive questions ("Caroline's X"),
                # deprioritize personal_attribute chunks where the stored subject is a
                # different person. Move mismatches to the end so the right person's
                # facts appear first in the prompt context window.
                if _possessor:
                    matched, mismatched = [], []
                    for ch in qdrant_chunks:
                        p = ch.get("payload", {})
                        ch_subject = str(p.get("subject") or "").lower()
                        ch_type = str(p.get("entity_type") or p.get("type") or "")
                        if ch_type == "personal_attribute" and ch_subject and ch_subject != _possessor:
                            mismatched.append(ch)
                        else:
                            matched.append(ch)
                    qdrant_chunks = matched + mismatched

                # Extract entity/memory ids that appear in Qdrant payload.
                # v2 writes store entity_ids (list); v1 writes store entity_id/memory_id (scalar).
                for chunk in qdrant_chunks:
                    payload = chunk.get("payload", {})
                    for key in ("entity_id", "memory_id", "id"):
                        val = payload.get(key)
                        if val:
                            seed_ids.append(str(val))
                    # v2 payload format: "entity_ids": ["entity_1", "entity_2", ...]
                    for eid in (payload.get("entity_ids") or []):
                        if eid:
                            seed_ids.append(str(eid))
                    # utterance_id is also a valid graph seed
                    utt_id = payload.get("utterance_id")
                    if utt_id:
                        seed_ids.append(str(utt_id))
            except Exception as exc:
                logger.warning("Qdrant read failed: %s", exc)

        result.qdrant_chunks = qdrant_chunks
        result.seed_entity_ids = seed_ids

        # 2. Graph traversal seeded from Qdrant hits
        if seed_ids and self._graph.is_available():
            try:
                nodes = await self._graph.fetch_subgraph(
                    tenant_id=tenant_id,
                    seed_ids=seed_ids,
                    hops=self._hops,
                    limit=self._top_m,
                )
                result.graph_nodes = nodes
            except Exception as exc:
                logger.warning("Graph read failed: %s", exc)

            # 2b. Also fetch Entity nodes directly for any entity-ID seeds.
            #     fetch_subgraph only follows FOLLOWED_BY edges on Utterances;
            #     entity-indexed Qdrant chunks (step 2c in write) have snake_case
            #     entity IDs that never match `:Utterance` nodes.
            try:
                entity_nodes = await self._graph.fetch_entities_by_ids(tenant_id, seed_ids)
                existing_ids = {n.get("id") for n in result.graph_nodes}
                for node in entity_nodes:
                    if node.get("id") not in existing_ids:
                        result.graph_nodes.insert(0, node)
                        existing_ids.add(node.get("id"))
            except Exception as exc:
                logger.warning("Entity direct fetch failed: %s", exc)

        # 3. Always include recent session utterances for context continuity
        if self._graph.is_available():
            try:
                recent = await self._graph.fetch_recent_utterances(
                    tenant_id, session_id, limit=6
                )
                # Merge with graph_nodes, dedup by id
                existing_ids = {n.get("id") for n in result.graph_nodes}
                for node in recent:
                    if node.get("id") not in existing_ids:
                        result.graph_nodes.append(node)
            except Exception as exc:
                logger.warning("Recent utterances fetch failed: %s", exc)

        # 4. Direct entity retrieval — fetch person profiles and temporal events
        #    by name/date extracted from the query, bypassing graph traversal.
        query_names = _extract_query_names(query)
        query_dates = _extract_query_dates(query)
        priority_nodes: list[dict[str, Any]] = []
        if self._graph.is_available():
            if query_names:
                try:
                    profiles = await self._graph.fetch_person_profiles(
                        tenant_id, query_names
                    )
                    priority_nodes.extend(profiles)
                except Exception as exc:
                    logger.warning("Profile fetch failed: %s", exc)
            else:
                # No explicit name in query — fetch top profiles as generic context
                try:
                    all_profiles = await self._graph.fetch_all_profiles(tenant_id, limit=5)
                    priority_nodes.extend(all_profiles)
                except Exception as exc:
                    logger.warning("All-profiles fetch failed: %s", exc)
            if query_dates or query_names:
                try:
                    # Use ASC order for "first/original/earliest" queries, DESC otherwise
                    q_low = query.lower()
                    wants_oldest = any(w in q_low for w in (
                        "first", "originally", "earliest", "initially", "when did",
                        "started", "began", "begin",
                    ))
                    if wants_oldest:
                        events = await self._graph.fetch_temporal_events_asc(
                            tenant_id, query_dates, query_names, limit=25
                        )
                    else:
                        events = await self._graph.fetch_temporal_events(
                            tenant_id, query_dates, query_names, limit=25
                        )
                    priority_nodes.extend(events)
                except Exception as exc:
                    logger.warning("Temporal event fetch failed: %s", exc)

        # Prepend priority nodes (profiles + temporal events) to graph_nodes
        # so they appear first in the prompt regardless of weight ranking.
        if priority_nodes:
            existing_ids = {n.get("id") for n in result.graph_nodes}
            for node in priority_nodes:
                if node.get("id") not in existing_ids:
                    result.graph_nodes.insert(0, node)
                    existing_ids.add(node.get("id"))

        # 5. Rerank: sort graph nodes by weight × recency (DNC signal) and
        #    Qdrant chunks by cosine similarity (already ordered, but apply
        #    recency tiebreak). No keyword boost — tested and hurts adversarial.
        #    Priority nodes skip reranking (already at front).
        n_priority = len(priority_nodes)
        reranked_qdrant, reranked_regular = rerank_context(
            query=query,
            qdrant_chunks=result.qdrant_chunks,
            graph_nodes=result.graph_nodes[n_priority:],
            top_qdrant=20,
            top_graph=25,
        )
        result.qdrant_chunks = reranked_qdrant
        result.graph_nodes = result.graph_nodes[:n_priority] + reranked_regular

        return result

    async def _qdrant_search(
        self, tenant_id: str, embedding: list[float]
    ) -> list[dict[str, Any]]:
        if not self._qdrant:
            return []
        try:
            # Use organization_id key to match v1 payload format so v2 can read
            # memories ingested by the v1 pipeline.  New v2 writes include both
            # organization_id and tenant_id in the payload (see _qdrant_upsert).
            hits = await self._qdrant.search(
                collection_name="memories",
                query_vector=embedding,
                limit=self._top_k,
                query_filter={"must": [{"key": "organization_id", "match": {"value": tenant_id}}]},
            )
            return [
                {"id": str(h.id), "score": h.score, "payload": h.payload or {}}
                for h in (hits or [])
            ]
        except Exception as exc:
            logger.warning("Qdrant search error: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Write weighting — Phase 3
    # ------------------------------------------------------------------

    async def write(
        self,
        tenant_id: str,
        session_id: str,
        utterance_text: str,
        role: str,
        prev_utterance_id: str | None = None,
        embedding: list[float] | None = None,
    ) -> WriteResult:
        result = WriteResult()

        if not self._graph.is_available():
            return result

        # 1. Extract entities from utterance
        entities: list[dict[str, Any]] = []
        temporal_ctx = parse_temporal_context(utterance_text)
        anchor_date = temporal_ctx.get("anchor_date")
        speaker = temporal_ctx.get("speaker")
        if self._extract_entities:
            try:
                entities = await self._extract_entities(utterance_text) or []
            except Exception as exc:
                logger.warning("Entity extraction failed: %s", exc)

        # 2. Upsert entity nodes into graph
        entity_ids: list[str] = []
        for ent in entities:
            eid = ent.get("id") or str(uuid.uuid4())
            name = ent.get("name", "unknown")
            etype = ent.get("type", "concept")
            try:
                await self._graph.upsert_entity(
                    tenant_id,
                    eid,
                    name,
                    etype,
                    attributes={
                        "content": ent.get("content"),
                        "subject": ent.get("subject"),
                        "attribute": ent.get("attribute"),
                        "value": ent.get("value"),
                        "canonical_date": ent.get("canonical_date"),
                    },
                )
                entity_ids.append(eid)
                result.graph_writes += 1
            except Exception as exc:
                logger.warning("Entity upsert failed: %s", exc)

        result.entity_ids = entity_ids

        # 2b. Accumulate personal_attribute entities into per-person profile nodes
        #     so facts survive beyond individual Utterance traversal depth.
        attr_by_subject: dict[str, list[str]] = {}
        high_signal_entities: list[dict[str, Any]] = []
        for ent in entities:
            etype = ent.get("type", "")
            if etype == "personal_attribute":
                subj = str(ent.get("subject") or "")
                attr = str(ent.get("attribute") or "")
                val = str(ent.get("value") or ent.get("name") or "")
                if subj and (attr or val):
                    fact = f"{attr}: {val}" if attr and val else (attr or val)
                    attr_by_subject.setdefault(subj, []).append(fact)
                    high_signal_entities.append(ent)
            elif etype == "temporal_event":
                high_signal_entities.append(ent)

        for person_name, facts in attr_by_subject.items():
            try:
                await self._graph.upsert_person_profile(tenant_id, person_name, facts)
            except Exception as exc:
                logger.warning("Profile upsert failed for %s: %s", person_name, exc)

        # 2c. Index high-signal entities separately in Qdrant for precision recall.
        #     Each personal_attribute and temporal_event gets its own short embedding
        #     so direct attribute questions ("What is X's hobby?") retrieve them precisely.
        if high_signal_entities and self._qdrant and self._embed:
            for ent in high_signal_entities[:20]:
                content = str(ent.get("content") or ent.get("name") or "")[:300]
                if not content:
                    continue
                # Use natural-language form for embedding to improve cosine similarity
                # with natural-language questions ("What is John's hobby?").
                etype = ent.get("type", "")
                if etype == "personal_attribute":
                    subj = str(ent.get("subject") or "")
                    attr = str(ent.get("attribute") or "").replace("_", " ")
                    val = str(ent.get("value") or "")
                    if subj and attr and val:
                        content = f"{subj}'s {attr} is {val}"
                try:
                    ent_embedding = await self._embed(content)
                    if ent_embedding:
                        ent_point_id = str(uuid.uuid4())
                        await self._qdrant_upsert(
                            tenant_id=tenant_id,
                            point_id=ent_point_id,
                            embedding=ent_embedding,
                            payload={
                                "text": content,
                                "type": ent.get("type", "entity"),
                                "entity_type": ent.get("type", "entity"),
                                "subject": str(ent.get("subject") or ""),
                                "attribute": str(ent.get("attribute") or ""),
                                "value": str(ent.get("value") or ""),
                                "canonical_date": str(ent.get("canonical_date") or ""),
                                "entity_id": str(ent.get("id") or ""),
                                "session_id": session_id,
                                "tenant_id": tenant_id,
                                "organization_id": tenant_id,
                                "anchor_date": anchor_date or "",
                                "speaker": speaker or "",
                                "created_at": int(time.time()),
                            },
                        )
                except Exception as exc:
                    logger.warning("Entity Qdrant index failed: %s", exc)

        # 3. Create Utterance node
        utt_id = str(uuid.uuid4())
        try:
            await self._graph.create_utterance(
                tenant_id,
                utt_id,
                utterance_text,
                role,
                session_id,
                anchor_date=anchor_date,
                speaker=speaker,
            )
            result.utterance_id = utt_id
            result.graph_writes += 1
        except Exception as exc:
            logger.warning("Utterance node creation failed: %s", exc)
            result.utterance_id = utt_id

        # 4. Link utterance → entities
        if entity_ids:
            try:
                await self._graph.link_utterance_to_entities(
                    tenant_id, utt_id, entity_ids
                )
                result.graph_writes += len(entity_ids)
            except Exception as exc:
                logger.warning("Utterance→entity linking failed: %s", exc)

        # 5. Chain utterances chronologically
        if prev_utterance_id:
            try:
                await self._graph.link_sequential_utterances(
                    tenant_id, prev_utterance_id, utt_id
                )
                result.graph_writes += 1
            except Exception as exc:
                logger.warning("Sequential utterance link failed: %s", exc)

        # 6. Upsert into Qdrant (episodic vector index)
        if embedding and self._qdrant:
            try:
                point_id = str(uuid.uuid4())
                await self._qdrant_upsert(
                    tenant_id=tenant_id,
                    point_id=point_id,
                    embedding=embedding,
                    payload={
                        "utterance_id": utt_id,
                        "session_id": session_id,
                        "role": role,
                        "text": utterance_text[:500],
                        "tenant_id": tenant_id,
                        "anchor_date": anchor_date or "",
                        "speaker": speaker or "",
                        # organization_id mirrors the v1 payload format so cross-engine
                        # reads work without re-ingesting (v1 stores with organization_id).
                        "organization_id": tenant_id,
                        "entity_ids": entity_ids,
                        "entity_names": [str(ent.get("name") or "") for ent in entities if ent.get("name")],
                        "created_at": int(time.time()),
                    },
                )
                result.qdrant_point_id = point_id
            except Exception as exc:
                logger.warning("Qdrant upsert failed: %s", exc)

        return result

    async def write_gist(
        self,
        tenant_id: str,
        session_id: str,
        gist_text: str,
        turn_start: int,
        turn_end: int,
    ) -> None:
        """Embed and store a dense segment summary in Qdrant for later retrieval."""
        if not self._qdrant or not gist_text:
            return
        embedding: list[float] = []
        if self._embed:
            try:
                embedding = await self._embed(gist_text)
            except Exception as exc:
                logger.warning("Gist embedding failed: %s", exc)
        if not embedding:
            return
        point_id = str(uuid.uuid4())
        await self._qdrant_upsert(
            tenant_id=tenant_id,
            point_id=point_id,
            embedding=embedding,
            payload={
                "text": gist_text,
                "type": "segment_gist",
                "tenant_id": tenant_id,
                "organization_id": tenant_id,
                "session_id": session_id,
                "turn_start": turn_start,
                "turn_end": turn_end,
                "created_at": int(time.time()),
            },
        )

    async def _qdrant_upsert(
        self,
        tenant_id: str,
        point_id: str,
        embedding: list[float],
        payload: dict[str, Any],
    ) -> None:
        if not self._qdrant:
            return
        try:
            from qdrant_client.models import PointStruct
            await self._qdrant.upsert(
                collection_name="memories",
                points=[PointStruct(id=point_id, vector=embedding, payload=payload)],
            )
        except Exception as exc:
            logger.warning("Qdrant upsert internal error: %s", exc)

    # ------------------------------------------------------------------
    # Decay / purge — Phase 3 tail
    # ------------------------------------------------------------------

    async def decay_and_prune(
        self,
        tenant_id: str,
        seed_ids: list[str],
    ) -> dict[str, int]:
        decayed = 0
        pruned = 0
        if not self._graph.is_available():
            return {"decayed": decayed, "pruned": pruned}
        if not seed_ids:
            return {"decayed": decayed, "pruned": pruned}
        try:
            decayed = await self._graph.decay_neighborhood(tenant_id, seed_ids)
        except Exception as exc:
            logger.warning("Decay step failed: %s", exc)
        try:
            pruned = await self._graph.prune_weak_edges(tenant_id)
        except Exception as exc:
            logger.warning("Prune step failed: %s", exc)
        return {"decayed": decayed, "pruned": pruned}
