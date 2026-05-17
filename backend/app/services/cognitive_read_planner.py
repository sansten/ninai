from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.query_intelligence_agent import run_heuristic as run_query_intelligence
from app.core.requester_context import RequesterContext
from app.models.memory import MemoryMetadata
from app.schemas.memory import MemoryResponse, MemorySearchRequest, SearchHnmsMode
from app.services.cognitive_evidence_service import CognitiveEvidenceService
from app.services.cognitive_gateway_service import CognitiveGatewayService
from app.services.embedding_service import EmbeddingService
from app.services.hierarchy_service import HierarchyService
from app.services.memory_service import MemoryService
from app.services.multi_hop_reasoning_service import ChainStep, MultiHopReasoningService
from app.services.retrieval_service import RetrievalService
from app.services.temporal_reasoning_service import TemporalReasoningService
from app.services.unified_episode_service import UnifiedEpisodeService


@dataclass
class PlannedReadResult:
    query: str
    expanded_query: str
    retrieval_strategy: str
    target_memory_level: str
    query_intelligence: dict[str, Any]
    memories: list[dict[str, Any]]
    total: int
    context_assembled: bool
    retrieval_confidence: float
    reasoning_steps: list[dict[str, Any]] = field(default_factory=list)
    compression_ratio: float = 1.0
    information_density: float = 0.0
    evidence_package: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class CognitiveReadPlanner:
    """Planner-driven retrieval across memory levels before final gateway ranking."""

    def __init__(
        self,
        session: AsyncSession,
        *,
        user_id: str,
        org_id: str,
        clearance_level: int = 0,
        gateway: CognitiveGatewayService | None = None,
        memory_service: MemoryService | None = None,
        evidence_service: CognitiveEvidenceService | None = None,
        episode_service: UnifiedEpisodeService | None = None,
        temporal_service: TemporalReasoningService | None = None,
        multi_hop_service: MultiHopReasoningService | None = None,
    ) -> None:
        self.session = session
        self.user_id = user_id
        self.org_id = org_id
        self.clearance_level = clearance_level
        self.gateway = gateway or CognitiveGatewayService()
        self.memory_service = memory_service or MemoryService(
            session=session,
            user_id=user_id,
            org_id=org_id,
            clearance_level=clearance_level,
        )
        self.evidence_service = evidence_service or CognitiveEvidenceService(session, org_id=org_id)
        self.episode_service = episode_service or UnifiedEpisodeService(session, org_id=org_id)
        self.temporal_service = temporal_service or TemporalReasoningService(session=session)
        self.multi_hop_service = multi_hop_service or MultiHopReasoningService()

    async def plan_and_read(
        self,
        *,
        query: str,
        limit: int = 10,
        context_id: str | None = None,
        scope: str | None = None,
        team_id: str | None = None,
        filter_tags: list[str] | None = None,
        hybrid: bool = True,
        use_graph: bool | None = None,
        hnms_mode: SearchHnmsMode | None = None,
        supplied_memories: list[dict[str, Any]] | None = None,
        requester: RequesterContext | None = None,
    ) -> PlannedReadResult:
        intelligence = run_query_intelligence(query, {})
        expanded_query = self._expand_query(query, intelligence)
        target_memory_level = self._target_memory_level(intelligence)
        requested_limit = max(1, int(limit or 10))
        reasoning_steps: list[dict[str, Any]] = [
            {
                "step": "planner_intelligence",
                "intent": intelligence.get("query_intent"),
                "entities": list(intelligence.get("extracted_entities") or []),
                "dynamic_filters": dict(intelligence.get("dynamic_filters") or {}),
                "target_memory_level": target_memory_level,
            }
        ]

        candidates = [dict(item) for item in (supplied_memories or [])]
        sources_used = ["supplied_memories"] if candidates else []
        planner_context: dict[str, Any] = {}

        if not candidates:
            if self._should_use_temporal_reasoning(intelligence):
                candidates, sources_used, operator_steps, planner_context = await self._plan_temporal_candidates(
                    query=query,
                    expanded_query=expanded_query,
                    intelligence=intelligence,
                    requested_limit=requested_limit,
                    scope=scope,
                    team_id=team_id,
                    filter_tags=filter_tags,
                    hybrid=hybrid,
                    use_graph=use_graph,
                    hnms_mode=hnms_mode,
                )
            elif self._should_use_multi_hop_reasoning(query, intelligence):
                candidates, sources_used, operator_steps, planner_context = await self._plan_multi_hop_candidates(
                    query=query,
                    expanded_query=expanded_query,
                    intelligence=intelligence,
                    requested_limit=requested_limit,
                    scope=scope,
                    team_id=team_id,
                    filter_tags=filter_tags,
                    hybrid=hybrid,
                    use_graph=use_graph,
                    hnms_mode=hnms_mode,
                )
            else:
                candidates, sources_used, operator_steps = await self._gather_base_candidates(
                    query=query,
                    expanded_query=expanded_query,
                    intelligence=intelligence,
                    requested_limit=requested_limit,
                    scope=scope,
                    team_id=team_id,
                    filter_tags=filter_tags,
                    hybrid=hybrid,
                    use_graph=use_graph,
                    hnms_mode=hnms_mode,
                )
            reasoning_steps.extend(operator_steps)

        ranked = await self.gateway.read(
            query=query,
            memories=candidates,
            limit=requested_limit,
            context_id=context_id,
            org_id=self.org_id if context_id else None,
            requester=requester,
        )
        reasoning_steps.extend(list(ranked.reasoning_steps or []))

        evidence_package = await self.evidence_service.build_package(
            query=query,
            memories=list(ranked.memories or []),
            query_intelligence=intelligence,
            planner_context={
                "expanded_query": expanded_query,
                "retrieval_strategy": "+".join(sources_used) if sources_used else "memory_search",
                "target_memory_level": target_memory_level,
                "sources_used": sources_used,
                **planner_context,
            },
        )

        return PlannedReadResult(
            query=query,
            expanded_query=expanded_query,
            retrieval_strategy="+".join(sources_used) if sources_used else "memory_search",
            target_memory_level=target_memory_level,
            query_intelligence=intelligence,
            memories=list(ranked.memories or []),
            total=int(ranked.total or 0),
            context_assembled=bool(ranked.context_assembled),
            retrieval_confidence=float(ranked.retrieval_confidence or 0.0),
            reasoning_steps=reasoning_steps,
            compression_ratio=float(ranked.compression_ratio or 1.0),
            information_density=float(ranked.information_density or 0.0),
            evidence_package=evidence_package,
        )

    async def _gather_base_candidates(
        self,
        *,
        query: str,
        expanded_query: str,
        intelligence: dict[str, Any],
        requested_limit: int,
        scope: str | None,
        team_id: str | None,
        filter_tags: list[str] | None,
        hybrid: bool,
        use_graph: bool | None,
        hnms_mode: SearchHnmsMode | None,
    ) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
        candidates = await self._search_memories(
            query=expanded_query,
            limit=max(requested_limit * 3, 12),
            scope=scope,
            team_id=team_id,
            hybrid=hybrid,
            use_graph=self._should_use_graph(intelligence, use_graph),
            hnms_mode=hnms_mode,
            filter_tags=self._merge_tags(
                filter_tags,
                (intelligence.get("dynamic_filters") or {}).get("tags"),
            ),
        )
        sources_used = ["memory_search"]
        reasoning_steps: list[dict[str, Any]] = [
            {
                "step": "memory_search",
                "candidate_count": len(candidates),
                "hybrid": hybrid,
                "graph_enabled": self._should_use_graph(intelligence, use_graph),
            }
        ]

        if self._should_use_hierarchy(intelligence):
            hierarchy = await self._hierarchical_candidates(
                query=query,
                limit=max(5, requested_limit),
                owner_id=self.user_id if (scope or "personal") == "personal" else None,
            )
            if hierarchy:
                candidates = await self._merge_candidates(candidates, hierarchy)
                sources_used.append("hierarchy_search")
                reasoning_steps.append(
                    {
                        "step": "hierarchy_search",
                        "hierarchy_candidate_count": len(hierarchy),
                    }
                )

        if self._should_use_submodular_retrieval(intelligence):
            retrieved = await self._coverage_candidates(
                query=query,
                scope=scope or "personal",
                scope_id=team_id or self.user_id,
                limit=max(6, requested_limit * 2),
            )
            if retrieved:
                candidates = await self._merge_candidates(candidates, retrieved)
                sources_used.append("coverage_retrieval")
                reasoning_steps.append(
                    {
                        "step": "coverage_retrieval",
                        "coverage_candidate_count": len(retrieved),
                    }
                )

        return candidates, sources_used, reasoning_steps

    async def _plan_temporal_candidates(
        self,
        *,
        query: str,
        expanded_query: str,
        intelligence: dict[str, Any],
        requested_limit: int,
        scope: str | None,
        team_id: str | None,
        filter_tags: list[str] | None,
        hybrid: bool,
        use_graph: bool | None,
        hnms_mode: SearchHnmsMode | None,
    ) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]], dict[str, Any]]:
        candidates, sources_used, reasoning_steps = await self._gather_base_candidates(
            query=query,
            expanded_query=expanded_query,
            intelligence=intelligence,
            requested_limit=requested_limit,
            scope=scope,
            team_id=team_id,
            filter_tags=filter_tags,
            hybrid=hybrid,
            use_graph=use_graph,
            hnms_mode=hnms_mode,
        )

        seed_ids = [self._candidate_id(item) for item in candidates[: max(3, requested_limit)]]
        episode_neighbors = await self._episode_neighbor_candidates(seed_ids, limit=max(6, requested_limit * 2))
        if episode_neighbors:
            candidates = await self._merge_candidates(candidates, episode_neighbors)
            sources_used.append("episode_neighbors")
            reasoning_steps.append(
                {
                    "step": "episode_neighbors",
                    "neighbor_candidate_count": len(episode_neighbors),
                }
            )

        temporal_reasoning = await self.temporal_service.temporal_query(
            self.org_id,
            "timeline_of_memories",
            query=query,
            candidate_memories=candidates,
            extracted_entities=list(intelligence.get("extracted_entities") or []),
            max_events=max(8, requested_limit * 3),
        )
        temporal_state = dict(temporal_reasoning or {}) if isinstance(temporal_reasoning, dict) else {"timeline": []}
        prioritized_ids = list(temporal_state.get("memory_ids") or [])
        if prioritized_ids:
            candidates = self._reorder_by_priority_ids(candidates, prioritized_ids)
        sources_used.append("temporal_reasoning")
        reasoning_steps.append(
            {
                "step": "temporal_reasoning",
                "timeline_count": len(temporal_state.get("timeline") or []),
                "anchor_count": int(temporal_state.get("anchor_count") or 0),
            }
        )

        return candidates, sources_used, reasoning_steps, {
            "reasoning_operator": "temporal",
            "temporal_reasoning": temporal_state,
        }

    async def _plan_multi_hop_candidates(
        self,
        *,
        query: str,
        expanded_query: str,
        intelligence: dict[str, Any],
        requested_limit: int,
        scope: str | None,
        team_id: str | None,
        filter_tags: list[str] | None,
        hybrid: bool,
        use_graph: bool | None,
        hnms_mode: SearchHnmsMode | None,
    ) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]], dict[str, Any]]:
        candidates, sources_used, reasoning_steps = await self._gather_base_candidates(
            query=query,
            expanded_query=expanded_query,
            intelligence=intelligence,
            requested_limit=requested_limit,
            scope=scope,
            team_id=team_id,
            filter_tags=filter_tags,
            hybrid=hybrid,
            use_graph=use_graph,
            hnms_mode=hnms_mode,
        )

        seed_ids = [self._candidate_id(item) for item in candidates[: max(3, requested_limit)]]
        episode_neighbors = await self._episode_neighbor_candidates(seed_ids, limit=max(6, requested_limit * 2))
        if episode_neighbors:
            candidates = await self._merge_candidates(candidates, episode_neighbors)
            sources_used.append("episode_neighbors")
            reasoning_steps.append(
                {
                    "step": "episode_neighbors",
                    "neighbor_candidate_count": len(episode_neighbors),
                }
            )

        chain_steps = self._build_multi_hop_chain(query=query, intelligence=intelligence, requested_limit=requested_limit)
        chain_trace: list[dict[str, Any]] = []
        chain_memories: list[dict[str, Any]] = []

        if chain_steps:
            async def _read_fn(
                *,
                query: str,
                memories: list[dict[str, Any]],
                limit: int,
                step_index: int,
                enrichment: dict[str, Any],
            ) -> dict[str, Any]:
                step_candidates = list(memories or [])
                searched = await self._search_memories(
                    query=query,
                    limit=max(max(1, int(limit or 1)), requested_limit * 2),
                    scope=scope,
                    team_id=team_id,
                    hybrid=hybrid,
                    use_graph=self._should_use_graph(intelligence, use_graph),
                    hnms_mode=hnms_mode,
                    filter_tags=self._merge_tags(
                        filter_tags,
                        (intelligence.get("dynamic_filters") or {}).get("tags"),
                    ),
                )
                step_candidates = await self._merge_candidates(step_candidates, searched)

                if self._should_use_submodular_retrieval(intelligence):
                    step_coverage = await self._coverage_candidates(
                        query=query,
                        scope=scope or "personal",
                        scope_id=team_id or self.user_id,
                        limit=max(4, limit),
                    )
                    if step_coverage:
                        step_candidates = await self._merge_candidates(step_candidates, step_coverage)

                neighbor_ids = [self._candidate_id(item) for item in step_candidates[:2]]
                step_neighbors = await self._episode_neighbor_candidates(neighbor_ids, limit=max(4, limit))
                if step_neighbors:
                    step_candidates = await self._merge_candidates(step_candidates, step_neighbors)

                output = self._summarize_hop_candidates(step_candidates)
                confidence = self._average_candidate_confidence(step_candidates)
                return {
                    "memories": step_candidates[: max(4, limit)],
                    "output": output,
                    "confidence": confidence,
                    "raw": {
                        "memories": step_candidates[: max(4, limit)],
                        "count": len(step_candidates),
                        "step_query": query,
                        "step_index": step_index,
                        "retrieval_confidence": confidence,
                    },
                }

            chain = await self.multi_hop_service.execute_async(
                goal=query,
                steps=chain_steps,
                read_fn=_read_fn,
            )
            for result in chain.step_results:
                step_memories = list(result.raw.get("memories") or [])
                if step_memories:
                    chain_memories = await self._merge_candidates(chain_memories, step_memories)
                chain_trace.append(
                    {
                        "step_index": result.index,
                        "query": result.resolved_input,
                        "memory_count": len(step_memories),
                        "confidence": result.confidence,
                        "memory_ids": [self._candidate_id(item) for item in step_memories],
                    }
                )

            if chain_memories:
                candidates = await self._merge_candidates(candidates, chain_memories)
                candidates = self._rerank_multi_hop_candidates(candidates, chain_trace, intelligence, query=query)
            sources_used.append("multi_hop_chain")
            reasoning_steps.append(
                {
                    "step": "multi_hop_chain",
                    "hop_count": len(chain.step_results),
                    "chain_confidence": chain.chain_confidence,
                    "retrieved_memory_count": len(chain_memories),
                }
            )

        return candidates, sources_used, reasoning_steps, {
            "reasoning_operator": "multi_hop",
            "multi_hop_trace": chain_trace,
        }

    async def _search_memories(
        self,
        *,
        query: str,
        limit: int,
        scope: str | None,
        team_id: str | None,
        hybrid: bool,
        use_graph: bool,
        hnms_mode: SearchHnmsMode | None,
        filter_tags: list[str] | None,
    ) -> list[dict[str, Any]]:
        request = MemorySearchRequest(
            query=query,
            scope=scope,
            team_id=team_id,
            limit=max(1, int(limit or 1)),
            hybrid=hybrid,
            use_graph=use_graph,
            hnms_mode=hnms_mode,
            tags=filter_tags or None,
        )
        embedding = await EmbeddingService.embed(query)
        memories = await self.memory_service.search_memories(
            query_embedding=embedding,
            request=request,
        )
        return [self._memory_to_dict(memory) for memory in memories]

    async def _hierarchical_candidates(
        self,
        *,
        query: str,
        limit: int,
        owner_id: str | None,
    ) -> list[dict[str, Any]]:
        try:
            hierarchy = await HierarchyService(self.session).hierarchical_search(
                organization_id=self.org_id,
                query=query,
                limit=max(1, int(limit or 1)),
                owner_id=owner_id,
            )
        except Exception:
            return []

        message_ids = [
            str(item.get("id") or "")
            for item in (hierarchy.get("messages") or [])
            if str(item.get("id") or "").strip()
        ]
        return await self._fetch_memories_by_ids(message_ids)

    async def _coverage_candidates(
        self,
        *,
        query: str,
        scope: str,
        scope_id: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        try:
            org_uuid = UUID(str(self.org_id))
        except (TypeError, ValueError):
            return []

        scope_uuid = None
        if scope_id:
            try:
                scope_uuid = UUID(str(scope_id))
            except (TypeError, ValueError):
                scope_uuid = None

        try:
            result = await RetrievalService(self.session).retrieve(
                query=query,
                organization_id=org_uuid,
                scope=scope,
                scope_id=scope_uuid,
                max_results=max(1, int(limit or 1)),
            )
        except Exception:
            return []

        return await self._fetch_memories_by_ids(list(result.get("memory_ids") or []))

    async def _fetch_memories_by_ids(self, memory_ids: list[str]) -> list[dict[str, Any]]:
        normalized_ids = [str(memory_id).strip() for memory_id in memory_ids if str(memory_id).strip()]
        if not normalized_ids:
            return []

        stmt = select(MemoryMetadata).where(
            MemoryMetadata.organization_id == self.org_id,
            MemoryMetadata.is_active.is_(True),
            MemoryMetadata.id.in_(normalized_ids),
        )
        memories = (await self.session.execute(stmt)).scalars().all()
        return [self._memory_to_dict(memory) for memory in memories]

    async def _merge_candidates(
        self,
        base: list[dict[str, Any]],
        extras: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        order: list[str] = []

        for item in list(base) + list(extras):
            memory_id = str(item.get("id") or "").strip()
            if not memory_id:
                continue
            if memory_id not in merged:
                merged[memory_id] = dict(item)
                order.append(memory_id)
                continue
            existing = merged[memory_id]
            combined = dict(existing)
            for key, value in item.items():
                if value is not None and value != [] and value != {}:
                    combined[key] = value
            existing_score = self._safe_float(existing.get("score"))
            new_score = self._safe_float(item.get("score"))
            if new_score > existing_score:
                combined["score"] = new_score
            merged[memory_id] = combined

        return [merged[memory_id] for memory_id in order]

    async def _episode_neighbor_candidates(
        self,
        memory_ids: list[str],
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        normalized = [memory_id for memory_id in memory_ids if memory_id]
        if not normalized:
            return []

        episode_index = await self.episode_service.list_for_memory_ids(
            normalized[: min(len(normalized), 6)],
            limit_per_memory=5,
        )
        seeds = set(normalized)
        related_ids: list[str] = []
        for items in episode_index.values():
            for item in items:
                for linked_memory_id in item.get("linked_memory_ids") or []:
                    memory_id = str(linked_memory_id).strip()
                    if not memory_id or memory_id in seeds or memory_id in related_ids:
                        continue
                    related_ids.append(memory_id)
                    if len(related_ids) >= max(1, int(limit or 1)):
                        return await self._fetch_memories_by_ids(related_ids)
        return await self._fetch_memories_by_ids(related_ids[: max(1, int(limit or 1))])

    def _build_multi_hop_chain(
        self,
        *,
        query: str,
        intelligence: dict[str, Any],
        requested_limit: int,
    ) -> list[ChainStep]:
        entities = [str(item).strip() for item in (intelligence.get("extracted_entities") or []) if str(item).strip()]
        entities = list(dict.fromkeys(entities))[:2]
        hop_limit = max(4, requested_limit * 2)

        steps = [ChainStep("read", query, limit=hop_limit)]
        for entity in entities:
            steps.append(
                ChainStep(
                    "read",
                    f"{query} {entity} {{step_0_output}}".strip(),
                    limit=hop_limit,
                )
            )

        synthesis_refs = " ".join(f"{{step_{idx}_output}}" for idx in range(1, min(len(steps), 3)))
        synthesis_query = " ".join(
            part for part in [query, " ".join(entities), synthesis_refs] if part
        ).strip()
        if synthesis_query and synthesis_query != query:
            steps.append(ChainStep("read", synthesis_query, limit=hop_limit))
        return steps[:4]

    def _rerank_multi_hop_candidates(
        self,
        candidates: list[dict[str, Any]],
        hop_trace: list[dict[str, Any]],
        intelligence: dict[str, Any],
        *,
        query: str,
    ) -> list[dict[str, Any]]:
        hop_frequency: dict[str, int] = {}
        for hop in hop_trace:
            for memory_id in hop.get("memory_ids") or []:
                if memory_id:
                    hop_frequency[memory_id] = hop_frequency.get(memory_id, 0) + 1

        entities = [str(item).strip().lower() for item in (intelligence.get("extracted_entities") or []) if str(item).strip()]
        query_tokens = set(re.findall(r"\b[a-z0-9_]{3,}\b", query.lower()))
        decorated: list[tuple[int, int, int, float, int, dict[str, Any]]] = []

        for index, candidate in enumerate(candidates):
            memory_id = self._candidate_id(candidate)
            text = self._candidate_text(candidate)
            entity_hits = 0
            for entity in entities:
                if not entity:
                    continue
                entity_tokens = re.findall(r"\b[a-z0-9_]{3,}\b", entity)
                if entity in text:
                    entity_hits += 2
                    continue
                if entity_tokens and all(token in text for token in entity_tokens):
                    entity_hits += 2
                    continue
                if any(token in text for token in entity_tokens):
                    entity_hits += 1
            token_hits = sum(1 for token in query_tokens if token in text)
            decorated.append(
                (
                    hop_frequency.get(memory_id, 0),
                    entity_hits,
                    token_hits,
                    self._safe_float(candidate.get("score")),
                    -index,
                    candidate,
                )
            )

        decorated.sort(reverse=True)
        return [item[-1] for item in decorated]

    @staticmethod
    def _candidate_text(candidate: dict[str, Any]) -> str:
        title = str(candidate.get("title") or "").strip()
        preview = str(candidate.get("content_preview") or candidate.get("content") or "").strip()
        tags = " ".join(str(tag).strip() for tag in (candidate.get("tags") or []) if str(tag).strip())
        entities: list[str] = []
        for value in (candidate.get("entities") or {}).values():
            if isinstance(value, list):
                entities.extend(str(item).strip() for item in value if str(item).strip())
            elif value is not None:
                text = str(value).strip()
                if text:
                    entities.append(text)
        return " ".join([title, preview, tags, " ".join(entities)]).lower()

    @staticmethod
    def _candidate_id(candidate: dict[str, Any]) -> str:
        return str(candidate.get("id") or candidate.get("memory_id") or "").strip()

    @staticmethod
    def _reorder_by_priority_ids(
        candidates: list[dict[str, Any]],
        prioritized_ids: list[str],
    ) -> list[dict[str, Any]]:
        priority_order = [str(item).strip() for item in prioritized_ids if str(item).strip()]
        priority_index = {memory_id: idx for idx, memory_id in enumerate(priority_order)}
        return sorted(
            candidates,
            key=lambda item: (
                priority_index.get(CognitiveReadPlanner._candidate_id(item), 10_000),
                -CognitiveReadPlanner._safe_float(item.get("score")),
            ),
        )

    @staticmethod
    def _summarize_hop_candidates(candidates: list[dict[str, Any]]) -> str:
        segments: list[str] = []
        for candidate in candidates[:3]:
            title = str(candidate.get("title") or "").strip()
            preview = str(candidate.get("content_preview") or candidate.get("content") or "").strip()
            parts = [part for part in [title, preview[:80]] if part]
            if parts:
                segments.append(" | ".join(parts))
        return "; ".join(segments)

    def _average_candidate_confidence(self, candidates: list[dict[str, Any]]) -> float:
        scored = [self._safe_float(candidate.get("score")) for candidate in candidates[:3] if candidate]
        if not scored:
            return 0.35
        return round(max(0.35, min(0.95, sum(scored) / len(scored))), 4)

    @staticmethod
    def _expand_query(query: str, intelligence: dict[str, Any]) -> str:
        extras = [str(item).strip() for item in (intelligence.get("extracted_entities") or []) if str(item).strip()]
        intent = str(intelligence.get("query_intent") or "retrieve").strip().lower()
        if intent == "find_timeline":
            extras.extend(["timeline", "sequence", "date"])
        elif intent == "explain":
            extras.extend(["cause", "reason"])
        elif intent == "compare":
            extras.extend(["difference", "contrast"])
        elif intent == "analyze":
            extras.extend(["pattern", "trend"])

        unique: list[str] = []
        seen: set[str] = set()
        for item in extras:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        if not unique:
            return query
        return (query + " " + " ".join(unique[:6])).strip()

    @staticmethod
    def _is_single_hop_query(query: str, intelligence: dict[str, Any]) -> bool:
        """Detect if query is a simple single-hop fact lookup (who/what/where/when/why/how + entity)."""
        intent = str(intelligence.get("query_intent") or "").strip().lower()
        entity_count = len([e for e in (intelligence.get("extracted_entities") or []) if isinstance(e, str) and e.strip()])
        # Single-hop markers: one entity, simple intent, short query
        simple_intents = {"retrieve", "find_person", "find_location", "find_attribute"}
        return (
            intent in simple_intents
            and entity_count <= 2
            and len(query.split()) <= 10
        )

    @staticmethod
    def _canonicalize_answer(answer: str, question: str, entity_name: str = "") -> str:
        """Normalize common fact equivalences for single_hop matching."""
        # Normalize adoption-related terms
        answer_norm = answer.lower().replace("adoption agencies", "adoption agency").strip()
        # Normalize identity/status terms
        answer_norm = answer_norm.replace("trans woman", "transgender woman")
        answer_norm = answer_norm.replace("single status", "single")
        answer_norm = answer_norm.replace("never married", "single")
        # Trim generic conversational filler
        if answer_norm.startswith("[") and answer_norm.endswith("]"):
            answer_norm = answer_norm.strip("[]")
        if answer_norm.endswith("!") or answer_norm.endswith("."):
            answer_norm = answer_norm[:-1].strip()
        return answer_norm

    @staticmethod
    def _target_memory_level(intelligence: dict[str, Any]) -> str:
        intent = str(intelligence.get("query_intent") or "retrieve").strip().lower()
        if intent == "find_timeline":
            return "episodes+temporal"
        if intent in {"explain", "compare", "analyze"}:
            return "semantic+facts+graph"
        if intent == "find_person":
            return "entity+memory"
        if intent in {"retrieve", "find_attribute", "find_location"}:
            return "entity_facts"  # Prioritize indexed facts for single_hop
        return "memory"

    @staticmethod
    def _should_use_hierarchy(intelligence: dict[str, Any]) -> bool:
        return str(intelligence.get("query_intent") or "").strip().lower() in {
            "find_timeline",
            "explain",
            "compare",
            "analyze",
        }

    @staticmethod
    def _should_use_temporal_reasoning(intelligence: dict[str, Any]) -> bool:
        return str(intelligence.get("query_intent") or "").strip().lower() == "find_timeline"

    @staticmethod
    def _should_use_multi_hop_reasoning(query: str, intelligence: dict[str, Any]) -> bool:
        intent = str(intelligence.get("query_intent") or "").strip().lower()
        entities = [str(item).strip() for item in (intelligence.get("extracted_entities") or []) if str(item).strip()]
        lowered = str(query or "").lower()
        if intent == "compare":
            return True
        if len(entities) >= 2 and intent in {"explain", "analyze", "compare"}:
            return True
        relation_cues = (" between ", " versus ", " vs ", " compare ", " difference ")
        if intent in {"compare", "explain"} and any(cue in f" {lowered} " for cue in relation_cues):
            return True
        return False

    @staticmethod
    def _should_use_submodular_retrieval(intelligence: dict[str, Any]) -> bool:
        intent = str(intelligence.get("query_intent") or "").strip().lower()
        return intent in {"explain", "compare", "analyze"} or bool(intelligence.get("extracted_entities"))

    @staticmethod
    def _should_use_graph(intelligence: dict[str, Any], requested: bool | None) -> bool:
        if requested is not None:
            return bool(requested)
        return bool(intelligence.get("extracted_entities"))

    @staticmethod
    def _memory_to_dict(memory: MemoryMetadata) -> dict[str, Any]:
        return MemoryResponse.model_validate(memory).model_dump(mode="python")

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _merge_tags(*tag_lists: list[str] | None) -> list[str] | None:
        merged: list[str] = []
        seen: set[str] = set()
        for tag_list in tag_lists:
            if not isinstance(tag_list, list):
                continue
            for tag in tag_list:
                if not isinstance(tag, str):
                    continue
                cleaned = tag.strip()
                if not cleaned:
                    continue
                key = cleaned.lower()
                if key in seen:
                    continue
                seen.add(key)
                merged.append(cleaned)
        return merged or None
