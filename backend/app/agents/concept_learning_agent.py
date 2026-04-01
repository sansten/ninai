"""Self-supervised concept learning agent (Phase 59)."""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.llm.ollama_breaker import create_ollama_client
from app.agents.types import AgentContext, AgentResult
from app.core.config import settings


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b[a-z0-9_]+\b", (text or "").lower()))


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _memory_tokens(memory: dict[str, Any]) -> set[str]:
    content = str(memory.get("content") or memory.get("content_preview") or "")
    tags = memory.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    return _tokenize(content + " " + " ".join(str(t) for t in tags))


def _top_terms(token_sets: list[set[str]], limit: int = 5) -> list[str]:
    counts: Counter[str] = Counter()
    for tokens in token_sets:
        counts.update(tokens)
    return [term for term, _ in counts.most_common(limit)]


def _concept_name(terms: list[str]) -> str:
    if not terms:
        return "general_concept"
    if len(terms) == 1:
        return terms[0]
    return f"{terms[0]}_{terms[1]}"


def _avg_pairwise_jaccard(token_sets: list[set[str]]) -> float:
    if len(token_sets) <= 1:
        return 1.0
    total = 0.0
    count = 0
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            total += jaccard_similarity(token_sets[i], token_sets[j])
            count += 1
    return total / max(1, count)


class _UnionFind:
    def __init__(self, ids: list[str]):
        self.parent = {i: i for i in ids}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def run_heuristic(
    *,
    memories: list[dict[str, Any]],
    existing_concepts: list[dict[str, Any]],
    min_cluster_size: int = 3,
) -> dict[str, Any]:
    memory_list = list(memories or [])
    min_size = max(1, int(min_cluster_size or 3))

    tokens_by_id: dict[str, set[str]] = {}
    memories_by_id: dict[str, dict[str, Any]] = {}
    for mem in memory_list:
        mem_id = str(mem.get("id") or "")
        if not mem_id:
            continue
        memories_by_id[mem_id] = mem
        tokens_by_id[mem_id] = _memory_tokens(mem)

    updated_concepts: list[dict[str, Any]] = []
    assigned_ids: set[str] = set()

    for concept in existing_concepts or []:
        name = str(concept.get("concept_name") or "").strip()
        if not name:
            continue
        canonical = {str(t).strip().lower() for t in (concept.get("canonical_terms") or []) if str(t).strip()}
        members = [str(m) for m in (concept.get("member_memory_ids") or concept.get("member_ids") or []) if str(m)]

        newly_absorbed: list[str] = []
        for mem_id, tokens in tokens_by_id.items():
            if mem_id in assigned_ids:
                continue
            sim = jaccard_similarity(tokens, canonical)
            if sim >= 0.3:
                newly_absorbed.append(mem_id)
                assigned_ids.add(mem_id)

        if newly_absorbed:
            full_members = list(dict.fromkeys(members + newly_absorbed))
            concept_tokens = [tokens_by_id[mid] for mid in full_members if mid in tokens_by_id]
            canonical_terms = _top_terms(concept_tokens, limit=5)
            updated_concepts.append(
                {
                    "concept_name": name,
                    "member_ids": full_members,
                    "new_member_ids": newly_absorbed,
                    "canonical_terms": canonical_terms,
                    "confidence": round(_avg_pairwise_jaccard(concept_tokens), 4),
                }
            )

    pending_ids = [mid for mid in tokens_by_id if mid not in assigned_ids]
    uf = _UnionFind(pending_ids)

    for i in range(len(pending_ids)):
        for j in range(i + 1, len(pending_ids)):
            a = pending_ids[i]
            b = pending_ids[j]
            if jaccard_similarity(tokens_by_id[a], tokens_by_id[b]) >= 0.3:
                uf.union(a, b)

    groups: dict[str, list[str]] = {}
    for mid in pending_ids:
        root = uf.find(mid)
        groups.setdefault(root, []).append(mid)

    new_concepts: list[dict[str, Any]] = []
    concept_member_ids: set[str] = set()

    for component in groups.values():
        if len(component) < min_size:
            continue
        component_tokens = [tokens_by_id[mid] for mid in component]
        canonical_terms = _top_terms(component_tokens, limit=5)
        new_concepts.append(
            {
                "concept_name": _concept_name(canonical_terms),
                "member_ids": component,
                "canonical_terms": canonical_terms,
                "confidence": round(_avg_pairwise_jaccard(component_tokens), 4),
            }
        )
        concept_member_ids.update(component)

    noise_memories = [mid for mid in pending_ids if mid not in concept_member_ids]
    total = len(new_concepts) + len(updated_concepts)

    return {
        "new_concepts": new_concepts,
        "updated_concepts": updated_concepts,
        "noise_memories": noise_memories,
        "total_concepts_found": total,
        "confidence": round(min(0.9, 0.5 + 0.05 * total), 4),
        "rationale": "heuristic",
    }


class ConceptLearningAgent(BaseAgent):
    name = "ConceptLearningAgent"
    version = "v1"

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}

        if not isinstance(outputs.get("new_concepts"), list):
            raise ValueError("new_concepts must be a list")
        if not isinstance(outputs.get("updated_concepts"), list):
            raise ValueError("updated_concepts must be a list")
        if not isinstance(outputs.get("noise_memories"), list):
            raise ValueError("noise_memories must be a list")
        if not isinstance(outputs.get("total_concepts_found"), int):
            raise ValueError("total_concepts_found must be an int")

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        enrichment = (context.get("memory") or {}).get("enrichment") or {}

        memories = list(enrichment.get("memories") or [])
        existing_concepts = list(enrichment.get("existing_concepts") or [])
        min_cluster_size = int(enrichment.get("min_cluster_size") or 3)

        strategy = getattr(settings, "CONCEPT_LEARNING_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        if strategy == "heuristic":
            outputs = run_heuristic(
                memories=memories,
                existing_concepts=existing_concepts,
                min_cluster_size=min_cluster_size,
            )
        else:
            prompt = (
                "You perform self-supervised concept learning over memory items. Output JSON only.\n\n"
                f"MEMORIES: {memories[:40]}\n"
                f"EXISTING_CONCEPTS: {existing_concepts[:20]}\n"
                f"MIN_CLUSTER_SIZE: {min_cluster_size}\n\n"
                "Return JSON with keys:\n"
                "- new_concepts: list[{concept_name, member_ids, canonical_terms, confidence}]\n"
                "- updated_concepts: list[{concept_name, member_ids, canonical_terms, confidence}]\n"
                "- noise_memories: list[str]\n"
                "- total_concepts_found: int\n"
                "- confidence: float\n"
                "- rationale: str"
            )
            client = create_ollama_client(
                base_url=str(getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")),
                model=str(settings.get_ollama_model("agents")),
                timeout_seconds=float(getattr(settings, "OLLAMA_TIMEOUT_SECONDS", 5.0)),
                max_concurrency=int(getattr(settings, "OLLAMA_MAX_CONCURRENCY", 2)),
            )
            resp = await client.complete_json(
                prompt=prompt,
                schema_hint={},
                tool_event_sink=context.get("tool_event_sink"),
            )
            if (
                isinstance(resp, dict)
                and isinstance(resp.get("new_concepts"), list)
                and isinstance(resp.get("updated_concepts"), list)
                and isinstance(resp.get("noise_memories"), list)
                and isinstance(resp.get("total_concepts_found"), int)
            ):
                outputs = resp
            else:
                outputs = run_heuristic(
                    memories=memories,
                    existing_concepts=existing_concepts,
                    min_cluster_size=min_cluster_size,
                )

        finished_at = datetime.now(timezone.utc)
        result = AgentResult(
            agent_name=self.name,
            agent_version=self.version,
            memory_id=memory_id,
            status="success",
            confidence=float(outputs.get("confidence") or 0.5),
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=started_at,
            finished_at=finished_at,
            trace_id=trace_id,
        )
        self.validate_outputs(result)
        return result
