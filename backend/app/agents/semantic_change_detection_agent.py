"""Semantic change detection agent (Phase 57)."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.llm.ollama_breaker import create_ollama_client
from app.agents.types import AgentContext, AgentResult
from app.core.config import settings

_NEGATION_TOKENS = {
    "not",
    "no",
    "never",
    "none",
    "without",
    "cannot",
    "cant",
    "isnt",
    "wasnt",
    "wont",
    "dont",
    "didnt",
}


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b[a-z0-9_]+\b", (text or "").lower()))


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _has_negation_shift(new_tokens: set[str], old_tokens: set[str]) -> bool:
    new_has = bool(new_tokens & _NEGATION_TOKENS)
    old_has = bool(old_tokens & _NEGATION_TOKENS)
    return new_has != old_has


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def run_heuristic(
    *,
    new_content: str,
    existing_memories: list[dict[str, Any]],
    change_threshold: float = 0.4,
) -> dict[str, Any]:
    memories = list(existing_memories or [])
    threshold = float(change_threshold)
    new_tokens = _tokenize(new_content)

    if not memories:
        return {
            "change_detected": False,
            "changed_memories": [],
            "change_type": "unrelated",
            "semantic_drift_score": 0.0,
            "recommended_action": "ignore",
            "confidence": 0.9,
            "rationale": "heuristic",
        }

    changed_memories: list[dict[str, Any]] = []
    per_memory_types: list[str] = []
    similarities: list[float] = []

    for memory in memories:
        content = str(memory.get("content") or memory.get("content_preview") or "")
        old_tokens = _tokenize(content)
        similarity = jaccard_similarity(new_tokens, old_tokens)
        similarities.append(similarity)

        mem_change_type: str
        if _has_negation_shift(new_tokens, old_tokens):
            mem_change_type = "contradiction"
        elif similarity < threshold:
            mem_change_type = "update"
        elif similarity < 0.8:
            mem_change_type = "update"
        else:
            mem_change_type = "extension"

        per_memory_types.append(mem_change_type)

        if similarity < threshold:
            changed_memories.append(
                {
                    "memory_id": str(memory.get("id") or ""),
                    "old_content_snippet": content[:120],
                    "similarity": round(similarity, 4),
                    "change_type": mem_change_type,
                }
            )

    max_similarity = max(similarities) if similarities else 1.0
    semantic_drift_score = _clip01(1.0 - max_similarity)
    change_detected = any(sim < threshold for sim in similarities)

    if "contradiction" in per_memory_types:
        change_type = "contradiction"
    elif "update" in per_memory_types:
        change_type = "update"
    elif "extension" in per_memory_types:
        change_type = "extension"
    else:
        change_type = "unrelated"

    action_map = {
        "contradiction": "supersede",
        "update": "flag_review",
        "extension": "append",
        "unrelated": "ignore",
    }
    recommended_action = action_map[change_type]

    return {
        "change_detected": change_detected,
        "changed_memories": changed_memories,
        "change_type": change_type,
        "semantic_drift_score": round(semantic_drift_score, 4),
        "recommended_action": recommended_action,
        "confidence": 0.75 if change_detected else 0.9,
        "rationale": "heuristic",
    }


class SemanticChangeDetectionAgent(BaseAgent):
    name = "SemanticChangeDetectionAgent"
    version = "v1"

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}

        if not isinstance(outputs.get("change_detected"), bool):
            raise ValueError("change_detected must be a bool")

        if not isinstance(outputs.get("changed_memories"), list):
            raise ValueError("changed_memories must be a list")

        change_type = outputs.get("change_type")
        if change_type not in {"contradiction", "update", "extension", "unrelated"}:
            raise ValueError("change_type must be contradiction|update|extension|unrelated")

        drift = outputs.get("semantic_drift_score")
        if not isinstance(drift, (int, float)) or not (0.0 <= float(drift) <= 1.0):
            raise ValueError("semantic_drift_score must be float in [0,1]")

        action = outputs.get("recommended_action")
        if action not in {"supersede", "flag_review", "append", "ignore"}:
            raise ValueError("recommended_action must be supersede|flag_review|append|ignore")

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        enrichment = (context.get("memory") or {}).get("enrichment") or {}

        new_content = str(enrichment.get("new_content") or "")
        existing_memories = list(enrichment.get("existing_memories") or [])
        change_threshold = float(enrichment.get("change_threshold") or 0.4)

        strategy = getattr(settings, "SEMANTIC_CHANGE_DETECTION_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        if strategy == "heuristic":
            outputs = run_heuristic(
                new_content=new_content,
                existing_memories=existing_memories,
                change_threshold=change_threshold,
            )
        else:
            prompt = (
                "You detect semantic change between new and existing memory text. Output JSON only.\n\n"
                f"NEW_CONTENT: {new_content}\n"
                f"CHANGE_THRESHOLD: {change_threshold}\n"
                f"EXISTING_MEMORIES: {existing_memories[:20]}\n\n"
                "Return JSON with keys:\n"
                "- change_detected: bool\n"
                "- changed_memories: list[{memory_id, old_content_snippet, similarity, change_type}]\n"
                "- change_type: contradiction|update|extension|unrelated\n"
                "- semantic_drift_score: float in [0,1]\n"
                "- recommended_action: supersede|flag_review|append|ignore\n"
                "- confidence: float in [0,1]\n"
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
                and isinstance(resp.get("change_detected"), bool)
                and isinstance(resp.get("changed_memories"), list)
                and resp.get("change_type") in {"contradiction", "update", "extension", "unrelated"}
                and isinstance(resp.get("semantic_drift_score"), (int, float))
                and resp.get("recommended_action") in {"supersede", "flag_review", "append", "ignore"}
            ):
                outputs = resp
            else:
                outputs = run_heuristic(
                    new_content=new_content,
                    existing_memories=existing_memories,
                    change_threshold=change_threshold,
                )

        finished_at = datetime.now(timezone.utc)
        result = AgentResult(
            agent_name=self.name,
            agent_version=self.version,
            memory_id=memory_id,
            status="success",
            confidence=float(outputs.get("confidence") or 0.75),
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=started_at,
            finished_at=finished_at,
            trace_id=trace_id,
        )
        self.validate_outputs(result)
        return result
