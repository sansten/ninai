"""Semantic role inference agent (Phase 69)."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.llm.ollama_breaker import create_ollama_client
from app.agents.types import AgentContext, AgentResult
from app.core.config import settings


_ROLE_SIGNALS = {
    "deployer": ["deploy", "release", "rollout", "push", "ship"],
    "reviewer": ["review", "approve", "lgtm", "merge", "pr"],
    "incident_owner": ["incident", "postmortem", "rca", "on-call", "pager"],
    "architect": ["design", "architecture", "rfc", "proposal", "schema"],
    "approver": ["approved", "sign-off", "authorized", "granted"],
    "escalation_target": ["escalate", "escalated to", "notify", "alert"],
}
_WORD_RE = re.compile(r"\b[a-z0-9_\-]+\b", re.IGNORECASE)


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in _WORD_RE.findall(text or "")}


def _memory_text(memory: dict[str, Any]) -> str:
    parts = [str(memory.get("content") or ""), str(memory.get("action_type") or "")]
    tags = memory.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    parts.extend(str(t) for t in tags)
    return " ".join(parts)


def _confidence(evidence_count: int) -> float:
    return min(0.95, round(0.4 + 0.1 * max(0, int(evidence_count)), 4))


def run_heuristic(*, memories: list[dict[str, Any]], existing_roles: list[dict[str, Any]]) -> dict[str, Any]:
    memory_list = list(memories or [])

    baseline: dict[tuple[str, str], int] = defaultdict(int)
    for role in existing_roles or []:
        entity_id = str(role.get("entity_id") or "").strip()
        role_label = str(role.get("role_label") or "").strip().lower()
        if not entity_id or not role_label:
            continue
        baseline[(entity_id, role_label)] = max(
            baseline[(entity_id, role_label)],
            int(role.get("evidence_count") or 0),
        )

    evidence: Counter[tuple[str, str]] = Counter()
    users_in_memories: set[str] = set()

    signal_sets = {k: {_s.lower() for _s in v} for k, v in _ROLE_SIGNALS.items()}

    for memory in memory_list:
        user_id = str(memory.get("user_id") or "").strip()
        if not user_id:
            continue
        users_in_memories.add(user_id)

        tokens = _tokenize(_memory_text(memory))

        for role_label, signals in signal_sets.items():
            if tokens & signals:
                evidence[(user_id, role_label)] += 1

    inferred_roles: list[dict[str, Any]] = []
    by_user_role_count: defaultdict[str, dict[str, int]] = defaultdict(dict)

    all_keys = set(baseline.keys()) | set(evidence.keys())
    for entity_id, role_label in sorted(all_keys):
        count = baseline.get((entity_id, role_label), 0) + evidence.get((entity_id, role_label), 0)
        if count <= 0:
            continue
        conf = _confidence(count)
        inferred_roles.append(
            {
                "entity_id": entity_id,
                "entity_type": "user",
                "role_label": role_label,
                "evidence_count": count,
                "confidence": conf,
            }
        )
        by_user_role_count[entity_id][role_label] = count

    users_with_any_role = {r["entity_id"] for r in inferred_roles}
    role_coverage = round(len(users_with_any_role) / max(1, len(users_in_memories)), 4) if users_in_memories else 0.0

    conflicts: list[dict[str, Any]] = []
    for user_id, role_counts in by_user_role_count.items():
        deploy_count = int(role_counts.get("deployer", 0))
        review_count = int(role_counts.get("reviewer", 0))
        if deploy_count >= 2 and review_count >= 2:
            conflicts.append(
                {
                    "entity_id": user_id,
                    "roles": ["deployer", "reviewer"],
                    "deployer_evidence": deploy_count,
                    "reviewer_evidence": review_count,
                }
            )

    return {
        "inferred_roles": inferred_roles,
        "role_coverage": role_coverage,
        "conflicts": conflicts,
        "confidence": 0.8,
        "rationale": "heuristic",
    }


class SemanticRoleInferenceAgent(BaseAgent):
    name = "SemanticRoleInferenceAgent"
    version = "v1"

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}

        if not isinstance(outputs.get("inferred_roles"), list):
            raise ValueError("inferred_roles must be a list")
        coverage = outputs.get("role_coverage")
        if not isinstance(coverage, (int, float)) or not (0.0 <= float(coverage) <= 1.0):
            raise ValueError("role_coverage must be float between 0 and 1")
        if not isinstance(outputs.get("conflicts"), list):
            raise ValueError("conflicts must be a list")

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        enrichment = (context.get("memory") or {}).get("enrichment") or {}

        memories = list(enrichment.get("memories") or [])
        existing_roles = list(enrichment.get("existing_roles") or [])

        strategy = getattr(settings, "SEMANTIC_ROLE_INFERENCE_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        if strategy == "heuristic":
            outputs = run_heuristic(memories=memories, existing_roles=existing_roles)
        else:
            prompt = (
                "You infer semantic organizational roles from memory events. Output JSON only.\n\n"
                f"MEMORIES: {memories[:100]}\n"
                f"EXISTING_ROLES: {existing_roles[:100]}\n\n"
                "Return JSON with keys:\n"
                "- inferred_roles: list[{entity_id, entity_type, role_label, evidence_count, confidence}]\n"
                "- role_coverage: float\n"
                "- conflicts: list[dict]\n"
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
                and isinstance(resp.get("inferred_roles"), list)
                and isinstance(resp.get("role_coverage"), (int, float))
                and isinstance(resp.get("conflicts"), list)
            ):
                outputs = resp
            else:
                outputs = run_heuristic(memories=memories, existing_roles=existing_roles)

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
