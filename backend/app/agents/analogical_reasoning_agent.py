"""Analogical reasoning agent — skill transfer via structural similarity (Phase 54)."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.llm.llm_breaker import create_llm_client
from app.agents.types import AgentContext, AgentResult
from app.core.config import settings

# ---------------------------------------------------------------------------
# Domain substitution map (static)
# ---------------------------------------------------------------------------

_DOMAIN_SUBSTITUTIONS: dict[str, str] = {
    "database": "cache",
    "index": "cache_key",
    "query": "request",
    "table": "bucket",
    "row": "entry",
    "postgres": "redis",
    "mysql": "memcached",
    "latency": "response_time",
    "timeout": "ttl",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> set[str]:
    """Lower-case word tokens from a string."""
    return set(re.findall(r"\b\w+\b", text.lower()))


def _candidate_tokens(analogue: dict[str, Any]) -> set[str]:
    """Combine tags + content fields from a candidate analogue into a token set."""
    parts: list[str] = []
    tags = analogue.get("tags") or []
    if isinstance(tags, list):
        parts.extend(str(t) for t in tags)
    elif isinstance(tags, str):
        parts.append(tags)
    for key in ("content", "description", "problem", "solution"):
        val = analogue.get(key)
        if val:
            parts.append(str(val))
    return _tokenize(" ".join(parts))


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def apply_substitutions(text: str) -> tuple[str, list[dict[str, str]]]:
    """Apply _DOMAIN_SUBSTITUTIONS to *text* and return (adapted_text, mappings)."""
    adapted = text
    mappings: list[dict[str, str]] = []
    for source, target in _DOMAIN_SUBSTITUTIONS.items():
        pattern = re.compile(rf"\b{re.escape(source)}\b", re.IGNORECASE)
        if pattern.search(adapted):
            adapted = pattern.sub(target, adapted)
            mappings.append({"source_term": source, "target_term": target})
    return adapted, mappings


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class AnalogicalReasoningAgent(BaseAgent):
    """Phase 54: map a current problem to structurally similar past solutions."""

    name = "AnalogicalReasoningAgent"
    version = "v1"

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}
        if "best_analogue" not in outputs:
            raise ValueError("best_analogue key required")
        if not isinstance(outputs.get("analogy_score"), float):
            raise ValueError("analogy_score must be a float")
        if not isinstance(outputs.get("transferred_solution"), str):
            raise ValueError("transferred_solution must be a str")
        if not isinstance(outputs.get("mapping"), list):
            raise ValueError("mapping must be a list")
        if not isinstance(outputs.get("confidence"), float):
            raise ValueError("confidence must be a float")
        if not isinstance(outputs.get("novel_elements"), list):
            raise ValueError("novel_elements must be a list")

    def _heuristic(
        self,
        *,
        source_problem: str,
        candidate_analogues: list[dict[str, Any]],
        structural_features: list[str],
    ) -> dict[str, Any]:
        features_set = set(f.lower() for f in structural_features)
        if not candidate_analogues:
            return {
                "best_analogue": None,
                "analogy_score": 0.0,
                "transferred_solution": "",
                "mapping": [],
                "confidence": 0.0,
                "novel_elements": list(features_set),
                "rationale": "heuristic",
            }

        # Score each candidate
        best_analogue = None
        best_score = -1.0
        best_tokens: set[str] = set()
        for analogue in candidate_analogues:
            tokens = _candidate_tokens(analogue)
            score = jaccard_similarity(features_set, tokens)
            if score > best_score:
                best_score = score
                best_analogue = analogue
                best_tokens = tokens

        analogy_score = round(best_score, 6)

        # Transfer solution via substitutions
        raw_solution = str((best_analogue or {}).get("solution") or "")
        transferred_solution, mapping = apply_substitutions(raw_solution)

        # Novel elements = features not present in best analogue's tokens
        novel_elements = [f for f in structural_features if f.lower() not in best_tokens]

        # Confidence penalised for novel elements
        novel_ratio = len(novel_elements) / max(len(structural_features), 1)
        confidence = round(analogy_score * (1.0 - novel_ratio), 6)

        return {
            "best_analogue": best_analogue,
            "analogy_score": analogy_score,
            "transferred_solution": transferred_solution,
            "mapping": mapping,
            "confidence": confidence,
            "novel_elements": novel_elements,
            "rationale": "heuristic",
        }

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        enrichment = (context.get("memory") or {}).get("enrichment") or {}
        source_problem: str = str(enrichment.get("source_problem") or "")
        candidate_analogues: list[dict[str, Any]] = list(
            enrichment.get("candidate_analogues") or []
        )
        structural_features: list[str] = list(
            enrichment.get("structural_features") or []
        )

        if settings.AGENT_STRATEGY == "heuristic":
            outputs = self._heuristic(
                source_problem=source_problem,
                candidate_analogues=candidate_analogues,
                structural_features=structural_features,
            )
            finished_at = datetime.now(timezone.utc)
            result = AgentResult(
                agent_name=self.name,
                agent_version=self.version,
                memory_id=memory_id,
                status="success",
                confidence=outputs["confidence"],
                outputs=outputs,
                warnings=[],
                errors=[],
                started_at=started_at,
                finished_at=finished_at,
                trace_id=str(trace_id) if trace_id else None,
            )
            self.validate_outputs(result)
            return result

        # LLM path
        try:
            client = create_llm_client()
            prompt = (
                "You are an analogical reasoning expert.\n\n"
                f"SOURCE PROBLEM: {source_problem}\n"
                f"STRUCTURAL FEATURES: {structural_features}\n"
                f"CANDIDATE ANALOGUES: {candidate_analogues}\n\n"
                "Respond as JSON with keys: best_analogue (dict|null), analogy_score (float), "
                "transferred_solution (str), mapping (list[{source_term,target_term}]), "
                "confidence (float), novel_elements (list[str])."
            )
            resp = await client.generate(model=settings.VLLM_MODEL, prompt=prompt)
            import json

            parsed = json.loads(resp.get("response", "{}"))
            if not isinstance(parsed.get("analogy_score"), (int, float)):
                raise ValueError("invalid llm response")
            parsed["analogy_score"] = float(parsed["analogy_score"])
            parsed["confidence"] = float(parsed.get("confidence", 0.5))
            parsed.setdefault("rationale", "llm")
            finished_at = datetime.now(timezone.utc)
            result = AgentResult(
                agent_name=self.name,
                agent_version=self.version,
                memory_id=memory_id,
                status="success",
                confidence=parsed["confidence"],
                outputs=parsed,
                warnings=[],
                errors=[],
                started_at=started_at,
                finished_at=finished_at,
                trace_id=str(trace_id) if trace_id else None,
            )
            self.validate_outputs(result)
            return result
        except Exception:
            outputs = self._heuristic(
                source_problem=source_problem,
                candidate_analogues=candidate_analogues,
                structural_features=structural_features,
            )
            finished_at = datetime.now(timezone.utc)
            result = AgentResult(
                agent_name=self.name,
                agent_version=self.version,
                memory_id=memory_id,
                status="success",
                confidence=outputs["confidence"],
                outputs=outputs,
                warnings=[],
                errors=[],
                started_at=started_at,
                finished_at=finished_at,
                trace_id=str(trace_id) if trace_id else None,
            )
            self.validate_outputs(result)
            return result
