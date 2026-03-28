"""Entity Resolution Agent — Phase 7.

Resolves extracted entities against the enterprise ontology to surface canonical
names and cross-silo links (same concept appearing across departments).

Depends on Phase 6 (SemanticNormalizationAgent) via memory enrichment
semantic_relationships field.

Outputs power downstream phases:
  - Phase 4 (World Model Graph): resolved canonical nodes
  - Phase 8 (Context Amplifier): cross-silo fan-out on expert reads
  - Phase 9 (Silo Propagation): routes signals via domain membership
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.agents.llm.ollama_breaker import create_ollama_client
from app.core.config import settings


# ---------------------------------------------------------------------------
# Built-in ontology — canonical enterprise entities + aliases
# (tenant-specific ontology injected at runtime from OntologyEntry DB rows)
# ---------------------------------------------------------------------------

_BUILTIN_ONTOLOGY: list[dict[str, Any]] = [
    {
        "canonical": "customer",
        "entity_type": "role",
        "aliases": ["client", "account", "end-user", "end user", "buyer", "subscriber"],
        "domains": ["sales", "finance", "support"],
    },
    {
        "canonical": "revenue",
        "entity_type": "metric",
        "aliases": ["arr", "mrr", "income", "bookings", "sales revenue"],
        "domains": ["sales", "finance"],
    },
    {
        "canonical": "project",
        "entity_type": "project",
        "aliases": ["initiative", "program", "effort", "workstream", "epic"],
        "domains": ["engineering", "product", "operations"],
    },
    {
        "canonical": "incident",
        "entity_type": "event",
        "aliases": ["outage", "issue", "problem", "failure", "defect"],
        "domains": ["engineering", "support", "operations"],
    },
    {
        "canonical": "contract",
        "entity_type": "document",
        "aliases": ["agreement", "nda", "sow", "statement of work", "msa"],
        "domains": ["sales", "legal", "finance"],
    },
    {
        "canonical": "employee",
        "entity_type": "role",
        "aliases": ["staff", "team member", "headcount", "hire", "worker", "colleague"],
        "domains": ["hr", "finance", "operations"],
    },
    {
        "canonical": "vendor",
        "entity_type": "organization",
        "aliases": ["supplier", "partner", "third-party", "third party", "service provider"],
        "domains": ["operations", "finance", "legal"],
    },
    {
        "canonical": "budget",
        "entity_type": "metric",
        "aliases": ["cost", "spend", "expense", "allocation", "forecast", "capex", "opex"],
        "domains": ["finance", "operations", "engineering"],
    },
    {
        "canonical": "deployment",
        "entity_type": "event",
        "aliases": ["release", "rollout", "launch", "ship", "publish"],
        "domains": ["engineering", "product", "operations"],
    },
    {
        "canonical": "sprint",
        "entity_type": "time_period",
        "aliases": ["iteration", "cycle", "milestone", "delivery"],
        "domains": ["product", "engineering"],
    },
]


def _build_alias_index(
    ontology: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build a flat alias→entry lookup from an ontology list."""
    index: dict[str, dict[str, Any]] = {}
    for entry in ontology:
        index[entry["canonical"].lower()] = entry
        for alias in entry.get("aliases", []):
            key = alias.lower().strip()
            if key and key not in index:
                index[key] = entry
    return index


# Pre-built index for the built-in ontology
_BUILTIN_INDEX: dict[str, dict[str, Any]] = _build_alias_index(_BUILTIN_ONTOLOGY)


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower())


def resolve_entity(
    name: str,
    extra_ontology: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Resolve a single entity name against built-in + extra ontology.

    Match priority:
    1. Exact (canonical name match)       → confidence 1.0, match_type "exact"
    2. Alias (known alias match)          → confidence 0.85, match_type "alias"
    3. Prefix/substring (fuzzy fallback)  → confidence 0.65, match_type "prefix"

    Returns a resolution dict or None if unresolved.
    """
    key = _normalize(name)
    if not key:
        return None

    # Merge lookup: extra ontology aliases take precedence over builtin
    lookup = dict(_BUILTIN_INDEX)
    if extra_ontology:
        extra_index = _build_alias_index(extra_ontology)
        lookup.update(extra_index)

    # Exact / alias match
    if key in lookup:
        entry = lookup[key]
        match_type = "exact" if key == entry["canonical"].lower() else "alias"
        confidence = 1.0 if match_type == "exact" else 0.85
        return {
            "original": name,
            "canonical": entry["canonical"],
            "entity_type": entry["entity_type"],
            "domains": entry.get("domains", []),
            "match_type": match_type,
            "confidence": confidence,
        }

    # Prefix / substring fuzzy match (cheap, no external dependency)
    for alias_key, entry in lookup.items():
        if len(alias_key) >= 3 and (alias_key.startswith(key) or key.startswith(alias_key)):
            return {
                "original": name,
                "canonical": entry["canonical"],
                "entity_type": entry["entity_type"],
                "domains": entry.get("domains", []),
                "match_type": "prefix",
                "confidence": 0.65,
            }

    return None


def extract_entities_from_content(text: str) -> list[str]:
    """Heuristic: extract capitalized noun phrases as candidate entities.

    Targets proper-noun-like tokens (e.g. "Acme Corp", "Q1 Budget").
    Caps at 20 to avoid noise.
    """
    if not text:
        return []
    tokens = re.findall(r"\b([A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*)\b", text)
    seen: set[str] = set()
    result: list[str] = []
    for t in tokens:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            result.append(t)
    return result[:20]


def build_cross_silo_links(
    resolved: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Surface entities whose ontology spans 2+ business domains (cross-silo).

    These are the structural links between silos — e.g. "customer" appears
    in sales, finance, and support. Returned once per canonical name.
    """
    links: list[dict[str, Any]] = []
    seen_canonicals: set[str] = set()
    for entity in resolved:
        canonical = entity["canonical"]
        domains = entity.get("domains", [])
        if len(domains) >= 2 and canonical not in seen_canonicals:
            seen_canonicals.add(canonical)
            links.append(
                {
                    "canonical": canonical,
                    "entity_type": entity["entity_type"],
                    "domains": domains,
                    "silo_count": len(domains),
                }
            )
    return links


class EntityResolutionAgent(BaseAgent):
    """Phase 7: Resolve entities against the enterprise ontology.

    Inputs  (via context["memory"]["enrichment"]):
      - semantic_relationships: list[{type, concept}] — from Phase 6

    Outputs:
      - resolved_entities:   list[{original, canonical, entity_type, domains,
                                   match_type, confidence}]
      - unresolved_entities: list[str]
      - cross_silo_links:    list[{canonical, entity_type, domains, silo_count}]
      - confidence:          float
      - rationale:           "heuristic" | "llm"
    """

    name = "EntityResolutionAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return ["SemanticNormalizationAgent"]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}
        if not isinstance(outputs.get("resolved_entities", []), list):
            raise ValueError("resolved_entities must be a list")
        if not isinstance(outputs.get("cross_silo_links", []), list):
            raise ValueError("cross_silo_links must be a list")
        if not isinstance(outputs.get("unresolved_entities", []), list):
            raise ValueError("unresolved_entities must be a list")
        for entity in outputs.get("resolved_entities", []):
            if not isinstance(entity, dict):
                raise ValueError("each resolved_entity must be a dict")
            if "canonical" not in entity or "entity_type" not in entity:
                raise ValueError("each resolved_entity must have canonical and entity_type")

    def _heuristic(
        self,
        *,
        content: str,
        relationships: list[dict[str, str]],
        extra_ontology: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        # Collect candidate entity names from Phase 6 relationships + raw content
        candidate_names: list[str] = []
        seen_names: set[str] = set()

        for rel in relationships:
            concept = rel.get("concept", "")
            if concept and _normalize(concept) not in seen_names:
                candidate_names.append(concept)
                seen_names.add(_normalize(concept))

        for entity in extract_entities_from_content(content):
            if _normalize(entity) not in seen_names:
                candidate_names.append(entity)
                seen_names.add(_normalize(entity))

        resolved: list[dict[str, Any]] = []
        unresolved: list[str] = []

        for name in candidate_names[:30]:
            resolution = resolve_entity(name, extra_ontology)
            if resolution:
                resolved.append(resolution)
            else:
                unresolved.append(name)

        cross_silo = build_cross_silo_links(resolved)

        total = len(candidate_names)
        if total == 0:
            confidence = 0.5
        else:
            confidence = min(0.9, round(0.4 + 0.5 * (len(resolved) / total), 3))

        return {
            "resolved_entities": resolved,
            "unresolved_entities": unresolved,
            "cross_silo_links": cross_silo,
            "confidence": confidence,
            "rationale": "heuristic",
        }

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        content = (context.get("memory") or {}).get("content", "")
        enrichment = (context.get("memory") or {}).get("enrichment") or {}

        # Phase 6 semantic_relationships fed as enrichment
        relationships: list[dict[str, str]] = enrichment.get("semantic_relationships", [])

        strategy = getattr(settings, "ENTITY_RESOLUTION_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        outputs: dict[str, Any]

        if strategy == "heuristic" or not content:
            outputs = self._heuristic(content=content, relationships=relationships)
        else:
            prompt = (
                "You are an enterprise ontology resolution engine. "
                "Output JSON only. Do not hallucinate.\n\n"
                "Identify and resolve named entities in the content against known enterprise concepts:\n"
                "- resolved_entities: list of {original, canonical, entity_type, domains, match_type, confidence}\n"
                "- unresolved_entities: list of entity name strings that could not be resolved\n"
                "- cross_silo_links: list of {canonical, entity_type, domains, silo_count} "
                "for entities spanning multiple departments\n"
                "- confidence: float 0..1\n"
                "- rationale: brief explanation\n\n"
                f"CONTENT:\n{content}\n\n"
                "Return JSON with keys: resolved_entities, unresolved_entities, "
                "cross_silo_links, confidence, rationale"
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
                and isinstance(resp.get("resolved_entities"), list)
                and isinstance(resp.get("unresolved_entities"), list)
                and isinstance(resp.get("cross_silo_links"), list)
            ):
                outputs = resp
            else:
                outputs = self._heuristic(content=content, relationships=relationships)

        finished_at = datetime.now(timezone.utc)

        result = AgentResult(
            agent_name=self.name,
            agent_version=self.version,
            memory_id=memory_id,
            status="success",
            confidence=float(outputs.get("confidence", 0.5)),
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=started_at,
            finished_at=finished_at,
            trace_id=str(trace_id) if trace_id else None,
        )
        self.validate_outputs(result)
        return result
