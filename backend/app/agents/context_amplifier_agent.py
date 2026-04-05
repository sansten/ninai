"""Context Amplifier Agent — Phase 8.

On expert read, fan-out across cross-silo entity links and assemble
a role-scoped context bundle showing signals from other business domains.

Depends on Phase 7 (EntityResolutionAgent) via memory enrichment
resolved_entities and cross_silo_links fields.
Also uses Phase 6 (SemanticNormalizationAgent) business_domain.

Outputs power downstream phases:
  - Phase 9 (Silo Propagation): context_bundle signals route to other teams
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.agents.llm.ollama_breaker import create_ollama_client
from app.core.config import settings


# ---------------------------------------------------------------------------
# Role → relevant business domains
# ---------------------------------------------------------------------------

_ROLE_DOMAIN_MAP: dict[str, list[str]] = {
    "cfo": ["finance", "sales", "hr", "legal"],
    "finance": ["finance", "sales", "hr", "legal"],
    "engineer": ["engineering", "operations", "product"],
    "sre": ["engineering", "operations"],
    "developer": ["engineering", "product"],
    "sales": ["sales", "finance", "support"],
    "support": ["support", "sales", "engineering"],
    "hr": ["hr", "finance", "operations"],
    "legal": ["legal", "finance", "sales"],
    "product": ["product", "engineering", "sales"],
    "ops": ["operations", "engineering", "finance"],
    "operations": ["operations", "engineering", "finance"],
    "marketing": ["marketing", "sales", "product"],
    "ceo": ["sales", "finance", "engineering", "hr", "legal", "product", "operations"],
    "admin": ["sales", "finance", "engineering", "hr", "legal", "product", "operations"],
}

_ALL_DOMAINS = [
    "engineering", "sales", "finance", "support",
    "hr", "legal", "product", "marketing", "operations",
]


def get_relevant_domains(roles: list[str]) -> list[str]:
    """Return relevant domains for the given actor roles.

    Unknown or empty roles → all domains (no restriction).
    """
    if not roles:
        return list(_ALL_DOMAINS)
    domains: set[str] = set()
    for role in roles:
        role_lower = role.lower().strip()
        if role_lower in _ROLE_DOMAIN_MAP:
            domains.update(_ROLE_DOMAIN_MAP[role_lower])
    if not domains:
        return list(_ALL_DOMAINS)
    return sorted(domains)


def amplify_context(
    *,
    resolved_entities: list[dict[str, Any]],
    cross_silo_links: list[dict[str, Any]],
    primary_domain: str,
    relevant_domains: list[str],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Build a role-scoped context bundle by fanning out cross-silo entities.

    For each cross-silo link that overlaps with relevant_domains, emits one
    bundle entry per foreign domain (i.e. every domain except primary_domain).
    Also includes resolved-entity signals for the primary domain.

    Returns (context_bundle, amplified_entities, domains_touched).
    """
    bundle: list[dict[str, Any]] = []
    amplified: list[str] = []
    seen_amplified: set[str] = set()
    domains_touched: set[str] = set()

    # Index resolved entities by canonical name for enriching cross-silo entries
    resolved_by_canonical: dict[str, dict[str, Any]] = {}
    for entity in resolved_entities:
        canonical = entity.get("canonical", "")
        if canonical and canonical not in resolved_by_canonical:
            resolved_by_canonical[canonical] = entity

    # Fan-out: surface foreign-domain signals for every relevant cross-silo entity
    for link in cross_silo_links:
        canonical = link.get("canonical", "")
        link_domains = link.get("domains", [])

        # At least one of the link's domains must be relevant to the actor
        if not any(d in relevant_domains for d in link_domains):
            continue

        entity_info = resolved_by_canonical.get(canonical, {})
        entity_type = entity_info.get("entity_type") or link.get("entity_type", "concept")
        original = entity_info.get("original", canonical)

        for domain in link_domains:
            if domain == primary_domain:
                continue  # actor already has the primary domain
            if domain not in relevant_domains:
                continue
            bundle.append(
                {
                    "entity": canonical,
                    "original": original,
                    "entity_type": entity_type,
                    "domain": domain,
                    "signal": f"{canonical} ({entity_type}) is also active in {domain}",
                    "source": "cross_silo",
                    "silo_count": link.get("silo_count", len(link_domains)),
                }
            )
            domains_touched.add(domain)

        if canonical not in seen_amplified:
            amplified.append(canonical)
            seen_amplified.add(canonical)

    # Direct signals: resolved entities whose domains include the primary domain
    for entity in resolved_entities:
        canonical = entity.get("canonical", "")
        entity_domains = entity.get("domains", [])
        if primary_domain in entity_domains or not entity_domains:
            confidence = entity.get("confidence", 0.5)
            bundle.append(
                {
                    "entity": canonical,
                    "original": entity.get("original", canonical),
                    "entity_type": entity.get("entity_type", "concept"),
                    "domain": primary_domain,
                    "signal": (
                        f"{canonical} resolved from '{entity.get('original', canonical)}' "
                        f"(confidence {confidence:.2f})"
                    ),
                    "source": "resolved",
                    "silo_count": len(entity_domains),
                }
            )
            domains_touched.add(primary_domain)

    return bundle[:50], amplified, sorted(domains_touched)


class ContextAmplifierAgent(BaseAgent):
    """Phase 8: Fan-out cross-silo entity signals for expert context assembly.

    Inputs (via context["memory"]["enrichment"]):
      - resolved_entities:  list[{original, canonical, entity_type, domains,
                                  match_type, confidence}]   — from Phase 7
      - cross_silo_links:   list[{canonical, entity_type, domains, silo_count}] — Phase 7
      - business_domain:    str — primary domain from Phase 6

    Inputs (via context["actor"]):
      - roles: list[str] — determines which domains the actor cares about

    Outputs:
      - context_bundle:      list[{entity, original, entity_type, domain,
                                   signal, source, silo_count}]
      - amplified_entities:  list[str]
      - domains_touched:     list[str]
      - confidence:          float
      - rationale:           "heuristic" | "llm"
    """

    name = "ContextAmplifierAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return ["EntityResolutionAgent"]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}
        if not isinstance(outputs.get("context_bundle", []), list):
            raise ValueError("context_bundle must be a list")
        if not isinstance(outputs.get("amplified_entities", []), list):
            raise ValueError("amplified_entities must be a list")
        if not isinstance(outputs.get("domains_touched", []), list):
            raise ValueError("domains_touched must be a list")
        for item in outputs.get("context_bundle", []):
            if not isinstance(item, dict):
                raise ValueError("each context_bundle item must be a dict")
            if "entity" not in item or "domain" not in item:
                raise ValueError("each context_bundle item must have entity and domain")

    def _heuristic(
        self,
        *,
        resolved_entities: list[dict[str, Any]],
        cross_silo_links: list[dict[str, Any]],
        primary_domain: str,
        relevant_domains: list[str],
    ) -> dict[str, Any]:
        bundle, amplified, domains_touched = amplify_context(
            resolved_entities=resolved_entities,
            cross_silo_links=cross_silo_links,
            primary_domain=primary_domain,
            relevant_domains=relevant_domains,
        )

        total = len(resolved_entities)
        if total == 0:
            confidence = 0.4
        else:
            confidence = min(0.9, round(0.4 + 0.4 * (len(amplified) / max(total, 1)), 3))

        return {
            "context_bundle": bundle,
            "amplified_entities": amplified,
            "domains_touched": domains_touched,
            "confidence": confidence,
            "rationale": "heuristic",
        }

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        content = (context.get("memory") or {}).get("content", "")
        enrichment = (context.get("memory") or {}).get("enrichment") or {}
        actor = context.get("actor") or {}

        # Phase 7 outputs
        resolved_entities: list[dict[str, Any]] = enrichment.get("resolved_entities", [])
        cross_silo_links: list[dict[str, Any]] = enrichment.get("cross_silo_links", [])

        # Phase 6 primary domain
        primary_domain: str = enrichment.get("business_domain", "general")

        # Actor roles → relevant domains
        roles: list[str] = list(actor.get("roles", []))
        relevant_domains = get_relevant_domains(roles)

        strategy = getattr(settings, "CONTEXT_AMPLIFIER_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        outputs: dict[str, Any]

        if strategy == "heuristic" or not content:
            outputs = self._heuristic(
                resolved_entities=resolved_entities,
                cross_silo_links=cross_silo_links,
                primary_domain=primary_domain,
                relevant_domains=relevant_domains,
            )
        else:
            prompt = (
                "You are an expert context amplifier for an enterprise Cognitive OS. "
                "Output JSON only. Do not hallucinate.\n\n"
                "Given resolved entities and cross-silo links, assemble a role-scoped context bundle "
                "showing signals from other business domains relevant to the actor's roles.\n\n"
                f"ACTOR ROLES: {roles}\n"
                f"PRIMARY DOMAIN: {primary_domain}\n"
                f"RELEVANT DOMAINS: {relevant_domains}\n"
                f"RESOLVED ENTITIES: {resolved_entities}\n"
                f"CROSS SILO LINKS: {cross_silo_links}\n\n"
                "Return JSON with keys:\n"
                "- context_bundle: list of {entity, original, entity_type, domain, signal, source, silo_count}\n"
                "- amplified_entities: list of canonical entity names that were amplified\n"
                "- domains_touched: list of domain strings surfaced in the bundle\n"
                "- confidence: float 0..1\n"
                "- rationale: brief explanation"
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
                and isinstance(resp.get("context_bundle"), list)
                and isinstance(resp.get("amplified_entities"), list)
                and isinstance(resp.get("domains_touched"), list)
            ):
                outputs = resp
            else:
                outputs = self._heuristic(
                    resolved_entities=resolved_entities,
                    cross_silo_links=cross_silo_links,
                    primary_domain=primary_domain,
                    relevant_domains=relevant_domains,
                )

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
