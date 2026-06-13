"""Episodic Memory Grouping — Phase 18.

Groups each enriched memory into a coherent episode — a named narrative arc
shared by memories with overlapping entities, domain, and temporal window.
Assigns each memory an episode key, role, cohesion score, and label, and
detects links to related episodes via shared root entities.

Depends on:
  - SemanticNormalizationAgent (Phase 6)  — domain
  - EntityResolutionAgent (Phase 7)       — entities
  - TemporalReasoningAgent (Phase 17)     — temporal_cluster

Inputs (via context["memory"]):
  - content:           raw text of the memory
  - created_at:        ISO timestamp of creation

Inputs (via context["memory"]["enrichment"]):
  - domain:            domain string (Phase 6)
  - entities:          resolved entities dict (Phase 7)
  - temporal_cluster:  ISO year-week string (Phase 17)
  - freshness_score:   float (Phase 15)
  - episode_siblings:  list[dict] — pre-fetched memories in the same episode
                       each: {id, created_at, domain, entities, freshness_score}
  - cross_episode_memories: list[dict] — memories from related episodes sharing
                       root entities; each: {id, episode_key, domain}

Outputs:
  - episode_key:           str — stable hash key for this episode cluster
  - episode_role:          "opener" | "body" | "climax" | "closer"
  - episode_cohesion:      float 0..1 — tightness of the episode cluster
  - episode_label:         str — human-readable e.g. "Engineering · acme · 2026-W10"
  - cross_episode_links:   list[str] — related episode keys via shared entities
  - confidence:            float
  - rationale:             "heuristic" | "llm"
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.agents.llm.llm_breaker import create_llm_client
from app.core.config import settings


# Temporal density: episode span at which density score hits 0.5.
_DENSITY_WINDOW_DAYS = 14.0

# Role thresholds: fraction of episode time span.
_OPENER_PCT = 0.20
_CLOSER_PCT = 0.80
_CLIMAX_FRESHNESS_MIN = 0.65  # minimum freshness to qualify as climax

_VALID_ROLES = frozenset({"opener", "body", "climax", "closer"})


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------

def _parse_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _extract_entity_names(entities: Any) -> set[str]:
    """Return a flat set of normalized entity names from heterogeneous payloads.

    Supports:
      - dicts (including nested lists)
      - list[dict] resolved entities ({canonical, original})
      - list[str]
      - scalar strings
    """
    names: set[str] = set()

    if isinstance(entities, dict):
        for key, value in entities.items():
            key_s = str(key).lower().strip()
            if key_s:
                names.add(key_s)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        for candidate in (item.get("canonical"), item.get("original"), item.get("name")):
                            if candidate:
                                names.add(str(candidate).lower().strip())
                    else:
                        names.add(str(item).lower().strip())
            elif isinstance(value, dict):
                for candidate in (value.get("canonical"), value.get("original"), value.get("name")):
                    if candidate:
                        names.add(str(candidate).lower().strip())
            elif value is not None:
                names.add(str(value).lower().strip())

    elif isinstance(entities, list):
        for item in entities:
            if isinstance(item, dict):
                for candidate in (
                    item.get("canonical"),
                    item.get("original"),
                    item.get("name"),
                    item.get("entity"),
                ):
                    if candidate:
                        names.add(str(candidate).lower().strip())
            else:
                names.add(str(item).lower().strip())

    elif entities is not None:
        names.add(str(entities).lower().strip())

    return {n for n in names if n}


def extract_root_entities(
    *,
    entities: Any,
    siblings: list[dict[str, Any]],
    limit: int = 3,
) -> list[str]:
    """Select stable root entities for episode grouping and cross-linking.

    Root entities are ranked by how many sibling memories they overlap with,
    then by lexical order for stability.
    """
    self_entities = _extract_entity_names(entities)
    if not self_entities:
        return []

    counts: dict[str, int] = {name: 1 for name in self_entities}
    for sibling in siblings or []:
        sib_entities = _extract_entity_names(
            sibling.get("entities") or sibling.get("resolved_entities") or {}
        )
        for name in self_entities:
            if name in sib_entities:
                counts[name] = counts.get(name, 1) + 1

    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    shared = [name for name, count in ranked if count >= 2]
    if shared:
        return shared[:limit]

    return [name for name, _ in ranked[:limit]]


def compute_episode_date_range(
    *,
    created_at: datetime,
    siblings: list[dict[str, Any]],
    temporal_cluster: str,
) -> str:
    """Build compact date-range text for episode labeling."""
    sibling_dts = [_parse_dt(s.get("created_at")) for s in siblings or []]
    all_dts = [created_at] + [dt for dt in sibling_dts if dt is not None]

    start_dt = min(all_dts)
    end_dt = max(all_dts)

    if start_dt.date() == end_dt.date():
        return start_dt.strftime("%Y-%m-%d")

    start_iso = start_dt.isocalendar()
    end_iso = end_dt.isocalendar()
    if start_iso.year == end_iso.year and start_iso.week == end_iso.week and temporal_cluster:
        return temporal_cluster

    return f"{start_dt.strftime('%Y-%m-%d')}..{end_dt.strftime('%Y-%m-%d')}"


def build_episode_key(
    *,
    domain: str,
    entities: Any,
    temporal_cluster: str,
    siblings: list[dict[str, Any]],
) -> str:
    """Stable 16-char hash key grouping memories in the same episode.

    Key is derived from domain + sorted entity names + temporal_cluster so
    that two memories with the same domain, overlapping entities, and the
    same ISO week resolve to the same episode.
    """
    roots = extract_root_entities(entities=entities, siblings=siblings)
    parts = [str(domain or "").lower().strip(), str(temporal_cluster or "").strip()]
    parts.extend(roots)
    canonical = "|".join(t for t in parts if t)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def build_episode_label(
    *,
    domain: str,
    entities: Any,
    temporal_cluster: str,
    created_at: datetime,
    siblings: list[dict[str, Any]],
) -> str:
    """Human-readable label from domain + top entities + date range."""
    domain_part = str(domain or "general").lower()
    root_entities = extract_root_entities(entities=entities, siblings=siblings)
    top_entities = root_entities[:2]
    range_part = compute_episode_date_range(
        created_at=created_at,
        siblings=siblings,
        temporal_cluster=temporal_cluster,
    )

    if top_entities:
        return f"{domain_part} · {', '.join(top_entities)} · {range_part}"
    return f"{domain_part} · {range_part}"


def compute_episode_role(
    *,
    created_at: datetime,
    freshness_score: float,
    siblings: list[dict[str, Any]],
) -> str:
    """Assign a narrative role based on position in the episode's time span.

    Rules (in priority order):
    1. opener  — no siblings, or this memory is in the first 20% of the span
    2. closer  — this memory is in the last 20% of the span
    3. climax  — highest freshness among middle members (freshness ≥ threshold)
    4. body    — all other middle positions
    """
    sibling_dts = [_parse_dt(s.get("created_at")) for s in siblings or []]
    sibling_dts = [dt for dt in sibling_dts if dt is not None]

    if not sibling_dts:
        return "opener"

    all_dts = sorted(sibling_dts + [created_at])
    span_seconds = (all_dts[-1] - all_dts[0]).total_seconds()

    if span_seconds <= 0:
        return "opener"

    pct = (created_at - all_dts[0]).total_seconds() / span_seconds
    pct = max(0.0, min(1.0, pct))

    if pct <= _OPENER_PCT:
        return "opener"
    if pct >= _CLOSER_PCT:
        return "closer"

    # Check for climax: highest freshness in the middle zone with enough signal.
    sibling_freshnesses = [
        float(s.get("freshness_score") or 0.0)
        for s in siblings
        if _parse_dt(s.get("created_at")) is not None
    ]
    if sibling_freshnesses:
        max_sib = max(sibling_freshnesses)
        if float(freshness_score) >= _CLIMAX_FRESHNESS_MIN and float(freshness_score) >= max_sib:
            return "climax"

    return "body"


def compute_entity_overlap(
    entities: dict[str, Any],
    siblings: list[dict[str, Any]],
) -> float:
    """Fraction of siblings sharing at least one entity with this memory."""
    if not siblings:
        return 0.0
    my_names = _extract_entity_names(entities)
    if not my_names:
        return 0.0
    matches = sum(
        1 for s in siblings if my_names & _extract_entity_names(s.get("entities") or {})
    )
    return round(matches / len(siblings), 4)


def compute_temporal_density(
    created_at: datetime,
    siblings: list[dict[str, Any]],
    *,
    window_days: float = _DENSITY_WINDOW_DAYS,
) -> float:
    """Measure how tightly packed the episode is in time.

    Returns 1.0 when all members are within window_days of each other,
    decaying toward 0 as the episode span grows.
    density = window_days / (window_days + span_days)
    """
    sibling_dts = [_parse_dt(s.get("created_at")) for s in siblings or []]
    sibling_dts = [dt for dt in sibling_dts if dt is not None]
    if not sibling_dts:
        return 0.0
    all_dts = [created_at] + sibling_dts
    span_days = (max(all_dts) - min(all_dts)).total_seconds() / 86400.0
    if span_days <= 0:
        return 1.0
    density = window_days / (window_days + span_days)
    return round(min(max(density, 0.0), 1.0), 4)


def compute_domain_consistency(
    domain: str,
    siblings: list[dict[str, Any]],
) -> float:
    """Fraction of siblings in the same domain as this memory."""
    if not siblings:
        return 0.0
    this_domain = str(domain or "").lower()
    same = sum(
        1 for s in siblings if str(s.get("domain") or "").lower() == this_domain
    )
    return round(same / len(siblings), 4)


def compute_episode_cohesion(
    *,
    entity_overlap: float,
    temporal_density: float,
    domain_consistency: float,
) -> float:
    """Cohesion = entity_overlap × temporal_density × domain_consistency."""
    score = float(entity_overlap) * float(temporal_density) * float(domain_consistency)
    return round(min(max(score, 0.0), 1.0), 4)


def extract_cross_episode_links(
    cross_episode_memories: list[dict[str, Any]],
    *,
    this_episode_key: str,
    root_entities: set[str],
) -> list[str]:
    """Collect distinct episode keys from related cross-episode memories.

    Excludes this memory's own episode key and deduplicates.
    """
    seen: set[str] = set()
    links: list[str] = []
    for mem in cross_episode_memories or []:
        key = str(mem.get("episode_key") or "")
        if not key or key == this_episode_key or key in seen:
            continue

        mem_entities = _extract_entity_names(
            mem.get("root_entities")
            or mem.get("entities")
            or mem.get("resolved_entities")
            or mem.get("root_entity")
            or {}
        )
        if root_entities and not (root_entities & mem_entities):
            continue

        seen.add(key)
        links.append(key)
    return links


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class EpisodicGroupingAgent(BaseAgent):
    """Phase 18: Group memories into coherent episodes with roles and labels.

    Assigns each memory to an episode cluster identified by a stable key,
    determines its narrative role within the episode, scores the episode's
    cohesion, and surfaces links to related episodes across silos.
    """

    name = "EpisodicGroupingAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return ["SemanticNormalizationAgent", "EntityResolutionAgent", "TemporalReasoningAgent"]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}

        role = outputs.get("episode_role")
        if role not in _VALID_ROLES:
            raise ValueError(f"invalid episode_role: {role!r}")

        cohesion = outputs.get("episode_cohesion")
        if not isinstance(cohesion, (int, float)) or not (0.0 <= float(cohesion) <= 1.0):
            raise ValueError("episode_cohesion must be a float in [0, 1]")

        if not isinstance(outputs.get("episode_key"), str):
            raise ValueError("episode_key must be a str")

        if not isinstance(outputs.get("episode_label"), str):
            raise ValueError("episode_label must be a str")

        if not isinstance(outputs.get("cross_episode_links"), list):
            raise ValueError("cross_episode_links must be a list")

    def _heuristic(
        self,
        *,
        created_at: datetime,
        domain: str,
        entities: Any,
        temporal_cluster: str,
        freshness_score: float,
        siblings: list[dict[str, Any]],
        cross_episode_memories: list[dict[str, Any]],
    ) -> dict[str, Any]:
        episode_key = build_episode_key(
            domain=domain,
            entities=entities,
            temporal_cluster=temporal_cluster,
            siblings=siblings,
        )
        root_entities = set(extract_root_entities(entities=entities, siblings=siblings))
        episode_label = build_episode_label(
            domain=domain,
            entities=entities,
            temporal_cluster=temporal_cluster,
            created_at=created_at,
            siblings=siblings,
        )
        episode_role = compute_episode_role(
            created_at=created_at,
            freshness_score=freshness_score,
            siblings=siblings,
        )

        entity_overlap = compute_entity_overlap(entities, siblings)
        temporal_density = compute_temporal_density(created_at, siblings)
        domain_consistency = compute_domain_consistency(domain, siblings)
        episode_cohesion = compute_episode_cohesion(
            entity_overlap=entity_overlap,
            temporal_density=temporal_density,
            domain_consistency=domain_consistency,
        )

        cross_episode_links = extract_cross_episode_links(
            cross_episode_memories,
            this_episode_key=episode_key,
            root_entities=root_entities,
        )

        if episode_cohesion >= 0.70:
            confidence = 0.85
        elif episode_cohesion >= 0.50:
            confidence = 0.75
        elif siblings:
            confidence = 0.70
        else:
            confidence = 0.60

        return {
            "episode_key": episode_key,
            "episode_role": episode_role,
            "episode_cohesion": episode_cohesion,
            "episode_label": episode_label,
            "cross_episode_links": cross_episode_links,
            "confidence": confidence,
            "rationale": "heuristic",
        }

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        memory = context.get("memory") or {}
        enrichment = memory.get("enrichment") or {}
        content = str(memory.get("content") or "")

        now = datetime.now(timezone.utc)
        created_at = _parse_dt(memory.get("created_at")) or now
        domain = str(enrichment.get("domain") or "")
        entities = enrichment.get("entities")
        if not entities:
            entities = enrichment.get("resolved_entities") or {}

        temporal_cluster = str(enrichment.get("temporal_cluster") or "").strip()
        if not temporal_cluster:
            iso = created_at.isocalendar()
            temporal_cluster = f"{iso.year}-W{iso.week:02d}"
        freshness_score = float(enrichment.get("freshness_score") or 0.5)
        siblings = list(enrichment.get("episode_siblings") or [])
        cross_episode_memories = list(enrichment.get("cross_episode_memories") or [])

        strategy = getattr(settings, "EPISODIC_GROUPING_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        outputs: dict[str, Any]

        if strategy == "heuristic" or not content:
            outputs = self._heuristic(
                created_at=created_at,
                domain=domain,
                entities=entities,
                temporal_cluster=temporal_cluster,
                freshness_score=freshness_score,
                siblings=siblings,
                cross_episode_memories=cross_episode_memories,
            )
        else:
            sib_json = siblings[:5]
            cross_json = cross_episode_memories[:5]
            prompt = (
                "You are an episodic memory grouping engine for an enterprise Cognitive OS. "
                "Output JSON only. Do not hallucinate.\n\n"
                "Assign this memory to an episode and determine its role.\n\n"
                f"CREATED_AT: {created_at.isoformat()}\n"
                f"DOMAIN: {domain}\n"
                f"ENTITIES: {sorted(list(_extract_entity_names(entities)))[:5]}\n"
                f"TEMPORAL_CLUSTER: {temporal_cluster}\n"
                f"FRESHNESS_SCORE: {freshness_score}\n"
                f"EPISODE_SIBLINGS: {sib_json}\n"
                f"CROSS_EPISODE_MEMORIES: {cross_json}\n"
                f"CONTENT: {content[:300]}\n\n"
                "Return JSON with keys:\n"
                "- episode_key: 16-char stable hash string\n"
                "- episode_role: one of opener|body|climax|closer\n"
                "- episode_cohesion: float 0..1\n"
                "- episode_label: human-readable string\n"
                "- cross_episode_links: list of episode key strings\n"
                "- confidence: float 0..1\n"
                "- rationale: brief explanation"
            )
            client = create_llm_client(
                base_url=str(getattr(settings, "VLLM_BASE_URL", "http://localhost:11434")),
                model=str(settings.get_llm_model("agents")),
                timeout_seconds=float(getattr(settings, "VLLM_TIMEOUT_SECONDS", 5.0)),
                max_concurrency=int(getattr(settings, "VLLM_MAX_CONCURRENCY", 2)),
            )
            resp = await client.complete_json(
                prompt=prompt,
                schema_hint={},
                tool_event_sink=context.get("tool_event_sink"),
            )
            if (
                isinstance(resp, dict)
                and resp.get("episode_role") in _VALID_ROLES
                and isinstance(resp.get("episode_key"), str)
                and isinstance(resp.get("episode_label"), str)
                and isinstance(resp.get("cross_episode_links"), list)
                and isinstance(resp.get("episode_cohesion"), (int, float))
            ):
                outputs = resp
            else:
                outputs = self._heuristic(
                    created_at=created_at,
                    domain=domain,
                    entities=entities,
                    temporal_cluster=temporal_cluster,
                    freshness_score=freshness_score,
                    siblings=siblings,
                    cross_episode_memories=cross_episode_memories,
                )

        finished_at = datetime.now(timezone.utc)

        result = AgentResult(
            agent_name=self.name,
            agent_version=self.version,
            memory_id=memory_id,
            status="success",
            confidence=float(outputs.get("confidence", 0.6)),
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=started_at,
            finished_at=finished_at,
            trace_id=str(trace_id) if trace_id else None,
        )
        self.validate_outputs(result)
        return result
