from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any


_GOAL_PHRASES = re.compile(
    r"\b(plan|goal|objective|aim|target|milestone|intend|propose|"
    r"strategy|roadmap|initiative|mission|step[s]?|phase[s]?|stage[s]?)\b",
    re.IGNORECASE,
)
_NEED_TO = re.compile(r"\b(need\s+to|going\s+to|want\s+to|have\s+to|will|should|must)\b", re.IGNORECASE)
_NUMBERED_ITEM = re.compile(r"^\s*\d+[.)]\s+(.+)$", re.MULTILINE)
_BULLET_ITEM = re.compile(r"^\s*[-*•]\s+(.+)$", re.MULTILINE)
_STEP_ITEM = re.compile(r"\bstep\s*\d+[:\s]+([^\n.!?]{3,})", re.IGNORECASE)
_ACTION_VERBS = re.compile(
    r"\b(analyze|create|build|deploy|fix|update|review|test|design|"
    r"implement|check|verify|run|setup|configure|migrate|monitor|"
    r"evaluate|schedule|contact|send|document|prepare|plan|define|"
    r"resolve|investigate|refactor|optimize|integrate|validate|launch)\b",
    re.IGNORECASE,
)
_BLOCKING_RE = re.compile(
    r"\b(blocked|blocking|stuck|waiting\s+on|pending|cannot|can'?t|failed|"
    r"failure|error|issue|problem|delayed|delay|on\s+hold|stalled|missing|"
    r"incomplete|not\s+done|unresolved|blocker)\b",
    re.IGNORECASE,
)
_MAX_SUBTASKS = 10
_MAX_SUBTASK_LEN = 200

_PRIMARY_SOURCE_TYPES = frozenset({"api", "system", "internal", "webhook"})
_SECONDARY_SOURCE_TYPES = frozenset({"document", "import", "integration", "upload", "file"})
_TERTIARY_SOURCE_TYPES = frozenset({"scrape", "crawl", "aggregated", "derived"})
_PRIMARY_AUTHOR_ROLES = frozenset({
    "engineer", "developer", "analyst", "manager", "owner", "admin", "author",
    "contributor", "lead", "architect", "system", "service",
})
_SECONDARY_AUTHOR_ROLES = frozenset({
    "reporter", "observer", "reviewer", "auditor", "external_partner", "vendor", "consultant",
})
_TERTIARY_AUTHOR_ROLES = frozenset({"anonymous", "aggregator", "bot", "unknown"})
_HEARSAY_PATTERNS = (
    re.compile(r"\bi heard\b", re.IGNORECASE),
    re.compile(r"\breportedly\b", re.IGNORECASE),
    re.compile(r"\bapparently\b", re.IGNORECASE),
    re.compile(r"\brumored?\b", re.IGNORECASE),
    re.compile(r"\bsomeone said\b", re.IGNORECASE),
    re.compile(r"\bword is\b", re.IGNORECASE),
)
_TIER_BASE = {"primary": 0.80, "secondary": 0.60, "tertiary": 0.35, "unknown": 0.45}

_DOMAIN_HALF_LIFE = {
    "incident": 7.0,
    "support": 14.0,
    "sprint": 14.0,
    "deployment": 21.0,
    "sales": 30.0,
    "engineering": 45.0,
    "project": 60.0,
    "finance": 90.0,
    "hr": 90.0,
    "strategy": 180.0,
    "document": 365.0,
}
_DEFAULT_HALF_LIFE = 30.0

_CAUSE_TYPE_WEIGHT = {
    "person": 1.0,
    "org": 0.95,
    "organization": 0.95,
    "customer": 0.90,
    "project": 0.80,
    "system": 0.75,
    "deployment": 0.70,
    "sprint": 0.65,
    "event": 0.60,
    "incident": 0.55,
    "document": 0.50,
    "concept": 0.40,
    "metric": 0.30,
}
_DEFAULT_CAUSE_WEIGHT = 0.45
_MAX_CHAINS = 10
_MAX_HOPS = 3

_WORD_RE = re.compile(r"\b[a-z0-9_]+\b", re.IGNORECASE)
_ESCALATION_TOKENS = {"critical", "alert", "failure", "down", "error"}

_LOW_CONFIDENCE_THRESHOLD = 0.55
_STALE_DECAY_THRESHOLD = 0.30
_LEVEL_THRESHOLDS = [
    (0.80, "critical"),
    (0.55, "high"),
    (0.30, "medium"),
    (0.00, "low"),
]


def detect_goal(content: str) -> bool:
    if not content or len(content.strip()) < 5:
        return False
    if _GOAL_PHRASES.search(content) or (_NEED_TO.search(content) and _ACTION_VERBS.search(content)):
        return True
    return bool(_NUMBERED_ITEM.search(content))


def _is_action_sentence(text: str) -> bool:
    return len(text.strip()) >= 8 and bool(_ACTION_VERBS.search(text))


def extract_subtasks(content: str, max_subtasks: int = _MAX_SUBTASKS) -> list[str]:
    if not content:
        return []

    def _clean(items: list[str]) -> list[str]:
        seen: list[str] = []
        for item in items:
            cleaned = item.strip()[:_MAX_SUBTASK_LEN]
            if cleaned and cleaned not in seen:
                seen.append(cleaned)
        return seen[:max_subtasks]

    for matches in (_NUMBERED_ITEM.findall(content), _STEP_ITEM.findall(content), _BULLET_ITEM.findall(content)):
        if matches:
            return _clean(matches)
    return _clean([segment for segment in re.split(r"[.!?;]\s+", content) if _is_action_sentence(segment)])


def detect_blocking_subtask(subtasks: list[str], content: str) -> str | None:
    if not subtasks:
        return None
    for subtask in subtasks:
        if _BLOCKING_RE.search(subtask):
            return subtask[:_MAX_SUBTASK_LEN]
    block_match = _BLOCKING_RE.search(content)
    if not block_match:
        return None
    window = content[max(0, block_match.start() - 120): block_match.start() + 120].lower()
    best_subtask: str | None = None
    best_overlap = 0
    for subtask in subtasks:
        tokens = set(re.findall(r"\b\w{3,}\b", subtask.lower()))
        overlap = sum(1 for token in tokens if token in window)
        if overlap > best_overlap:
            best_overlap = overlap
            best_subtask = subtask
    if best_subtask and best_overlap > 0:
        return best_subtask[:_MAX_SUBTASK_LEN]
    return None


def classify_source_tier(*, source_type: str, author_role: str, content: str) -> str:
    source = (source_type or "").strip().lower()
    role = (author_role or "").strip().lower()
    if source in _PRIMARY_SOURCE_TYPES:
        return "primary"
    if source in _SECONDARY_SOURCE_TYPES:
        return "tertiary" if role in _TERTIARY_AUTHOR_ROLES else "secondary"
    if source in _TERTIARY_SOURCE_TYPES:
        return "tertiary"
    if role in _PRIMARY_AUTHOR_ROLES:
        return "primary"
    if role in _SECONDARY_AUTHOR_ROLES:
        return "secondary"
    if role in _TERTIARY_AUTHOR_ROLES:
        return "tertiary"
    if any(pattern.search(content) for pattern in _HEARSAY_PATTERNS):
        return "tertiary"
    return "unknown"


def compute_credibility_score(
    *,
    source_tier: str,
    citation_depth: int,
    corroboration_count: int,
    high_severity_conflict_count: int,
) -> float:
    base = _TIER_BASE.get(source_tier, _TIER_BASE["unknown"])
    citation_bonus = min(0.10, 0.02 * citation_depth)
    corroboration_bonus = min(0.10, 0.025 * corroboration_count)
    conflict_penalty = 0.05 * min(3, high_severity_conflict_count)
    score = base + citation_bonus + corroboration_bonus - conflict_penalty
    return round(min(max(score, 0.05), 0.95), 4)


def domain_half_life(domain: str) -> float:
    return _DOMAIN_HALF_LIFE.get(str(domain or "").lower(), _DEFAULT_HALF_LIFE)


def compute_decay_rate(*, half_life: float) -> float:
    safe_half_life = max(float(half_life), 0.01)
    return round(math.log(2) / safe_half_life, 6)


def compute_base_freshness(*, age_days: float, decay_rate: float) -> float:
    score = math.exp(-float(decay_rate) * max(float(age_days), 0.0))
    return round(min(max(score, 0.0), 1.0), 4)


def _cause_weight(entity_type: str) -> float:
    return _CAUSE_TYPE_WEIGHT.get((entity_type or "").lower(), _DEFAULT_CAUSE_WEIGHT)


def identify_effect_candidates(*, world_nodes: list[dict[str, Any]], push_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    node_lookup = {node["entity"]: node for node in world_nodes}
    scores: dict[str, float] = {}
    entity_meta: dict[str, dict[str, Any]] = {}
    for candidate in push_candidates:
        topic = candidate.get("topic", "")
        urgency = float(candidate.get("urgency_score", 0.0))
        if topic and urgency >= 0.5:
            scores[topic] = max(scores.get(topic, 0.0), urgency)
            entity_meta[topic] = node_lookup.get(topic, {"entity": topic, "entity_type": "concept", "domains": []})
    for node in world_nodes:
        entity = node["entity"]
        silo_count = int(node.get("silo_count", 1))
        if silo_count >= 2:
            scores[entity] = max(scores.get(entity, 0.0), min(0.9, 0.4 + 0.15 * (silo_count - 1)))
            entity_meta[entity] = node
    candidates = [
        {
            "entity": entity,
            "entity_type": entity_meta[entity].get("entity_type", "concept"),
            "domains": entity_meta[entity].get("domains", []),
            "effect_score": round(score, 3),
        }
        for entity, score in scores.items()
    ]
    candidates.sort(key=lambda item: -item["effect_score"])
    return candidates[:20]


def build_adjacency(entity_edges: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    adjacency: dict[str, list[dict[str, Any]]] = {}
    for edge in entity_edges:
        left = edge.get("from_entity", "")
        right = edge.get("to_entity", "")
        weight = float(edge.get("weight", 0.5))
        if not left or not right:
            continue
        adjacency.setdefault(left, []).append({"neighbor": right, "weight": weight})
        adjacency.setdefault(right, []).append({"neighbor": left, "weight": weight})
    return adjacency


def trace_causal_chain(
    *,
    effect_entity: str,
    adjacency: dict[str, list[dict[str, Any]]],
    node_lookup: dict[str, dict[str, Any]],
    propagated_signal_entities: set[str],
    max_hops: int = _MAX_HOPS,
) -> list[dict[str, Any]]:
    completed: list[dict[str, Any]] = []
    frontier: list[tuple[str, list[str], float]] = [(effect_entity, [effect_entity], 1.0)]
    visited: set[tuple[str, ...]] = set()
    for _ in range(max_hops):
        if not frontier:
            break
        next_frontier: list[tuple[str, list[str], float]] = []
        for current, path, strength in frontier:
            current_weight = _cause_weight(node_lookup.get(current, {}).get("entity_type", "concept"))
            for neighbor_info in adjacency.get(current, []):
                neighbor = neighbor_info["neighbor"]
                if neighbor in path:
                    continue
                neighbor_weight = _cause_weight(node_lookup.get(neighbor, {}).get("entity_type", "concept"))
                if neighbor_weight <= current_weight and neighbor not in propagated_signal_entities:
                    continue
                new_path = path + [neighbor]
                path_key = tuple(new_path)
                if path_key in visited:
                    continue
                visited.add(path_key)
                new_strength = round(strength * neighbor_info["weight"] * neighbor_weight, 4)
                completed.append({
                    "causal_path": new_path,
                    "hop_count": len(new_path) - 1,
                    "chain_strength": new_strength,
                })
                next_frontier.append((neighbor, new_path, new_strength))
        frontier = next_frontier
    results: list[dict[str, Any]] = []
    for item in completed:
        silos: set[str] = set()
        for entity in item["causal_path"]:
            silos.update(node_lookup.get(entity, {}).get("domains", []))
        item["silo_span"] = len(silos)
        results.append(item)
    results.sort(key=lambda item: -item["chain_strength"])
    return results[:_MAX_CHAINS]


def extract_root_causes(chains: list[dict[str, Any]], node_lookup: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    for chain in chains:
        path = chain.get("causal_path", [])
        if len(path) < 2:
            continue
        root = path[-1]
        scores[root] = max(scores.get(root, 0.0), chain["chain_strength"])
    results = [
        {
            "entity": entity,
            "entity_type": node_lookup.get(entity, {}).get("entity_type", "concept"),
            "cause_score": round(score, 3),
            "domains": node_lookup.get(entity, {}).get("domains", []),
        }
        for entity, score in scores.items()
    ]
    results.sort(key=lambda item: -item["cause_score"])
    return results


def generate_counterfactual_hint(root_causes: list[dict[str, Any]], top_effect: str | None) -> str | None:
    if not root_causes or not top_effect:
        return None
    root = root_causes[0]
    domains = root.get("domains", [])
    domain_suffix = f" [{', '.join(domains)}]" if domains else ""
    return (
        f"Addressing {root['entity']}{domain_suffix} could mitigate the impact on {top_effect} "
        f"(cause_score={root['cause_score']:.2f})"
    )


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in _WORD_RE.findall(text or "")}


def _episode_text(episode: dict[str, Any]) -> str:
    parts = [
        str(episode.get("content") or ""),
        str(episode.get("summary") or ""),
        str(episode.get("event_description") or ""),
    ]
    tags = episode.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    parts.extend(str(tag) for tag in tags)
    return " ".join(parts)


def _overlap_score(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _dominant_tag(episode: dict[str, Any]) -> str:
    tags = episode.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    cleaned = [str(tag).strip().lower() for tag in tags if str(tag).strip()]
    return cleaned[0] if cleaned else "untagged"


def _severity_change(text: str) -> str:
    tokens = _tokenize(text)
    if tokens & _ESCALATION_TOKENS:
        return "increase"
    if any(token in tokens for token in {"resolved", "stable", "recovered", "fixed", "mitigated"}):
        return "decrease"
    return "stable"


def _extract_entities(current_state: dict[str, Any], template: dict[str, Any]) -> list[str]:
    entities = current_state.get("entities") or []
    if isinstance(entities, str):
        entities = [entities]
    cleaned = [str(entity).strip() for entity in entities if str(entity).strip()]
    if cleaned:
        return cleaned[:5]
    tags = template.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    return [str(tag).strip() for tag in tags if str(tag).strip()][:5]


def future_run_heuristic(
    *,
    current_state: dict[str, Any],
    planned_action: str,
    historical_episodes: list[dict[str, Any]],
    simulation_steps: int = 3,
) -> dict[str, Any]:
    action = str(planned_action or "").strip() or "Perform planned action"
    steps = max(1, int(simulation_steps if simulation_steps is not None else 3))
    action_tokens = _tokenize(action)
    history = list(historical_episodes or [])

    matching_indices: list[int] = []
    for index, episode in enumerate(history):
        if _overlap_score(action_tokens, _tokenize(_episode_text(episode))) >= 0.2:
            matching_indices.append(index)

    templates: list[dict[str, Any]] = []
    for index in matching_indices:
        next_index = index + 1
        if next_index < len(history):
            templates.append(history[next_index])

    deduped_templates: list[dict[str, Any]] = []
    seen_tags: set[str] = set()
    for template in templates:
        key = _dominant_tag(template)
        if key in seen_tags:
            continue
        seen_tags.add(key)
        deduped_templates.append(template)

    simulated_episodes = [{
        "step": 0,
        "event_description": action,
        "probability": 0.9,
        "severity_change": "stable",
        "entities_affected": _extract_entities(current_state, {}),
    }]

    base_probability = 0.75
    for step in range(1, steps):
        if deduped_templates:
            template = deduped_templates[(step - 1) % len(deduped_templates)]
            description = str(
                template.get("event_description")
                or template.get("content")
                or template.get("summary")
                or f"Follow-up event after action: {action}"
            )
        else:
            template = {}
            description = f"Likely follow-up step {step} after: {action}"
        simulated_episodes.append({
            "step": step,
            "event_description": description,
            "probability": round(max(0.01, base_probability * (0.8 ** step)), 4),
            "severity_change": _severity_change(description),
            "entities_affected": _extract_entities(current_state, template),
        })

    risk_events = [
        episode
        for episode in simulated_episodes
        if episode.get("severity_change") == "increase" and float(episode.get("probability") or 0.0) > 0.5
    ]
    max_risk_probability = max((float(event.get("probability") or 0.0) for event in risk_events), default=0.0)
    success_probability = round(max(0.0, min(1.0, 0.9 * (1.0 - max_risk_probability))), 4)

    recommended_precautions: list[str] = []
    for event in risk_events:
        entities_text = ", ".join(event.get("entities_affected") or []) or "key systems"
        description = str(event.get("event_description") or "").strip()
        recommended_precautions.append(f"Monitor {entities_text} closely before proceeding.")
        recommended_precautions.append(f"Roll back if {description[:50]} occurs.")

    seen_precautions: Counter[str] = Counter()
    deduped_precautions: list[str] = []
    for precaution in recommended_precautions:
        if seen_precautions[precaution] > 0:
            continue
        seen_precautions[precaution] += 1
        deduped_precautions.append(precaution)

    return {
        "simulated_episodes": simulated_episodes,
        "success_probability": success_probability,
        "risk_events": risk_events,
        "recommended_precautions": deduped_precautions,
        "confidence": 0.6,
        "rationale": "heuristic",
    }


def collect_unresolved_conflicts(enrichment: dict[str, Any]) -> list[dict[str, Any]]:
    conflicts = enrichment.get("conflicts") or []
    resolution_rate = float(enrichment.get("resolution_rate") or 0.0)
    escalation_targets = list(enrichment.get("escalation_targets") or [])
    if not conflicts or resolution_rate >= 1.0:
        return []
    unresolved: list[dict[str, Any]] = []
    for conflict in conflicts:
        if not isinstance(conflict, dict):
            continue
        severity = str(conflict.get("severity") or "low").lower()
        entity = str(conflict.get("entity") or "unknown")
        conflict_type = str(conflict.get("conflict_type") or "unknown")
        is_high = severity == "high"
        is_escalated = entity in escalation_targets or conflict_type in escalation_targets
        if is_high or is_escalated or resolution_rate < 0.5:
            unresolved.append({"entity": entity, "severity": severity, "conflict_type": conflict_type})
    return unresolved


def _is_float(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def collect_low_confidence_fields(enrichment: dict[str, Any]) -> list[str]:
    fields: list[str] = []
    if (score := enrichment.get("credibility_score")) is not None and _is_float(score) and float(score) < _LOW_CONFIDENCE_THRESHOLD:
        fields.append("credibility_score")
    if (score := enrichment.get("playbook_confidence")) is not None and _is_float(score) and float(score) < _LOW_CONFIDENCE_THRESHOLD:
        fields.append("playbook_confidence")
    if (score := enrichment.get("temporal_confidence")) is not None and _is_float(score) and float(score) < _LOW_CONFIDENCE_THRESHOLD:
        fields.append("temporal_confidence")
    if (score := enrichment.get("causal_confidence")) is not None and _is_float(score) and float(score) < _LOW_CONFIDENCE_THRESHOLD:
        fields.append("causal_confidence")
    if (score := enrichment.get("decay_factor")) is not None and _is_float(score) and float(score) < _STALE_DECAY_THRESHOLD:
        fields.append("decay_factor")
    if enrichment.get("blocking_subtask"):
        fields.append("goal_completion")
    return fields


def _compute_uncertainty_score(
    *,
    unresolved_count: int,
    low_confidence_count: int,
    high_severity_count: int,
    credibility_score: float | None,
    is_stale: bool,
) -> float:
    score = min(0.30, unresolved_count * 0.10)
    score += min(0.20, high_severity_count * 0.10)
    score += min(0.20, low_confidence_count * 0.05)
    if credibility_score is not None:
        score += round(max(0.0, _LOW_CONFIDENCE_THRESHOLD - credibility_score) * 0.30, 4)
    if is_stale:
        score += 0.10
    return round(min(1.0, score), 4)


def _classify_uncertainty_level(score: float) -> str:
    for threshold, label in _LEVEL_THRESHOLDS:
        if score >= threshold:
            return label
    return "low"


def uncertainty_run_heuristic(enrichment: dict[str, Any]) -> dict[str, Any]:
    unresolved = collect_unresolved_conflicts(enrichment)
    low_fields = collect_low_confidence_fields(enrichment)
    high_severity_conflicts = enrichment.get("high_severity_conflicts") or []
    high_severity_count = len([conflict for conflict in high_severity_conflicts if isinstance(conflict, dict)])
    credibility_raw = enrichment.get("credibility_score")
    credibility_score = float(credibility_raw) if _is_float(credibility_raw) else None
    decay_raw = enrichment.get("decay_factor")
    is_stale = bool(enrichment.get("is_stale")) or (_is_float(decay_raw) and float(decay_raw) < _STALE_DECAY_THRESHOLD)
    score = _compute_uncertainty_score(
        unresolved_count=len(unresolved),
        low_confidence_count=len(low_fields),
        high_severity_count=high_severity_count,
        credibility_score=credibility_score,
        is_stale=is_stale,
    )
    level = _classify_uncertainty_level(score)
    signal_count = len(unresolved) + len(low_fields)
    if level == "critical":
        confidence = 0.85
    elif level == "high":
        confidence = 0.80
    elif level == "medium":
        confidence = 0.70
    else:
        confidence = 0.65
    if signal_count == 0:
        confidence -= 0.10
    elif signal_count >= 4:
        confidence += 0.05
    confidence = round(min(0.90, max(0.40, confidence)), 4)
    return {
        "uncertainty_level": level,
        "unresolved_conflicts": unresolved,
        "low_confidence_fields": low_fields,
        "review_recommended": score >= 0.35,
        "uncertainty_score": score,
        "confidence": confidence,
        "rationale": "heuristic",
    }