"""Query Intelligence Layer — Phase 27.

Enriches the read path by analysing a memory's content and existing
enrichment signals to detect query intent, extract entities, assemble
dynamic retrieval filters, and recommend which downstream agents are
most relevant for the memory's subject matter.

Depends on (reads enrichment from):
  - EntityResolutionAgent (Phase 7)      — entity_name, entity_type, resolved_aliases
  - UncertaintyReportingAgent (Phase 22) — uncertainty_level, low_confidence_fields
  - NarrativeSynthesisAgent (Phase 23)   — key_entities, action_items

Outputs:
  - query_intent:       str        — locate | explain | compare | analyze |
                                     generate | find_person | find_timeline | retrieve
  - extracted_entities: list[str]  — entities from content + enrichment
  - dynamic_filters:    dict       — filters to narrow retrieval
  - suggested_agents:   list[str]  — agents most relevant for this intent
  - confidence:         float      0..1
  - rationale:          "heuristic" | "llm"
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.agents.llm.llm_breaker import create_llm_client
from app.core.config import settings


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_INTENTS = {
    "locate",
    "explain",
    "compare",
    "analyze",
    "generate",
    "find_person",
    "find_timeline",
    "retrieve",
}

_DEFAULT_INTENT = "retrieve"

_INTENT_KEYWORDS: dict[str, list[str]] = {
    "locate": ["find", "search", "locate", "where", "which", "look for", "fetch"],
    "explain": ["why", "because", "cause", "reason", "explain", "how did", "due to", "what caused"],
    "compare": ["compare", "difference", "vs", "versus", "better", "worse", "between", "contrast"],
    "analyze": ["analyze", "analyse", "trend", "pattern", "metric", "performance", "insight", "report"],
    "generate": ["create", "generate", "write", "draft", "produce", "build", "summarize", "summarise"],
    "find_person": ["who", "person", "people", "contact", "team", "staff", "employee", "user", "owner"],
    "find_timeline": ["when", "timeline", "schedule", "deadline", "date", "history", "sequence", "order"],
}

_INTENT_AGENT_MAP: dict[str, list[str]] = {
    "locate": ["EntityResolutionAgent", "WorldModelAgent", "SiloPropagationAgent"],
    "explain": ["CausalReasoningAgent", "UncertaintyReportingAgent", "EpisodicGroupingAgent"],
    "compare": ["ConflictDetectionAgent", "NarrativeSynthesisAgent", "TemporalReasoningAgent"],
    "analyze": ["AnomalyDetectionAgent", "TemporalReasoningAgent", "CredibilityAgent"],
    "generate": ["NarrativeSynthesisAgent", "PlaybookAgent", "GoalDecompositionAgent"],
    "find_person": ["EntityResolutionAgent", "SiloPropagationAgent", "OrgAttentionAgent"],
    "find_timeline": ["TemporalReasoningAgent", "EpisodicGroupingAgent", "ProactiveMemoryPushAgent"],
    "retrieve": ["EntityResolutionAgent", "CredibilityAgent", "MemoryConsolidationAgent"],
}

_MAX_ENTITIES = 20
_MAX_TAGS = 5

_ENRICHMENT_SIGNAL_KEYS = [
    "entity_name",
    "uncertainty_level",
    "key_entities",
    "tags",
    "credibility_score",
    "temporal_confidence",
    "normalized_tags",
    "entity_type",
]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def detect_intent(text: str) -> str:
    """Keyword-vote intent classification. Returns the intent with the most matches."""
    if not text or not text.strip():
        return _DEFAULT_INTENT
    lower = text.lower()
    scores: dict[str, int] = {intent: 0 for intent in _INTENT_KEYWORDS}
    for intent, keywords in _INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                scores[intent] += 1
    best_intent = max(scores, key=lambda k: scores[k])
    if scores[best_intent] == 0:
        return _DEFAULT_INTENT
    return best_intent


def extract_entities(text: str, enrichment: dict[str, Any]) -> list[str]:
    """Merge entities from enrichment signals and a simple content scan."""
    entities: set[str] = set()

    # From EntityResolutionAgent
    entity_name = enrichment.get("entity_name")
    if entity_name and isinstance(entity_name, str):
        entities.add(entity_name.strip())

    aliases = enrichment.get("resolved_aliases") or []
    if isinstance(aliases, list):
        for a in aliases:
            if isinstance(a, str) and a.strip():
                entities.add(a.strip())

    # From NarrativeSynthesisAgent
    key_entities = enrichment.get("key_entities") or []
    if isinstance(key_entities, list):
        for e in key_entities:
            if isinstance(e, str) and e.strip():
                entities.add(e.strip())

    # Simple content scan: capitalised words not at sentence start
    if text and text.strip():
        words = text.split()
        for i, word in enumerate(words):
            clean = word.strip(".,;:!?\"'()[]{}—-")
            if len(clean) >= 2 and clean[0].isupper() and i > 0:
                entities.add(clean)

    return sorted(entities)[:_MAX_ENTITIES]


def build_dynamic_filters(intent: str, enrichment: dict[str, Any]) -> dict[str, Any]:
    """Assemble retrieval filters from enrichment signals and detected intent."""
    filters: dict[str, Any] = {}

    entity_type = enrichment.get("entity_type")
    if entity_type and isinstance(entity_type, str):
        filters["entity_type"] = entity_type

    tags = enrichment.get("tags") or enrichment.get("normalized_tags") or []
    if isinstance(tags, list) and tags:
        filters["tags"] = [t for t in tags if isinstance(t, str)][:_MAX_TAGS]

    uncertainty_level = str(enrichment.get("uncertainty_level") or "").lower()
    if uncertainty_level in {"high", "critical"}:
        filters["exclude_uncertainty_levels"] = ["high", "critical"]

    if intent == "find_timeline":
        filters["has_temporal_data"] = True
    elif intent == "find_person":
        filters["entity_type_hint"] = "person"
    elif intent == "analyze":
        filters["min_credibility"] = 0.5

    return filters


def suggest_agents(intent: str) -> list[str]:
    """Return agent recommendations for the given intent."""
    return list(_INTENT_AGENT_MAP.get(intent, _INTENT_AGENT_MAP[_DEFAULT_INTENT]))


def count_enrichment_signals(enrichment: dict[str, Any]) -> int:
    """Count how many known enrichment keys are populated."""
    return sum(1 for k in _ENRICHMENT_SIGNAL_KEYS if enrichment.get(k) is not None)


def compute_confidence(
    has_content: bool,
    entity_count: int,
    enrichment_signals: int,
    intent: str,
) -> float:
    """Confidence in the query intelligence result."""
    base = 0.50
    if has_content:
        base += 0.15
    if entity_count >= 1:
        base += 0.10
    if entity_count >= 3:
        base += 0.05
    if enrichment_signals >= 3:
        base += 0.10
    if intent != _DEFAULT_INTENT:
        base += 0.05
    return round(min(0.90, max(0.40, base)), 4)


def run_heuristic(content: str, enrichment: dict[str, Any]) -> dict[str, Any]:
    """Full keyword + enrichment heuristic for query intelligence."""
    intent = detect_intent(content)
    entities = extract_entities(content, enrichment)
    filters = build_dynamic_filters(intent, enrichment)
    agents = suggest_agents(intent)
    signals = count_enrichment_signals(enrichment)
    confidence = compute_confidence(
        has_content=bool(content and content.strip()),
        entity_count=len(entities),
        enrichment_signals=signals,
        intent=intent,
    )
    return {
        "query_intent": intent,
        "extracted_entities": entities,
        "dynamic_filters": filters,
        "suggested_agents": agents,
        "confidence": confidence,
        "rationale": "heuristic",
    }


async def run_llm_only_query_intelligence(
    content: str,
    enrichment: dict[str, Any],
    tool_event_sink: Any | None = None,
) -> dict[str, Any]:
    """Run query intelligence using LLM only (no heuristic fallback)."""
    prompt = (
        "You are an enterprise memory query intelligence engine. "
        "Output JSON only. Do not hallucinate.\n\n"
        "Analyse the memory content and enrichment signals below. "
        "Detect the primary intent of this memory, extract named entities, "
        "suggest retrieval filters, and recommend relevant enrichment agents.\n\n"
        f"CONTENT: {content[:500]}\n"
        f"ENTITY_NAME: {enrichment.get('entity_name')}\n"
        f"ENTITY_TYPE: {enrichment.get('entity_type')}\n"
        f"KEY_ENTITIES: {enrichment.get('key_entities')}\n"
        f"UNCERTAINTY_LEVEL: {enrichment.get('uncertainty_level')}\n"
        f"TAGS: {enrichment.get('tags') or enrichment.get('normalized_tags')}\n\n"
        "Return JSON with keys:\n"
        "- query_intent: one of locate|explain|compare|analyze|generate|find_person|find_timeline|retrieve\n"
        "- extracted_entities: list of entity name strings\n"
        "- dynamic_filters: dict of retrieval filter key-value pairs\n"
        "- suggested_agents: list of agent name strings\n"
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
        tool_event_sink=tool_event_sink,
    )
    if not _valid_llm_response(resp):
        raise ValueError("invalid llm response")

    out = dict(resp)
    out.setdefault("rationale", "llm")
    return out


def _valid_llm_response(resp: Any) -> bool:
    if not isinstance(resp, dict):
        return False
    if resp.get("query_intent") not in _VALID_INTENTS:
        return False
    if not isinstance(resp.get("extracted_entities"), list):
        return False
    if not isinstance(resp.get("dynamic_filters"), dict):
        return False
    if not isinstance(resp.get("suggested_agents"), list):
        return False
    return True


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class QueryIntelligenceAgent(BaseAgent):
    """Phase 27: Enrich the read path with intent, entities, and filters.

    Analyses memory content and prior enrichment to detect what the memory is
    about, extract entities, assemble dynamic retrieval filters, and surface
    which downstream agents are most useful for the memory's subject matter.
    """

    name = "QueryIntelligenceAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return [
            "EntityResolutionAgent",
            "UncertaintyReportingAgent",
            "NarrativeSynthesisAgent",
        ]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}

        intent = outputs.get("query_intent")
        if intent not in _VALID_INTENTS:
            raise ValueError(
                f"query_intent must be one of {sorted(_VALID_INTENTS)}, got {intent!r}"
            )

        if not isinstance(outputs.get("extracted_entities"), list):
            raise ValueError("extracted_entities must be a list")

        if not isinstance(outputs.get("dynamic_filters"), dict):
            raise ValueError("dynamic_filters must be a dict")

        if not isinstance(outputs.get("suggested_agents"), list):
            raise ValueError("suggested_agents must be a list")

        confidence = outputs.get("confidence")
        if not isinstance(confidence, (int, float)) or not (0.0 <= float(confidence) <= 1.0):
            raise ValueError("confidence must be a float in [0, 1]")

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        memory = context.get("memory") or {}
        enrichment: dict[str, Any] = memory.get("enrichment") or {}

        content = str(
            memory.get("content")
            or memory.get("text")
            or memory.get("body")
            or enrichment.get("normalized_text")
            or ""
        )

        strategy = getattr(settings, "QUERY_INTELLIGENCE_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        outputs: dict[str, Any]

        if strategy == "heuristic":
            outputs = run_heuristic(content, enrichment)
        else:
            prompt = (
                "You are an enterprise memory query intelligence engine. "
                "Output JSON only. Do not hallucinate.\n\n"
                "Analyse the memory content and enrichment signals below. "
                "Detect the primary intent of this memory, extract named entities, "
                "suggest retrieval filters, and recommend relevant enrichment agents.\n\n"
                f"CONTENT: {content[:500]}\n"
                f"ENTITY_NAME: {enrichment.get('entity_name')}\n"
                f"ENTITY_TYPE: {enrichment.get('entity_type')}\n"
                f"KEY_ENTITIES: {enrichment.get('key_entities')}\n"
                f"UNCERTAINTY_LEVEL: {enrichment.get('uncertainty_level')}\n"
                f"TAGS: {enrichment.get('tags') or enrichment.get('normalized_tags')}\n\n"
                "Return JSON with keys:\n"
                "- query_intent: one of locate|explain|compare|analyze|generate|find_person|find_timeline|retrieve\n"
                "- extracted_entities: list of entity name strings\n"
                "- dynamic_filters: dict of retrieval filter key-value pairs\n"
                "- suggested_agents: list of agent name strings\n"
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
            if _valid_llm_response(resp):
                outputs = resp
            else:
                outputs = run_heuristic(content, enrichment)

        finished_at = datetime.now(timezone.utc)

        result = AgentResult(
            agent_name=self.name,
            agent_version=self.version,
            memory_id=memory_id,
            status="success",
            confidence=float(outputs.get("confidence", 0.65)),
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=started_at,
            finished_at=finished_at,
            trace_id=str(trace_id) if trace_id else None,
        )
        self.validate_outputs(result)
        return result
