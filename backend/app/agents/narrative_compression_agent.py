"""Narrative memory compression agent (Phase 56)."""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.llm.ollama_breaker import create_ollama_client
from app.agents.types import AgentContext, AgentResult
from app.core.config import settings


def _parse_created_at(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.min.replace(tzinfo=timezone.utc)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b[a-z0-9_]+\b", (text or "").lower()))


def dominant_tag(episodes: list[dict[str, Any]]) -> str:
    counts: Counter[str] = Counter()
    for episode in episodes:
        tags = episode.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        for tag in tags:
            tag_text = str(tag).strip().lower()
            if tag_text:
                counts[tag_text] += 1
    if not counts:
        return "general"
    return counts.most_common(1)[0][0]


def _topic_overlap_score(episode: dict[str, Any], topic_tokens: set[str]) -> int:
    tags = episode.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    tag_tokens: set[str] = set()
    for tag in tags:
        tag_tokens |= _tokenize(str(tag))
    return len(tag_tokens & topic_tokens)


def select_key_events(episodes: list[dict[str, Any]], topic: str) -> list[dict[str, Any]]:
    topic_tokens = _tokenize(topic)

    scored: list[tuple[int, datetime, dict[str, Any]]] = []
    for episode in episodes:
        score = _topic_overlap_score(episode, topic_tokens)
        created = _parse_created_at(episode.get("created_at"))
        scored.append((score, created, episode))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored[:5]]


def _build_timespan(sorted_episodes: list[dict[str, Any]]) -> dict[str, str | None]:
    if not sorted_episodes:
        return {"from": None, "to": None}
    first_dt = _parse_created_at(sorted_episodes[0].get("created_at"))
    last_dt = _parse_created_at(sorted_episodes[-1].get("created_at"))
    return {"from": first_dt.isoformat(), "to": last_dt.isoformat()}


def run_heuristic(
    *, episodes: list[dict[str, Any]], topic: str, max_sentences: int = 3
) -> dict[str, Any]:
    if not episodes:
        return {
            "compressed_narrative": "",
            "compression_ratio": 0.0,
            "key_events": [],
            "time_span": {"from": None, "to": None},
            "archived_ids": [],
            "confidence": 0.5,
            "rationale": "heuristic",
        }

    sorted_episodes = sorted(episodes, key=lambda ep: _parse_created_at(ep.get("created_at")))
    span = _build_timespan(sorted_episodes)
    selected = select_key_events(sorted_episodes, topic)
    key_event_snippets = [str(ep.get("content") or "").strip()[:100] for ep in selected if str(ep.get("content") or "").strip()]
    dominant = dominant_tag(sorted_episodes)

    first_key = key_event_snippets[0] if key_event_snippets else "no key events"
    second_key = key_event_snippets[1] if len(key_event_snippets) > 1 else ""

    sentences: list[str] = [
        f"From {span['from']} to {span['to']}, {len(sorted_episodes)} events occurred related to {topic}.",
        f"Key events: {first_key}.",
    ]
    if second_key:
        sentences.append(f"Additional event: {second_key}.")
    sentences.append(f"Overall pattern: {dominant}.")
    bounded = sentences[: max(1, int(max_sentences or 3))]
    narrative = " ".join(bounded)

    key_ids = {str(ep.get("id")) for ep in selected if ep.get("id")}
    archived_ids = [
        str(ep.get("id"))
        for ep in sorted_episodes
        if ep.get("id") and str(ep.get("id")) not in key_ids
    ]

    ratio = len(sorted_episodes) / max(1, int(max_sentences or 3))
    confidence = min(0.9, 0.5 + 0.05 * len(sorted_episodes))

    return {
        "compressed_narrative": narrative,
        "compression_ratio": round(ratio, 4),
        "key_events": key_event_snippets[:5],
        "time_span": span,
        "archived_ids": archived_ids,
        "confidence": round(confidence, 4),
        "rationale": "heuristic",
    }


class NarrativeCompressionAgent(BaseAgent):
    name = "NarrativeCompressionAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return ["NarrativeSynthesisAgent"]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}

        if not isinstance(outputs.get("compressed_narrative"), str):
            raise ValueError("compressed_narrative must be a string")
        if not isinstance(outputs.get("compression_ratio"), (int, float)):
            raise ValueError("compression_ratio must be numeric")
        if not isinstance(outputs.get("key_events"), list):
            raise ValueError("key_events must be a list")
        if not isinstance(outputs.get("archived_ids"), list):
            raise ValueError("archived_ids must be a list")

        span = outputs.get("time_span")
        if not isinstance(span, dict):
            raise ValueError("time_span must be a dict")
        if "from" not in span or "to" not in span:
            raise ValueError("time_span must include from/to")

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        enrichment = (context.get("memory") or {}).get("enrichment") or {}

        episodes = list(enrichment.get("episodes") or [])
        topic = str(enrichment.get("topic") or "general")
        max_sentences = int(enrichment.get("max_sentences") or 3)

        strategy = getattr(settings, "NARRATIVE_COMPRESSION_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        if strategy == "heuristic":
            outputs = run_heuristic(
                episodes=episodes,
                topic=topic,
                max_sentences=max_sentences,
            )
        else:
            prompt = (
                "You are a narrative compression engine. Output JSON only.\n\n"
                f"TOPIC: {topic}\n"
                f"MAX_SENTENCES: {max_sentences}\n"
                f"EPISODES: {episodes[:20]}\n\n"
                "Return JSON with keys:\n"
                "- compressed_narrative: str\n"
                "- compression_ratio: float\n"
                "- key_events: list[str]\n"
                "- time_span: {from, to}\n"
                "- archived_ids: list[str]\n"
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
                and isinstance(resp.get("compressed_narrative"), str)
                and isinstance(resp.get("compression_ratio"), (int, float))
                and isinstance(resp.get("key_events"), list)
                and isinstance(resp.get("time_span"), dict)
                and isinstance(resp.get("archived_ids"), list)
            ):
                outputs = resp
            else:
                outputs = run_heuristic(
                    episodes=episodes,
                    topic=topic,
                    max_sentences=max_sentences,
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
