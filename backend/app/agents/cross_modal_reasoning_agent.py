"""Cross-modal reasoning agent (Phase 76)."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.llm.llm_breaker import create_llm_client
from app.agents.types import AgentContext, AgentResult
from app.core.config import settings


_WORD_RE = re.compile(r"\b[a-z0-9_]+\b", re.IGNORECASE)


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in _WORD_RE.findall(text or "")}


def _as_tag_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return _tokenize(value)
    if isinstance(value, (list, tuple, set)):
        tags: set[str] = set()
        for item in value:
            tags.update(_tokenize(str(item)))
        return tags
    return _tokenize(str(value))


def _parse_dt(value: Any) -> datetime:
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
    return datetime.now(timezone.utc)


def _query_overlap(tags: set[str], query_tokens: set[str]) -> float:
    if not tags or not query_tokens:
        return 0.0
    return len(tags & query_tokens) / len(query_tokens)


def _modality_label(item: dict[str, Any], fallback: str) -> str:
    mod = str(item.get("modality") or fallback).strip().lower()
    if mod in {"image", "video", "visual"}:
        return "visual"
    if mod in {"audio", "voice"}:
        return "audio"
    if mod == "text":
        return "text"
    return fallback


def run_heuristic(
    *,
    text_memories: list[dict[str, Any]],
    visual_memories: list[dict[str, Any]],
    audio_memories: list[dict[str, Any]],
    query: str,
    time_window_minutes: int = 60,
) -> dict[str, Any]:
    query_text = str(query or "").strip()
    query_tokens = _tokenize(query_text)
    window_minutes = max(1, int(time_window_minutes if time_window_minutes is not None else 60))
    window_seconds = float(window_minutes * 60)

    text_rows = list(text_memories or [])
    visual_rows = list(visual_memories or [])
    audio_rows = list(audio_memories or [])

    cross_modal_links: list[dict[str, Any]] = []
    modalities_used: set[str] = set()

    candidates: list[tuple[str, dict[str, Any]]] = []
    for vm in visual_rows:
        candidates.append(("visual", vm))
    for am in audio_rows:
        candidates.append(("audio", am))

    for text in text_rows:
        text_tags = _as_tag_set(text.get("tags"))
        overlap = _query_overlap(text_tags, query_tokens)
        if overlap < 0.2:
            continue

        modalities_used.add("text")
        text_dt = _parse_dt(text.get("created_at"))
        text_id = str(text.get("id") or "")

        for fallback_modality, item in candidates:
            modality = _modality_label(item, fallback_modality)
            candidate_tags = _as_tag_set(item.get("searchable_tags") or item.get("tags"))
            shared_tags = sorted(text_tags & candidate_tags)
            if not shared_tags:
                continue

            candidate_dt = _parse_dt(item.get("created_at"))
            gap_seconds = abs((text_dt - candidate_dt).total_seconds())
            if gap_seconds > window_seconds:
                continue

            correlation = overlap * max(0.0, 1.0 - (gap_seconds / window_seconds))
            if correlation <= 0.1:
                continue

            link: dict[str, Any] = {
                "text_id": text_id,
                "link_type": "temporal_co_occurrence" if gap_seconds < 300 else "thematic_co_occurrence",
                "shared_tags": shared_tags,
                "temporal_gap_seconds": round(gap_seconds, 6),
                "correlation_score": round(correlation, 6),
            }

            candidate_id = str(item.get("id") or "")
            if modality == "audio":
                link["audio_id"] = candidate_id
            else:
                link["visual_id"] = candidate_id

            cross_modal_links.append(link)
            modalities_used.add(modality)

    modalities = sorted(modalities_used)
    evidence_strength = min(1.0, len(cross_modal_links) / 5.0)
    confidence = min(0.9, 0.5 + 0.1 * len(cross_modal_links))

    unified_conclusion = (
        f"Analysis of {len(modalities)} modalities found "
        f"{len(cross_modal_links)} correlated signals related to {query_text[:50]}."
    )

    return {
        "cross_modal_links": cross_modal_links,
        "unified_conclusion": unified_conclusion,
        "modalities_used": modalities,
        "evidence_strength": round(evidence_strength, 6),
        "confidence": round(confidence, 6),
        "rationale": "heuristic",
    }


class CrossModalReasoningAgent(BaseAgent):
    name = "CrossModalReasoningAgent"
    version = "v1"

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return

        outputs = result.outputs or {}
        if not isinstance(outputs.get("cross_modal_links"), list):
            raise ValueError("cross_modal_links must be a list")
        if not isinstance(outputs.get("unified_conclusion"), str):
            raise ValueError("unified_conclusion must be a string")
        if not isinstance(outputs.get("modalities_used"), list):
            raise ValueError("modalities_used must be a list")

        strength = outputs.get("evidence_strength")
        if not isinstance(strength, (int, float)) or not (0.0 <= float(strength) <= 1.0):
            raise ValueError("evidence_strength must be float between 0 and 1")

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        enrichment = (context.get("memory") or {}).get("enrichment") or {}

        text_memories = list(enrichment.get("text_memories") or [])
        visual_memories = list(enrichment.get("visual_memories") or [])
        audio_memories = list(enrichment.get("audio_memories") or [])
        query = str(enrichment.get("query") or "")
        time_window_minutes = int(enrichment.get("time_window_minutes") or 60)

        strategy = getattr(settings, "CROSS_MODAL_REASONING_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        if strategy == "heuristic":
            outputs = run_heuristic(
                text_memories=text_memories,
                visual_memories=visual_memories,
                audio_memories=audio_memories,
                query=query,
                time_window_minutes=time_window_minutes,
            )
        else:
            prompt = (
                "You correlate evidence across text/visual/audio memories. Return JSON only.\n\n"
                f"QUERY: {query}\n"
                f"TIME_WINDOW_MINUTES: {time_window_minutes}\n"
                f"TEXT_MEMORIES: {text_memories[:80]}\n"
                f"VISUAL_MEMORIES: {visual_memories[:80]}\n"
                f"AUDIO_MEMORIES: {audio_memories[:80]}\n\n"
                "Return keys: cross_modal_links, unified_conclusion, modalities_used, evidence_strength, confidence, rationale"
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
                and isinstance(resp.get("cross_modal_links"), list)
                and isinstance(resp.get("unified_conclusion"), str)
                and isinstance(resp.get("modalities_used"), list)
                and isinstance(resp.get("evidence_strength"), (int, float))
            ):
                outputs = resp
            else:
                outputs = run_heuristic(
                    text_memories=text_memories,
                    visual_memories=visual_memories,
                    audio_memories=audio_memories,
                    query=query,
                    time_window_minutes=time_window_minutes,
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
