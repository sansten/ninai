"""Temporal pattern miner agent (Phase 62)."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from statistics import mean, pstdev
from typing import Any

from app.agents.base import BaseAgent
from app.agents.llm.llm_breaker import create_llm_client
from app.agents.types import AgentContext, AgentResult
from app.core.config import settings


def _parse_dt(value: Any) -> datetime | None:
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
            return None
    return None


def _top_tags(memories: list[dict[str, Any]], indices: list[int], limit: int = 3) -> list[str]:
    c: Counter[str] = Counter()
    for idx in indices:
        tags = memories[idx].get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        for t in tags:
            t_text = str(t).strip().lower()
            if t_text:
                c[t_text] += 1
    return [k for k, _ in c.most_common(limit)]


def _avg_severity(memories: list[dict[str, Any]], indices: list[int]) -> float:
    vals = []
    for idx in indices:
        val = memories[idx].get("severity")
        vals.append(float(val) if isinstance(val, (int, float)) else 0.5)
    return round(mean(vals) if vals else 0.5, 4)


def run_heuristic(
    *,
    memories: list[dict[str, Any]],
    analysis_window_days: int = 90,
    min_occurrences: int = 3,
) -> dict[str, Any]:
    mems = list(memories or [])
    if not mems:
        return {
            "patterns": [],
            "dominant_pattern": None,
            "anomalous_times": [],
            "total_events_analysed": 0,
            "confidence": 0.5,
            "rationale": "heuristic",
        }

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max(1, int(analysis_window_days or 90)))

    filtered: list[dict[str, Any]] = []
    for m in mems:
        dt = _parse_dt(m.get("created_at"))
        if dt is None:
            continue
        if dt >= cutoff:
            m2 = dict(m)
            m2["_dt"] = dt
            filtered.append(m2)

    if not filtered:
        return {
            "patterns": [],
            "dominant_pattern": None,
            "anomalous_times": [],
            "total_events_analysed": 0,
            "confidence": 0.5,
            "rationale": "heuristic",
        }

    hour_counts: dict[int, list[int]] = defaultdict(list)
    dow_counts: dict[int, list[int]] = defaultdict(list)
    for idx, m in enumerate(filtered):
        dt = m["_dt"]
        hour_counts[dt.hour].append(idx)
        dow_counts[dt.weekday()].append(idx)

    patterns: list[dict[str, Any]] = []

    denom = max(1.0, float(analysis_window_days) / 7.0)
    for hour, indices in hour_counts.items():
        occ = len(indices)
        if occ < min_occurrences:
            continue
        conf = min(0.95, occ / denom)
        patterns.append(
            {
                "pattern_type": "hour_of_day",
                "pattern_key": f"hour_{hour}",
                "topic_tags": _top_tags(filtered, indices, limit=3),
                "occurrence_count": occ,
                "avg_severity": _avg_severity(filtered, indices),
                "confidence": round(conf, 4),
            }
        )

    for dow, indices in dow_counts.items():
        occ = len(indices)
        if occ < min_occurrences:
            continue
        conf = min(0.95, occ / denom)
        patterns.append(
            {
                "pattern_type": "day_of_week",
                "pattern_key": f"dow_{dow}",
                "topic_tags": _top_tags(filtered, indices, limit=3),
                "occurrence_count": occ,
                "avg_severity": _avg_severity(filtered, indices),
                "confidence": round(conf, 4),
            }
        )

    hour_values = [len(hour_counts.get(h, [])) for h in range(24)]
    mu = mean(hour_values)
    sigma = pstdev(hour_values) if len(hour_values) > 1 else 0.0
    anomalous_times = []
    if sigma > 0:
        threshold = mu + 2 * sigma
        for h, count in enumerate(hour_values):
            if count > threshold:
                anomalous_times.append(f"hour_{h}")

    dominant_pattern = None
    if patterns:
        max_occ = max(p["occurrence_count"] for p in patterns)
        tied = [p for p in patterns if p["occurrence_count"] == max_occ]
        dominant_pattern = tied[0] if len(tied) == 1 else None

    overall_conf = round(mean([p["confidence"] for p in patterns]), 4) if patterns else 0.5

    return {
        "patterns": patterns,
        "dominant_pattern": dominant_pattern,
        "anomalous_times": anomalous_times,
        "total_events_analysed": len(filtered),
        "confidence": overall_conf,
        "rationale": "heuristic",
    }


class TemporalPatternMinerAgent(BaseAgent):
    name = "TemporalPatternMinerAgent"
    version = "v1"

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}

        if not isinstance(outputs.get("patterns"), list):
            raise ValueError("patterns must be a list")
        if outputs.get("dominant_pattern") is not None and not isinstance(outputs.get("dominant_pattern"), dict):
            raise ValueError("dominant_pattern must be a dict or None")
        if not isinstance(outputs.get("anomalous_times"), list):
            raise ValueError("anomalous_times must be a list")
        if not isinstance(outputs.get("total_events_analysed"), int):
            raise ValueError("total_events_analysed must be int")

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        enrichment = (context.get("memory") or {}).get("enrichment") or {}

        memories = list(enrichment.get("memories") or [])
        analysis_window_days = int(enrichment.get("analysis_window_days") or 90)
        min_occurrences = int(enrichment.get("min_occurrences") or 3)

        strategy = getattr(settings, "TEMPORAL_PATTERN_MINER_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        if strategy == "heuristic":
            outputs = run_heuristic(
                memories=memories,
                analysis_window_days=analysis_window_days,
                min_occurrences=min_occurrences,
            )
        else:
            prompt = (
                "You mine recurring temporal patterns from timestamped memory events. Output JSON only.\n\n"
                f"MEMORIES: {memories[:80]}\n"
                f"ANALYSIS_WINDOW_DAYS: {analysis_window_days}\n"
                f"MIN_OCCURRENCES: {min_occurrences}\n\n"
                "Return JSON with keys:\n"
                "- patterns: list[{pattern_type, pattern_key, topic_tags, occurrence_count, avg_severity, confidence}]\n"
                "- dominant_pattern: dict or null\n"
                "- anomalous_times: list[str]\n"
                "- total_events_analysed: int\n"
                "- confidence: float\n"
                "- rationale: str"
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
                and isinstance(resp.get("patterns"), list)
                and isinstance(resp.get("anomalous_times"), list)
                and isinstance(resp.get("total_events_analysed"), int)
            ):
                outputs = resp
            else:
                outputs = run_heuristic(
                    memories=memories,
                    analysis_window_days=analysis_window_days,
                    min_occurrences=min_occurrences,
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
