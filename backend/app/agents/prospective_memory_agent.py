"""Prospective memory agent — deadline detection and reminder suggestion (Phase 53)."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.llm.ollama_breaker import create_ollama_client
from app.agents.types import AgentContext, AgentResult
from app.core.config import settings

# ---------------------------------------------------------------------------
# Token → time-offset helpers
# ---------------------------------------------------------------------------

_DEADLINE_KEYWORDS = [
    "by",
    "before",
    "within",
    "due",
    "deadline",
    "expires",
    "until",
    "no later than",
]

# Regexes ordered from most-specific to least-specific.
_PATTERN_HOURS = re.compile(
    r"within\s+(\d+(?:\.\d+)?)\s+hours?", re.IGNORECASE
)
_PATTERN_DAYS = re.compile(
    r"within\s+(\d+(?:\.\d+)?)\s+days?", re.IGNORECASE
)
_PATTERN_TOMORROW = re.compile(r"\btomorrow\b", re.IGNORECASE)
_PATTERN_END_OF_WEEK = re.compile(r"\bend[\s\-]of[\s\-]week\b", re.IGNORECASE)
_PATTERN_FRIDAY = re.compile(r"\b(?:by\s+)?friday\b", re.IGNORECASE)
_PATTERN_MONDAY = re.compile(r"\b(?:by\s+)?monday\b", re.IGNORECASE)
_PATTERN_HOUR_PLAIN = re.compile(
    r"\b(\d+(?:\.\d+)?)\s+hours?\b", re.IGNORECASE
)
_PATTERN_DAY_PLAIN = re.compile(
    r"\b(\d+(?:\.\d+)?)\s+days?\b", re.IGNORECASE
)
_PATTERN_WEEK_PLAIN = re.compile(
    r"\b(\d+(?:\.\d+)?)\s+weeks?\b", re.IGNORECASE
)


def _hours_until_weekday(
    current_time: datetime,
    target_weekday: int,  # 0=Mon … 6=Sun
    target_hour: int = 17,
) -> float:
    """Return hours from *current_time* until the next occurrence of *target_weekday* at *target_hour*."""
    days_ahead = (target_weekday - current_time.weekday()) % 7
    if days_ahead == 0 and current_time.hour >= target_hour:
        days_ahead = 7
    target_dt = current_time.replace(
        hour=target_hour, minute=0, second=0, microsecond=0
    ) + timedelta(days=days_ahead)
    delta = target_dt - current_time
    return max(0.0, delta.total_seconds() / 3600)


def infer_offset_hours(token: str, current_time: datetime) -> float | None:
    """Return hours offset for a single deadline token/phrase, or None if unparseable."""
    m = _PATTERN_HOURS.search(token)
    if m:
        return float(m.group(1))
    m = _PATTERN_DAYS.search(token)
    if m:
        return float(m.group(1)) * 24
    if _PATTERN_TOMORROW.search(token):
        return 24.0
    if _PATTERN_END_OF_WEEK.search(token):
        return _hours_until_weekday(current_time, 4)  # Friday
    if _PATTERN_FRIDAY.search(token):
        return _hours_until_weekday(current_time, 4)
    if _PATTERN_MONDAY.search(token):
        return _hours_until_weekday(current_time, 0)
    m = _PATTERN_HOUR_PLAIN.search(token)
    if m:
        return float(m.group(1))
    m = _PATTERN_DAY_PLAIN.search(token)
    if m:
        return float(m.group(1)) * 24
    m = _PATTERN_WEEK_PLAIN.search(token)
    if m:
        return float(m.group(1)) * 168
    return None


def classify_urgency(offset_hours: float) -> str:
    if offset_hours < 4:
        return "high"
    if offset_hours < 48:
        return "medium"
    return "low"


def extract_deadline_tokens(content: str) -> list[str]:
    """Return substring snippets that contain deadline keywords."""
    tokens: list[str] = []
    lower = content.lower()
    for kw in _DEADLINE_KEYWORDS:
        idx = lower.find(kw)
        while idx != -1:
            # Grab up to 60 chars around the keyword for context.
            snippet = content[idx : idx + 60].strip()
            tokens.append(snippet)
            idx = lower.find(kw, idx + 1)
    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for t in tokens:
        low = t.lower()
        if low not in seen:
            seen.add(low)
            unique.append(t)
    return unique


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ProspectiveMemoryAgent(BaseAgent):
    """Phase 53: detect deadlines and suggest prospective reminders."""

    name = "ProspectiveMemoryAgent"
    version = "v1"

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}
        if not isinstance(outputs.get("reminders_suggested"), list):
            raise ValueError("reminders_suggested must be a list")
        if not isinstance(outputs.get("deadline_detected"), bool):
            raise ValueError("deadline_detected must be a bool")
        if not isinstance(outputs.get("deadline_tokens"), list):
            raise ValueError("deadline_tokens must be a list")
        if not isinstance(outputs.get("confidence"), float):
            raise ValueError("confidence must be a float")

    def _heuristic(
        self,
        *,
        content: str,
        current_time: datetime,
        existing_reminders: list[dict[str, Any]],
    ) -> dict[str, Any]:
        tokens = extract_deadline_tokens(content)
        if not tokens:
            return {
                "reminders_suggested": [],
                "deadline_detected": False,
                "deadline_tokens": [],
                "confidence": 0.95,
            }

        suggestions: list[dict[str, Any]] = []
        for tok in tokens:
            offset = infer_offset_hours(tok, current_time)
            if offset is None:
                # Keyword found but no parseable time → include with a default offset
                offset = 168.0  # 1 week — conservative default
            urgency = classify_urgency(offset)
            suggestions.append(
                {
                    "trigger_type": "time",
                    "trigger_at_offset_hours": round(offset, 2),
                    "reminder_content": f"Deadline detected: {tok.strip()}",
                    "urgency": urgency,
                }
            )

        # Suppress duplicates against existing reminders (same reminder_content)
        existing_contents = {
            r.get("reminder_content", "").lower().rstrip(".,;:!?") for r in existing_reminders
        }
        suggestions = [
            s
            for s in suggestions
            if s["reminder_content"].lower().rstrip(".,;:!?") not in existing_contents
        ]

        confidence = min(0.95, 0.5 + 0.15 * len(tokens))
        return {
            "reminders_suggested": suggestions,
            "deadline_detected": True,
            "deadline_tokens": tokens,
            "confidence": round(confidence, 4),
        }

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        enrichment = (context.get("memory") or {}).get("enrichment") or {}
        content: str = str(enrichment.get("content") or "")
        existing_reminders: list[dict[str, Any]] = list(
            enrichment.get("existing_reminders") or []
        )
        current_time_raw = enrichment.get("current_time")
        if isinstance(current_time_raw, datetime):
            current_time = current_time_raw
        else:
            current_time = datetime.now(timezone.utc)

        if settings.AGENT_STRATEGY == "heuristic":
            outputs = self._heuristic(
                content=content,
                current_time=current_time,
                existing_reminders=existing_reminders,
            )
            outputs["rationale"] = "heuristic"
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
            client = create_ollama_client()
            prompt = (
                "You are a deadline-detection assistant. Given a text snippet, "
                "identify any deadline phrases and estimate offset hours.\n\n"
                f"Text: {content}\n\n"
                "Respond as JSON with keys: reminders_suggested (list of "
                "{trigger_type, trigger_at_offset_hours, reminder_content, urgency}), "
                "deadline_detected (bool), deadline_tokens (list[str]), confidence (float)."
            )
            resp = await client.generate(model=settings.OLLAMA_MODEL, prompt=prompt)
            import json

            parsed = json.loads(resp.get("response", "{}"))
            if not isinstance(parsed.get("deadline_detected"), bool):
                raise ValueError("invalid llm response")
            parsed["rationale"] = "llm"
            finished_at = datetime.now(timezone.utc)
            result = AgentResult(
                agent_name=self.name,
                agent_version=self.version,
                memory_id=memory_id,
                status="success",
                confidence=float(parsed.get("confidence", 0.7)),
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
                content=content,
                current_time=current_time,
                existing_reminders=existing_reminders,
            )
            outputs["rationale"] = "heuristic"
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
