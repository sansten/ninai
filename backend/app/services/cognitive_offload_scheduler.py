"""Cognitive offload scheduler (Phase 73)."""

from __future__ import annotations


class CognitiveOffloadScheduler:
    KEEP_TYPES = {"decision", "incident_summary", "goal", "playbook", "causal_insight"}
    COMPRESS_TYPES = {"log", "event_stream", "raw_data"}
    OFFLOAD_TYPES = {"documentation", "changelog", "static_config"}
    DISCARD_LOW_TYPES = {"test", "debug_trace"}

    @staticmethod
    def _as_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _as_int(value: object, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    def decide(
        self,
        *,
        content_type: str,
        importance_score: float,
        reference_count: int,
        is_near_duplicate: bool,
    ) -> str:
        ctype = str(content_type or "").strip().lower()
        importance = max(0.0, min(1.0, self._as_float(importance_score, 0.0)))
        refs = max(0, self._as_int(reference_count, 0))

        if bool(is_near_duplicate):
            return "discard"

        if ctype in self.DISCARD_LOW_TYPES and importance < 0.1:
            return "discard"

        if refs >= 2:
            return "keep"

        if ctype in self.KEEP_TYPES and importance >= 0.5:
            return "keep"

        if ctype in self.COMPRESS_TYPES and 0.2 <= importance < 0.5 and refs == 0:
            return "compress"

        if ctype in self.OFFLOAD_TYPES and importance < 0.2:
            return "offload"

        if importance >= 0.6:
            return "keep"
        if importance >= 0.2:
            return "compress"
        return "offload"

    def compress_content(self, content: str, max_chars: int = 200) -> str:
        text = (content or "").strip()
        limit = max(1, int(max_chars if max_chars is not None else 200))

        if len(text) <= limit:
            return text

        head = text[:limit].rstrip()

        best_end = -1
        for mark in (". ", "! ", "? ", ".", "!", "?"):
            idx = head.rfind(mark)
            if idx > best_end:
                best_end = idx

        if best_end >= 0:
            if head[best_end] in ".!?":
                trimmed = head[: best_end + 1].strip()
            else:
                trimmed = head[:best_end].strip()
        else:
            trimmed = head

        return f"{trimmed} [compressed]"

    def offload_pointer(self, *, content: str, source_url: str | None) -> dict:
        text = (content or "").strip()
        return {
            "type": "pointer",
            "summary": text[:100],
            "source_url": source_url,
            "retrievable": True,
        }

    def batch_decide(self, *, memories: list[dict]) -> dict[str, list[dict]]:
        buckets: dict[str, list[dict]] = {
            "keep": [],
            "compress": [],
            "offload": [],
            "discard": [],
        }

        for memory in memories or []:
            decision = self.decide(
                content_type=str(memory.get("content_type", "")),
                importance_score=self._as_float(memory.get("importance_score"), 0.0),
                reference_count=self._as_int(memory.get("reference_count"), 0),
                is_near_duplicate=bool(memory.get("is_near_duplicate", False)),
            )
            buckets[decision].append(dict(memory))

        return buckets
