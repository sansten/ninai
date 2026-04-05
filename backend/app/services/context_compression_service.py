"""Hierarchical contextual compression service (Feature 24.4).

Compresses retrieved memory content to a token budget while preserving
high-signal elements:
- entities
- causal markers / causal links
- high-credibility claims

Returns compression ratio and information-density metadata for API responses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _token_count(text: str) -> int:
    return len(str(text or "").split())


def _split_sentences(text: str) -> list[str]:
    raw = str(text or "").replace("\n", " ")
    for sep in ("?", "!", ";"):
        raw = raw.replace(sep, ".")
    return [chunk.strip() for chunk in raw.split(".") if chunk.strip()]


@dataclass
class CompressionResult:
    memories: list[dict[str, Any]]
    compression_ratio: float
    information_density: float


class ContextCompressionService:
    """Token-budget compression with basic signal-preservation heuristics."""

    def _preservation_score(self, memory: dict[str, Any]) -> int:
        score = 0
        cred = memory.get("credibility_score")
        if cred is None:
            cred = memory.get("enrichment", {}).get("credibility_score")
        try:
            cred_f = float(cred) if cred is not None else 0.5
        except (TypeError, ValueError):
            cred_f = 0.5

        if cred_f >= 0.75:
            score += 2

        entities = memory.get("entities") or memory.get("enrichment", {}).get("entities")
        if isinstance(entities, dict) and len(entities) > 0:
            score += 2

        causal_links = memory.get("causal_links") or memory.get("causal_chain")
        if isinstance(causal_links, (list, dict, str)) and causal_links:
            score += 1

        return score

    def _select_sentences(self, memory: dict[str, Any], token_budget: int) -> str:
        content = str(memory.get("content") or "")
        if _token_count(content) <= token_budget:
            return content

        entities = memory.get("entities") or memory.get("enrichment", {}).get("entities")
        entity_tokens: set[str] = set()
        if isinstance(entities, dict):
            for value in entities.values():
                if isinstance(value, str):
                    entity_tokens.update(value.lower().split())
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            entity_tokens.update(item.lower().split())

        causal_markers = {"because", "therefore", "caused", "cause", "resulted", "impact", "due"}
        sentences = _split_sentences(content)

        prioritized: list[tuple[int, str]] = []
        for sentence in sentences:
            lower = sentence.lower()
            rank = 0
            if entity_tokens and any(tok in lower for tok in entity_tokens):
                rank += 2
            if any(marker in lower for marker in causal_markers):
                rank += 1
            prioritized.append((rank, sentence))

        prioritized.sort(key=lambda item: item[0], reverse=True)

        selected: list[str] = []
        used = 0
        for _, sentence in prioritized:
            count = _token_count(sentence)
            if used + count > token_budget:
                continue
            selected.append(sentence)
            used += count
            if used >= token_budget:
                break

        if not selected:
            words = content.split()
            return " ".join(words[: max(1, token_budget)])
        return ". ".join(selected)

    def compress(self, *, memories: list[dict[str, Any]], token_budget: int = 240) -> CompressionResult:
        mems = list(memories or [])
        if not mems:
            return CompressionResult(memories=[], compression_ratio=1.0, information_density=0.0)

        budget = max(40, int(token_budget or 240))
        total_original = sum(_token_count(str(m.get("content") or "")) for m in mems)
        if total_original <= budget:
            return CompressionResult(memories=mems, compression_ratio=1.0, information_density=0.0)

        weighted = sorted(mems, key=self._preservation_score, reverse=True)
        per_memory_budget = max(8, budget // max(1, len(weighted)))

        compressed: list[dict[str, Any]] = []
        preserved_signals = 0
        total_compressed = 0

        for memory in weighted:
            ctext = self._select_sentences(memory, per_memory_budget)
            original_tokens = _token_count(str(memory.get("content") or ""))
            compressed_tokens = _token_count(ctext)
            total_compressed += compressed_tokens

            if compressed_tokens > 0 and self._preservation_score(memory) > 0:
                preserved_signals += 1

            compressed.append(
                {
                    **memory,
                    "content": ctext,
                    "_compressed": compressed_tokens < original_tokens,
                    "_original_tokens": original_tokens,
                    "_compressed_tokens": compressed_tokens,
                }
            )

        ratio = round(total_compressed / max(1, total_original), 4)
        density = round(preserved_signals / max(1, total_compressed), 4)
        return CompressionResult(
            memories=compressed,
            compression_ratio=ratio,
            information_density=density,
        )
