"""Self-RAG verification service (Feature 24.3).

Provides a reflection-token-like verification pass over retrieved memories:
- is_supported: does memory contain query-supported terms?
- is_grounded: does memory include enough content/credibility to ground a claim?
- is_useful: combined utility score for keeping the memory in context.

Outputs reasoning steps suitable for explainable response payloads.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PUNCT = re.compile(r"[^\w\s]")


@dataclass
class SelfRagVerificationResult:
    verified_memories: list[dict]
    reasoning_steps: list[dict]


def _tokens(value: str) -> set[str]:
    """Tokenise text into a lowercase word set, stripping punctuation."""
    return {t for t in _PUNCT.sub(" ", str(value or "")).lower().split() if t}


class SelfRagService:
    """Verify and filter retrieved context before answer generation."""

    def _score(self, *, query: str, memory: dict) -> dict:
        query_tokens = _tokens(query)
        content = str(memory.get("content") or "")
        content_tokens = _tokens(content)
        overlap = len(query_tokens & content_tokens)

        supported = overlap > 0
        grounded = len(content_tokens) >= 3

        cred = memory.get("credibility_score")
        if cred is None:
            cred = memory.get("enrichment", {}).get("credibility_score")
        try:
            cred_f = float(cred) if cred is not None else 1.0
        except (TypeError, ValueError):
            cred_f = 1.0
        cred_f = max(0.1, min(1.0, cred_f))

        support_score = min(1.0, overlap / max(1, len(query_tokens)))
        grounded_score = 0.6 if grounded else 0.2
        useful_score = round(0.5 * support_score + 0.3 * grounded_score + 0.2 * cred_f, 4)

        return {
            "is_supported": supported,
            "is_grounded": grounded,
            "is_useful": useful_score >= 0.35,
            "support_score": round(support_score, 4),
            "grounded_score": round(grounded_score, 4),
            "useful_score": useful_score,
        }

    def verify_and_filter(
        self,
        *,
        query: str,
        memories: list[dict],
        strict_support: bool = True,
    ) -> SelfRagVerificationResult:
        verified: list[dict] = []
        steps: list[dict] = []

        for mem in memories:
            result = self._score(query=query, memory=mem)
            if strict_support:
                keep = bool(result["is_supported"] and result["is_grounded"] and result["is_useful"])
            else:
                # In semantic vector mode, lexical support can be low while relevance is high.
                keep = bool(result["is_grounded"] and result["is_useful"])

            steps.append(
                {
                    "memory_id": mem.get("id"),
                    "decision": "kept" if keep else "filtered",
                    "scores": result,
                }
            )

            if keep:
                verified.append(mem)

        return SelfRagVerificationResult(verified_memories=verified, reasoning_steps=steps)
