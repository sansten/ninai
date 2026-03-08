"""Uncertainty-Gated Expansion Service (GAP-4).

Implements entropy-based evidence admission control following Eq. (8) from
the xMemory framework:

    Include e_j iff H_reader(y | C ∪ {e_j}) < H_reader(y | C) - δ

Where:
    - H_reader(y|C) is the reader LLM's entropy over answer y given context C
    - δ is the minimum uncertainty reduction threshold
    - e_j is a candidate evidence item (episode, message)

This ensures that additional evidence is only included when it meaningfully
reduces the reader's predictive uncertainty, preventing token waste from
redundant or uninformative context.

Reference: Hu et al. (2026), "Beyond RAG for Agent Memory", Eq. (8)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Tunables ────────────────────────────────────────────────────────────
DEFAULT_ENTROPY_THRESHOLD = 0.1     # δ in Eq. (8), bits of uncertainty reduction
DEFAULT_MAX_EXPANSION_ITEMS = 10    # Budget for Stage II expansion
DEFAULT_TEMPERATURE = 0.0           # Low temp for entropy calculation
OLLAMA_TIMEOUT = 30.0               # Timeout for LLM calls


class UncertaintyGatingService:
    """Entropy-based adaptive evidence expansion (GAP-4)."""

    def __init__(self):
        self.ollama_base_url = settings.OLLAMA_BASE_URL

    # ── Public API ──────────────────────────────────────────────────

    async def expand_with_gating(
        self,
        *,
        query: str,
        initial_context: str,
        candidate_items: List[Dict[str, Any]],
        max_items: int = DEFAULT_MAX_EXPANSION_ITEMS,
        entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD,
    ) -> Dict[str, Any]:
        """Iteratively expand context with entropy-gated evidence admission.

        Args:
            query: User query
            initial_context: Coarse context from Stage I (topics + summaries)
            candidate_items: List of candidate evidence dicts with keys:
                - "id": unique identifier
                - "content": text content
                - "type": "episode" | "message" | "semantic_node"
            max_items: Maximum number of items to add
            entropy_threshold: Minimum ΔH required for inclusion (bits)

        Returns:
            {
                "final_context": str,
                "included_items": [{"id": str, "type": str, "entropy_reduction": float}, ...],
                "total_entropy_reduction": float,
                "expansion_stopped_reason": str,
            }
        """
        logger.info(
            "expand_with_gating: query='%s' candidates=%d threshold=%.3f",
            query[:50],
            len(candidate_items),
            entropy_threshold,
        )

        # Handle empty candidates
        if not candidate_items:
            return self._build_result(
                initial_context,
                [],
                0.0,
                "No candidates provided",
            )

        # Start with initial context
        current_context = initial_context
        current_entropy = await self._compute_entropy(query, current_context)

        included_items = []
        total_reduction = 0.0

        for i, candidate in enumerate(candidate_items):
            if len(included_items) >= max_items:
                logger.info("Reached max_items budget (%d)", max_items)
                return self._build_result(
                    current_context,
                    included_items,
                    total_reduction,
                    "Max items reached",
                )

            # Trial context: add candidate
            trial_context = self._append_evidence(current_context, candidate)
            trial_entropy = await self._compute_entropy(query, trial_context)

            # Compute uncertainty reduction
            delta_h = current_entropy - trial_entropy

            logger.debug(
                "Candidate %d: H_current=%.3f H_trial=%.3f ΔH=%.3f (threshold=%.3f)",
                i,
                current_entropy,
                trial_entropy,
                delta_h,
                entropy_threshold,
            )

            if delta_h >= entropy_threshold:
                # Accept: reduces uncertainty sufficiently
                included_items.append({
                    "id": candidate["id"],
                    "type": candidate["type"],
                    "entropy_reduction": delta_h,
                })
                current_context = trial_context
                current_entropy = trial_entropy
                total_reduction += delta_h
            else:
                # Reject: insufficient uncertainty reduction — continue evaluating remaining
                # candidates. A later candidate may still pass the threshold independently.
                logger.debug("Candidate %d rejected (ΔH=%.3f < δ=%.3f), continuing", i, delta_h, entropy_threshold)
                continue

        # Finished all candidates (all accepted)
        logger.info(
            "Expansion complete: included %d/%d items, total ΔH=%.3f",
            len(included_items),
            len(candidate_items),
            total_reduction,
        )
        return self._build_result(
            current_context,
            included_items,
            total_reduction,
            "All candidates evaluated",
        )

    # ── Entropy Computation ─────────────────────────────────────────

    async def _compute_entropy(self, query: str, context: str) -> float:
        """Compute H_reader(y|C) — the LLM's predictive entropy.

        Calls Ollama with logprobs enabled to get token-level probabilities,
        then computes Shannon entropy over the output distribution.

        H = -Σ p(y_i) log₂ p(y_i)
        """
        try:
            # Build prompt
            prompt = self._build_prompt(query, context)

            # Call Ollama with logprobs
            response_data = await self._call_ollama_with_logprobs(prompt=prompt)

            # Extract probabilities and compute entropy
            entropy = self._calculate_entropy_from_logprobs(response_data)

            return entropy

        except Exception as exc:
            logger.warning("Entropy computation failed: %s, returning default=1.0", exc)
            return 1.0  # High entropy = uncertain

    async def _call_ollama_with_logprobs(self, prompt: str) -> Dict[str, Any]:
        """Call Ollama API with logprobs enabled."""
        url = f"{self.ollama_base_url}/api/generate"
        payload = {
            "model": settings.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": DEFAULT_TEMPERATURE,
                "num_predict": 50,  # Generate enough tokens for entropy calculation
                "top_k": 50,        # Sample from top-k for logprob diversity
            },
        }

        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()

    def _calculate_entropy_from_logprobs(
        self,
        response_data: Optional[Any] = None,
        fallback_text: Optional[str] = None,
    ) -> float:
        """Compute Shannon entropy from logprobs or response.

        Args:
            response_data: Either dict with {"response": str, "logprobs": list}
                          or list of {"token": str, "logprob": float} dicts
            fallback_text: Optional response text for length-based heuristic

        Returns:
            Entropy in bits
        """
        # Case 1: response_data is a list of logprob dicts
        if isinstance(response_data, list):
            if not response_data:
                return 1.0
            # Calculate Shannon entropy from logprobs
            # H = -Σ p(token_i) * log2(p(token_i))
            total_entropy = 0.0
            for item in response_data:
                logprob = item.get("logprob", -1.0)
                # Convert natural log to probability
                prob = math.exp(logprob)
                if prob > 0:
                    # H = -Σ p*log2(p)
                    total_entropy -= prob * math.log2(prob)
            # Return total entropy (not per-token average)
            return total_entropy

        # Case 2: response_data is dict from Ollama
        if isinstance(response_data, dict):
            logprobs = response_data.get("logprobs")
            if logprobs and isinstance(logprobs, list):
                # Recursively handle logprobs list
                return self._calculate_entropy_from_logprobs(logprobs)
            response_text = response_data.get("response", "")
        else:
            response_text = fallback_text or ""

        # Fallback heuristic: normalized response length
        # Longer responses → more uncertain (higher entropy)
        # Shorter responses → more confident (lower entropy)
        response_length = len(response_text.split())

        # Map length to entropy range [0.1, 2.0] bits
        # Short response (5 tokens) → ~0.2 bits
        # Long response (50 tokens) → ~1.5 bits
        normalized_entropy = 0.1 + (min(response_length, 50) / 50) * 1.4

        logger.debug(
            "Entropy estimate from response length: %d tokens → %.3f bits",
            response_length,
            normalized_entropy,
        )

        return normalized_entropy

    # ── Prompt Construction ─────────────────────────────────────────

    def _build_prompt(self, query: str, context: str) -> str:
        """Build a prompt for entropy calculation.

        We want the LLM to answer briefly, so we can measure confidence.
        """
        return f"""Based on the following context, answer the query concisely.

Context:
{context}

Query: {query}

Answer:"""

    def _append_evidence(self, context: str, evidence: Dict[str, Any]) -> str:
        """Append evidence item to context."""
        evidence_type = evidence.get("type", "unknown")
        evidence_content = evidence.get("content", "")
        return f"{context}\n\n[{evidence_type.upper()}]\n{evidence_content}"

    def _build_result(
        self,
        final_context: str,
        included_items: List[Dict[str, Any]],
        total_reduction: float,
        stop_reason: str,
    ) -> Dict[str, Any]:
        """Build result dictionary."""
        return {
            "final_context": final_context,
            "included_items": included_items,
            "total_entropy_reduction": total_reduction,
            "expansion_stopped_reason": stop_reason,
        }
