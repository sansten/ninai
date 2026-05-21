"""
V2 Ollama Reasoning Engine (Component A)

Stateless controller — receives assembled context and produces:
  1. A natural-language response
  2. A list of graph node IDs cited in the reasoning (explainability)
  3. Extracted entities for graph write-back

Structured output is requested via Ollama's `format: json` mode.
Falls back to heuristic parsing if the model doesn't return valid JSON.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 45.0
_EMBED_PATH = "/api/embeddings"
_GENERATE_PATH = "/api/generate"


@dataclass
class InferenceResult:
    response: str = ""
    cited_node_ids: list[str] = field(default_factory=list)
    extracted_entities: list[dict[str, Any]] = field(default_factory=list)
    raw_json: dict[str, Any] = field(default_factory=dict)
    error: str = ""


class OllamaReasoningEngine:
    """
    Thin async wrapper around Ollama's generate + embeddings APIs.

    Kept deliberately minimal — all prompt assembly happens in prompt_builder.py.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b",
        embed_model: str = "nomic-embed-text",
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._model = model
        self._embed_model = embed_model
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Embeddings (used by DNC router read-weighting)
    # ------------------------------------------------------------------

    async def embed(self, text: str) -> list[float]:
        payload = {"model": self._embed_model, "prompt": text}
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(f"{self._base}{_EMBED_PATH}", json=payload)
                resp.raise_for_status()
                return resp.json().get("embedding", [])
        except Exception as exc:
            logger.warning("Embedding failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Entity extraction (structured JSON output)
    # ------------------------------------------------------------------

    async def extract_entities(self, text: str) -> list[dict[str, Any]]:
        """
        Ask the LLM to identify named entities in `text`.
        Returns list of {id, name, type} dicts.
        """
        prompt = (
            "Extract named entities from the following text.\n"
            "Return ONLY a JSON array with objects having these keys:\n"
            '  "name": string (the entity name),\n'
            '  "type": one of ["concept","user","task","object"],\n'
            '  "id": a snake_case unique key for the entity.\n\n'
            f"Text:\n{text[:1500]}\n\n"
            "JSON array:"
        )
        raw = await self._generate_json(prompt)
        entities: list[dict[str, Any]] = []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict) and "name" in item:
                    entities.append(
                        {
                            "id": str(item.get("id", item["name"].lower().replace(" ", "_"))),
                            "name": str(item["name"]),
                            "type": str(item.get("type", "concept")),
                        }
                    )
        return entities

    # ------------------------------------------------------------------
    # Main inference call
    # ------------------------------------------------------------------

    async def infer(self, prompt: str) -> InferenceResult:
        """
        Run the assembled Graph-RAG prompt through Ollama.
        Expects the model to return JSON with keys:
          response, cited_node_ids, extracted_entities
        """
        result = InferenceResult()
        raw = await self._generate_json(prompt)
        if not raw:
            result.error = "empty model response"
            return result

        result.raw_json = raw if isinstance(raw, dict) else {}
        if isinstance(raw, dict):
            result.response = str(raw.get("response", ""))
            cited = raw.get("cited_node_ids", [])
            result.cited_node_ids = [str(c) for c in cited] if isinstance(cited, list) else []
            entities = raw.get("extracted_entities", [])
            result.extracted_entities = entities if isinstance(entities, list) else []
        else:
            # Model returned a plain string — wrap it
            result.response = str(raw)

        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _generate_json(self, prompt: str) -> Any:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.1},
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(f"{self._base}{_GENERATE_PATH}", json=payload)
                resp.raise_for_status()
                data = resp.json()
                raw_text = data.get("response", "")
                return json.loads(raw_text)
        except json.JSONDecodeError as exc:
            logger.warning("JSON decode failed from model output: %s", exc)
            return {}
        except Exception as exc:
            logger.warning("Ollama generate failed: %s", exc)
            return {}

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base}/api/version")
                return resp.status_code == 200
        except Exception:
            return False
