"""Multimodal Deep Memory — Phase 43.

Provides semantic understanding of non-text attachments (images, diagrams,
screenshots, audio, video) so that memories containing visual or recorded
content can be retrieved by their semantic meaning rather than just filename.

Example: "the screenshot where the button was red" → retrievable by visual
content because this agent tagged the memory with colour and UI-element
context at ingest time.

Builds on:
  - VisionMemoryService (PR-9)       — image embeddings, OCR, object detection
  - SpatialMemory model (PR-9)       — UI element layout extracted from screenshots
  - MemoryAttachment model           — attachment metadata
  - PlaybookAgent (Phase 20)         — procedural knowledge from video walkthroughs

Inputs (via context["memory"]):
  - content:  raw text of the memory (may be empty for pure-attachment memories)

Inputs (via context["memory"]["enrichment"]):
  - attachments:       list[dict]  — each: {id, file_name, content_type,
                                            size_bytes}
  - visual_embeddings: list[dict]  — pre-extracted per Phase PR-9 service;
                                     each: {modality, extracted_text,
                                            detected_objects, object_confidence_scores,
                                            embedding_confidence}
  - spatial_data:      dict        — UI-element layout from SpatialMemory;
                                     keys: elements, layout_description, dominant_colors
  - audio_analysis:    dict        — from AudioAnalysis model;
                                     keys: transcription, emotion_detected, summary

Outputs:
  - primary_modality:    str        — dominant modality of the memory's attachments
  - modalities_present:  list[str]  — all distinct modalities detected
  - extracted_text:      str        — combined OCR / transcription text
  - detected_objects:    list[str]  — objects / UI elements identified
  - dominant_colors:     list[str]  — colour palette (screenshots/images)
  - searchable_tags:     list[str]  — normalised tags for retrieval
  - visual_summary:      str        — one-sentence natural language description
  - embedding_confidence:float      — quality of visual embedding data
  - confidence:          float
  - rationale:           "heuristic" | "llm"
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.agents.llm.ollama_breaker import create_ollama_client
from app.core.config import settings


# ---------------------------------------------------------------------------
# Modality classification
# ---------------------------------------------------------------------------

# Maps MIME type prefix / suffix → modality label
_MIME_TO_MODALITY: dict[str, str] = {
    "image/png": "screenshot",
    "image/jpeg": "image",
    "image/jpg": "image",
    "image/gif": "image",
    "image/webp": "image",
    "image/svg+xml": "diagram",
    "image/bmp": "image",
    "video/mp4": "video",
    "video/webm": "video",
    "video/avi": "video",
    "video/mov": "video",
    "audio/mpeg": "audio",
    "audio/mp3": "audio",
    "audio/wav": "audio",
    "audio/ogg": "audio",
    "audio/m4a": "audio",
    "application/pdf": "document",
}

# Extension fallback when content_type is absent
_EXT_TO_MODALITY: dict[str, str] = {
    ".png": "screenshot",
    ".jpg": "image",
    ".jpeg": "image",
    ".gif": "image",
    ".webp": "image",
    ".svg": "diagram",
    ".bmp": "image",
    ".mp4": "video",
    ".webm": "video",
    ".avi": "video",
    ".mov": "video",
    ".mp3": "audio",
    ".wav": "audio",
    ".ogg": "audio",
    ".m4a": "audio",
    ".pdf": "document",
    ".docx": "document",
    ".xlsx": "document",
}

# Name hints for diagram vs screenshot classification
_DIAGRAM_HINTS = frozenset({
    "diagram", "flowchart", "chart", "graph", "architecture", "schema",
    "erd", "uml", "workflow", "design", "wireframe", "mockup",
})

_SCREENSHOT_HINTS = frozenset({
    "screenshot", "screen", "capture", "snap", "screengrab",
    "ui", "window", "desktop",
})


def classify_modality(file_name: str, content_type: str | None) -> str:
    """Return a modality label for an attachment based on MIME type and filename."""
    ct = (content_type or "").lower().strip()
    if ct in _MIME_TO_MODALITY:
        modality = _MIME_TO_MODALITY[ct]
    elif ct.startswith("image/"):
        modality = "image"
    elif ct.startswith("video/"):
        modality = "video"
    elif ct.startswith("audio/"):
        modality = "audio"
    else:
        name_lower = (file_name or "").lower()
        ext = ""
        if "." in name_lower:
            ext = "." + name_lower.rsplit(".", 1)[-1]
        modality = _EXT_TO_MODALITY.get(ext, "document")

    # Refine image → diagram or screenshot via filename hints
    if modality in {"image", "screenshot", "diagram"}:
        name_lower = (file_name or "").lower()
        tokens = set(re.findall(r"[a-z]+", name_lower))
        if tokens & _DIAGRAM_HINTS:
            modality = "diagram"
        elif tokens & _SCREENSHOT_HINTS:
            modality = "screenshot"

    return modality


# ---------------------------------------------------------------------------
# Tag extraction
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset({
    "the", "and", "for", "are", "was", "this", "that", "with", "from",
    "have", "been", "not", "but", "will", "can", "all", "one", "has",
    "its", "also", "then", "than", "when", "they", "their", "about",
    "more", "some", "each", "these", "over", "other", "our", "out",
})


def extract_tags(
    *,
    detected_objects: list[str],
    extracted_text: str,
    modalities: list[str],
    dominant_colors: list[str],
) -> list[str]:
    """Build a deduplicated list of normalised searchable tags."""
    tags: list[str] = list(modalities)

    # Objects → normalise and add
    for obj in detected_objects:
        token = re.sub(r"[^a-z0-9_]", "_", str(obj).lower().strip())
        if token and token not in tags:
            tags.append(token)

    # Colour hints
    for colour in dominant_colors:
        token = re.sub(r"[^a-z0-9_]", "_", str(colour).lower().strip())
        if token and token not in tags:
            tags.append(token)

    # Significant words from extracted text
    if extracted_text:
        words = re.findall(r"\b[a-z]{3,}\b", extracted_text.lower())
        for w in words:
            if w not in _STOP_WORDS and w not in tags:
                tags.append(w)
            if len(tags) >= 40:
                break

    return tags[:40]


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def build_visual_summary(
    *,
    primary_modality: str,
    detected_objects: list[str],
    extracted_text: str,
    dominant_colors: list[str],
    content: str,
) -> str:
    """Produce a one-sentence natural language description of the attachment."""
    parts: list[str] = []

    modality_label = primary_modality.replace("_", " ")
    parts.append(f"A {modality_label}")

    if detected_objects:
        obj_str = ", ".join(detected_objects[:5])
        parts.append(f"showing {obj_str}")

    if dominant_colors:
        colour_str = ", ".join(dominant_colors[:3])
        parts.append(f"with dominant colours: {colour_str}")

    if extracted_text:
        snippet = extracted_text.strip()[:120].replace("\n", " ")
        parts.append(f'containing text: "{snippet}"')
    elif content:
        snippet = content.strip()[:80].replace("\n", " ")
        parts.append(f'described as: "{snippet}"')

    return " ".join(parts) + "."


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

def compute_embedding_confidence(visual_embeddings: list[dict[str, Any]]) -> float:
    """Average embedding_confidence across all visual embedding records."""
    scores = [
        float(e.get("embedding_confidence") or 0.0)
        for e in visual_embeddings
        if isinstance(e, dict)
    ]
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 4)


def compute_confidence(
    *,
    attachment_count: int,
    embedding_count: int,
    has_extracted_text: bool,
    has_spatial_data: bool,
) -> float:
    """Overall agent confidence."""
    if attachment_count == 0:
        return 0.30
    score = 0.40
    if embedding_count > 0:
        score += 0.25
    if has_extracted_text:
        score += 0.20
    if has_spatial_data:
        score += 0.10
    if attachment_count >= 2:
        score += 0.05
    return min(round(score, 2), 0.95)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class MultimodalDeepMemoryAgent(BaseAgent):
    """Phase 43: Tag memories with semantic understanding of attachments.

    Classifies modality, extracts objects/colours/text from pre-computed
    visual embeddings and spatial data, and emits searchable_tags that make
    non-text memories retrievable by visual content.
    """

    name = "MultimodalDeepMemoryAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return ["PlaybookAgent"]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}

        pm = outputs.get("primary_modality")
        if not isinstance(pm, str) or not pm:
            raise ValueError("primary_modality must be a non-empty str")

        mp = outputs.get("modalities_present")
        if not isinstance(mp, list):
            raise ValueError("modalities_present must be a list")

        if not isinstance(outputs.get("extracted_text"), str):
            raise ValueError("extracted_text must be a str")

        do = outputs.get("detected_objects")
        if not isinstance(do, list):
            raise ValueError("detected_objects must be a list")

        dc = outputs.get("dominant_colors")
        if not isinstance(dc, list):
            raise ValueError("dominant_colors must be a list")

        st = outputs.get("searchable_tags")
        if not isinstance(st, list):
            raise ValueError("searchable_tags must be a list")

        if not isinstance(outputs.get("visual_summary"), str):
            raise ValueError("visual_summary must be a str")

        ec = outputs.get("embedding_confidence")
        if not isinstance(ec, (int, float)) or not (0.0 <= float(ec) <= 1.0):
            raise ValueError("embedding_confidence must be a float in [0, 1]")

        conf = outputs.get("confidence")
        if not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
            raise ValueError("confidence must be a float in [0, 1]")

    def _heuristic(
        self,
        *,
        content: str,
        attachments: list[dict[str, Any]],
        visual_embeddings: list[dict[str, Any]],
        spatial_data: dict[str, Any],
        audio_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        # ---- Modalities ------------------------------------------------
        modalities: list[str] = []
        for att in attachments:
            m = classify_modality(
                file_name=str(att.get("file_name") or ""),
                content_type=att.get("content_type"),
            )
            if m not in modalities:
                modalities.append(m)

        # Honour explicit modality fields from visual_embeddings
        for ve in visual_embeddings:
            m = str(ve.get("modality") or "").lower()
            if m and m not in modalities:
                modalities.append(m)

        if not modalities:
            modalities = ["document"]

        primary_modality = modalities[0]

        # ---- Extracted text ------------------------------------------------
        text_parts: list[str] = []
        for ve in visual_embeddings:
            et = str(ve.get("extracted_text") or "").strip()
            if et:
                text_parts.append(et)
        if audio_analysis:
            tr = str(audio_analysis.get("transcription") or "").strip()
            if tr:
                text_parts.append(tr)
            summary = str(audio_analysis.get("summary") or "").strip()
            if summary:
                text_parts.append(summary)
        extracted_text = " ".join(text_parts)[:2000]

        # ---- Detected objects ------------------------------------------------
        detected_objects: list[str] = []
        for ve in visual_embeddings:
            objs = ve.get("detected_objects") or []
            if isinstance(objs, list):
                for obj in objs:
                    o = str(obj).strip()
                    if o and o not in detected_objects:
                        detected_objects.append(o)

        # Spatial elements as additional object hints
        if spatial_data:
            elements = spatial_data.get("elements") or []
            if isinstance(elements, list):
                for el in elements[:10]:
                    if isinstance(el, dict):
                        label = str(el.get("type") or el.get("label") or "").strip()
                        if label and label not in detected_objects:
                            detected_objects.append(label)

        detected_objects = detected_objects[:30]

        # ---- Dominant colours ------------------------------------------------
        dominant_colors: list[str] = []
        if spatial_data:
            dc = spatial_data.get("dominant_colors") or []
            if isinstance(dc, list):
                dominant_colors = [str(c) for c in dc[:8]]
        if not dominant_colors:
            for ve in visual_embeddings:
                meta = ve.get("extracted_metadata") or {}
                if isinstance(meta, dict):
                    dc = meta.get("dominant_colors") or []
                    if isinstance(dc, list):
                        dominant_colors = [str(c) for c in dc[:8]]
                    if dominant_colors:
                        break

        # ---- Tags ------------------------------------------------
        searchable_tags = extract_tags(
            detected_objects=detected_objects,
            extracted_text=extracted_text,
            modalities=modalities,
            dominant_colors=dominant_colors,
        )

        # ---- Summary ------------------------------------------------
        visual_summary = build_visual_summary(
            primary_modality=primary_modality,
            detected_objects=detected_objects,
            extracted_text=extracted_text,
            dominant_colors=dominant_colors,
            content=content,
        )

        # ---- Confidence ------------------------------------------------
        embedding_confidence = compute_embedding_confidence(visual_embeddings)
        confidence = compute_confidence(
            attachment_count=len(attachments),
            embedding_count=len(visual_embeddings),
            has_extracted_text=bool(extracted_text),
            has_spatial_data=bool(spatial_data),
        )

        return {
            "primary_modality": primary_modality,
            "modalities_present": modalities,
            "extracted_text": extracted_text,
            "detected_objects": detected_objects,
            "dominant_colors": dominant_colors,
            "searchable_tags": searchable_tags,
            "visual_summary": visual_summary,
            "embedding_confidence": embedding_confidence,
            "confidence": confidence,
            "rationale": "heuristic",
        }

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        memory = context.get("memory") or {}
        enrichment = memory.get("enrichment") or {}

        content = str(memory.get("content") or "")
        attachments = list(enrichment.get("attachments") or [])
        visual_embeddings = list(enrichment.get("visual_embeddings") or [])
        spatial_data = dict(enrichment.get("spatial_data") or {})
        audio_analysis = dict(enrichment.get("audio_analysis") or {})

        strategy = getattr(settings, "MULTIMODAL_DEEP_MEMORY_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        outputs: dict[str, Any]

        if strategy == "heuristic" or (not attachments and not visual_embeddings):
            outputs = self._heuristic(
                content=content,
                attachments=attachments,
                visual_embeddings=visual_embeddings,
                spatial_data=spatial_data,
                audio_analysis=audio_analysis,
            )
        else:
            prompt = (
                "You are a multimodal memory indexing engine for an enterprise Cognitive OS. "
                "Output JSON only. Do not hallucinate.\n\n"
                "Analyse the attachment metadata and any pre-extracted visual data. "
                "Return a semantic index with the following fields:\n"
                "- primary_modality: str — dominant modality (image/screenshot/diagram/audio/video/document)\n"
                "- modalities_present: list[str] — all distinct modalities\n"
                "- extracted_text: str — OCR / transcription text combined\n"
                "- detected_objects: list[str] — objects or UI elements identified\n"
                "- dominant_colors: list[str] — colour palette\n"
                "- searchable_tags: list[str] — normalised retrieval tags\n"
                "- visual_summary: str — one-sentence description\n"
                "- embedding_confidence: float 0..1\n"
                "- confidence: float 0..1\n"
                "- rationale: brief explanation\n\n"
                f"CONTENT: {content[:300]}\n"
                f"ATTACHMENTS: {attachments[:5]}\n"
                f"VISUAL_EMBEDDINGS: {visual_embeddings[:3]}\n"
                f"SPATIAL_DATA: {spatial_data}\n"
                f"AUDIO_ANALYSIS: {audio_analysis}\n"
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
                and isinstance(resp.get("primary_modality"), str)
                and isinstance(resp.get("modalities_present"), list)
                and isinstance(resp.get("searchable_tags"), list)
            ):
                outputs = resp
            else:
                outputs = self._heuristic(
                    content=content,
                    attachments=attachments,
                    visual_embeddings=visual_embeddings,
                    spatial_data=spatial_data,
                    audio_analysis=audio_analysis,
                )

        finished_at = datetime.now(timezone.utc)

        result = AgentResult(
            agent_name=self.name,
            agent_version=self.version,
            memory_id=memory_id,
            status="success",
            confidence=float(outputs.get("confidence", 0.50)),
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=started_at,
            finished_at=finished_at,
            trace_id=str(trace_id) if trace_id else None,
        )
        self.validate_outputs(result)
        return result
