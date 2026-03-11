"""Tests for MultimodalDeepMemoryAgent — Phase 43."""

from __future__ import annotations

import pytest

from app.agents.multimodal_deep_memory_agent import (
    MultimodalDeepMemoryAgent,
    classify_modality,
    extract_tags,
    build_visual_summary,
    compute_embedding_confidence,
    compute_confidence,
)
from app.agents.types import AgentContext


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

AGENT = MultimodalDeepMemoryAgent()


def _ctx(
    content: str = "",
    attachments: list | None = None,
    visual_embeddings: list | None = None,
    spatial_data: dict | None = None,
    audio_analysis: dict | None = None,
) -> AgentContext:
    return {
        "memory": {
            "content": content,
            "enrichment": {
                "attachments": attachments or [],
                "visual_embeddings": visual_embeddings or [],
                "spatial_data": spatial_data or {},
                "audio_analysis": audio_analysis or {},
            },
        },
        "runtime": {"job_id": "trace-43"},
    }


ATT_PNG = {"id": "att-1", "file_name": "screenshot.png", "content_type": "image/png", "size_bytes": 50000}
ATT_JPG = {"id": "att-2", "file_name": "photo.jpg", "content_type": "image/jpeg", "size_bytes": 120000}
ATT_SVG = {"id": "att-3", "file_name": "architecture_diagram.svg", "content_type": "image/svg+xml", "size_bytes": 8000}
ATT_MP4 = {"id": "att-4", "file_name": "walkthrough.mp4", "content_type": "video/mp4", "size_bytes": 4000000}
ATT_MP3 = {"id": "att-5", "file_name": "meeting.mp3", "content_type": "audio/mpeg", "size_bytes": 300000}
ATT_PDF = {"id": "att-6", "file_name": "report.pdf", "content_type": "application/pdf", "size_bytes": 200000}
ATT_NOCT = {"id": "att-7", "file_name": "capture.png", "content_type": None, "size_bytes": 40000}

VE_IMAGE = {
    "modality": "image",
    "extracted_text": "Click the submit button to continue",
    "detected_objects": ["button", "form", "text_field"],
    "object_confidence_scores": {"button": 0.9, "form": 0.8},
    "embedding_confidence": 0.85,
}

VE_SCREENSHOT = {
    "modality": "screenshot",
    "extracted_text": "Error: connection refused on port 5432",
    "detected_objects": ["error_message", "dialog", "retry_button"],
    "embedding_confidence": 0.72,
}

SPATIAL = {
    "elements": [
        {"type": "BUTTON", "label": "Submit"},
        {"type": "ERROR_MESSAGE", "label": "Connection error"},
    ],
    "layout_description": "Modal dialog with error and retry option",
    "dominant_colors": ["red", "white", "grey"],
}

AUDIO = {
    "transcription": "Today we discuss the deployment pipeline and rollback strategy",
    "emotion_detected": "neutral",
    "summary": "Discussion of deployment and rollback procedures",
}


# ---------------------------------------------------------------------------
# classify_modality
# ---------------------------------------------------------------------------

def test_classify_png_screenshot():
    assert classify_modality("screenshot.png", "image/png") == "screenshot"


def test_classify_jpg_image():
    assert classify_modality("photo.jpg", "image/jpeg") == "image"


def test_classify_svg_diagram():
    assert classify_modality("architecture.svg", "image/svg+xml") == "diagram"


def test_classify_mp4_video():
    assert classify_modality("video.mp4", "video/mp4") == "video"


def test_classify_mp3_audio():
    assert classify_modality("audio.mp3", "audio/mpeg") == "audio"


def test_classify_pdf_document():
    assert classify_modality("doc.pdf", "application/pdf") == "document"


def test_classify_diagram_by_filename_hint():
    assert classify_modality("architecture_diagram.png", "image/png") == "diagram"


def test_classify_screenshot_by_filename_hint():
    assert classify_modality("screen_capture.jpg", "image/jpeg") == "screenshot"


def test_classify_no_content_type_uses_extension():
    assert classify_modality("image.png", None) == "screenshot"


def test_classify_no_content_type_mp4():
    assert classify_modality("video.mp4", None) == "video"


def test_classify_unknown_falls_back_to_document():
    assert classify_modality("data.xyz", None) == "document"


def test_classify_unknown_mime_falls_back_to_document():
    assert classify_modality("data.bin", "application/octet-stream") == "document"


def test_classify_flowchart_hint():
    assert classify_modality("flowchart_v2.png", "image/png") == "diagram"


def test_classify_wireframe_hint():
    assert classify_modality("wireframe_design.jpg", "image/jpeg") == "diagram"


# ---------------------------------------------------------------------------
# extract_tags
# ---------------------------------------------------------------------------

def test_extract_tags_includes_modalities():
    tags = extract_tags(
        detected_objects=[],
        extracted_text="",
        modalities=["screenshot"],
        dominant_colors=[],
    )
    assert "screenshot" in tags


def test_extract_tags_includes_detected_objects():
    tags = extract_tags(
        detected_objects=["button", "form"],
        extracted_text="",
        modalities=["screenshot"],
        dominant_colors=[],
    )
    assert "button" in tags
    assert "form" in tags


def test_extract_tags_includes_colors():
    tags = extract_tags(
        detected_objects=[],
        extracted_text="",
        modalities=["image"],
        dominant_colors=["red", "white"],
    )
    assert "red" in tags
    assert "white" in tags


def test_extract_tags_from_text():
    tags = extract_tags(
        detected_objects=[],
        extracted_text="connection refused database error",
        modalities=["screenshot"],
        dominant_colors=[],
    )
    assert "connection" in tags or "database" in tags or "error" in tags


def test_extract_tags_deduplicates():
    tags = extract_tags(
        detected_objects=["button", "button"],
        extracted_text="",
        modalities=["screenshot"],
        dominant_colors=[],
    )
    assert tags.count("button") == 1


def test_extract_tags_max_40():
    tags = extract_tags(
        detected_objects=[f"obj_{i}" for i in range(50)],
        extracted_text="lots of text " * 20,
        modalities=["image"],
        dominant_colors=["red", "blue", "green"],
    )
    assert len(tags) <= 40


def test_extract_tags_empty_inputs():
    tags = extract_tags(detected_objects=[], extracted_text="", modalities=[], dominant_colors=[])
    assert isinstance(tags, list)


# ---------------------------------------------------------------------------
# build_visual_summary
# ---------------------------------------------------------------------------

def test_summary_mentions_modality():
    summary = build_visual_summary(
        primary_modality="screenshot",
        detected_objects=[],
        extracted_text="",
        dominant_colors=[],
        content="",
    )
    assert "screenshot" in summary.lower()


def test_summary_mentions_objects():
    summary = build_visual_summary(
        primary_modality="image",
        detected_objects=["button", "form"],
        extracted_text="",
        dominant_colors=[],
        content="",
    )
    assert "button" in summary or "form" in summary


def test_summary_mentions_colors():
    summary = build_visual_summary(
        primary_modality="screenshot",
        detected_objects=[],
        extracted_text="",
        dominant_colors=["red", "white"],
        content="",
    )
    assert "red" in summary


def test_summary_includes_text_snippet():
    summary = build_visual_summary(
        primary_modality="screenshot",
        detected_objects=[],
        extracted_text="Error: connection refused",
        dominant_colors=[],
        content="",
    )
    assert "connection" in summary or "Error" in summary


def test_summary_falls_back_to_content():
    summary = build_visual_summary(
        primary_modality="document",
        detected_objects=[],
        extracted_text="",
        dominant_colors=[],
        content="quarterly report results",
    )
    assert "quarterly" in summary or "report" in summary


def test_summary_ends_with_period():
    summary = build_visual_summary(
        primary_modality="video",
        detected_objects=[],
        extracted_text="",
        dominant_colors=[],
        content="",
    )
    assert summary.endswith(".")


# ---------------------------------------------------------------------------
# compute_embedding_confidence
# ---------------------------------------------------------------------------

def test_embedding_confidence_average():
    ves = [{"embedding_confidence": 0.8}, {"embedding_confidence": 0.6}]
    conf = compute_embedding_confidence(ves)
    assert conf == pytest.approx(0.7, abs=0.01)


def test_embedding_confidence_empty():
    assert compute_embedding_confidence([]) == 0.0


def test_embedding_confidence_single():
    assert compute_embedding_confidence([{"embedding_confidence": 0.9}]) == pytest.approx(0.9)


def test_embedding_confidence_missing_field():
    ves = [{"modality": "image"}]  # no embedding_confidence
    conf = compute_embedding_confidence(ves)
    assert conf == 0.0


# ---------------------------------------------------------------------------
# compute_confidence
# ---------------------------------------------------------------------------

def test_confidence_no_attachments():
    conf = compute_confidence(
        attachment_count=0, embedding_count=0,
        has_extracted_text=False, has_spatial_data=False
    )
    assert conf == 0.30


def test_confidence_with_all_signals():
    conf = compute_confidence(
        attachment_count=2, embedding_count=2,
        has_extracted_text=True, has_spatial_data=True
    )
    assert conf >= 0.90


def test_confidence_partial_signals():
    conf = compute_confidence(
        attachment_count=1, embedding_count=1,
        has_extracted_text=False, has_spatial_data=False
    )
    assert 0.60 <= conf <= 0.80


def test_confidence_capped_at_095():
    conf = compute_confidence(
        attachment_count=10, embedding_count=10,
        has_extracted_text=True, has_spatial_data=True
    )
    assert conf <= 0.95


# ---------------------------------------------------------------------------
# Agent.validate_outputs
# ---------------------------------------------------------------------------

def _good_outputs() -> dict:
    return {
        "primary_modality": "screenshot",
        "modalities_present": ["screenshot"],
        "extracted_text": "Submit button clicked",
        "detected_objects": ["button"],
        "dominant_colors": ["blue"],
        "searchable_tags": ["screenshot", "button"],
        "visual_summary": "A screenshot showing a button.",
        "embedding_confidence": 0.8,
        "confidence": 0.75,
        "rationale": "heuristic",
    }


def _make_result(outputs: dict):
    from app.agents.types import AgentResult
    from datetime import datetime, timezone
    return AgentResult(
        agent_name="MultimodalDeepMemoryAgent",
        agent_version="v1",
        memory_id="m1",
        status="success",
        confidence=float(outputs.get("confidence", 0.7)),
        outputs=outputs,
        warnings=[],
        errors=[],
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        trace_id=None,
    )


def test_validate_good():
    AGENT.validate_outputs(_make_result(_good_outputs()))


def test_validate_missing_primary_modality():
    outputs = _good_outputs()
    outputs["primary_modality"] = ""
    with pytest.raises(ValueError, match="primary_modality"):
        AGENT.validate_outputs(_make_result(outputs))


def test_validate_bad_modalities_present():
    outputs = _good_outputs()
    outputs["modalities_present"] = "screenshot"
    with pytest.raises(ValueError, match="modalities_present"):
        AGENT.validate_outputs(_make_result(outputs))


def test_validate_bad_detected_objects():
    outputs = _good_outputs()
    outputs["detected_objects"] = "button"
    with pytest.raises(ValueError, match="detected_objects"):
        AGENT.validate_outputs(_make_result(outputs))


def test_validate_bad_searchable_tags():
    outputs = _good_outputs()
    outputs["searchable_tags"] = None
    with pytest.raises(ValueError, match="searchable_tags"):
        AGENT.validate_outputs(_make_result(outputs))


def test_validate_bad_embedding_confidence():
    outputs = _good_outputs()
    outputs["embedding_confidence"] = 1.5
    with pytest.raises(ValueError, match="embedding_confidence"):
        AGENT.validate_outputs(_make_result(outputs))


def test_validate_bad_confidence():
    from app.agents.types import AgentResult
    from datetime import datetime, timezone
    outputs = _good_outputs()
    outputs["confidence"] = 2.0
    result = AgentResult(
        agent_name="MultimodalDeepMemoryAgent",
        agent_version="v1",
        memory_id="m1",
        status="success",
        confidence=0.7,
        outputs=outputs,
        warnings=[],
        errors=[],
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        trace_id=None,
    )
    with pytest.raises(ValueError, match="confidence"):
        AGENT.validate_outputs(result)


def test_validate_skipped_on_non_success():
    from app.agents.types import AgentResult
    from datetime import datetime, timezone
    result = AgentResult(
        agent_name="MultimodalDeepMemoryAgent",
        agent_version="v1",
        memory_id="m1",
        status="failed",
        confidence=0.0,
        outputs={},
        warnings=[],
        errors=["fail"],
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        trace_id=None,
    )
    AGENT.validate_outputs(result)  # should not raise


# ---------------------------------------------------------------------------
# Agent.run — heuristic path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_single_png_attachment():
    result = await AGENT.run("m1", _ctx(attachments=[ATT_PNG]))
    assert result.status == "success"
    assert result.outputs["primary_modality"] == "screenshot"
    assert "screenshot" in result.outputs["modalities_present"]


@pytest.mark.asyncio
async def test_run_jpg_attachment():
    result = await AGENT.run("m2", _ctx(attachments=[ATT_JPG]))
    assert result.outputs["primary_modality"] == "image"


@pytest.mark.asyncio
async def test_run_svg_diagram():
    result = await AGENT.run("m3", _ctx(attachments=[ATT_SVG]))
    assert result.outputs["primary_modality"] == "diagram"


@pytest.mark.asyncio
async def test_run_video_attachment():
    result = await AGENT.run("m4", _ctx(attachments=[ATT_MP4]))
    assert result.outputs["primary_modality"] == "video"


@pytest.mark.asyncio
async def test_run_audio_attachment():
    result = await AGENT.run("m5", _ctx(attachments=[ATT_MP3]))
    assert result.outputs["primary_modality"] == "audio"


@pytest.mark.asyncio
async def test_run_pdf_attachment():
    result = await AGENT.run("m6", _ctx(attachments=[ATT_PDF]))
    assert result.outputs["primary_modality"] == "document"


@pytest.mark.asyncio
async def test_run_with_visual_embeddings():
    result = await AGENT.run("m7", _ctx(
        attachments=[ATT_PNG],
        visual_embeddings=[VE_IMAGE],
    ))
    assert "button" in result.outputs["detected_objects"]
    assert result.outputs["extracted_text"] != ""
    assert result.outputs["embedding_confidence"] == pytest.approx(0.85)


@pytest.mark.asyncio
async def test_run_extracted_text_from_ve():
    result = await AGENT.run("m8", _ctx(
        attachments=[ATT_PNG],
        visual_embeddings=[VE_SCREENSHOT],
    ))
    assert "connection" in result.outputs["extracted_text"].lower()


@pytest.mark.asyncio
async def test_run_spatial_data_adds_colors():
    result = await AGENT.run("m9", _ctx(
        attachments=[ATT_PNG],
        spatial_data=SPATIAL,
    ))
    assert "red" in result.outputs["dominant_colors"]


@pytest.mark.asyncio
async def test_run_spatial_elements_in_objects():
    result = await AGENT.run("m10", _ctx(
        attachments=[ATT_PNG],
        spatial_data=SPATIAL,
    ))
    # BUTTON or ERROR_MESSAGE from spatial elements
    objects_lower = [o.lower() for o in result.outputs["detected_objects"]]
    assert any("button" in o or "error" in o for o in objects_lower)


@pytest.mark.asyncio
async def test_run_audio_transcription():
    result = await AGENT.run("m11", _ctx(
        attachments=[ATT_MP3],
        audio_analysis=AUDIO,
    ))
    assert "deployment" in result.outputs["extracted_text"].lower()


@pytest.mark.asyncio
async def test_run_searchable_tags_not_empty():
    result = await AGENT.run("m12", _ctx(
        attachments=[ATT_PNG],
        visual_embeddings=[VE_IMAGE],
    ))
    assert len(result.outputs["searchable_tags"]) > 0


@pytest.mark.asyncio
async def test_run_visual_summary_is_string():
    result = await AGENT.run("m13", _ctx(attachments=[ATT_PNG]))
    assert isinstance(result.outputs["visual_summary"], str)
    assert len(result.outputs["visual_summary"]) > 0


@pytest.mark.asyncio
async def test_run_multiple_attachments_multiple_modalities():
    result = await AGENT.run("m14", _ctx(attachments=[ATT_PNG, ATT_MP3]))
    assert len(result.outputs["modalities_present"]) >= 2


@pytest.mark.asyncio
async def test_run_no_attachments_fallback():
    result = await AGENT.run("m15", _ctx())
    assert result.status == "success"
    assert result.outputs["primary_modality"] == "document"
    assert result.outputs["confidence"] == 0.30


@pytest.mark.asyncio
async def test_run_no_content_type_attachment():
    result = await AGENT.run("m16", _ctx(attachments=[ATT_NOCT]))
    assert result.status == "success"
    assert result.outputs["primary_modality"] in {"screenshot", "image", "document"}


@pytest.mark.asyncio
async def test_run_confidence_in_range():
    result = await AGENT.run("m17", _ctx(
        attachments=[ATT_PNG],
        visual_embeddings=[VE_IMAGE],
        spatial_data=SPATIAL,
    ))
    assert 0.0 <= result.outputs["confidence"] <= 1.0


@pytest.mark.asyncio
async def test_run_embedding_confidence_zero_without_ve():
    result = await AGENT.run("m18", _ctx(attachments=[ATT_PNG]))
    assert result.outputs["embedding_confidence"] == 0.0


@pytest.mark.asyncio
async def test_run_rationale_heuristic():
    result = await AGENT.run("m19", _ctx(attachments=[ATT_JPG]))
    assert result.outputs["rationale"] == "heuristic"


@pytest.mark.asyncio
async def test_run_trace_id_propagated():
    result = await AGENT.run("m20", _ctx(attachments=[ATT_PNG]))
    assert result.trace_id == "trace-43"


@pytest.mark.asyncio
async def test_run_all_output_keys_present():
    result = await AGENT.run("m21", _ctx(attachments=[ATT_PNG]))
    required = {
        "primary_modality", "modalities_present", "extracted_text",
        "detected_objects", "dominant_colors", "searchable_tags",
        "visual_summary", "embedding_confidence", "confidence", "rationale",
    }
    assert required.issubset(result.outputs.keys())


@pytest.mark.asyncio
async def test_run_combined_ve_and_audio():
    result = await AGENT.run("m22", _ctx(
        content="product demo walkthrough",
        attachments=[ATT_MP4],
        visual_embeddings=[VE_IMAGE],
        audio_analysis=AUDIO,
    ))
    assert result.status == "success"
    assert len(result.outputs["searchable_tags"]) > 2


@pytest.mark.asyncio
async def test_run_tags_contain_screenshot_for_png():
    result = await AGENT.run("m23", _ctx(attachments=[ATT_PNG]))
    assert "screenshot" in result.outputs["searchable_tags"]


@pytest.mark.asyncio
async def test_run_content_used_in_summary_when_no_extracted_text():
    result = await AGENT.run(
        "m24",
        _ctx(content="architecture overview diagram", attachments=[ATT_SVG]),
    )
    summary = result.outputs["visual_summary"].lower()
    assert "diagram" in summary or "architecture" in summary


# ---------------------------------------------------------------------------
# Agent metadata
# ---------------------------------------------------------------------------

def test_agent_name():
    assert AGENT.name == "MultimodalDeepMemoryAgent"


def test_agent_version():
    assert AGENT.version == "v1"


def test_agent_dependencies():
    assert "PlaybookAgent" in AGENT.dependencies()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_lookup():
    from app.agents.registry import get_agent
    agent = get_agent("multimodal_deep_memory")
    assert isinstance(agent, MultimodalDeepMemoryAgent)


def test_registry_camel():
    from app.agents.registry import get_agent
    agent = get_agent("multimodaldeepmemoryyagent")
    assert agent is None  # typo → miss


def test_registry_alias():
    from app.agents.registry import get_agent
    agent = get_agent("multimodal_memory")
    assert isinstance(agent, MultimodalDeepMemoryAgent)


def test_registry_miss():
    from app.agents.registry import get_agent
    assert get_agent("not_a_real_agent_43") is None
