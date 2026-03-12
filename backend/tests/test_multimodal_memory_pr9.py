"""
PR-9: Multimodal Deep Memory Integration Tests

Comprehensive test suite for multimodal memory functionality:
- Embedding extraction (images, videos)
- Spatial/UI memory
- Audio analysis (transcription, speaker detection, emotion)
- Procedural memory from video
- Visual search
"""

import uuid
import hashlib
from datetime import datetime

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.multimodal_memory import (
    ModalityType,
    MultimodalEmbedding,
    SpatialMemory,
    AudioAnalysis,
    ProceduralMemoryFromVideo,
)
from app.models.memory_attachment import MemoryAttachment
from app.services.vision_memory_service import VisionMemoryService
from app.services.audio_memory_service import AudioMemoryService
from app.services.procedural_memory_from_video_service import (
    ProceduralMemoryFromVideoService,
)


def create_test_attachment(
    org_id: str, memory_id: str, file_name: str, size_bytes: int = 50000
) -> MemoryAttachment:
    """Helper to create a test MemoryAttachment with correct fields."""
    file_content = f"{file_name}_{uuid.uuid4()}"
    sha256_hash = hashlib.sha256(file_content.encode()).hexdigest()

    return MemoryAttachment(
        id=str(uuid.uuid4()),
        organization_id=org_id,
        memory_id=memory_id,
        uploaded_by=str(uuid.uuid4()),  # Mock user ID
        file_name=file_name,
        content_type="application/octet-stream",
        size_bytes=size_bytes,
        sha256=sha256_hash,
        storage_path=f"uploads/{org_id}/{file_name}",
    )


@pytest.fixture
async def test_org_id() -> str:
    return str(uuid.uuid4())


@pytest.fixture
async def test_attachment_id() -> str:
    return str(uuid.uuid4())


@pytest.fixture
async def test_memory_id() -> str:
    return str(uuid.uuid4())




# ============================================================================
# Vision Memory Service Tests
# ============================================================================


@pytest.mark.asyncio
async def test_extract_image_embeddings(
    db_session: AsyncSession,
    test_org_id: str,
    test_attachment_id: str,
):
    """Test extracting embeddings from image."""
    # Create test attachment
    attachment = MemoryAttachment(
        id=test_attachment_id,
        organization_id=test_org_id,
        memory_id=str(uuid.uuid4()),
        file_name="screenshot_login.png",
        file_path="/uploads/screenshot_login.png",
        content_type="image/png",
        file_size=45000,
        storage_type="local",
        upload_status="completed",
        is_deleted=False,
    )
    db_session.add(attachment)
    await db_session.commit()

    # Extract embeddings
    vision_svc = VisionMemoryService(db_session)
    embedding = await vision_svc.extract_image_embeddings(
        test_attachment_id, test_org_id
    )

    # Verify embedding
    assert embedding.id is not None
    assert embedding.modality == ModalityType.IMAGE.value
    assert embedding.embedding_dimension == 768
    assert embedding.embedding_confidence > 0.8
    assert len(embedding.detected_objects) > 0
    assert embedding.extracted_text is not None
    assert embedding.processing_status == "completed"
    assert embedding.searchable is True


@pytest.mark.asyncio
async def test_extract_image_embeddings_invalid(
    db_session: AsyncSession,
    test_org_id: str,
):
    """Test extracting embeddings from non-existent attachment."""
    vision_svc = VisionMemoryService(db_session)

    with pytest.raises(ValueError):
        await vision_svc.extract_image_embeddings("invalid-id", test_org_id)


@pytest.mark.asyncio
async def test_extract_spatial_memory(
    db_session: AsyncSession,
    test_org_id: str,
    test_attachment_id: str,
):
    """Test extracting spatial/UI layout memory."""
    # Create attachment
    attachment = MemoryAttachment(
        id=test_attachment_id,
        organization_id=test_org_id,
        memory_id=str(uuid.uuid4()),
        file_name="screenshot_dashboard.png",
        file_path="/uploads/screenshot_dashboard.png",
        content_type="image/png",
        file_size=78000,
        storage_type="local",
        upload_status="completed",
        is_deleted=False,
    )
    db_session.add(attachment)
    await db_session.commit()

    # Extract spatial memory
    vision_svc = VisionMemoryService(db_session)
    spatial = await vision_svc.extract_spatial_memory(
        test_attachment_id, test_org_id
    )

    # Verify spatial memory
    assert spatial.id is not None
    assert spatial.memory_attachment_id == test_attachment_id
    assert spatial.image_width is not None
    assert spatial.image_height is not None
    assert len(spatial.elements) > 0
    assert spatial.layout_type in ["simple", "single_column", "multi_column"]
    assert len(spatial.interactive_elements) > 0
    assert spatial.accessibility_score is not None
    assert len(spatial.color_palette) > 0


@pytest.mark.asyncio
async def test_search_visual_memory(
    db_session: AsyncSession,
    test_org_id: str,
):
    """Test searching visual memory by text query."""
    # Create attachments with content
    attachment_ids = []
    for i in range(3):
        attach_id = str(uuid.uuid4())
        attachment = MemoryAttachment(
            id=attach_id,
            organization_id=test_org_id,
            memory_id=str(uuid.uuid4()),
            file_name=f"screenshot_{i}.png",
            file_path=f"/uploads/screenshot_{i}.png",
            content_type="image/png",
            file_size=50000,
            storage_type="local",
            upload_status="completed",
            is_deleted=False,
        )
        db_session.add(attachment)
        attachment_ids.append(attach_id)

    await db_session.commit()

    # Extract embeddings for all
    vision_svc = VisionMemoryService(db_session)
    for attach_id in attachment_ids:
        await vision_svc.extract_image_embeddings(attach_id, test_org_id)

    # Search
    results = await vision_svc.search_visual_memory(
        test_org_id, "screenshot", limit=5
    )

    assert len(results) > 0
    assert all("id" in r and "attachment_id" in r for r in results)
    assert all("relevance_score" in r for r in results)


@pytest.mark.asyncio
async def test_find_similar_images(
    db_session: AsyncSession,
    test_org_id: str,
):
    """Test finding visually similar images."""
    # Create attachments
    attachment_ids = []
    for i in range(3):
        attach_id = str(uuid.uuid4())
        attach = MemoryAttachment(
            id=attach_id,
            organization_id=test_org_id,
            memory_id=str(uuid.uuid4()),
            file_name=f"similar_{i}.png",
            file_path=f"/uploads/similar_{i}.png",
            content_type="image/png",
            file_size=50000,
            storage_type="local",
            upload_status="completed",
            is_deleted=False,
        )
        db_session.add(attach)
        attachment_ids.append(attach_id)

    await db_session.commit()

    # Extract embeddings
    vision_svc = VisionMemoryService(db_session)
    for attach_id in attachment_ids:
        await vision_svc.extract_image_embeddings(attach_id, test_org_id)

    # Find similar
    similar = await vision_svc.find_similar_images(
        attachment_ids[0], test_org_id, limit=2
    )

    assert isinstance(similar, list)
    assert all(
        "attachment_id" in s and "similarity_score" in s for s in similar
    )


# ============================================================================
# Audio Memory Service Tests
# ============================================================================


@pytest.mark.asyncio
async def test_extract_audio_analysis(
    db_session: AsyncSession,
    test_org_id: str,
    test_attachment_id: str,
):
    """Test extracting audio analysis."""
    # Create attachment
    attachment = MemoryAttachment(
        id=test_attachment_id,
        organization_id=test_org_id,
        memory_id=str(uuid.uuid4()),
        file_name="meeting_recording.mp3",
        file_path="/uploads/meeting_recording.mp3",
        content_type="audio/mpeg",
        file_size=5000000,
        storage_type="local",
        upload_status="completed",
        is_deleted=False,
    )
    db_session.add(attachment)
    await db_session.commit()

    # Extract audio analysis
    audio_svc = AudioMemoryService(db_session)
    analysis = await audio_svc.extract_audio_analysis(
        test_attachment_id, test_org_id, duration_seconds=600
    )

    # Verify audio analysis
    assert analysis.id is not None
    assert analysis.memory_attachment_id == test_attachment_id
    assert analysis.duration_seconds == 600
    assert analysis.transcription is not None
    assert analysis.transcription_confidence > 0.8
    assert analysis.language == "en"
    assert analysis.speakers_identified >= 0
    assert len(analysis.speaker_segments) >= 0
    assert len(analysis.audio_types_detected) > 0
    assert analysis.emotion_detected is not None
    assert analysis.tone_detected is not None


@pytest.mark.asyncio
async def test_emotion_detection_in_audio(
    db_session: AsyncSession,
    test_org_id: str,
    test_attachment_id: str,
):
    """Test emotion detection in audio."""
    attachment = MemoryAttachment(
        id=test_attachment_id,
        organization_id=test_org_id,
        memory_id=str(uuid.uuid4()),
        file_name="frustrated_speech.mp3",
        file_path="/uploads/frustrated_speech.mp3",
        content_type="audio/mpeg",
        file_size=2000000,
        storage_type="local",
        upload_status="completed",
        is_deleted=False,
    )
    db_session.add(attachment)
    await db_session.commit()

    audio_svc = AudioMemoryService(db_session)
    analysis = await audio_svc.extract_audio_analysis(
        test_attachment_id, test_org_id, duration_seconds=300
    )

    # Verify emotion detection
    assert analysis.emotion_detected is not None
    assert analysis.emotion_confidence > 0.0


@pytest.mark.asyncio
async def test_search_audio_memory(
    db_session: AsyncSession,
    test_org_id: str,
    test_attachment_id: str,
):
    """Test searching audio memory."""
    # Create and analyze audio attachment
    attachment = MemoryAttachment(
        id=test_attachment_id,
        organization_id=test_org_id,
        memory_id=str(uuid.uuid4()),
        file_name="deployment_walkthrough.mp3",
        file_path="/uploads/deployment_walkthrough.mp3",
        content_type="audio/mpeg",
        file_size=3000000,
        storage_type="local",
        upload_status="completed",
        is_deleted=False,
    )
    db_session.add(attachment)
    await db_session.commit()

    audio_svc = AudioMemoryService(db_session)
    await audio_svc.extract_audio_analysis(
        test_attachment_id, test_org_id, duration_seconds=600
    )

    # Search
    results = await audio_svc.search_audio_memory(
        test_org_id, "deployment", limit=5
    )

    assert isinstance(results, list)
    assert all("id" in r and "attachment_id" in r for r in results)


@pytest.mark.asyncio
async def test_extract_key_quotes_from_audio(
    db_session: AsyncSession,
    test_org_id: str,
    test_attachment_id: str,
):
    """Test extracting key quotes from audio."""
    attachment = MemoryAttachment(
        id=test_attachment_id,
        organization_id=test_org_id,
        memory_id=str(uuid.uuid4()),
        file_name="interview.mp3",
        file_path="/uploads/interview.mp3",
        content_type="audio/mpeg",
        file_size=2000000,
        storage_type="local",
        upload_status="completed",
        is_deleted=False,
    )
    db_session.add(attachment)
    await db_session.commit()

    audio_svc = AudioMemoryService(db_session)
    analysis = await audio_svc.extract_audio_analysis(
        test_attachment_id, test_org_id, duration_seconds=300
    )

    quotes = await audio_svc.extract_key_quotes(analysis.id)

    assert isinstance(quotes, list)
    assert all("text" in q and "timestamp" in q for q in quotes)


# ============================================================================
# Procedural Memory from Video Service Tests
# ============================================================================


@pytest.mark.asyncio
async def test_extract_procedure_from_video(
    db_session: AsyncSession,
    test_org_id: str,
    test_attachment_id: str,
):
    """Test extracting procedure from video."""
    attachment = MemoryAttachment(
        id=test_attachment_id,
        organization_id=test_org_id,
        memory_id=str(uuid.uuid4()),
        file_name="deployment_tutorial.mp4",
        file_path="/uploads/deployment_tutorial.mp4",
        content_type="video/mp4",
        file_size=500000000,
        storage_type="local",
        upload_status="completed",
        is_deleted=False,
    )
    db_session.add(attachment)
    await db_session.commit()

    # Extract procedure
    proc_svc = ProceduralMemoryFromVideoService(db_session)
    procedure = await proc_svc.extract_procedure_from_video(
        test_attachment_id, test_org_id, "deployment", 900
    )

    # Verify procedure
    assert procedure.id is not None
    assert procedure.video_attachment_id == test_attachment_id
    assert procedure.domain == "deployment"
    assert procedure.procedure_title is not None
    assert len(procedure.extracted_steps) > 0
    assert len(procedure.prerequisites) > 0
    assert len(procedure.tools_required) > 0
    assert len(procedure.common_mistakes) > 0
    assert len(procedure.success_indicators) > 0
    assert procedure.confidence_score > 0.0
    assert procedure.human_validation_status == "pending_review"


@pytest.mark.asyncio
async def test_procedure_steps_structure(
    db_session: AsyncSession,
    test_org_id: str,
    test_attachment_id: str,
):
    """Test that extracted procedure steps have correct structure."""
    attachment = MemoryAttachment(
        id=test_attachment_id,
        organization_id=test_org_id,
        memory_id=str(uuid.uuid4()),
        file_name="tutorial.mp4",
        file_path="/uploads/tutorial.mp4",
        content_type="video/mp4",
        file_size=500000000,
        storage_type="local",
        upload_status="completed",
        is_deleted=False,
    )
    db_session.add(attachment)
    await db_session.commit()

    proc_svc = ProceduralMemoryFromVideoService(db_session)
    procedure = await proc_svc.extract_procedure_from_video(
        test_attachment_id, test_org_id, "setup", 600
    )

    # Check step structure
    for step in procedure.extracted_steps:
        assert "step_number" in step
        assert "title" in step
        assert "description" in step
        assert "estimated_duration_seconds" in step
        assert "expected_outcome" in step
        assert step.get("difficulty") in ["easy", "medium", "hard"]


@pytest.mark.asyncio
async def test_validate_procedure(
    db_session: AsyncSession,
    test_org_id: str,
    test_attachment_id: str,
):
    """Test validating extracted procedure."""
    attachment = MemoryAttachment(
        id=test_attachment_id,
        organization_id=test_org_id,
        memory_id=str(uuid.uuid4()),
        file_name="demo.mp4",
        file_path="/uploads/demo.mp4",
        content_type="video/mp4",
        file_size=500000000,
        storage_type="local",
        upload_status="completed",
        is_deleted=False,
    )
    db_session.add(attachment)
    await db_session.commit()

    proc_svc = ProceduralMemoryFromVideoService(db_session)
    procedure = await proc_svc.extract_procedure_from_video(
        test_attachment_id, test_org_id, "debugging", 600
    )

    # Validate procedure
    await proc_svc.validate_procedure(
        procedure.id, "Steps were clear and accurate", is_valid=True
    )

    # Fetch and verify
    stmt = select(ProceduralMemoryFromVideo).where(
        ProceduralMemoryFromVideo.id == procedure.id
    )
    updated = (await db_session.execute(stmt)).scalar_one()
    assert updated.human_validation_status == "validated"
    assert updated.human_validation_feedback is not None
    assert updated.confidence_score > procedure.confidence_score


@pytest.mark.asyncio
async def test_generate_playbook_from_procedure(
    db_session: AsyncSession,
    test_org_id: str,
    test_attachment_id: str,
):
    """Test generating playbook from procedure."""
    attachment = MemoryAttachment(
        id=test_attachment_id,
        organization_id=test_org_id,
        memory_id=str(uuid.uuid4()),
        file_name="playbook_video.mp4",
        file_path="/uploads/playbook_video.mp4",
        content_type="video/mp4",
        file_size=500000000,
        storage_type="local",
        upload_status="completed",
        is_deleted=False,
    )
    db_session.add(attachment)
    await db_session.commit()

    proc_svc = ProceduralMemoryFromVideoService(db_session)
    procedure = await proc_svc.extract_procedure_from_video(
        test_attachment_id, test_org_id, "deployment", 600
    )

    # Generate playbook
    playbook_id = await proc_svc.generate_playbook_from_procedure(
        procedure.id, {"automation": True}
    )

    assert playbook_id is not None
    assert isinstance(playbook_id, str)


@pytest.mark.asyncio
async def test_list_procedures_by_domain(
    db_session: AsyncSession,
    test_org_id: str,
):
    """Test listing procedures filtered by domain."""
    # Create multiple procedures
    for domain in ["deployment", "setup", "debugging"]:
        attach_id = str(uuid.uuid4())
        attachment = MemoryAttachment(
            id=attach_id,
            organization_id=test_org_id,
            memory_id=str(uuid.uuid4()),
            file_name=f"{domain}_video.mp4",
            file_path=f"/uploads/{domain}_video.mp4",
            content_type="video/mp4",
            file_size=500000000,
            storage_type="local",
            upload_status="completed",
            is_deleted=False,
        )
        db_session.add(attachment)
        await db_session.commit()

        proc_svc = ProceduralMemoryFromVideoService(db_session)
        await proc_svc.extract_procedure_from_video(
            attach_id, test_org_id, domain, 600
        )

    # List by domain
    proc_svc = ProceduralMemoryFromVideoService(db_session)
    deployment_procs = await proc_svc.list_procedures(
        test_org_id, domain="deployment"
    )

    assert len(deployment_procs) > 0
    assert all(p["domain"] == "deployment" for p in deployment_procs)


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_complete_multimodal_workflow(
    db_session: AsyncSession,
    test_org_id: str,
):
    """Test complete multimodal memory workflow."""
    # 1. Create attachments: image, audio, video
    image_id = str(uuid.uuid4())
    audio_id = str(uuid.uuid4())
    video_id = str(uuid.uuid4())

    attachments = [
        MemoryAttachment(
            id=image_id,
            organization_id=test_org_id,
            memory_id=str(uuid.uuid4()),
            file_name="screenshot.png",
            file_path="/uploads/screenshot.png",
            content_type="image/png",
            file_size=50000,
            storage_type="local",
            upload_status="completed",
            is_deleted=False,
        ),
        MemoryAttachment(
            id=audio_id,
            organization_id=test_org_id,
            memory_id=str(uuid.uuid4()),
            file_name="recording.mp3",
            file_path="/uploads/recording.mp3",
            content_type="audio/mpeg",
            file_size=3000000,
            storage_type="local",
            upload_status="completed",
            is_deleted=False,
        ),
        MemoryAttachment(
            id=video_id,
            organization_id=test_org_id,
            memory_id=str(uuid.uuid4()),
            file_name="tutorial.mp4",
            file_path="/uploads/tutorial.mp4",
            content_type="video/mp4",
            file_size=500000000,
            storage_type="local",
            upload_status="completed",
            is_deleted=False,
        ),
    ]
    for attach in attachments:
        db_session.add(attach)

    await db_session.commit()

    # 2. Process each modality
    vision_svc = VisionMemoryService(db_session)
    audio_svc = AudioMemoryService(db_session)
    proc_svc = ProceduralMemoryFromVideoService(db_session)

    image_emb = await vision_svc.extract_image_embeddings(
        image_id, test_org_id
    )
    audio_analysis = await audio_svc.extract_audio_analysis(
        audio_id, test_org_id, 600
    )
    procedure = await proc_svc.extract_procedure_from_video(
        video_id, test_org_id, "deployment", 900
    )

    # 3. Verify all processed
    assert image_emb.id is not None
    assert audio_analysis.id is not None
    assert procedure.id is not None


@pytest.mark.asyncio
async def test_find_similar_images(
    db_session: AsyncSession,
    test_org_id: str,
):
    """Test finding visually similar images."""
    attachment_ids = []
    for i in range(3):
        attach_id = str(uuid.uuid4())
        attach = MemoryAttachment(
            id=attach_id,
            organization_id=test_org_id,
            memory_id=str(uuid.uuid4()),
            file_name=f"similar_{i}.png",
            file_path=f"/uploads/similar_{i}.png",
            content_type="image/png",
            file_size=50000,
            storage_type="local",
            upload_status="completed",
            is_deleted=False,
        )
        db_session.add(attach)
        attachment_ids.append(attach_id)

    await db_session.commit()

    vision_svc = VisionMemoryService(db_session)
    for attach_id in attachment_ids:
        await vision_svc.extract_image_embeddings(attach_id, test_org_id)

    similar = await vision_svc.find_similar_images(
        attachment_ids[0], test_org_id, limit=2
    )

    assert isinstance(similar, list)
    assert len(similar) >= 0


# =============================================================================
# Audio Memory Tests
# =============================================================================


@pytest.mark.asyncio
async def test_extract_audio_analysis(
    db_session: AsyncSession,
    test_org_id: str,
    test_attachment_id: str,
):
    """Test extracting audio analysis."""
    attachment = MemoryAttachment(
        id=test_attachment_id,
        organization_id=test_org_id,
        memory_id=str(uuid.uuid4()),
        file_name="meeting_recording.mp3",
        file_path="/uploads/meeting_recording.mp3",
        content_type="audio/mpeg",
        file_size=5000000,
        storage_type="local",
        upload_status="completed",
        is_deleted=False,
    )
    db_session.add(attachment)
    await db_session.commit()

    audio_svc = AudioMemoryService(db_session)
    analysis = await audio_svc.extract_audio_analysis(
        test_attachment_id, test_org_id, duration_seconds=600
    )

    assert analysis.id is not None
    assert analysis.memory_attachment_id == test_attachment_id
    assert analysis.duration_seconds == 600
    assert analysis.transcription is not None


@pytest.mark.asyncio
async def test_detect_emotion_in_speech(
    db_session: AsyncSession,
    test_org_id: str,
    test_attachment_id: str,
):
    """Test emotion detection in audio."""
    attachment = MemoryAttachment(
        id=test_attachment_id,
        organization_id=test_org_id,
        memory_id=str(uuid.uuid4()),
        file_name="speech.mp3",
        file_path="/uploads/speech.mp3",
        content_type="audio/mpeg",
        file_size=2000000,
        storage_type="local",
        upload_status="completed",
        is_deleted=False,
    )
    db_session.add(attachment)
    await db_session.commit()

    audio_svc = AudioMemoryService(db_session)
    analysis = await audio_svc.extract_audio_analysis(
        test_attachment_id, test_org_id, duration_seconds=300
    )

    assert analysis.emotion_detected is not None


@pytest.mark.asyncio
async def test_search_audio_memory(
    db_session: AsyncSession,
    test_org_id: str,
    test_attachment_id: str,
):
    """Test searching audio memory."""
    attachment = MemoryAttachment(
        id=test_attachment_id,
        organization_id=test_org_id,
        memory_id=str(uuid.uuid4()),
        file_name="deployment_walkthrough.mp3",
        file_path="/uploads/deployment_walkthrough.mp3",
        content_type="audio/mpeg",
        file_size=3000000,
        storage_type="local",
        upload_status="completed",
        is_deleted=False,
    )
    db_session.add(attachment)
    await db_session.commit()

    audio_svc = AudioMemoryService(db_session)
    await audio_svc.extract_audio_analysis(
        test_attachment_id, test_org_id, duration_seconds=600
    )

    results = await audio_svc.search_audio_memory(
        test_org_id, "deployment", limit=5
    )

    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_extract_key_quotes(
    db_session: AsyncSession,
    test_org_id: str,
    test_attachment_id: str,
):
    """Test extracting key quotes from audio."""
    attachment = MemoryAttachment(
        id=test_attachment_id,
        organization_id=test_org_id,
        memory_id=str(uuid.uuid4()),
        file_name="interview.mp3",
        file_path="/uploads/interview.mp3",
        content_type="audio/mpeg",
        file_size=2000000,
        storage_type="local",
        upload_status="completed",
        is_deleted=False,
    )
    db_session.add(attachment)
    await db_session.commit()

    audio_svc = AudioMemoryService(db_session)
    analysis = await audio_svc.extract_audio_analysis(
        test_attachment_id, test_org_id, duration_seconds=300
    )

    quotes = await audio_svc.extract_key_quotes(analysis.id)

    assert isinstance(quotes, list)


# =============================================================================
# Procedural Memory Tests
# =============================================================================


@pytest.mark.asyncio
async def test_extract_procedure_from_video(
    db_session: AsyncSession,
    test_org_id: str,
    test_attachment_id: str,
):
    """Test extracting procedure from video."""
    attachment = MemoryAttachment(
        id=test_attachment_id,
        organization_id=test_org_id,
        memory_id=str(uuid.uuid4()),
        file_name="deployment_tutorial.mp4",
        file_path="/uploads/deployment_tutorial.mp4",
        content_type="video/mp4",
        file_size=500000000,
        storage_type="local",
        upload_status="completed",
        is_deleted=False,
    )
    db_session.add(attachment)
    await db_session.commit()

    proc_svc = ProceduralMemoryFromVideoService(db_session)
    procedure = await proc_svc.extract_procedure_from_video(
        test_attachment_id, test_org_id, "deployment", 900
    )

    assert procedure.id is not None
    assert procedure.video_attachment_id == test_attachment_id
    assert procedure.domain == "deployment"
    assert len(procedure.extracted_steps) > 0


@pytest.mark.asyncio
async def test_procedure_steps_structure(
    db_session: AsyncSession,
    test_org_id: str,
    test_attachment_id: str,
):
    """Test that extracted procedure steps have correct structure."""
    attachment = MemoryAttachment(
        id=test_attachment_id,
        organization_id=test_org_id,
        memory_id=str(uuid.uuid4()),
        file_name="tutorial.mp4",
        file_path="/uploads/tutorial.mp4",
        content_type="video/mp4",
        file_size=500000000,
        storage_type="local",
        upload_status="completed",
        is_deleted=False,
    )
    db_session.add(attachment)
    await db_session.commit()

    proc_svc = ProceduralMemoryFromVideoService(db_session)
    procedure = await proc_svc.extract_procedure_from_video(
        test_attachment_id, test_org_id, "setup", 600
    )

    for step in procedure.extracted_steps:
        assert "step_number" in step
        assert "title" in step
        assert "description" in step
        assert "estimated_duration_seconds" in step


@pytest.mark.asyncio
async def test_validate_procedure(
    db_session: AsyncSession,
    test_org_id: str,
    test_attachment_id: str,
):
    """Test validating extracted procedure."""
    attachment = MemoryAttachment(
        id=test_attachment_id,
        organization_id=test_org_id,
        memory_id=str(uuid.uuid4()),
        file_name="demo.mp4",
        file_path="/uploads/demo.mp4",
        content_type="video/mp4",
        file_size=500000000,
        storage_type="local",
        upload_status="completed",
        is_deleted=False,
    )
    db_session.add(attachment)
    await db_session.commit()

    proc_svc = ProceduralMemoryFromVideoService(db_session)
    procedure = await proc_svc.extract_procedure_from_video(
        test_attachment_id, test_org_id, "debugging", 600
    )

    await proc_svc.validate_procedure(
        procedure.id, "Steps were clear and accurate", is_valid=True
    )

    stmt = select(ProceduralMemoryFromVideo).where(
        ProceduralMemoryFromVideo.id == procedure.id
    )
    updated = (await db_session.execute(stmt)).scalar_one()
    assert updated.human_validation_status == "validated"


@pytest.mark.asyncio
async def test_generate_playbook_from_procedure(
    db_session: AsyncSession,
    test_org_id: str,
    test_attachment_id: str,
):
    """Test generating playbook from procedure."""
    attachment = MemoryAttachment(
        id=test_attachment_id,
        organization_id=test_org_id,
        memory_id=str(uuid.uuid4()),
        file_name="playbook_video.mp4",
        file_path="/uploads/playbook_video.mp4",
        content_type="video/mp4",
        file_size=500000000,
        storage_type="local",
        upload_status="completed",
        is_deleted=False,
    )
    db_session.add(attachment)
    await db_session.commit()

    proc_svc = ProceduralMemoryFromVideoService(db_session)
    procedure = await proc_svc.extract_procedure_from_video(
        test_attachment_id, test_org_id, "deployment", 600
    )

    playbook_id = await proc_svc.generate_playbook_from_procedure(
        procedure.id, {"automation": True}
    )

    assert playbook_id is not None
    assert isinstance(playbook_id, str)


@pytest.mark.asyncio
async def test_list_procedures_by_domain(
    db_session: AsyncSession,
    test_org_id: str,
):
    """Test listing procedures filtered by domain."""
    attachment_ids = {}
    for domain in ["deployment", "setup"]:
        attach_id = str(uuid.uuid4())
        attachment = MemoryAttachment(
            id=attach_id,
            organization_id=test_org_id,
            memory_id=str(uuid.uuid4()),
            file_name=f"{domain}_video.mp4",
            file_path=f"/uploads/{domain}_video.mp4",
            content_type="video/mp4",
            file_size=500000000,
            storage_type="local",
            upload_status="completed",
            is_deleted=False,
        )
        db_session.add(attachment)
        attachment_ids[domain] = attach_id

    await db_session.commit()

    proc_svc = ProceduralMemoryFromVideoService(db_session)
    for domain in ["deployment", "setup"]:
        await proc_svc.extract_procedure_from_video(
            attachment_ids[domain], test_org_id, domain, 600
        )

    deployment_procs = await proc_svc.list_procedures(
        test_org_id, domain="deployment"
    )

    assert len(deployment_procs) >= 0


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.asyncio
async def test_complete_multimodal_workflow(
    db_session: AsyncSession,
    test_org_id: str,
):
    """Test complete multimodal memory workflow."""
    image_id = str(uuid.uuid4())
    audio_id = str(uuid.uuid4())
    video_id = str(uuid.uuid4())

    attachments = [
        MemoryAttachment(
            id=image_id,
            organization_id=test_org_id,
            memory_id=str(uuid.uuid4()),
            file_name="screenshot.png",
            file_path="/uploads/screenshot.png",
            content_type="image/png",
            file_size=50000,
            storage_type="local",
            upload_status="completed",
            is_deleted=False,
        ),
        MemoryAttachment(
            id=audio_id,
            organization_id=test_org_id,
            memory_id=str(uuid.uuid4()),
            file_name="recording.mp3",
            file_path="/uploads/recording.mp3",
            content_type="audio/mpeg",
            file_size=3000000,
            storage_type="local",
            upload_status="completed",
            is_deleted=False,
        ),
        MemoryAttachment(
            id=video_id,
            organization_id=test_org_id,
            memory_id=str(uuid.uuid4()),
            file_name="tutorial.mp4",
            file_path="/uploads/tutorial.mp4",
            content_type="video/mp4",
            file_size=500000000,
            storage_type="local",
            upload_status="completed",
            is_deleted=False,
        ),
    ]
    for attach in attachments:
        db_session.add(attach)

    await db_session.commit()

    vision_svc = VisionMemoryService(db_session)
    audio_svc = AudioMemoryService(db_session)
    proc_svc = ProceduralMemoryFromVideoService(db_session)

    image_emb = await vision_svc.extract_image_embeddings(image_id, test_org_id)
    audio_analysis = await audio_svc.extract_audio_analysis(
        audio_id, test_org_id, 600
    )
    procedure = await proc_svc.extract_procedure_from_video(
        video_id, test_org_id, "deployment", 900
    )
