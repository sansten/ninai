"""
PR-9: Multimodal Deep Memory Integration API Schemas
"""

from datetime import datetime
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field


# ============ Multimodal Embedding Schemas ============

class DetectedObject(BaseModel):
    """Detected object in image/video."""

    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bounding_box: Optional[Dict[str, float]] = None


class MultimodalEmbeddingResponse(BaseModel):
    """Response containing multimodal embedding."""

    id: str
    attachment_id: str
    modality: str = Field(..., description="image | audio | video | diagram | screenshot")
    extracted_text: Optional[str] = Field(
        None, description="Extracted text via OCR from visual content"
    )
    detected_objects: List[str] = Field(..., description="Detected objects/entities")
    object_confidence_scores: Dict[str, float]
    embedding_dimension: int
    embedding_confidence: float = Field(..., ge=0.0, le=1.0)
    thumbnail_url: Optional[str] = None
    duration_seconds: Optional[int] = None
    processing_status: str = Field(..., description="completed | failed | processing")
    created_at: Optional[str] = None


class ExtractEmbeddingRequest(BaseModel):
    """Request to extract embeddings from attachment."""

    attachment_id: str = Field(..., description="Memory attachment ID")
    modality_hint: Optional[str] = Field(
        None, description="Optional hint about content type"
    )


class ExtractEmbeddingResponse(BaseModel):
    """Response after embedding extraction."""

    success: bool
    embedding_id: str
    modality: str
    extracted_text: Optional[str]
    detected_objects: List[str]
    processing_time_ms: int


# ============ Spatial Memory Schemas ============

class UIElement(BaseModel):
    """UI element in screenshot."""

    id: str
    element_type: str = Field(
        ...,
        description="button | text_field | dropdown | menu | image | heading | link",
    )
    label: Optional[str] = None
    position: Dict[str, int] = Field(
        ..., description="x, y, width, height coordinates"
    )
    color: Optional[str] = None
    interactive: bool = False
    accessibility_label: Optional[str] = None


class TextRegion(BaseModel):
    """Text-containing region in image."""

    text: str
    position: Dict[str, int]
    font_size: Optional[int] = None
    color: Optional[str] = None


class SpatialMemoryResponse(BaseModel):
    """Spatial/UI memory response."""

    id: str
    attachment_id: str
    image_width: Optional[int]
    image_height: Optional[int]
    elements: List[UIElement] = Field(..., description="UI elements detected")
    layout_description: Optional[str]
    layout_type: str = Field(
        ..., description="single_column | multi_column | grid | simple"
    )
    interactive_elements: List[str] = Field(..., description="IDs of interactive elements")
    text_regions: List[TextRegion] = Field(..., description="Text areas extracted")
    color_palette: List[str]
    dominant_colors: Dict[str, float]
    accessibility_score: Optional[float]
    accessibility_issues: List[str]
    created_at: Optional[str]


class ExtractSpatialMemoryRequest(BaseModel):
    """Request to extract spatial memory from image."""

    attachment_id: str
    include_accessibility_check: bool = True


class ExtractSpatialMemoryResponse(BaseModel):
    """Response after spatial extraction."""

    success: bool
    spatial_memory_id: str
    element_count: int
    interactive_element_count: int
    layout_type: str
    accessibility_score: Optional[float]


# ============ Audio Analysis Schemas ============

class SpeakerSegment(BaseModel):
    """Speaker segment in audio."""

    speaker_id: str
    speaker_name: Optional[str] = None
    start_seconds: int
    end_seconds: int
    duration_seconds: int = 0


class KeyMoment(BaseModel):
    """Significant moment in audio."""

    timestamp: Dict[str, Any] = Field(..., description="Start time in seconds")
    description: str
    importance: float = Field(..., ge=0.0, le=1.0)


class AudioAnalysisResponse(BaseModel):
    """Response containing audio analysis."""

    id: str
    attachment_id: str
    duration_seconds: int
    transcription: Optional[str] = Field(
        None, description="Full transcript of audio content"
    )
    transcription_confidence: Optional[float] = None
    language: Optional[str] = None
    speakers_identified: int
    speaker_segments: List[SpeakerSegment]
    audio_types_detected: List[str] = Field(
        ..., description="speech | music | background_noise | sound_effect"
    )
    emotion_detected: Optional[str] = Field(
        None, description="happy | angry | sad | calm | anxious | neutral"
    )
    emotion_confidence: Optional[float] = None
    tone_detected: Optional[str] = Field(
        None, description="professional | casual | instructional | supportive"
    )
    tone_confidence: Optional[float] = None
    key_moments: List[KeyMoment]
    summary: Optional[str]
    fidelity_score: Optional[float]
    created_at: Optional[str]


class ExtractAudioRequest(BaseModel):
    """Request to analyze audio content."""

    attachment_id: str
    extract_transcription: bool = True
    detect_speakers: bool = True
    detect_emotion: bool = True


class ExtractAudioResponse(BaseModel):
    """Response after audio analysis."""

    success: bool
    audio_analysis_id: str
    duration_seconds: int
    transcription_length: int
    speakers_detected: int
    emotion: Optional[str]
    processing_time_ms: int


# ============ Procedural Memory from Video Schemas ============

class ProcedureStep(BaseModel):
    """Single step in a procedure."""

    step_number: int
    title: str
    description: str
    estimated_duration_seconds: int
    expected_outcome: str
    difficulty: str = Field(..., description="easy | medium | hard")
    key_action: Optional[str] = None
    notes: Optional[str] = None


class VideoSegment(BaseModel):
    """Video segment corresponding to procedure step."""

    step_number: int
    start_seconds: int
    end_seconds: int
    title: str
    relevance: float = Field(..., ge=0.0, le=1.0)


class ProceduralMemoryResponse(BaseModel):
    """Procedural knowledge extracted from video."""

    id: str
    video_attachment_id: str
    domain: str
    procedure_title: str
    procedure_description: Optional[str]
    extracted_steps: List[ProcedureStep]
    step_count: int
    estimated_duration_minutes: Optional[int]
    prerequisites: List[str]
    tools_required: List[str]
    common_mistakes: List[str]
    success_indicators: List[str]
    confidence_score: float
    video_segments: List[VideoSegment]
    validation_status: Optional[str] = Field(
        None, description="pending_review | validated | rejected"
    )
    created_at: Optional[str]


class ExtractProcedureRequest(BaseModel):
    """Request to extract procedure from video."""

    video_attachment_id: str
    domain: str = Field(
        ..., description="deployment | setup | debugging | other"
    )
    duration_seconds: Optional[int] = None


class ExtractProcedureResponse(BaseModel):
    """Response after procedure extraction."""

    success: bool
    procedure_id: str
    domain: str
    step_count: int
    estimated_duration_minutes: Optional[int]
    confidence_score: float


class ValidateProcedureRequest(BaseModel):
    """Request to validate extracted procedure."""

    feedback: str
    is_valid: bool


class ValidateProcedureResponse(BaseModel):
    """Response after procedure validation."""

    success: bool
    procedure_id: str
    validation_status: str
    updated_confidence_score: float


class GeneratePlaybookRequest(BaseModel):
    """Request to generate playbook from procedure."""

    procedure_id: str
    playbook_config: Optional[Dict[str, Any]] = None


class GeneratePlaybookResponse(BaseModel):
    """Response with generated playbook."""

    success: bool
    procedure_id: str
    playbook_id: str
    playbook_title: str


# ============ Visual Search Schemas ============

class VisualSearchResult(BaseModel):
    """Single result from visual search."""

    id: str
    attachment_id: str
    modality: str
    relevance_score: float
    extracted_text: Optional[str]
    detected_objects: List[str]
    summary: Optional[str] = None


class SearchVisualMemoryRequest(BaseModel):
    """Request to search visual content."""

    query: str = Field(..., description="Search query")
    modality_filter: Optional[str] = Field(
        None, description="Optional filter: image | audio | video"
    )
    limit: int = Field(10, ge=1, le=50)


class SearchVisualMemoryResponse(BaseModel):
    """Visual search results."""

    query: str
    total_results: int
    results: List[VisualSearchResult]
    search_time_ms: int


class FindSimilarImagesRequest(BaseModel):
    """Request to find similar images."""

    attachment_id: str
    limit: int = Field(5, ge=1, le=20)


class FindSimilarImagesResponse(BaseModel):
    """Similar images found."""

    reference_attachment_id: str
    similar_results: List[VisualSearchResult]


class GetVisualContextRequest(BaseModel):
    """Request visual context for a query."""

    query: str


class GetVisualContextResponse(BaseModel):
    """Visual context for query."""

    query: str
    context: Optional[str]
    reference_count: int


# ============ Batch Operations Schemas ============

class BatchProcessAttachmentRequest(BaseModel):
    """Request to process multiple attachments."""

    attachment_ids: List[str]
    extract_embedding: bool = True
    extract_spatial: bool = False
    extract_audio: bool = False


class BatchProcessAttachmentResponse(BaseModel):
    """Response after batch processing."""

    success: bool
    processed_count: int
    failed_count: int
    total_processing_time_ms: int
    results: List[Dict[str, Any]]


# ============ Analytics Schemas ============

class MultimodalAnalyticsResponse(BaseModel):
    """Analytics about multimodal memory usage."""

    total_embeddings_indexed: int
    total_spatial_memories: int
    total_audio_analyses: int
    total_procedures_extracted: int
    average_search_time_ms: float
    unique_modalities: List[str]
    most_detected_objects: List[str]
    processing_stats: Dict[str, Any]
