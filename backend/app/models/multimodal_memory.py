"""
PR-9: Multimodal Deep Memory Integration Models

Models for semantic understanding of visual content (images, diagrams, videos),
audio content (transcriptions, sound analysis), and spatial/UI memory.
Enables rich, multimodal retrieval and understanding.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any

from sqlalchemy import Float, Index, Integer, String, TIMESTAMP, ForeignKey, JSON, Boolean, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDMixin, TimestampMixin


class ModalityType(str, Enum):
    """Types of multimodal content."""

    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DIAGRAM = "diagram"
    SCREENSHOT = "screenshot"
    DOCUMENT = "document"


class ObjectDetectionConfidence(str, Enum):
    """Confidence levels for object detection."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ElementType(str, Enum):
    """Types of UI/spatial elements."""

    BUTTON = "button"
    TEXT_FIELD = "text_field"
    DROPDOWN = "dropdown"
    MENU = "menu"
    ERROR_MESSAGE = "error_message"
    SUCCESS_MESSAGE = "success_message"
    IMAGE = "image"
    TABLE = "table"
    CODE_BLOCK = "code_block"
    HEADING = "heading"
    LINK = "link"


class EmbeddingModel(str, Enum):
    """Embedding models used for multimodal content."""

    CLIP_VIT = "clip-vit"
    WHISPER = "whisper"
    VIDEO_CLIP = "video-clip"
    RESNET50 = "resnet50"
    DINO = "dino"


class AudioType(str, Enum):
    """Types of audio content."""

    SPEECH = "speech"
    BACKGROUND_NOISE = "background_noise"
    MUSIC = "music"
    SOUND_EFFECT = "sound_effect"
    SILENCE = "silence"


class MultimodalEmbedding(Base, UUIDMixin, TimestampMixin):
    """
    Semantic embedding for non-text multimodal content (images, audio, video).

    Enables similarity search and semantic understanding across modalities.
    """

    __tablename__ = "multimodal_embeddings"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    memory_attachment_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("memory_attachments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    modality: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    embedding: Mapped[List[float]] = mapped_column(JSONB, nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(100), nullable=False)
    embedding_dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding_confidence: Mapped[float] = mapped_column(Float, default=0.9, nullable=False)
    extracted_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    detected_objects: Mapped[List[str]] = mapped_column(JSONB, default=[], nullable=False)
    object_confidence_scores: Mapped[Dict[str, float]] = mapped_column(JSONB, default={}, nullable=False)
    extracted_metadata: Mapped[Dict[str, Any]] = mapped_column(JSONB, default={}, nullable=False)
    thumbnail_url: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)
    duration_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    processing_time_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    processing_status: Mapped[str] = mapped_column(String(50), default="completed", nullable=False)
    processing_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    searchable: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    __table_args__ = (
        Index("idx_multimodal_org_modality", "organization_id", "modality"),
        Index("idx_multimodal_org_searchable", "organization_id", "searchable"),
    )


class SpatialMemory(Base, UUIDMixin, TimestampMixin):
    """
    Spatial/UI memory for visual layouts and interactive elements.

    Captures positioning, relationships, and interaction points in visual content
    (screenshots, diagrams, UIs). Enables spatial reasoning and UI-based retrieval.
    """

    __tablename__ = "spatial_memories"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    memory_attachment_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("memory_attachments.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    image_width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    image_height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    elements: Mapped[List[Dict[str, Any]]] = mapped_column(JSONB, default=[], nullable=False)
    layout_description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    layout_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    interactive_elements: Mapped[List[Dict[str, Any]]] = mapped_column(JSONB, default=[], nullable=False)
    text_regions: Mapped[List[Dict[str, Any]]] = mapped_column(JSONB, default=[], nullable=False)
    color_palette: Mapped[List[str]] = mapped_column(JSONB, default=[], nullable=False)
    dominant_colors: Mapped[Dict[str, float]] = mapped_column(JSONB, default={}, nullable=False)
    spatial_graph: Mapped[Dict[str, Any]] = mapped_column(JSONB, default={}, nullable=False)
    ui_hierarchy: Mapped[List[Dict[str, Any]]] = mapped_column(JSONB, default=[], nullable=False)
    accessibility_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    accessibility_issues: Mapped[List[str]] = mapped_column(JSONB, default=[], nullable=False)

    __table_args__ = (
        Index("idx_spatial_org_attachment", "organization_id", "memory_attachment_id"),
        Index("idx_spatial_org_layout", "organization_id", "layout_type"),
    )


class AudioAnalysis(Base, UUIDMixin, TimestampMixin):
    """
    Semantic analysis of audio content (speech, sound effects, music).

    Includes transcriptions, speaker identification, emotion/tone detection,
    and audio fingerprinting for similarity search.
    """

    __tablename__ = "audio_analyses"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    memory_attachment_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("memory_attachments.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    duration_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    transcription: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    transcription_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    language: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    speakers_identified: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    speaker_segments: Mapped[List[Dict[str, Any]]] = mapped_column(JSONB, default=[], nullable=False)
    audio_types_detected: Mapped[List[str]] = mapped_column(JSONB, default=[], nullable=False)
    emotion_detected: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    emotion_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tone_detected: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    tone_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    audio_fingerprint: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    key_moments: Mapped[List[Dict[str, Any]]] = mapped_column(JSONB, default=[], nullable=False)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    fidelity_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    __table_args__ = (
        Index("idx_audio_org_language", "organization_id", "language"),
        Index("idx_audio_org_emotion", "organization_id", "emotion_detected"),
    )


class ProceduralMemoryFromVideo(Base, UUIDMixin, TimestampMixin):
    """
    Procedural/domain knowledge extracted from video demonstrations.

    Captures step-by-step procedures learned by watching video walkthroughs,
    screen recordings, or tutorials. Enables automation and knowledge transfer.
    """

    __tablename__ = "procedural_memories_from_video"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    video_attachment_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("memory_attachments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    domain: Mapped[str] = mapped_column(String(200), nullable=False)
    procedure_title: Mapped[str] = mapped_column(String(500), nullable=False)
    procedure_description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    extracted_steps: Mapped[List[Dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    step_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    estimated_duration_minutes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    prerequisites: Mapped[List[str]] = mapped_column(JSONB, default=[], nullable=False)
    tools_required: Mapped[List[str]] = mapped_column(JSONB, default=[], nullable=False)
    common_mistakes: Mapped[List[str]] = mapped_column(JSONB, default=[], nullable=False)
    success_indicators: Mapped[List[str]] = mapped_column(JSONB, default=[], nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.7, nullable=False)
    video_segments: Mapped[List[Dict[str, Any]]] = mapped_column(JSONB, default=[], nullable=False)
    generated_playbook_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("playbooks.id", ondelete="SET NULL"),
        nullable=True,
    )
    human_validation_status: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    human_validation_feedback: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("idx_procedural_org_domain", "organization_id", "domain"),
        Index("idx_procedural_org_title", "organization_id", "procedure_title"),
    )


class VisualSearchQuery(Base, UUIDMixin, TimestampMixin):
    """
    Tracking for visual search queries (for analytics and optimization).

    Records when users search for images/video content by description,
    enabling improvement of visual indexing and retrieval.
    """

    __tablename__ = "visual_search_queries"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    query_type: Mapped[str] = mapped_column(String(50), nullable=False)  # "text" | "visual_similarity"
    results_returned: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    results_clicked: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    user_feedback: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # "relevant" | "irrelevant"
    search_duration_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    modalities_searched: Mapped[List[str]] = mapped_column(JSONB, default=[], nullable=False)

    __table_args__ = (
        Index("idx_visual_search_org", "organization_id"),
        Index("idx_visual_search_feedback", "organization_id", "user_feedback"),
    )
