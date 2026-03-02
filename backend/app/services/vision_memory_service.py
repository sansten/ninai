"""
Vision Memory Service - PR-9

Semantic understanding of visual content (images, screenshots, diagrams, videos).
Handles embeddings, OCR, object detection, and visual search.
"""

import json
import re
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.multimodal_memory import (
    MultimodalEmbedding,
    SpatialMemory,
    VisualSearchQuery,
    ModalityType,
    EmbeddingModel,
)
from app.models.memory_attachment import MemoryAttachment


class VisionMemoryService:
    """
    Semantic understanding of visual content.
    Extracts embeddings, performs OCR, detects objects, and enables visual search.
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.ocr_keywords = {
            "error": ["error", "failed", "failure", "exception", "critical", "fatal"],
            "success": ["success", "completed", "done", "finished", "ok", "passed"],
            "warning": ["warning", "caution", "alert", "attention", "deprecated"],
            "button": ["button", "click", "press", "submit", "save", "cancel", "next"],
            "input": ["input", "field", "text", "password", "email", "search"],
        }

    async def extract_image_embeddings(
        self,
        attachment_id: str,
        organization_id: str,
    ) -> MultimodalEmbedding:
        """
        Extract semantic embeddings from image/screenshot.

        Uses simulated CLIP embeddings and object detection.
        In production, integrate with real vision models (CLIP, YOLOv8, etc).
        """
        # Fetch attachment
        stmt = select(MemoryAttachment).where(
            MemoryAttachment.id == attachment_id
        )
        attachment = (await self.session.execute(stmt)).scalar_one_or_none()
        if not attachment:
            raise ValueError(f"Attachment {attachment_id} not found")

        # Simulate embedding extraction
        embedding = self._generate_simulated_embedding(768)  # CLIP dimension
        detected_objects = self._detect_objects_from_content(
            attachment.file_name or "",
            attachment.content_type or "",
        )

        # Extract text via OCR (simulated)
        extracted_text = self._extract_text_from_image(attachment.file_name)

        # Create embedding record
        multimodal_emb = MultimodalEmbedding(
            id=self._generate_uuid(),
            organization_id=organization_id,
            memory_attachment_id=attachment_id,
            modality=ModalityType.IMAGE.value,
            embedding=embedding,
            embedding_model=EmbeddingModel.CLIP_VIT.value,
            embedding_dimension=len(embedding),
            embedding_confidence=0.92,
            extracted_text=extracted_text,
            detected_objects=detected_objects,
            object_confidence_scores=self._generate_confidence_scores(detected_objects),
            extracted_metadata={
                "has_text": bool(extracted_text),
                "object_count": len(detected_objects),
                "dominant_colors": ["blue", "white", "gray"],
            },
            thumbnail_url=f"/attachments/{attachment_id}/thumbnail",
            file_size_bytes=attachment.file_size,
            processing_time_ms=150,
            processing_status="completed",
            searchable=True,
        )

        self.session.add(multimodal_emb)
        await self.session.flush()
        return multimodal_emb

    async def extract_spatial_memory(
        self,
        attachment_id: str,
        organization_id: str,
    ) -> SpatialMemory:
        """
        Extract spatial/UI layout information from screenshot or diagram.

        Identifies elements (buttons, text fields, menus) and their positions.
        """
        # Fetch attachment
        stmt = select(MemoryAttachment).where(
            MemoryAttachment.id == attachment_id
        )
        attachment = (await self.session.execute(stmt)).scalar_one_or_none()
        if not attachment:
            raise ValueError(f"Attachment {attachment_id} not found")

        # Simulate spatial analysis
        elements = self._detect_ui_elements(attachment.file_name or "")
        interactive_elements = [e for e in elements if self._is_interactive(e)]
        text_regions = self._extract_text_regions(elements)

        spatial_mem = SpatialMemory(
            id=self._generate_uuid(),
            organization_id=organization_id,
            memory_attachment_id=attachment_id,
            image_width=1920,  # Simulated
            image_height=1080,  # Simulated
            elements=elements,
            layout_description=self._describe_layout(elements),
            layout_type=self._detect_layout_type(elements),
            interactive_elements=interactive_elements,
            text_regions=text_regions,
            color_palette=["#0066CC", "#FFFFFF", "#F0F0F0", "#666666"],
            dominant_colors={"white": 0.45, "blue": 0.30, "gray": 0.25},
            spatial_graph=self._build_spatial_graph(elements),
            ui_hierarchy=self._build_hierarchy(elements),
            accessibility_score=0.82,
            accessibility_issues=["low_contrast_on_buttons", "missing_alt_text"],
        )

        self.session.add(spatial_mem)
        await self.session.flush()
        return spatial_mem

    async def search_visual_memory(
        self,
        organization_id: str,
        query: str,
        modality_filter: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search visual memory by text query.

        Uses semantic similarity of extracted text and detected objects.
        """
        # In production: embed query, search embeddings table via pgvector
        # For now: keyword-based search on extracted_text and detected_objects

        query_lower = query.lower()
        keywords = re.findall(r"\b\w+\b", query_lower)

        # Query embeddings
        stmt = select(MultimodalEmbedding).where(
            MultimodalEmbedding.organization_id == organization_id,
            MultimodalEmbedding.searchable.is_(True),
        )
        if modality_filter:
            stmt = stmt.where(MultimodalEmbedding.modality == modality_filter)

        results = (await self.session.execute(stmt)).scalars().all()

        # Score results by keyword match
        scored_results = []
        for emb in results:
            score = 0.0
            text_content = (emb.extracted_text or "").lower()
            for keyword in keywords:
                if keyword in text_content:
                    score += 1.0
                if keyword in [obj.lower() for obj in emb.detected_objects]:
                    score += 1.0

            if score > 0:
                scored_results.append((score, emb))

        # Sort by score and limit
        scored_results.sort(key=lambda x: x[0], reverse=True)
        results_list = [emb for _, emb in scored_results[:limit]]

        # Log search query
        await self._log_search_query(
            organization_id,
            query,
            len(results_list),
            modality_filter or "all",
        )

        return [
            {
                "id": r.id,
                "attachment_id": r.memory_attachment_id,
                "modality": r.modality,
                "extracted_text": r.extracted_text,
                "detected_objects": r.detected_objects,
                "relevance_score": 0.85,
            }
            for r in results_list
        ]

    async def visual_context_for_query(
        self,
        organization_id: str,
        query: str,
    ) -> Optional[str]:
        """
        Find relevant visual context for a query.

        Returns formatted context string with image descriptions.
        """
        results = await self.search_visual_memory(organization_id, query, limit=3)

        if not results:
            return None

        context_parts = []
        for i, result in enumerate(results, 1):
            part = f"Visual Reference {i}:\n"
            part += f"  Type: {result['modality']}\n"
            if result["extracted_text"]:
                part += f"  Content: {result['extracted_text'][:200]}...\n"
            if result["detected_objects"]:
                part += f"  Objects: {', '.join(result['detected_objects'][:5])}\n"
            context_parts.append(part)

        return "\n".join(context_parts)

    async def find_similar_images(
        self,
        attachment_id: str,
        organization_id: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find visually similar images using embedding similarity.

        In production: use pgvector <-> operator for cosine distance.
        """
        # Fetch reference embedding
        stmt = select(MultimodalEmbedding).where(
            MultimodalEmbedding.memory_attachment_id == attachment_id,
            MultimodalEmbedding.organization_id == organization_id,
        )
        ref_emb = (await self.session.execute(stmt)).scalar_one_or_none()
        if not ref_emb:
            return []

        # Find all embeddings for this org
        stmt = select(MultimodalEmbedding).where(
            MultimodalEmbedding.organization_id == organization_id,
            MultimodalEmbedding.id != ref_emb.id,
            MultimodalEmbedding.modality == ref_emb.modality,
        )
        other_embeddings = (await self.session.execute(stmt)).scalars().all()

        # Simple similarity: count matching objects
        similarities = []
        ref_objects = set(ref_emb.detected_objects)
        for emb in other_embeddings:
            intersection = len(ref_objects & set(emb.detected_objects))
            union = len(ref_objects | set(emb.detected_objects))
            jaccard = intersection / union if union > 0 else 0.0
            if jaccard > 0.3:
                similarities.append((jaccard, emb))

        similarities.sort(key=lambda x: x[0], reverse=True)

        return [
            {
                "id": emb.id,
                "attachment_id": emb.memory_attachment_id,
                "similarity_score": score,
                "detected_objects": emb.detected_objects,
            }
            for score, emb in similarities[:limit]
        ]

    # ============ Helper Methods ============

    def _generate_simulated_embedding(self, dimension: int) -> List[float]:
        """Generate simulated CLIP embedding."""
        import hashlib
        seed = int(hashlib.md5(b"simulated").hexdigest(), 16)
        return [
            ((seed + i) % 1000) / 1000.0 - 0.5 for i in range(dimension)
        ]

    def _detect_objects_from_content(
        self,
        file_name: str,
        content_type: str,
    ) -> List[str]:
        """Detect objects based on filename/content type."""
        objects = []

        if "screenshot" in file_name.lower() or "screen" in file_name.lower():
            objects.extend(["screenshot", "user_interface", "ui_elements"])

        if "error" in file_name.lower() or "exception" in file_name.lower():
            objects.extend(["error_message", "alert", "exception"])

        if "code" in file_name.lower() or "source" in file_name.lower():
            objects.extend(["code", "syntax", "programming"])

        if "chart" in file_name.lower() or "graph" in file_name.lower():
            objects.extend(["chart", "data_visualization", "metric"])

        if not objects:
            objects = ["image", "visual_content"]

        return objects

    def _extract_text_from_image(self, file_name: str) -> Optional[str]:
        """Simulate OCR text extraction."""
        # In production: use Tesseract, EasyOCR, or cloud APIs
        if "chart" in file_name.lower():
            return "Sales Performance Q1 2026: Revenue $1.2M, Growth 15%"
        elif "error" in file_name.lower():
            return "RuntimeError: Connection timeout after 30s. Stack trace: ..."
        elif "code" in file_name.lower():
            return "def analyze_sentiment(text): # Python sentiment analysis ..."
        else:
            return f"Content from {file_name}"

    def _detect_ui_elements(self, file_name: str) -> List[Dict[str, Any]]:
        """Detect UI elements in screenshot."""
        elements = [
            {
                "id": "elem_1",
                "type": "button",
                "label": "Submit",
                "position": {"x": 400, "y": 500, "width": 100, "height": 40},
                "color": "#0066CC",
            },
            {
                "id": "elem_2",
                "type": "text_field",
                "label": "Email Address",
                "position": {"x": 200, "y": 300, "width": 400, "height": 40},
                "color": "#FFFFFF",
            },
            {
                "id": "elem_3",
                "type": "heading",
                "label": "Login Form",
                "position": {"x": 200, "y": 50, "width": 400, "height": 50},
                "color": "#000000",
            },
        ]
        return elements

    def _is_interactive(self, element: Dict[str, Any]) -> bool:
        """Check if element is interactive."""
        interactive_types = {"button", "text_field", "dropdown", "menu", "link"}
        return element.get("type") in interactive_types

    def _extract_text_regions(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract text-containing regions."""
        return [e for e in elements if e.get("type") in {"heading", "text_field", "button"}]

    def _describe_layout(self, elements: List[Dict[str, Any]]) -> str:
        """Describe layout of elements."""
        element_types = set(e.get("type") for e in elements)
        return f"Layout with {len(elements)} elements: {', '.join(sorted(element_types))}"

    def _detect_layout_type(self, elements: List[Dict[str, Any]]) -> str:
        """Detect layout type (single_column, two_column, grid, etc)."""
        if len(elements) <= 3:
            return "simple"
        elif any(e["position"]["x"] > 500 for e in elements):
            return "multi_column"
        else:
            return "single_column"

    def _build_spatial_graph(self, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build spatial relationship graph."""
        return {
            "nodes": [e["id"] for e in elements],
            "edges": [
                {"from": "elem_1", "to": "elem_2", "relationship": "below"},
                {"from": "elem_2", "to": "elem_3", "relationship": "contains"},
            ],
        }

    def _build_hierarchy(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build UI hierarchy."""
        return [
            {
                "level": 0,
                "element": "elem_3",
                "type": "heading",
                "label": "Login Form",
                "children": ["elem_2", "elem_1"],
            },
            {
                "level": 1,
                "element": "elem_2",
                "type": "text_field",
                "label": "Email Address",
                "children": [],
            },
            {
                "level": 1,
                "element": "elem_1",
                "type": "button",
                "label": "Submit",
                "children": [],
            },
        ]

    def _generate_confidence_scores(
        self,
        detected_objects: List[str],
    ) -> Dict[str, float]:
        """Generate confidence scores for detected objects."""
        return {obj: 0.85 + (hash(obj) % 15) / 100.0 for obj in detected_objects}

    async def _log_search_query(
        self,
        organization_id: str,
        query: str,
        results_count: int,
        modality: str,
    ) -> None:
        """Log visual search query for analytics."""
        from app.models.multimodal_memory import VisualSearchQuery

        log_entry = VisualSearchQuery(
            id=self._generate_uuid(),
            organization_id=organization_id,
            query_text=query,
            query_type="text",
            results_returned=results_count,
            results_clicked=0,
            search_duration_ms=50,
            modalities_searched=[modality],
        )
        self.session.add(log_entry)
        await self.session.flush()

    @staticmethod
    def _generate_uuid() -> str:
        """Generate UUID string."""
        import uuid
        return str(uuid.uuid4())
