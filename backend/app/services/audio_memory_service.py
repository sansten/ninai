"""
Audio Memory Service - PR-9

Semantic understanding of audio content.
Handles transcription, speaker identification, emotion detection, and audio search.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.multimodal_memory import AudioAnalysis, AudioType
from app.models.memory_attachment import MemoryAttachment


class AudioMemoryService:
    """
    Semantic understanding of audio content (speech, sound effects, music).
    Handles transcription, speaker identification, emotion/tone detection.
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.emotion_keywords = {
            "happy": ["happy", "joy", "delighted", "excited", "enthusiastic"],
            "angry": ["angry", "frustrated", "irritated", "upset", "furious"],
            "sad": ["sad", "unhappy", "depressed", "disappointed", "miserable"],
            "calm": ["calm", "relaxed", "peaceful", "serene", "tranquil"],
            "anxious": ["anxious", "nervous", "worried", "stressed", "tense"],
        }
        self.tone_keywords = {
            "professional": ["professional", "formal", "business", "technical"],
            "casual": ["casual", "friendly", "informal", "conversational"],
            "instructional": ["instructional", "explanatory", "teaching", "demonstrating"],
            "supportive": ["supportive", "empathetic", "understanding", "helpful"],
        }

    async def extract_audio_analysis(
        self,
        attachment_id: str,
        organization_id: str,
        duration_seconds: int = 120,
    ) -> AudioAnalysis:
        """
        Analyze audio content: transcription, emotion, tone, speaker count.

        Simulates audio processing. In production: use Whisper, Pyannote, etc.
        """
        # Fetch attachment
        stmt = select(MemoryAttachment).where(
            MemoryAttachment.id == attachment_id
        )
        attachment = (await self.session.execute(stmt)).scalar_one_or_none()
        if not attachment:
            raise ValueError(f"Attachment {attachment_id} not found")

        # Simulate transcription
        transcription = self._transcribe_audio(attachment.file_name or "")

        # Detect emotion and tone from transcription
        emotion = self._detect_emotion(transcription)
        tone = self._detect_tone(transcription)

        # Simulate speaker detection
        speaker_segments = self._detect_speakers(transcription, duration_seconds)
        speakers_count = len(set(s.get("speaker_id") for s in speaker_segments))

        # Create audio analysis record
        audio_analysis = AudioAnalysis(
            id=self._generate_uuid(),
            organization_id=organization_id,
            memory_attachment_id=attachment_id,
            duration_seconds=duration_seconds,
            transcription=transcription,
            transcription_confidence=0.91,
            language="en",
            speakers_identified=speakers_count,
            speaker_segments=speaker_segments,
            audio_types_detected=self._detect_audio_types(transcription),
            emotion_detected=emotion["emotion"],
            emotion_confidence=emotion["confidence"],
            tone_detected=tone["tone"],
            tone_confidence=tone["confidence"],
            audio_fingerprint=self._generate_fingerprint(attachment.file_name),
            key_moments=self._extract_key_moments(transcription),
            summary=self._summarize_audio(transcription),
            fidelity_score=0.88,
        )

        self.session.add(audio_analysis)
        await self.session.flush()
        return audio_analysis

    async def search_audio_memory(
        self,
        organization_id: str,
        query: str,
        emotion_filter: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search audio memory by transcription and metadata.
        """
        query_lower = query.lower()

        # Query audio analysis
        stmt = select(AudioAnalysis).where(
            AudioAnalysis.organization_id == organization_id,
        )
        if emotion_filter:
            stmt = stmt.where(AudioAnalysis.emotion_detected == emotion_filter)

        results = (await self.session.execute(stmt)).scalars().all()

        # Score by keyword match in transcription
        scored_results = []
        for audio in results:
            score = 0.0
            transcription = (audio.transcription or "").lower()
            if query_lower in transcription:
                score += 2.0
            # Simple tokenization
            query_words = query_lower.split()
            for word in query_words:
                if word in transcription:
                    score += 1.0

            if score > 0:
                scored_results.append((score, audio))

        scored_results.sort(key=lambda x: x[0], reverse=True)
        results_list = [audio for _, audio in scored_results[:limit]]

        return [
            {
                "id": a.id,
                "attachment_id": a.memory_attachment_id,
                "duration_seconds": a.duration_seconds,
                "transcription": a.transcription[:300] if a.transcription else None,
                "emotion": a.emotion_detected,
                "tone": a.tone_detected,
                "speakers": a.speakers_identified,
                "relevance_score": 0.82,
            }
            for a in results_list
        ]

    async def extract_key_quotes(
        self,
        audio_analysis_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Extract significant quotes/statements from audio.
        """
        stmt = select(AudioAnalysis).where(AudioAnalysis.id == audio_analysis_id)
        audio = (await self.session.execute(stmt)).scalar_one_or_none()
        if not audio or not audio.transcription:
            return []

        # Extract sentences (simulated)
        sentences = audio.transcription.split(". ")
        quotes = [
            {
                "text": sent.strip(),
                "timestamp": {"start_seconds": 10 + i * 20, "end_seconds": 30 + i * 20},
                "speaker": f"Speaker {(i % 2) + 1}",
                "significance": 0.75 + (i % 3) * 0.1,
            }
            for i, sent in enumerate(sentences[:5])
        ]
        return quotes

    # ============ Helper Methods ============

    def _transcribe_audio(self, file_name: str) -> str:
        """Simulate audio transcription."""
        # In production: use Whisper, Google Cloud Speech-to-Text, etc.
        if "meeting" in file_name.lower():
            return (
                "Speaker 1: Let's discuss the deployment process. Speaker 2: Sure, "
                "I think we should use blue-green deployment. Speaker 1: Good idea, "
                "that reduces downtime. We should test thoroughly before rollout."
            )
        elif "tutorial" in file_name.lower():
            return (
                "Today we'll learn about Kubernetes. First, you need to install kubectl. "
                "Then set up your cluster configuration. Finally, deploy your application "
                "using kubectl apply. Let me show you step by step."
            )
        else:
            return "Audio content transcription: [simulated content from file]"

    def _detect_emotion(self, text: str) -> Dict[str, Any]:
        """Detect emotion from transcript."""
        text_lower = text.lower()
        for emotion, keywords in self.emotion_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return {"emotion": emotion, "confidence": 0.85 + (hash(emotion) % 10) / 100.0}
        return {"emotion": "neutral", "confidence": 0.70}

    def _detect_tone(self, text: str) -> Dict[str, Any]:
        """Detect tone from transcript."""
        text_lower = text.lower()
        for tone, keywords in self.tone_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return {"tone": tone, "confidence": 0.83}
        return {"tone": "neutral", "confidence": 0.65}

    def _detect_speakers(self, text: str, duration: int) -> List[Dict[str, Any]]:
        """Simulate speaker detection and segmentation."""
        speakers = text.count("Speaker")
        segments = []
        segment_duration = duration // max(speakers, 1)

        for i in range(max(speakers, 1)):
            segments.append(
                {
                    "speaker_id": f"speaker_{i+1}",
                    "start_seconds": i * segment_duration,
                    "end_seconds": (i + 1) * segment_duration,
                    "speaker_name": f"Speaker {i+1}",
                }
            )
        return segments

    def _detect_audio_types(self, text: str) -> List[str]:
        """Detect types of audio present."""
        types = []
        if text and len(text) > 100:
            types.append(AudioType.SPEECH.value)
        if "ambient" in text.lower() or "background" in text.lower():
            types.append(AudioType.BACKGROUND_NOISE.value)
        if not types:
            types.append(AudioType.SPEECH.value)
        return types

    def _generate_fingerprint(self, file_name: str) -> str:
        """Generate audio fingerprint for similarity detection."""
        import hashlib
        hash_val = hashlib.md5((file_name or "unknown").encode()).hexdigest()
        return hash_val[:32]

    def _extract_key_moments(self, text: str) -> List[Dict[str, Any]]:
        """Extract key moments from transcript."""
        moment_indicators = ["important", "critical", "must", "remember", "note", "key"]
        sentences = text.split(". ")
        moments = []

        for i, sent in enumerate(sentences):
            if any(ind in sent.lower() for ind in moment_indicators):
                moments.append(
                    {
                        "timestamp": {"start_seconds": i * 20},
                        "description": sent.strip(),
                        "importance": 0.8,
                    }
                )

        return moments[:5]  # Top 5 moments

    def _summarize_audio(self, text: str) -> str:
        """Summarize audio content."""
        # Simple: take first 150 chars
        if text:
            return text[:150] + "..."
        return "No transcription available"

    @staticmethod
    def _generate_uuid() -> str:
        """Generate UUID string."""
        import uuid
        return str(uuid.uuid4())
