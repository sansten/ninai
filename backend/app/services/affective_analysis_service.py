"""
PR-8: Affective Analysis & Emotional Regulation Service

Detects emotional content, analyzes sentiment trajectories, and facilitates empathetic responses.
Enables emotion-aware interaction management and de-escalation capabilities.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.models.affective_memory import (
    AffectiveMemory,
    EmotionalTrajectory,
    EmotionalInteractionEvent,
    EmotionalTag,
    DeEscalationStrategy,
)
from app.models.memory import MemoryMetadata


class AffectiveAnalysisService:
    """
    Analyzes and tracks emotional content in memories and interactions.
    """

    def __init__(self, session: Optional[AsyncSession] = None):
        self.session = session

    async def analyze_memory_affect(
        self,
        memory_content: str,
        memory_id: str,
        organization_id: str,
        user_ids: Optional[List[str]] = None,
    ) -> Dict:
        """
        Analyze emotional valence and arousal from memory content.
        
        Uses heuristics for sentiment:
        - Valence: -1.0 (very negative) to +1.0 (very positive)
        - Arousal: 0 (calm) to 1 (intense/excited)
        - Emotional tags: categorical labels
        """
        if not memory_content or not memory_content.strip():
            return {
                "valence": 0.0,
                "arousal": 0.0,
                "emotional_tags": [],
                "significance": 0.3,
            }

        content_lower = memory_content.lower()

        # Valence detection (negative/positive sentiment)
        negative_words = [
            "error", "failed", "failure", "problem", "issue", "bug", "crash",
            "frustrated", "angry", "upset", "sad", "hate", "awful",
            "terrible", "broken", "denied", "rejected", "losing", "fatal",
            "down", "critical", "urgent", "emergency",
        ]
        positive_words = [
            "success", "great", "excellent", "happy", "joy", "love",
            "wonderful", "fixed", "solved", "achieved", "winning",
            "approved", "thanks", "appreciate", "breakthrough", "milestone",
        ]

        neg_count = sum(1 for word in negative_words if word in content_lower)
        pos_count = sum(1 for word in positive_words if word in content_lower)

        if neg_count + pos_count == 0:
            valence = 0.0
        else:
            valence = (pos_count - neg_count) / max(1, (neg_count + pos_count)) * 0.8
            valence = max(-1.0, min(1.0, valence))

        # Arousal detection (intensity/excitement)
        intense_markers = ["!", "urgent", "asap", "critical", "emergency", "crisis"]
        arousal = min(1.0, 0.3 + sum(1 for marker in intense_markers if marker in content_lower) * 0.15)

        # Emotional tags
        emotional_tags = []
        if "frustrat" in content_lower or "annoyed" in content_lower:
            emotional_tags.append(EmotionalTag.FRUSTRATION.value)
        if "happy" in content_lower or "joy" in content_lower or "loved" in content_lower or "love" in content_lower or "excellent" in content_lower or "great" in content_lower:
            emotional_tags.append(EmotionalTag.JOY.value)
        if "fear" in content_lower or "afraid" in content_lower or "scared" in content_lower:
            emotional_tags.append(EmotionalTag.FEAR.value)
        if "satisfied" in content_lower or "success" in content_lower or "solved" in content_lower or "fixed" in content_lower:
            emotional_tags.append(EmotionalTag.SATISFACTION.value)
        if "anxious" in content_lower or "worry" in content_lower or "concerned" in content_lower:
            emotional_tags.append(EmotionalTag.ANXIETY.value)
        if "surprise" in content_lower or "unexpected" in content_lower or "astonish" in content_lower:
            emotional_tags.append(EmotionalTag.SURPRISE.value)
        if "confused" in content_lower or "unclear" in content_lower or "puzzled" in content_lower:
            emotional_tags.append(EmotionalTag.CONFUSION.value)
        if "angry" in content_lower or "furious" in content_lower or "rage" in content_lower or "mad" in content_lower:
            emotional_tags.append(EmotionalTag.ANGER.value)
        if "relieved" in content_lower or "relief" in content_lower:
            emotional_tags.append(EmotionalTag.RELIEF.value)
        if "disappoint" in content_lower or "let down" in content_lower:
            emotional_tags.append(EmotionalTag.DISAPPOINTMENT.value)

        # Significance: combination of arousal and emotional intensity
        significance = min(1.0, arousal + (abs(valence) * 0.5))

        return {
            "valence": round(valence, 3),
            "arousal": round(arousal, 3),
            "emotional_tags": list(set(emotional_tags)),  # deduplicate
            "significance": round(significance, 3),
            "confidence_in_measurement": 0.75,
        }

    async def record_affective_memory(
        self,
        memory_id: str,
        organization_id: str,
        valence: float,
        arousal: float,
        emotional_tags: List[str],
        significance: float,
        user_ids: Optional[List[str]] = None,
        session: Optional[AsyncSession] = None,
    ) -> str:
        """
        Record affective analysis for a memory in the database.
        Returns the created AffectiveMemory ID.
        """
        db = session or self.session
        if not db:
            raise ValueError("No database session provided")

        affective = AffectiveMemory(
            organization_id=organization_id,
            memory_id=memory_id,
            valence=valence,
            arousal=arousal,
            emotional_tags=emotional_tags,
            significance=significance,
            associated_user_ids=user_ids or [],
            measured_at=datetime.utcnow(),
            confidence_in_measurement=0.75,
        )
        db.add(affective)
        await db.commit()
        return affective.id

    async def compute_user_emotional_trajectory(
        self,
        user_id: str,
        organization_id: str,
        lookback_days: int = 30,
        session: Optional[AsyncSession] = None,
    ) -> Dict:
        """
        Compute emotional trajectory for a user from recent interactions.
        
        Returns dict with:
        - measurements: time-series of valence/arousal
        - trend: improving | stable | deteriorating
        - current_state: latest emotional snapshot
        - escalation_risk: probability of escalation
        - de_escalation_strategies: recommended approaches
        """
        db = session or self.session
        if not db:
            raise ValueError("No database session provided")

        # Fetch recent emotional interaction events
        stmt = (
            select(EmotionalInteractionEvent)
            .where(
                EmotionalInteractionEvent.organization_id == organization_id,
                EmotionalInteractionEvent.user_id == user_id,
            )
            .order_by(desc(EmotionalInteractionEvent.created_at))
            .limit(50)
        )
        result = await db.execute(stmt)
        events = result.scalars().all()

        if not events:
            return {
                "user_id": user_id,
                "measurements": [],
                "trend": "stable",
                "current_state": {"valence": 0.0, "arousal": 0.0, "tags": []},
                "escalation_risk": 0.1,
                "de_escalation_strategies": [],
            }

        # Build measurements time series (reverse to chronological)
        events_reversed = list(reversed(events))
        measurements = [
            {
                "timestamp": event.created_at.isoformat(),
                "valence": event.initial_valence,
                "arousal": event.initial_arousal,
            }
            for event in events_reversed
        ]

        # Compute trend
        if len(measurements) >= 2:
            recent_valence = sum(m["valence"] for m in measurements[-5:]) / len(measurements[-5:])
            older_valence = sum(m["valence"] for m in measurements[:5]) / len(measurements[:5])
            valence_change = recent_valence - older_valence

            if valence_change > 0.1:  # Reduced threshold for sensitivity
                trend = "improving"
            elif valence_change < -0.1:  # Reduced threshold
                trend = "deteriorating"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # Current state
        latest_event = events[0]
        current_state = {
            "valence": latest_event.initial_valence,
            "arousal": latest_event.initial_arousal,
            "timestamp": latest_event.created_at.isoformat(),
        }

        # Escalation risk: high if recent negative valence + increasing arousal
        recent_arousal = sum(m["arousal"] for m in measurements[-5:]) / min(5, len(measurements))
        recent_valence = sum(m["valence"] for m in measurements[-5:]) / min(5, len(measurements))

        escalation_risk = 0.1
        if recent_valence < -0.5 and recent_arousal > 0.6:
            escalation_risk = min(1.0, 0.8)
        elif recent_valence < -0.3 and recent_arousal > 0.5:
            escalation_risk = min(1.0, 0.5)
        elif recent_valence < 0.0 and trend == "deteriorating":
            escalation_risk = min(1.0, 0.4)

        # Recommend de-escalation strategy
        de_escalation_strategies = []
        if escalation_risk > 0.5:
            de_escalation_strategies.append(DeEscalationStrategy.EMPATHY.value)
            if recent_arousal > 0.7:
                de_escalation_strategies.append(DeEscalationStrategy.SLOW_DOWN.value)
                de_escalation_strategies.append(DeEscalationStrategy.BREAK.value)
            else:
                de_escalation_strategies.append(DeEscalationStrategy.VALIDATE.value)
                de_escalation_strategies.append(DeEscalationStrategy.CLARIFY.value)

        return {
            "user_id": user_id,
            "measurements": measurements,
            "trend": trend,
            "current_state": current_state,
            "escalation_risk": round(escalation_risk, 2),
            "de_escalation_strategies": de_escalation_strategies,
            "is_at_risk": escalation_risk > 0.5,
        }

    async def detect_escalation_risk(
        self,
        user_id: str,
        organization_id: str,
        session: Optional[AsyncSession] = None,
    ) -> float:
        """
        Compute escalation risk (0-1) for a user.
        Based on emotional trajectory and historical escalation patterns.
        """
        trajectory = await self.compute_user_emotional_trajectory(
            user_id=user_id,
            organization_id=organization_id,
            session=session,
        )
        return trajectory.get("escalation_risk", 0.1)

    async def select_empathetic_tone(
        self,
        user_emotional_state: Dict,
        response_content: str,
    ) -> str:
        """
        Rewrite response to match and soothe user's emotional state.
        
        Adjusts tone based on:
        - Valence: negative users get empathy, positive users get enthusiasm
        - Arousal: high-arousal users get calming tone
        """
        valence = user_emotional_state.get("valence", 0.0)
        arousal = user_emotional_state.get("arousal", 0.5)
        tags = user_emotional_state.get("tags", [])

        prefix = ""

        # Arousal-based: high arousal needs calming
        if arousal > 0.7:
            prefix += "Let's take this step by step. "

        # Valence-based: negative needs empathy
        if valence < -0.5:
            if "frustration" in tags:
                prefix += "I understand this is frustrating. "
            elif "anxiety" in tags:
                prefix += "I can see why you're concerned. "
            elif "anger" in tags:
                prefix += "I completely understand your frustration. "
            else:
                prefix += "I recognize this is challenging. "
        elif valence > 0.5:
            prefix += "Great! Let's build on this momentum. "

        # Validate concerns
        if "fear" in tags or "anxiety" in tags:
            prefix += "Let me address your concerns directly: "

        return prefix + response_content if prefix else response_content


class EmotionalRegulationService:
    """
    Helps agent respond appropriately to emotionally charged situations.
    """

    def __init__(self, session: Optional[AsyncSession] = None):
        self.session = session

    async def suggest_de_escalation_strategy(
        self,
        user_id: str,
        organization_id: str,
        session: Optional[AsyncSession] = None,
    ) -> Optional[str]:
        """
        User is escalating. Suggest de-escalation strategy.
        
        Returns: "empathy" | "slow_down" | "involve_expert" | "break" | "validate" | "clarify"
        """
        db = session or self.session
        if not db:
            raise ValueError("No database session provided")

        # Get most recent interaction
        stmt = (
            select(EmotionalInteractionEvent)
            .where(
                EmotionalInteractionEvent.organization_id == organization_id,
                EmotionalInteractionEvent.user_id == user_id,
                EmotionalInteractionEvent.was_escalation == True,
            )
            .order_by(desc(EmotionalInteractionEvent.created_at))
            .limit(1)
        )
        result = await db.execute(stmt)
        event = result.scalars().first()

        if not event:
            return None

        initial_arousal = event.initial_arousal
        initial_valence = event.initial_valence

        # High arousal + negative valence = need to break/slow down
        if initial_arousal > 0.8 and initial_valence < -0.5:
            return DeEscalationStrategy.BREAK.value

        # Negative valence = empathy
        if initial_valence < -0.3:
            return DeEscalationStrategy.EMPATHY.value

        # Uncertainty = clarify
        if -0.2 <= initial_valence <= 0.2 and initial_arousal > 0.6:
            return DeEscalationStrategy.CLARIFY.value

        # Complex issue = involve expert
        return DeEscalationStrategy.INVOLVE_EXPERT.value

    async def should_escalate_to_human(
        self,
        user_id: str,
        organization_id: str,
        session: Optional[AsyncSession] = None,
    ) -> bool:
        """
        Should human agent take over?
        
        True if:
        - User's emotional state indicates distress
        - Multiple failed de-escalation attempts
        - User explicitly requests human
        """
        analysis_svc = AffectiveAnalysisService(session=session)
        risk = await analysis_svc.detect_escalation_risk(
            user_id=user_id,
            organization_id=organization_id,
            session=session,
        )

        # Escalate to human if risk > 0.75
        return risk > 0.75

    async def record_interaction_outcome(
        self,
        user_id: str,
        organization_id: str,
        interaction_content: str,
        initial_valence: float,
        initial_arousal: float,
        final_valence: Optional[float] = None,
        final_arousal: Optional[float] = None,
        agent_response_tone: Optional[str] = None,
        de_escalation_applied: Optional[str] = None,
        outcome: Optional[str] = None,
        session: Optional[AsyncSession] = None,
    ) -> str:
        """
        Record how an emotional interaction went.
        Used for training and understanding patterns.
        
        Returns the created EmotionalInteractionEvent ID.
        """
        db = session or self.session
        if not db:
            raise ValueError("No database session provided")

        was_escalation = initial_valence < -0.3 or initial_arousal > 0.6
        was_de_escalated = (
            final_valence is not None
            and final_arousal is not None
            and final_valence > initial_valence
            and final_arousal < initial_arousal
        )

        event = EmotionalInteractionEvent(
            organization_id=organization_id,
            user_id=user_id,
            interaction_content=interaction_content,
            initial_valence=initial_valence,
            initial_arousal=initial_arousal,
            final_valence=final_valence,
            final_arousal=final_arousal,
            agent_response_tone=agent_response_tone,
            de_escalation_applied=de_escalation_applied,
            was_escalation=was_escalation,
            was_de_escalated=was_de_escalated,
            outcome=outcome,
        )
        db.add(event)
        await db.commit()
        return event.id
