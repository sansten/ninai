"""
PR-5: Temporal Reasoning Service

Time-aware analysis, sequence detection, trajectory forecasting, and optimal timing.
"""

from datetime import datetime, timedelta, timezone
import re
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.temporal_reasoning import (
    TemporalFact,
    TemporalSequence,
    TemporalTrajectory,
    TemporalChangetype,
    TrendDirection,
    PatternType,
)


class TemporalReasoningService:
    """
    Time-aware intelligence layer.
    
    Temporal reasoning enables:
    - Understanding when facts are valid
    - Detecting recurring event sequences
    - Analyzing trends and making forecasts
    - Identifying inflection points (when things change)
    - Determining optimal timing for actions
    """
    
    def __init__(self, session: Optional[AsyncSession] = None):
        self.session = session
    
    async def tag_facts_with_temporal_validity(
        self,
        org_id: str,
        fact_id: str,
        valid_from: datetime,
        valid_to: Optional[datetime] = None,
        change_type: str = "stable",
        confidence: float = 0.8,
    ) -> Dict[str, Any]:
        """
        Tag a fact with temporal validity interval.
        
        Args:
            org_id: Organization ID
            fact_id: The fact being tagged
            valid_from: When fact starts being true
            valid_to: When fact stops being true (None = still current)
            change_type: onset | offset | stable | transient
            confidence: How sure about the validity interval?
        
        Returns:
            Dictionary with the temporal fact metadata
        """
        temporal_fact = {
            "id": str(uuid4()),
            "organization_id": org_id,
            "fact_id": fact_id,
            "valid_from": valid_from.isoformat(),
            "valid_to": valid_to.isoformat() if valid_to else None,
            "confidence_at_time": confidence,
            "change_type": change_type,
            "created_at": datetime.utcnow().isoformat(),
        }
        
        return temporal_fact

    async def build_memory_timeline(
        self,
        *,
        query: str,
        memories: List[Dict[str, Any]],
        facts: Optional[List[Dict[str, Any]]] = None,
        extracted_entities: Optional[List[str]] = None,
        max_events: int = 12,
    ) -> Dict[str, Any]:
        """Build a timeline-oriented evidence view from retrieved memories."""
        entities = [str(item).strip() for item in (extracted_entities or []) if str(item).strip()]
        query_tokens = self._query_tokens(query)

        ranked_events: list[dict[str, Any]] = []
        for memory in memories:
            occurred_at = self._coerce_datetime(
                memory.get("occurred_at")
                or (memory.get("extra_metadata") or {}).get("event_time")
                or memory.get("created_at")
            )
            title = str(memory.get("title") or "").strip()
            preview = str(memory.get("content_preview") or memory.get("content") or "").strip()
            text = " ".join(part for part in [title, preview] if part)
            memory_entities = self._flatten_entities(memory.get("entities"))

            entity_overlap = sum(
                1 for entity in entities
                if entity.lower() in text.lower() or entity.lower() in {item.lower() for item in memory_entities}
            )
            token_overlap = sum(1 for token in query_tokens if token and token in text.lower())
            base_score = self._safe_float(memory.get("score"))
            recency_anchor = occurred_at.timestamp() if occurred_at else 0.0

            ranked_events.append(
                {
                    "memory_id": str(memory.get("id") or memory.get("memory_id") or "").strip(),
                    "occurred_at": occurred_at.isoformat() if occurred_at else None,
                    "title": title or None,
                    "content_preview": preview or None,
                    "entities": memory_entities,
                    "score": base_score,
                    "entity_overlap": entity_overlap,
                    "token_overlap": token_overlap,
                    "temporal_anchor": recency_anchor,
                }
            )

        if not ranked_events:
            return {
                "timeline": [],
                "memory_ids": [],
                "anchor_count": 0,
                "ordering": "chronological",
                "earliest": None,
                "latest": None,
            }

        ranked_events.sort(
            key=lambda item: (
                int(item["occurred_at"] is not None),
                item["entity_overlap"],
                item["token_overlap"],
                item["score"],
                item["temporal_anchor"],
            ),
            reverse=True,
        )

        selected = ranked_events[: max(1, int(max_events or 1))]
        selected.sort(
            key=lambda item: (
                item["occurred_at"] is None,
                item["occurred_at"] or "",
                -self._safe_float(item.get("score")),
            )
        )

        fact_map: dict[str, list[str]] = {}
        for fact in facts or []:
            source_memory_id = str(fact.get("source_memory_id") or "").strip()
            if not source_memory_id:
                continue
            fact_map.setdefault(source_memory_id, []).append(
                f"{fact.get('subject')} {fact.get('predicate')} {fact.get('object')}"
            )

        timeline: list[dict[str, Any]] = []
        for item in selected:
            memory_id = item["memory_id"]
            timeline.append(
                {
                    "memory_id": memory_id,
                    "occurred_at": item["occurred_at"],
                    "title": item["title"],
                    "content_preview": item["content_preview"],
                    "entities": item["entities"],
                    "score": round(self._safe_float(item["score"]), 4),
                    "fact_summaries": fact_map.get(memory_id, [])[:3],
                }
            )

        anchored = [item["occurred_at"] for item in timeline if item.get("occurred_at")]
        return {
            "timeline": timeline,
            "memory_ids": [item["memory_id"] for item in timeline if item.get("memory_id")],
            "anchor_count": len(anchored),
            "ordering": "chronological",
            "earliest": anchored[0] if anchored else None,
            "latest": anchored[-1] if anchored else None,
            "query_has_relative_temporal_language": self._has_relative_temporal_language(query),
        }
    
    async def detect_sequences(
        self,
        org_id: str,
        entity_timeline: List[Tuple[str, datetime]],
        min_occurrences: int = 3,
    ) -> List[Dict]:
        """
        Find recurring patterns in ordered event sequences.
        
        Example:
        - Events: [issue_reported, support_assigned, user_satisfied, 2_weeks_quiet, issue_reported]
        - Detected: "Issue → Support → Satisfaction → 2 weeks" (pattern)
        
        Args:
            org_id: Organization ID
            entity_timeline: List of (entity_id, timestamp) tuples
            min_occurrences: Minimum repetitions to count as pattern
        
        Returns:
            List of detected sequences with pattern metadata
        """
        if len(entity_timeline) < min_occurrences * 2:
            return []
        
        sequences = []
        
        # Compute temporal gaps
        entities = [e[0] for e in entity_timeline]
        timestamps = [e[1] for e in entity_timeline]
        temporal_gaps = []
        
        for i in range(len(timestamps) - 1):
            gap_seconds = int((timestamps[i + 1] - timestamps[i]).total_seconds())
            temporal_gaps.append(gap_seconds)
        
        # Classify pattern type
        if len(temporal_gaps) > 0:
            gap_values = np.array(temporal_gaps)
            
            if np.std(gap_values) < np.mean(gap_values) * 0.1:
                pattern_type = PatternType.CYCLE.value
            elif np.max(gap_values) > np.min(gap_values) * 10:
                pattern_type = PatternType.ESCALATION.value
            elif all(g < 1000 for g in temporal_gaps):  # fast sequence
                pattern_type = PatternType.RESOLUTION.value
            else:
                pattern_type = PatternType.TREND.value
            
            # Compute pattern strength (consistency)
            if len(gap_values) > 1:
                pattern_strength = 1.0 / (1.0 + np.std(gap_values) / (np.mean(gap_values) + 1e-6))
            else:
                pattern_strength = 0.5
            
            sequence = {
                "id": str(uuid4()),
                "organization_id": org_id,
                "sequence_type": "event_sequence",
                "entities": entities,
                "temporal_gaps": temporal_gaps,
                "pattern_type": pattern_type,
                "pattern_strength": float(pattern_strength),
                "last_observed_at": timestamps[-1].isoformat(),
                "observation_count": 1,
                "created_at": datetime.utcnow().isoformat(),
            }
            sequences.append(sequence)
        
        return sequences
    
    async def compute_trajectories(
        self,
        org_id: str,
        entity_id: str,
        quantity: str,
        measurements: List[Tuple[datetime, float]],
    ) -> Dict:
        """
        Analyze how a quantity changes over time.
        
        Uses exponential moving average (EMA) for trend, simple linear fit,
        and seasonal decomposition detection.
        
        Args:
            org_id: Organization ID
            entity_id: What entity are we tracking? (user, problem, metric)
            quantity: What are we measuring? (sentiment, memory_strength, completion_rate)
            measurements: List of (timestamp, value) tuples
        
        Returns:
            Dictionary with trajectory analysis including trend, inflection points
        """
        if len(measurements) < 2:
            return {}
        
        # Extract timestamps and values
        timestamps = np.array([m[0].timestamp() for m in measurements])
        values = np.array([m[1] for m in measurements])
        
        # Compute trend using simple linear regression
        if len(values) >= 3:
            x = np.arange(len(values))
            coeffs = np.polyfit(x, values, 1)
            slope = coeffs[0]
            
            # Determine trend direction
            if abs(slope) < 0.01:
                trend_direction = TrendDirection.STABLE.value
            elif slope > 0:
                trend_direction = TrendDirection.INCREASING.value
            else:
                trend_direction = TrendDirection.DECREASING.value
            
            # Compute trend strength (R-squared)
            y_pred = np.polyval(coeffs, x)
            ss_res = np.sum((values - y_pred) ** 2)
            ss_tot = np.sum((values - np.mean(values)) ** 2)
            trend_strength = 1.0 - (ss_res / (ss_tot + 1e-6)) if ss_tot > 0 else 0.5
        else:
            trend_direction = TrendDirection.STABLE.value
            trend_strength = 0.5
        
        # Detect inflection points (changes in trend)
        inflection_points = []
        if len(values) >= 4:
            diffs = np.diff(values)
            for i in range(1, len(diffs)):
                if diffs[i] * diffs[i - 1] < 0:  # Sign change
                    inflection_points.append(datetime.fromtimestamp(float(timestamps[i])).isoformat())
        
        # Simple EMA for forecasting
        alpha = 0.3
        ema = values[0]
        forecast = []
        for v in values[1:]:
            ema = alpha * v + (1 - alpha) * ema
            forecast.append(({
                "timestamp": datetime.fromtimestamp(float(timestamps[-1]) + (len(forecast) + 1) * 3600).isoformat(),
                "predicted_value": float(ema),
                "confidence": 0.7 - (len(forecast) * 0.05),  # Confidence decreases further ahead
            }))
        
        trajectory = {
            "id": str(uuid4()),
            "organization_id": org_id,
            "entity_id": entity_id,
            "quantity": quantity,
            "measurements": [{"timestamp": m[0].isoformat(), "value": m[1]} for m in measurements],
            "trend_direction": trend_direction,
            "trend_strength": float(trend_strength),
            "predicted_future": forecast[:7],  # Next 7 periods
            "inflection_points": inflection_points,
            "last_computed_at": datetime.utcnow().isoformat(),
            "created_at": datetime.utcnow().isoformat(),
        }
        
        return trajectory
    
    async def forecast_trajectory(
        self,
        trajectory: Dict,
        horizon_periods: int = 7,
    ) -> List[Dict]:
        """
        Forecast future values for a trajectory.
        
        Takes current trajectory and extends predictions into future.
        
        Args:
            trajectory: Existing trajectory with trend and seasonality
            horizon_periods: How many periods ahead to forecast?
        
        Returns:
            List of [{timestamp, predicted_value, confidence_interval}]
        """
        forecasts = []
        
        if not trajectory.get("measurements"):
            return forecasts
        
        measurements = trajectory["measurements"]
        values = np.array([m["value"] for m in measurements])
        
        # Simple exponential smoothing
        alpha = 0.3
        ema = values[-1]
        last_timestamp = datetime.fromisoformat(measurements[-1]["timestamp"])
        
        for period in range(1, horizon_periods + 1):
            future_time = last_timestamp + timedelta(hours=period * 24)
            
            # Compute next predicted value
            ema = alpha * values[-1] + (1 - alpha) * ema
            
            # Add some variance
            std_dev = float(np.std(values)) if len(values) > 1 else 0.1
            ci_lower = ema - 1.96 * std_dev
            ci_upper = ema + 1.96 * std_dev
            
            # Confidence decreases further ahead
            confidence = max(0.5, 0.9 - (period * 0.08))
            
            forecasts.append({
                "timestamp": future_time.isoformat(),
                "predicted_value": float(ema),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "confidence": confidence,
            })
        
        return forecasts
    
    async def detect_inflection_points(
        self,
        trajectory: Dict,
        threshold_std: float = 1.5,
    ) -> List[Dict]:
        """
        Identify significant changes in trajectory behavior.
        
        Uses statistical measures to flag when trend shifts direction or magnitude.
        
        Args:
            trajectory: Trajectory with historical measurements
            threshold_std: How many std devs to flag as inflection?
        
        Returns:
            List of inflection point events with timestamps and descriptions
        """
        inflections = []
        
        if not trajectory.get("measurements") or len(trajectory["measurements"]) < 4:
            return inflections
        
        measurements = trajectory["measurements"]
        values = np.array([m["value"] for m in measurements])
        timestamps = [datetime.fromisoformat(m["timestamp"]) for m in measurements]
        
        # Compute rolling statistics
        window = 3
        for i in range(window, len(values) - window):
            before = values[i - window:i]
            after = values[i:i + window]
            
            before_mean = np.mean(before)
            after_mean = np.mean(after)
            overall_std = np.std(values)
            
            change = abs(after_mean - before_mean)
            
            if change > threshold_std * overall_std:
                inflections.append({
                    "timestamp": timestamps[i].isoformat(),
                    "change_magnitude": float(change),
                    "direction": "increase" if after_mean > before_mean else "decrease",
                    "confidence": min(1.0, change / (threshold_std * overall_std + 1e-6)),
                })
        
        return inflections
    
    async def temporal_query(
        self,
        org_id: str,
        query_type: str,
        **kwargs,
    ) -> Any:
        """
        Execute temporal queries.
        
        Examples:
        - "facts_valid_at_time": {timestamp: datetime} → facts true at this time
        - "facts_updated_after": {after: datetime} → facts changed after this time
        - "trajectory_crosses_threshold": {trajectory_id, threshold, direction}
        
        Args:
            org_id: Organization ID
            query_type: Type of query to execute
            **kwargs: Query parameters
        
        Returns:
            Results matching the query criteria
        """
        results: Any = []
        
        if query_type == "facts_valid_at_time":
            target_time = kwargs.get("timestamp")
            target_dt = self._coerce_datetime(target_time)
            facts = list(kwargs.get("candidate_facts") or [])
            valid_facts: list[dict[str, Any]] = []
            for fact in facts:
                valid_from = self._coerce_datetime(fact.get("valid_from"))
                valid_to = self._coerce_datetime(fact.get("valid_to"))
                if target_dt is None:
                    continue
                if valid_from and valid_from > target_dt:
                    continue
                if valid_to and valid_to < target_dt:
                    continue
                valid_facts.append(dict(fact))
            results = valid_facts
        
        elif query_type == "facts_updated_after":
            after_time = kwargs.get("after")
            after_dt = self._coerce_datetime(after_time)
            facts = list(kwargs.get("candidate_facts") or [])
            updated_facts: list[dict[str, Any]] = []
            for fact in facts:
                fact_dt = self._coerce_datetime(fact.get("valid_from") or fact.get("created_at"))
                if after_dt is None or fact_dt is None:
                    continue
                if fact_dt > after_dt:
                    updated_facts.append(dict(fact))
            results = updated_facts
        
        elif query_type == "trajectory_crosses_threshold":
            threshold = kwargs.get("threshold")
            direction = kwargs.get("direction", "above")  # "above" or "below"
            # Would analyze trajectory and find crossing points
            results = [{"message": f"Trajectory crossing {direction} {threshold}", "crossings": []}]

        elif query_type == "timeline_of_memories":
            results = await self.build_memory_timeline(
                query=str(kwargs.get("query") or ""),
                memories=list(kwargs.get("candidate_memories") or []),
                facts=list(kwargs.get("candidate_facts") or []),
                extracted_entities=list(kwargs.get("extracted_entities") or []),
                max_events=int(kwargs.get("max_events") or 12),
            )
        
        return results
    
    async def when_should_act(
        self,
        org_id: str,
        goal_context: Dict,
        trajectory: Dict,
        action_lead_time_hours: int = 24,
    ) -> Optional[Dict]:
        """
        Estimate optimal time to act given goal and trajectory context.
        
        Example:
        - Goal: "Reduce memory overhead"
        - Trajectory: "Memory usage trending up, will hit limit in 10 days"
        - Recommendation: "Act now (24h lead time), before limit reached"
        
        Args:
            org_id: Organization ID
            goal_context: Context of goal (type, domain, urgency)
            trajectory: Current trajectory analysis
            action_lead_time_hours: How much advance notice do we need?
        
        Returns:
            Dictionary with recommended action time and reasoning
        """
        if not trajectory.get("predicted_future"):
            return None
        
        recommendations = trajectory.get("predicted_future", [])
        if not recommendations:
            return None
        
        # Simple heuristic: if trending toward problem, act when we have lead time
        trend_direction = trajectory.get("trend_direction")
        critical_value = goal_context.get("critical_threshold", 0.9)
        
        for forecast in recommendations:
            if forecast["predicted_value"] >= critical_value:
                # Found when we'd hit the threshold
                forecast_time = datetime.fromisoformat(forecast["timestamp"])
                action_time = forecast_time - timedelta(hours=action_lead_time_hours)
                
                return {
                    "recommended_action_time": action_time.isoformat(),
                    "reason": f"Action needed before threshold breach at {forecast_time}",
                    "current_trajectory": trend_direction,
                    "forecast_confidence": forecast.get("confidence", 0.7),
                    "window_hours": action_lead_time_hours,
                }
        
        return None

    @staticmethod
    def _coerce_datetime(value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return None
            if cleaned.endswith("Z"):
                cleaned = cleaned[:-1] + "+00:00"
            try:
                parsed = datetime.fromisoformat(cleaned)
            except ValueError:
                return None
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed
        return None

    @staticmethod
    def _flatten_entities(entities: Any) -> List[str]:
        if not isinstance(entities, dict):
            return []
        flattened: list[str] = []
        for values in entities.values():
            if isinstance(values, list):
                flattened.extend(str(item).strip() for item in values if str(item).strip())
            elif values is not None:
                text = str(values).strip()
                if text:
                    flattened.append(text)
        return list(dict.fromkeys(flattened))

    @staticmethod
    def _query_tokens(query: str) -> List[str]:
        return re.findall(r"\b[a-z0-9_]{3,}\b", str(query or "").lower())

    @staticmethod
    def _has_relative_temporal_language(query: str) -> bool:
        lowered = str(query or "").lower()
        cues = (
            "before",
            "after",
            "latest",
            "last",
            "first",
            "earlier",
            "later",
            "previous",
            "next",
            "when",
        )
        return any(cue in lowered for cue in cues)

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0
