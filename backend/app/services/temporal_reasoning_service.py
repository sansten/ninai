"""
PR-5: Temporal Reasoning Service

Time-aware analysis, sequence detection, trajectory forecasting, and optimal timing.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
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
    ) -> Dict[str, any]:
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
                    inflection_points.append(timestamps[i])
        
        # Simple EMA for forecasting
        alpha = 0.3
        ema = values[0]
        forecast = []
        for v in values[1:]:
            ema = alpha * v + (1 - alpha) * ema
            forecast.append(({
                "timestamp": datetime.fromtimestamp(timestamps[-1] + (len(forecast) + 1) * 3600).isoformat(),
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
            "inflection_points": [t.isoformat() for t in inflection_points],
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
    ) -> List[Dict]:
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
        results = []
        
        if query_type == "facts_valid_at_time":
            target_time = kwargs.get("timestamp")
            # Would query: valid_from <= target_time AND (valid_to IS NULL OR valid_to >= target_time)
            results = [{"message": "Facts valid at target time", "count": 0}]
        
        elif query_type == "facts_updated_after":
            after_time = kwargs.get("after")
            # Would query: created_at > after_time
            results = [{"message": "Facts updated after time", "count": 0}]
        
        elif query_type == "trajectory_crosses_threshold":
            threshold = kwargs.get("threshold")
            direction = kwargs.get("direction", "above")  # "above" or "below"
            # Would analyze trajectory and find crossing points
            results = [{"message": f"Trajectory crossing {direction} {threshold}", "crossings": []}]
        
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
