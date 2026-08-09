"""
Intrinsic Motivation Service

Service for autonomous goal generation based on intrinsic motivation.
Detects knowledge gaps, generates curiosity goals, predicts user needs,
and supports continuous self-improvement through goal-based learning.
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional

from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    AutonomousGoal,
    KnowledgeGap,
    AutonomousGoalOutcome,
)
from app.models.audit import AuditEvent
from app.models.self_model import SelfModelProfile


class IntrinsicMotivationService:
    """
    Service for autonomous goal generation and intrinsic motivation.
    
    Implements methods to:
    - Detect knowledge gaps in agent's beliefs
    - Generate curiosity-driven goals
    - Predict user needs proactively
    - Identify self-improvement opportunities
    - Estimate goal value
    - Track goal outcomes for continuous learning
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def detect_knowledge_gaps(
        self,
        org_id: str,
        min_confidence: float = 0.5
    ) -> List[Dict]:
        """
        Detect gaps in agent's knowledge.
        
        Types of gaps:
        - missing_fact: beliefs needed but not present
        - contradiction: conflicting beliefs exist
        - outdated: old beliefs beyond TTL
        - low_confidence: beliefs with low confidence scores
        
        Args:
            org_id: Organization ID
            min_confidence: Minimum confidence threshold for gaps
        
        Returns:
            List of detected gaps with metadata
        """
        gaps = []
        
        # Get existing knowledge gaps that haven't been resolved
        stmt = select(KnowledgeGap).where(
            KnowledgeGap.organization_id == org_id,
            KnowledgeGap.resolved_at == None
        )
        result = await self.session.execute(stmt)
        existing_gaps = result.scalars().all()
        
        # Convert to dicts
        gaps = [
            {
                "id": gap.id,
                "gap_type": gap.gap_type,
                "domain": gap.domain,
                "description": gap.description,
                "confidence_in_gap": gap.confidence_in_gap,
                "related_memories": gap.related_memories,
                "suggested_learning_approach": gap.suggested_learning_approach,
                "discovered_at": gap.discovered_at,
            }
            for gap in existing_gaps
            if float(getattr(gap, "confidence_in_gap", 0.0) or 0.0) >= float(min_confidence)
        ]
        
        return gaps
    
    async def generate_curiosity_goals(
        self,
        org_id: str,
        knowledge_gaps: List[Dict]
    ) -> List[Dict]:
        """
        Convert knowledge gaps into curiosity-driven goals.
        
        Each gap becomes a goal with:
        - initiator: "curiosity"
        - description: derived from gap description
        - trigger_memory_ids: from gap.related_memories
        - expected_value: based on gap confidence and domain importance
        
        Args:
            org_id: Organization ID
            knowledge_gaps: List of knowledge gaps
        
        Returns:
            List of autonomous goal dicts with curiosity initiator
        """
        goals = []
        
        for gap in knowledge_gaps:
            # Convert gap to goal
            goal = {
                "id": str(uuid.uuid4()),
                "organization_id": org_id,
                "initiator": "curiosity",
                "title": f"Investigate: {gap['gap_type']} in {gap['domain']}",
                "description": f"Knowledge gap detected: {gap['description']}",
                "trigger_memory_ids": gap.get("related_memories", []),
                "expected_value": min(0.9, max(0.3, gap["confidence_in_gap"] * 0.9)),
                "urgency": 0.6,
                "confidence": gap["confidence_in_gap"],
                "status": "proposed",
                "created_at": datetime.utcnow(),
                "metadata": {
                    "gap_id": str(gap.get("id") or ""),
                    "original_gap_type": gap["gap_type"],
                    "suggested_approach": gap.get("suggested_learning_approach"),
                },
            }
            goals.append(goal)
        
        return goals
    
    async def predict_user_needs(
        self,
        org_id: str,
        lookback_days: int = 90
    ) -> List[Dict]:
        """Predict user needs by analyzing AuditEvent frequency per resource_type.

        Queries the real audit log for the org, groups events by resource_type,
        and generates anticipatory goals for the most-queried domains so the agent
        can prepare before the next request arrives.
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=lookback_days)

        # Count audit events by resource_type in the lookback window
        stmt = (
            select(
                AuditEvent.resource_type,
                func.count().label("event_count"),
            )
            .where(
                and_(
                    AuditEvent.organization_id == org_id,
                    AuditEvent.timestamp >= cutoff_date,
                    AuditEvent.resource_type.isnot(None),
                )
            )
            .group_by(AuditEvent.resource_type)
            .order_by(func.count().desc())
            .limit(5)
        )
        result = await self.session.execute(stmt)
        domain_counts = result.all()

        if not domain_counts:
            return []

        total_events = sum(row.event_count for row in domain_counts)
        goals = []

        for row in domain_counts:
            domain = row.resource_type
            count = row.event_count
            # Confidence scales with how dominant this domain is
            frequency_ratio = count / total_events if total_events > 0 else 0.0
            confidence = min(0.90, 0.40 + frequency_ratio * 0.50)
            # Estimated recurrence period in days (heuristic: lookback / count)
            expected_frequency_days = max(1, lookback_days // count)

            goal = {
                "id": str(uuid.uuid4()),
                "organization_id": org_id,
                "initiator": "prediction",
                "title": f"Prepare: Likely user request about {domain}",
                "description": (
                    f"Audit history ({lookback_days}d): {count} events in '{domain}'. "
                    f"Estimated recurrence every ~{expected_frequency_days} days. "
                    f"Pre-loading context may reduce latency."
                ),
                "trigger_memory_ids": [],
                "expected_value": 0.4 + confidence * 0.4,
                "urgency": frequency_ratio,
                "confidence": confidence,
                "status": "proposed",
                "created_at": datetime.now(timezone.utc),
                "metadata": {
                    "prediction_type": "user_need",
                    "domain": domain,
                    "event_count_lookback": count,
                    "expected_frequency_days": expected_frequency_days,
                },
            }
            goals.append(goal)

        return goals
    
    async def detect_self_improvement_needs(
        self,
        org_id: str,
        tool_success_threshold: float = 0.6
    ) -> List[Dict]:
        """Identify underperforming tools by reading the real SelfModelProfile.

        SelfModelProfile.tool_reliability is a JSONB dict keyed by tool_name,
        each entry containing EMA-smoothed success_rate_30d computed by
        SelfModelService. Tools whose success_rate_30d is below threshold become
        self-improvement goals.
        """
        stmt = select(SelfModelProfile).where(
            SelfModelProfile.organization_id == org_id
        )
        result = await self.session.execute(stmt)
        profile = result.scalar_one_or_none()

        if not profile or not profile.tool_reliability:
            return []

        goals = []
        for tool_name, reliability in profile.tool_reliability.items():
            success_rate = reliability.get("success_rate_30d", 1.0)
            sample_size = reliability.get("sample_size_30d", 0)
            if sample_size < 3:
                continue  # too few samples to be confident
            if success_rate >= tool_success_threshold:
                continue

            target_rate = min(1.0, success_rate + 0.25)
            # Confidence is higher when we have more samples
            confidence = min(0.95, 0.50 + sample_size / 100.0)

            goal = {
                "id": str(uuid.uuid4()),
                "organization_id": org_id,
                "initiator": "self_improvement",
                "title": f"Improve: {tool_name} reliability",
                "description": (
                    f"Tool '{tool_name}' has {success_rate:.0%} 30-day success rate "
                    f"({sample_size} calls). Target: {target_rate:.0%}. "
                    f"Investigate failure patterns and add guardrails."
                ),
                "trigger_memory_ids": [],
                "expected_value": min(1.0, (target_rate - success_rate) * 1.5),
                "urgency": 1.0 - success_rate,
                "confidence": confidence,
                "status": "proposed",
                "created_at": datetime.now(timezone.utc),
                "metadata": {
                    "improvement_type": "tool_reliability",
                    "target_area": tool_name,
                    "current_success_rate": success_rate,
                    "target_success_rate": target_rate,
                    "sample_size_30d": sample_size,
                },
            }
            goals.append(goal)

        # Sort by urgency (lowest success rate first)
        goals.sort(key=lambda g: g["urgency"], reverse=True)
        return goals
    
    async def estimate_goal_value(
        self,
        goal: Dict,
        importance_weight: float = 0.4,
        impact_weight: float = 0.4,
        effort_weight: float = 0.2
    ) -> float:
        """
        Estimate the value of a goal for prioritization.
        
        Formula: value = (importance × impact) / effort
        
        Args:
            goal: Goal dict with metadata
            importance_weight: Weight for importance (0-1)
            impact_weight: Weight for impact (0-1)
            effort_weight: Weight for effort (0-1)
        
        Returns:
            Value score (0-1)
        """
        # Base value from goal fields
        base_value = goal.get("expected_value", 0.5)
        
        # Extract metadata indicators
        if goal["initiator"] == "curiosity":
            importance = goal.get("confidence", 0.7)  # Gap confidence → importance
            impact = 0.6  # Moderate impact from filling gap
            effort = 0.7  # Takes time to learn
        
        elif goal["initiator"] == "prediction":
            importance = 0.7  # Proactive user service is important
            impact = 0.65  # Better user experience
            effort = 0.5  # Usually straightforward prep
        
        elif goal["initiator"] == "self_improvement":
            # Higher urgency for tools that fail often
            meta = goal.get("metadata")
            if not isinstance(meta, dict):
                meta = goal.get("meta")
            if not isinstance(meta, dict):
                meta = {}
            target_rate = meta.get("target_success_rate", 0.8)
            current_rate = meta.get("current_success_rate", 0.5)
            importance = min(1.0, 1.0 - current_rate)  # Lower success = higher importance
            impact = target_rate - current_rate  # Potential improvement
            effort = 0.6  # Moderate learning curve
        
        else:  # "priority_rebalance"
            importance = 0.5
            impact = 0.5
            effort = 0.4
        
        # Combine using weighted formula
        effort_adjustment = max(0.1, effort)  # Avoid division by zero
        value = (importance * importance_weight + 
                impact * impact_weight) / effort_adjustment
        
        # Normalize to 0-1
        value = min(1.0, max(0.0, value * 0.5))  # Scale to reasonable range
        
        return value
    
    async def record_goal_outcome(
        self,
        org_id: str,
        goal_id: str,
        outcome_type: str,
        impact_description: str = "",
        feedback_from: str = "auto_detection",
        was_user_expecting: bool = False
    ) -> Dict:
        """
        Record whether an autonomous goal was valuable.
        
        This feedback is used to improve future goal generation.
        
        Args:
            org_id: Organization ID
            goal_id: Goal ID
            outcome_type: "valuable" | "not_valuable" | "premature"
            impact_description: Description of impact
            feedback_from: Source of feedback
            was_user_expecting: Whether user also wanted this
        
        Returns:
            Recorded outcome dict
        """
        outcome = AutonomousGoalOutcome(
            organization_id=org_id,
            goal_id=goal_id,
            outcome_type=outcome_type,
            impact_description=impact_description,
            feedback_from=feedback_from,
            was_user_expecting=was_user_expecting,
        )
        
        self.session.add(outcome)
        await self.session.commit()
        
        return {
            "id": outcome.id,
            "goal_id": goal_id,
            "outcome_type": outcome_type,
            "created_at": outcome.created_at,
        }
    
    async def get_goal_success_metrics(
        self,
        org_id: str
    ) -> Dict:
        """
        Calculate success metrics for autonomous goal generation.
        
        Returns stats like:
        - Total goals generated
        - Percentage completed as valuable
        - User expectation alignment
        
        Args:
            org_id: Organization ID
        
        Returns:
            Dictionary with metrics
        """
        # Query outcomes
        stmt = select(AutonomousGoalOutcome).where(
            AutonomousGoalOutcome.organization_id == org_id
        )
        result = await self.session.execute(stmt)
        outcomes = result.scalars().all()
        
        if not outcomes:
            return {
                "total_outcomes": 0,
                "valuable_rate": 0.0,
                "user_expectation_alignment": 0.0,
                "average_outcome_days": 0,
            }
        
        valuable_count = sum(1 for o in outcomes if o.outcome_type == "valuable")
        user_expecting_count = sum(1 for o in outcomes if o.was_user_expecting)

        goal_ids = [
            str(o.goal_id) for o in outcomes
            if getattr(o, "goal_id", None)
        ]
        goals_by_id = {}
        if goal_ids:
            goal_stmt = select(AutonomousGoal).where(
                AutonomousGoal.organization_id == org_id,
                AutonomousGoal.id.in_(goal_ids),
            )
            goal_result = await self.session.execute(goal_stmt)
            goals_by_id = {
                str(goal.id): goal
                for goal in goal_result.scalars().all()
            }

        durations_days = []
        for outcome in outcomes:
            goal = goals_by_id.get(str(getattr(outcome, "goal_id", "") or ""))
            outcome_created = getattr(outcome, "created_at", None)
            goal_created = getattr(goal, "created_at", None) if goal is not None else None
            if goal_created is None or outcome_created is None:
                continue
            durations_days.append(max((outcome_created - goal_created).days, 0))
        
        return {
            "total_outcomes": len(outcomes),
            "valuable_rate": valuable_count / len(outcomes) if outcomes else 0.0,
            "user_expectation_alignment": user_expecting_count / len(outcomes) if outcomes else 0.0,
            "average_outcome_days": (
                sum(durations_days) / len(durations_days) if durations_days else 0
            ),
        }
