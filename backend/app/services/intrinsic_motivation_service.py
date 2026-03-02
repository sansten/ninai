"""
Intrinsic Motivation Service

Service for autonomous goal generation based on intrinsic motivation.
Detects knowledge gaps, generates curiosity goals, predicts user needs,
and supports continuous self-improvement through goal-based learning.
"""

import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import math
import random

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    AutonomousGoal,
    KnowledgeGap,
    AutonomousGoalOutcome,
)


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
                    "gap_id": str(gap["id"]),
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
        """
        Predict user needs based on historical patterns.
        
        Analyzes:
        - Frequency of user requests by domain
        - Seasonal patterns
        - Escalation patterns
        
        Generates anticipatory goals like:
        "User typically asks about billing every 3 months; they're due soon"
        
        Args:
            org_id: Organization ID
            lookback_days: Days of history to analyze
        
        Returns:
            List of predictive goals
        """
        goals = []
        
        # In a real implementation, this would:
        # 1. Query audit logs for user requests in lookback period
        # 2. Group by domain and calculate frequency
        # 3. Detect patterns and seasonality
        # 4. Generate proactive goals for predicted next need
        
        # For demo purposes, return a few example predictive goals
        prediction_domains = [
            {"domain": "billing", "frequency_days": 90, "confidence": 0.75},
            {"domain": "feature_usage", "frequency_days": 14, "confidence": 0.65},
            {"domain": "account_health", "frequency_days": 30, "confidence": 0.70},
        ]
        
        cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)
        
        for domain_pred in prediction_domains:
            goal = {
                "id": str(uuid.uuid4()),
                "organization_id": org_id,
                "initiator": "prediction",
                "title": f"Prepare: Likely user request about {domain_pred['domain']}",
                "description": (
                    f"Based on {lookback_days}-day history, user typically asks about "
                    f"{domain_pred['domain']} every {domain_pred['frequency_days']} days. "
                    f"Next request likely soon."
                ),
                "trigger_memory_ids": [],
                "expected_value": 0.5 + (domain_pred["confidence"] * 0.35),
                "urgency": 0.5,
                "confidence": domain_pred["confidence"],
                "status": "proposed",
                "created_at": datetime.utcnow(),
                "metadata": {
                    "prediction_type": "user_need",
                    "domain": domain_pred["domain"],
                    "expected_frequency_days": domain_pred["frequency_days"],
                },
            }
            goals.append(goal)
        
        return goals
    
    async def detect_self_improvement_needs(
        self,
        org_id: str,
        tool_success_threshold: float = 0.6
    ) -> List[Dict]:
        """
        Identify areas where agent should improve its capabilities.
        
        Looks for:
        - Tools with low success rates
        - Domains with poor understanding
        - Skills with declining performance
        
        Args:
            org_id: Organization ID
            tool_success_threshold: Min success rate to avoid improvement goal
        
        Returns:
            List of self-improvement goals
        """
        goals = []
        
        # In a real implementation, this would:
        # 1. Query tool_call_logs for success rates per tool
        # 2. Compare against threshold
        # 3. Generate improvement goals for low-performing tools
        
        # For demo purposes, return example self-improvement goals
        improvement_areas = [
            {
                "area": "customer_data_extraction",
                "current_success_rate": 0.55,
                "target_rate": 0.80,
                "confidence": 0.8,
            },
            {
                "area": "sentiment_analysis",
                "current_success_rate": 0.68,
                "target_rate": 0.90,
                "confidence": 0.70,
            },
        ]
        
        for area in improvement_areas:
            if area["current_success_rate"] < tool_success_threshold:
                goal = {
                    "id": str(uuid.uuid4()),
                    "organization_id": org_id,
                    "initiator": "self_improvement",
                    "title": f"Improve: {area['area']} capability",
                    "description": (
                        f"Tool '{area['area']}' has {area['current_success_rate']:.0%} success rate. "
                        f"Target: {area['target_rate']:.0%}. Need to study patterns and edge cases."
                    ),
                    "trigger_memory_ids": [],
                    "expected_value": 0.7,
                    "urgency": 0.7,
                    "confidence": area["confidence"],
                    "status": "proposed",
                    "created_at": datetime.utcnow(),
                    "metadata": {
                        "improvement_type": "tool_reliability",
                        "target_area": area["area"],
                        "current_success_rate": area["current_success_rate"],
                        "target_success_rate": area["target_rate"],
                    },
                }
                goals.append(goal)
        
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
            target_rate = goal.get("metadata", {}).get("target_success_rate", 0.8)
            current_rate = goal.get("metadata", {}).get("current_success_rate", 0.5)
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
        
        return {
            "total_outcomes": len(outcomes),
            "valuable_rate": valuable_count / len(outcomes) if outcomes else 0.0,
            "user_expectation_alignment": user_expecting_count / len(outcomes) if outcomes else 0.0,
            "average_outcome_days": sum(
                (o.created_at - o.created_at).days for o in outcomes
            ) / len(outcomes) if outcomes else 0,
        }
