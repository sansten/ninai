"""
Adaptive Strategy Service - PR-4

Implements tool capability learning and adaptive strategy selection.
Enables the agent to learn from tool usage patterns and optimize tool selection.
"""

from typing import List, Dict, Optional
from datetime import datetime, timedelta
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import uuid4

from app.models.tool_capability import (
    ToolCapability,
    StrategyAdaptation,
    CapabilityDiscovery,
    ToolType,
)


class AdaptiveStrategyService:
    """
    Service for managing tool capabilities and adaptive strategy selection.
    
    Tracks tool performance, learns from usage patterns, and recommends
    optimal tool selections for different goal types and scenarios.
    """
    
    def __init__(self, session: Optional[AsyncSession] = None):
        """Initialize the service with optional database session."""
        self.session = session
    
    async def register_tool_capability(
        self,
        org_id: str,
        tool_name: str,
        tool_type: str,
        description: str = "",
        supported_goal_types: List[str] = None,
        supported_domains: List[str] = None,
        known_limitations: Dict = None,
    ) -> Dict:
        """
        Register a new tool capability or update existing one.
        
        Args:
            org_id: Organization ID
            tool_name: Human-readable tool name
            tool_type: Type from ToolType enum
            description: What the tool does
            supported_goal_types: Goal types this tool can help with
            supported_domains: Knowledge domains this tool covers
            known_limitations: Limitations of this tool
        
        Returns:
            Tool capability dict
        """
        tool_type_enum = ToolType(tool_type)
        
        # Check if tool already exists
        if self.session:
            stmt = select(ToolCapability).where(
                (ToolCapability.organization_id == org_id) &
                (ToolCapability.tool_name == tool_name)
            )
            result = await self.session.execute(stmt)
            existing = result.scalar_one_or_none()
            
            if existing:
                existing.description = description
                existing.supported_goal_types = supported_goal_types or []
                existing.supported_domains = supported_domains or []
                existing.known_limitations = known_limitations or {}
                existing.last_evaluated = datetime.utcnow()
                self.session.add(existing)
                await self.session.commit()
                return {
                    "id": str(existing.id),
                    "tool_name": existing.tool_name,
                    "tool_type": existing.tool_type.value,
                    "success_rate": existing.success_rate,
                }
        
        # Create new tool
        tool = ToolCapability(
            organization_id=org_id,
            tool_name=tool_name,
            tool_type=tool_type_enum,
            description=description,
            supported_goal_types=supported_goal_types or [],
            supported_domains=supported_domains or [],
            known_limitations=known_limitations or {},
        )
        
        if self.session:
            self.session.add(tool)
            await self.session.commit()
        
        return {
            "id": str(tool.id),
            "tool_name": tool.tool_name,
            "tool_type": tool.tool_type.value,
            "success_rate": tool.success_rate,
        }
    
    async def record_tool_usage(
        self,
        org_id: str,
        tool_name: str,
        goal_id: str,
        success: bool,
        execution_time: float = 0.0,
        cost: float = 0.0,
    ) -> Dict:
        """
        Record tool usage and update performance metrics.
        
        Args:
            org_id: Organization ID
            tool_name: Tool that was used
            goal_id: Goal this was used for
            success: Whether tool usage succeeded
            execution_time: How long it took in seconds
            cost: Normalized cost (0-1)
        
        Returns:
            Updated capability metrics
        """
        if not self.session:
            return {"success": success}
        
        stmt = select(ToolCapability).where(
            (ToolCapability.organization_id == org_id) &
            (ToolCapability.tool_name == tool_name)
        )
        result = await self.session.execute(stmt)
        tool_cap = result.scalar_one_or_none()
        
        if not tool_cap:
            return {"error": f"Tool {tool_name} not found"}
        
        # Update metrics
        tool_cap.total_uses += 1
        tool_cap.last_used = datetime.utcnow()
        
        if success:
            tool_cap.successful_uses += 1
        else:
            tool_cap.failed_uses += 1
        
        # Update success rate
        tool_cap.success_rate = tool_cap.successful_uses / tool_cap.total_uses
        
        # Update execution time (exponential moving average)
        alpha = 0.3  # Smoothing factor
        if tool_cap.avg_execution_time == 0:
            tool_cap.avg_execution_time = execution_time
        else:
            tool_cap.avg_execution_time = (alpha * execution_time + 
                                           (1 - alpha) * tool_cap.avg_execution_time)
        
        # Update cost estimate
        if tool_cap.avg_cost == 0:
            tool_cap.avg_cost = cost
        else:
            tool_cap.avg_cost = (alpha * cost + (1 - alpha) * tool_cap.avg_cost)
        
        # Recalculate reliability score
        # Success rate (40%) + Consistency (30%) + Cost efficiency (30%)
        consistency = 1.0 - min(1.0, tool_cap.avg_execution_time / 10.0)  # Penalize slow tools
        cost_efficiency = 1.0 - tool_cap.avg_cost  # Prefer cheaper tools
        
        tool_cap.reliability_score = (
            0.4 * tool_cap.success_rate +
            0.3 * consistency +
            0.3 * cost_efficiency
        )
        
        self.session.add(tool_cap)
        await self.session.commit()
        
        return {
            "tool_name": tool_name,
            "success_rate": tool_cap.success_rate,
            "reliability_score": tool_cap.reliability_score,
            "total_uses": tool_cap.total_uses,
        }
    
    async def select_best_tool(
        self,
        org_id: str,
        goal_type: str,
        domain: str = "",
        exclude_tools: List[str] = None,
    ) -> Optional[Dict]:
        """
        Select the best tool for a given goal type and domain.
        
        Uses learned performance metrics to rank tools and recommend
        the most effective one.
        
        Args:
            org_id: Organization ID
            goal_type: Type of goal this tool will help with
            domain: Knowledge domain (optional)
            exclude_tools: Tool names to exclude from selection
        
        Returns:
            Best tool recommendation with reasoning
        """
        exclude_tools = exclude_tools or []
        
        if not self.session:
            return None
        
        # Build query for relevant tools
        stmt = select(ToolCapability).where(
            ToolCapability.organization_id == org_id
        )
        result = await self.session.execute(stmt)
        tools = result.scalars().all()
        
        # Filter and score tools
        candidates = []
        for tool in tools:
            if tool.tool_name in exclude_tools:
                continue
            
            # Check if tool supports this goal type
            if tool.supported_goal_types and goal_type not in tool.supported_goal_types:
                continue
            
            # Check if tool supports this domain
            if domain and tool.supported_domains and domain not in tool.supported_domains:
                continue
            
            # Score this candidate
            # Prefer higher reliability, but also consider exploration (tools with few uses)
            exploration_bonus = 0.1 if tool.total_uses < 5 else 0.0
            score = tool.reliability_score + exploration_bonus
            
            candidates.append({
                "tool_name": tool.tool_name,
                "tool_type": tool.tool_type.value,
                "score": score,
                "reliability_score": tool.reliability_score,
                "success_rate": tool.success_rate,
                "avg_execution_time": tool.avg_execution_time,
                "total_uses": tool.total_uses,
            })
        
        if not candidates:
            return None
        
        # Return best candidate
        best = max(candidates, key=lambda x: x["score"])
        return {
            "recommended_tool": best["tool_name"],
            "reasoning": f"Reliability: {best['reliability_score']:.2%}, Success Rate: {best['success_rate']:.2%}",
            "details": best,
        }
    
    async def record_strategy_adaptation(
        self,
        org_id: str,
        goal_id: str,
        goal_type: str,
        previous_strategy: Optional[str],
        new_strategy: str,
        reason: str,
        confidence: float = 0.5,
    ) -> Dict:
        """
        Record that the agent adapted its strategy for a goal.
        
        Args:
            org_id: Organization ID
            goal_id: Goal ID that triggered adaptation
            goal_type: Type of goal
            previous_strategy: What was tried before
            new_strategy: What to try now
            reason: Why the change was made
            confidence: Confidence in new strategy (0-1)
        
        Returns:
            Recorded adaptation info
        """
        adaptation = StrategyAdaptation(
            organization_id=org_id,
            goal_id=goal_id,
            goal_type=goal_type,
            previous_strategy=previous_strategy,
            new_strategy=new_strategy,
            adaptation_reason=reason,
            triggered_by="performance",
            confidence_in_adaptation=confidence,
        )
        
        if self.session:
            self.session.add(adaptation)
            await self.session.commit()
        
        return {
            "id": str(adaptation.id),
            "goal_id": str(goal_id),
            "from_strategy": previous_strategy,
            "to_strategy": new_strategy,
        }
    
    async def discover_capability(
        self,
        org_id: str,
        discovery_type: str,  # "new_tool", "capability_gap", "synergy", "limitation"
        tool_or_capability: str,
        description: str,
        potential_value: float = 0.5,
    ) -> Dict:
        """
        Record a discovered tool/capability for later investigation.
        
        Args:
            org_id: Organization ID
            discovery_type: Type of discovery
            tool_or_capability: Name of tool or capability
            description: What was discovered
            potential_value: Estimated value (0-1)
        
        Returns:
            Recorded discovery info
        """
        discovery = CapabilityDiscovery(
            organization_id=org_id,
            discovery_type=discovery_type,
            tool_or_capability=tool_or_capability,
            description=description,
            potential_value=potential_value,
            discovered_when="goal_execution",
        )
        
        if self.session:
            self.session.add(discovery)
            await self.session.commit()
        
        return {
            "id": str(discovery.id),
            "discovery_type": discovery_type,
            "tool_or_capability": tool_or_capability,
            "potential_value": potential_value,
        }
    
    async def get_tool_recommendations(
        self,
        org_id: str,
        goal_id: str,
        goal_type: str,
        domain: str = "",
        limit: int = 5,
    ) -> List[Dict]:
        """
        Get ranked list of tool recommendations for a goal.
        
        Args:
            org_id: Organization ID
            goal_id: Goal ID
            goal_type: Type of goal
            domain: Knowledge domain
            limit: Max recommendations to return
        
        Returns:
            List of tool recommendations ranked by score
        """
        if not self.session:
            return []
        
        stmt = select(ToolCapability).where(
            ToolCapability.organization_id == org_id
        )
        result = await self.session.execute(stmt)
        tools = result.scalars().all()
        
        recommendations = []
        for tool in tools:
            # Check relevance
            if tool.supported_goal_types and goal_type not in tool.supported_goal_types:
                continue
            if domain and tool.supported_domains and domain not in tool.supported_domains:
                continue
            
            score = tool.reliability_score
            recommendations.append({
                "tool_name": tool.tool_name,
                "tool_type": tool.tool_type.value,
                "score": score,
                "success_rate": tool.success_rate,
                "total_uses": tool.total_uses,
                "supported_goal_types": tool.supported_goal_types,
                "known_limitations": tool.known_limitations,
            })
        
        # Sort by score descending
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations[:limit]
    
    async def get_strategy_history(
        self,
        org_id: str,
        goal_id: str,
    ) -> List[Dict]:
        """
        Get strategy adaptation history for a specific goal.
        
        Args:
            org_id: Organization ID
            goal_id: Goal ID
        
        Returns:
            List of adaptations in chronological order
        """
        if not self.session:
            return []
        
        stmt = select(StrategyAdaptation).where(
            (StrategyAdaptation.organization_id == org_id) &
            (StrategyAdaptation.goal_id == goal_id)
        ).order_by(StrategyAdaptation.created_at.asc())
        
        result = await self.session.execute(stmt)
        adaptations = result.scalars().all()
        
        return [
            {
                "id": str(a.id),
                "from_strategy": a.previous_strategy,
                "to_strategy": a.new_strategy,
                "reason": a.adaptation_reason,
                "created_at": a.created_at.isoformat(),
            }
            for a in adaptations
        ]
    
    async def get_discovery_backlog(
        self,
        org_id: str,
        status: str = "unvalidated",
    ) -> List[Dict]:
        """
        Get pending capability discoveries for investigation.
        
        Args:
            org_id: Organization ID
            status: Filter by validation status
        
        Returns:
            List of discoveries
        """
        if not self.session:
            return []
        
        stmt = select(CapabilityDiscovery).where(
            (CapabilityDiscovery.organization_id == org_id) &
            (CapabilityDiscovery.validation_status == status)
        ).order_by(CapabilityDiscovery.created_at.desc())
        
        result = await self.session.execute(stmt)
        discoveries = result.scalars().all()
        
        return [
            {
                "id": str(d.id),
                "type": d.discovery_type,
                "tool_or_capability": d.tool_or_capability,
                "description": d.description,
                "potential_value": d.potential_value,
                "status": d.validation_status,
            }
            for d in discoveries
        ]
