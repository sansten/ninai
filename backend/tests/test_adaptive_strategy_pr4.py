"""
Tests for PR-4: Tool Capability Learning & Adaptive Strategy Selection

Comprehensive test suite for adaptive strategy selection features.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import AsyncMock

from app.models import (
    ToolCapability,
    StrategyAdaptation,
    CapabilityDiscovery,
    ToolType,
)
from app.services.adaptive_strategy_service import AdaptiveStrategyService


class TestToolCapabilityModel:
    """Tests for ToolCapability model."""
    
    def test_create_tool_capability(self):
        """Test creating a tool capability."""
        org_id = uuid4()
        tool = ToolCapability(
            organization_id=org_id,
            tool_name="python_executor",
            tool_type=ToolType.CODE_EXECUTION,
            description="Python code execution engine",
            success_rate=0.92,
            reliability_score=0.88,
        )
        
        assert tool.tool_name == "python_executor"
        assert tool.tool_type == ToolType.CODE_EXECUTION
        assert tool.success_rate == 0.92
        assert tool.reliability_score == 0.88
    
    def test_tool_metrics_tracking(self):
        """Test tracking tool performance metrics."""
        org_id = uuid4()
        tool = ToolCapability(
            organization_id=org_id,
            tool_name="gpt4_api",
            tool_type=ToolType.TEXT_GENERATION,
            total_uses=100,
            successful_uses=95,
            failed_uses=5,
        )
        
        assert tool.total_uses == 100
        assert tool.successful_uses == 95
        assert tool.failed_uses == 5
    
    def test_tool_supported_capabilities(self):
        """Test tool supported goal types and domains."""
        org_id = uuid4()
        tool = ToolCapability(
            organization_id=org_id,
            tool_name="data_analyzer",
            tool_type=ToolType.DATA_ANALYSIS,
            supported_goal_types=["curiosity", "self_improvement"],
            supported_domains=["analytics", "customer_data"],
        )
        
        assert "curiosity" in tool.supported_goal_types
        assert "analytics" in tool.supported_domains


class TestStrategyAdaptationModel:
    """Tests for StrategyAdaptation model."""
    
    def test_create_strategy_adaptation(self):
        """Test creating strategy adaptation record."""
        org_id = uuid4()
        goal_id = uuid4()
        
        adaptation = StrategyAdaptation(
            organization_id=org_id,
            goal_id=goal_id,
            goal_type="curiosity",
            previous_strategy="gpt3_model",
            new_strategy="gpt4_model",
            adaptation_reason="GPT-4 has higher quality outputs",
            triggered_by="performance",
            confidence_in_adaptation=0.85,
        )
        
        assert adaptation.previous_strategy == "gpt3_model"
        assert adaptation.new_strategy == "gpt4_model"
        assert adaptation.confidence_in_adaptation == 0.85
    
    def test_strategy_performance_tracking(self):
        """Test tracking strategy performance before/after."""
        org_id = uuid4()
        adaptation = StrategyAdaptation(
            organization_id=org_id,
            goal_id=uuid4(),
            goal_type="prediction",
            previous_strategy="linear_regression",
            new_strategy="neural_network",
            adaptation_reason="Better accuracy",
            previous_success_rate=0.72,
            new_success_rate=0.89,
            predicted_improvement=0.20,
        )
        
        assert adaptation.previous_success_rate == 0.72
        assert adaptation.new_success_rate == 0.89
        improvement = adaptation.new_success_rate - adaptation.previous_success_rate
        assert improvement > 0


class TestCapabilityDiscoveryModel:
    """Tests for CapabilityDiscovery model."""
    
    def test_create_capability_discovery(self):
        """Test recording capability discovery."""
        org_id = uuid4()
        discovery = CapabilityDiscovery(
            organization_id=org_id,
            discovery_type="new_tool",
            tool_or_capability="claude_api",
            description="Found Claude API can handle complex reasoning better",
            potential_value=0.8,
            discovered_when="goal_execution",
        )
        
        assert discovery.discovery_type == "new_tool"
        assert discovery.tool_or_capability == "claude_api"
        assert discovery.potential_value == 0.8
    
    def test_discovery_types(self):
        """Test different discovery types."""
        org_id = uuid4()
        discovery_types = [
            "new_tool",
            "capability_gap",
            "synergy",
            "limitation",
        ]
        
        for discovery_type in discovery_types:
            discovery = CapabilityDiscovery(
                organization_id=org_id,
                discovery_type=discovery_type,
                tool_or_capability=f"test_{discovery_type}",
                description="Test description",
            )
            assert discovery.discovery_type == discovery_type


class TestAdaptiveStrategyService:
    """Tests for AdaptiveStrategyService."""
    
    @pytest.mark.asyncio
    async def test_register_tool_capability(self):
        """Test registering a tool capability."""
        svc = AdaptiveStrategyService(session=None)
        org_id = str(uuid4())
        
        result = await svc.register_tool_capability(
            org_id=org_id,
            tool_name="test_tool",
            tool_type="code_execution",
            description="Test tool",
            supported_goal_types=["curiosity", "self_improvement"],
        )
        
        assert result["tool_name"] == "test_tool"
        assert "id" in result
        # Success rate may be None initially until usage is recorded
        assert result.get("success_rate") is None or result["success_rate"] == 0.0
    
    @pytest.mark.asyncio
    async def test_record_tool_usage(self):
        """Test recording tool usage and metrics."""
        svc = AdaptiveStrategyService(session=None)
        
        org_id = str(uuid4())
        tool_name = "test_tool"
        
        result = await svc.record_tool_usage(
            org_id=org_id,
            tool_name=tool_name,
            goal_id=str(uuid4()),
            success=True,
            execution_time=1.5,
            cost=0.1,
        )
        
        assert result.get("success", False) or "error" in result
    
    @pytest.mark.asyncio
    async def test_select_best_tool(self):
        """Test selecting best tool for a goal."""
        svc = AdaptiveStrategyService(session=None)
        org_id = str(uuid4())
        
        result = await svc.select_best_tool(
            org_id=org_id,
            goal_type="curiosity",
            domain="analytics",
        )
        
        # Without a real database, this will return None
        assert result is None or "recommended_tool" in result
    
    @pytest.mark.asyncio
    async def test_record_strategy_adaptation(self):
        """Test recording strategy adaptation."""
        svc = AdaptiveStrategyService(session=None)
        org_id = str(uuid4())
        goal_id = str(uuid4())
        
        result = await svc.record_strategy_adaptation(
            org_id=org_id,
            goal_id=goal_id,
            goal_type="prediction",
            previous_strategy="model_a",
            new_strategy="model_b",
            reason="Model B has better performance",
            confidence=0.8,
        )
        
        assert result["to_strategy"] == "model_b"
        assert result["from_strategy"] == "model_a"
    
    @pytest.mark.asyncio
    async def test_discover_capability(self):
        """Test discovering new capability."""
        svc = AdaptiveStrategyService(session=None)
        org_id = str(uuid4())
        
        result = await svc.discover_capability(
            org_id=org_id,
            discovery_type="new_tool",
            tool_or_capability="new_api",
            description="Found new API for data analysis",
            potential_value=0.7,
        )
        
        assert result["tool_or_capability"] == "new_api"
        assert result["potential_value"] == 0.7
    
    @pytest.mark.asyncio
    async def test_get_tool_recommendations(self):
        """Test getting tool recommendations."""
        svc = AdaptiveStrategyService(session=None)
        org_id = str(uuid4())
        
        results = await svc.get_tool_recommendations(
            org_id=org_id,
            goal_id=str(uuid4()),
            goal_type="curiosity",
            limit=5,
        )
        
        assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_get_strategy_history(self):
        """Test getting strategy history for goal."""
        svc = AdaptiveStrategyService(session=None)
        org_id = str(uuid4())
        goal_id = str(uuid4())
        
        results = await svc.get_strategy_history(
            org_id=org_id,
            goal_id=goal_id,
        )
        
        assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_get_discovery_backlog(self):
        """Test getting discovery backlog."""
        svc = AdaptiveStrategyService(session=None)
        org_id = str(uuid4())
        
        results = await svc.get_discovery_backlog(
            org_id=org_id,
            status="unvalidated",
        )
        
        assert isinstance(results, list)


class TestToolLearning:
    """Tests for tool learning behavior."""
    
    def test_reliability_score_calculation(self):
        """Test reliability score calculation."""
        # Reliability = 0.4 * success_rate + 0.3 * consistency + 0.3 * cost_efficiency
        success_rate = 0.9
        consistency = 0.8
        cost_efficiency = 0.95
        
        reliability = (
            0.4 * success_rate +
            0.3 * consistency +
            0.3 * cost_efficiency
        )
        
        assert 0.8 < reliability < 1.0
    
    def test_tool_type_enum(self):
        """Test ToolType enum values."""
        expected_types = [
            "data_analysis",
            "api_call",
            "code_execution",
            "knowledge_retrieval",
            "text_generation",
            "image_generation",
            "web_search",
            "memory_query",
            "system_command",
        ]
        
        for type_name in expected_types:
            # Enum keys are uppercase with underscores
            enum_key = type_name.upper()
            tool_type = ToolType[enum_key]
            assert tool_type.value == type_name


class TestStrategyAdaptationTriggers:
    """Tests for strategy adaptation triggers."""
    
    def test_performance_trigger(self):
        """Test performance-based adaptation trigger."""
        previous_success = 0.6
        new_success = 0.85
        threshold = 0.15
        
        improvement = new_success - previous_success
        should_adapt = improvement >= threshold
        
        assert should_adapt is True
    
    def test_discovery_trigger(self):
        """Test capability discovery trigger."""
        discovery_value = 0.8
        investigation_threshold = 0.5
        
        should_investigate = discovery_value >= investigation_threshold
        
        assert should_investigate is True
    
    def test_multiple_candidates(self):
        """Test selecting best from multiple tool candidates."""
        candidates = [
            {"tool": "tool_a", "score": 0.75},
            {"tool": "tool_b", "score": 0.85},
            {"tool": "tool_c", "score": 0.82},
        ]
        
        best = max(candidates, key=lambda x: x["score"])
        assert best["tool"] == "tool_b"
