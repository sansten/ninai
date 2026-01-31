"""
Tests for CrewAI and LangGraph integrations
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch


class TestCrewAIAdapter:
    """Test CrewAI memory adapter."""
    
    def test_crewai_adapter_structure(self):
        """Test CrewAI adapter can be imported."""
        try:
            from integrations.crewai_adapter import NinaiCrewAIMemory, create_crew_memory
            assert NinaiCrewAIMemory is not None
            assert create_crew_memory is not None
        except ImportError as e:
            if "crewai" in str(e).lower():
                pytest.skip("CrewAI not installed - this is expected in test environment")
            raise
    
    def test_crewai_adapter_initialization(self):
        """Test CrewAI adapter can be initialized with mock."""
        try:
            from integrations.crewai_adapter import NinaiCrewAIMemory
        except ImportError:
            pytest.skip("CrewAI not installed")
        
        # Mock CrewAI availability
        with patch('integrations.crewai_adapter.CREWAI_AVAILABLE', True):
            with patch('integrations.crewai_adapter.Memory', object):
                # Should initialize without errors
                try:
                    memory = NinaiCrewAIMemory(
                        api_base_url="http://localhost:8080/api/v1",
                        api_key="test-token",
                        crew_id="test-crew"
                    )
                    assert memory.crew_id == "test-crew"
                    assert memory.api_key == "test-token"
                except ImportError:
                    pytest.skip("CrewAI not available")


class TestLangGraphAdapter:
    """Test LangGraph checkpoint adapter."""
    
    def test_langgraph_adapter_structure(self):
        """Test LangGraph adapter can be imported."""
        try:
            from integrations.langgraph_adapter import NinaiLangGraphCheckpointSaver, create_checkpoint_saver
            assert NinaiLangGraphCheckpointSaver is not None
            assert create_checkpoint_saver is not None
        except ImportError as e:
            if "langgraph" in str(e).lower():
                pytest.skip("LangGraph not installed - this is expected in test environment")
            raise
    
    def test_langgraph_adapter_initialization(self):
        """Test LangGraph adapter can be initialized with mock."""
        try:
            from integrations.langgraph_adapter import NinaiLangGraphCheckpointSaver
        except ImportError:
            pytest.skip("LangGraph not installed")
        
        # Mock LangGraph availability
        with patch('integrations.langgraph_adapter.LANGGRAPH_AVAILABLE', True):
            with patch('integrations.langgraph_adapter.BaseCheckpointSaver', object):
                # Should initialize without errors
                try:
                    saver = NinaiLangGraphCheckpointSaver(
                        api_base_url="http://localhost:8080/api/v1",
                        api_key="test-token",
                        graph_id="test-graph"
                    )
                    assert saver.graph_id == "test-graph"
                    assert saver.api_key == "test-token"
                except ImportError:
                    pytest.skip("LangGraph not available")


class TestFrameworkIntegrationsList:
    """Test that all framework integrations are exported."""
    
    def test_all_integrations_exported(self):
        """Test __init__.py exports all adapters."""
        from integrations import __all__
        
        expected = [
            "NinaiLangChainMemory",
            "NinaiLlamaIndexVectorStore",
            "NinaiCrewAIMemory",
            "create_crew_memory",
            "NinaiLangGraphCheckpointSaver",
            "create_checkpoint_saver",
        ]
        
        for integration in expected:
            assert integration in __all__, f"{integration} not exported from integrations package"
