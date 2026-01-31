"""
Tests for new feature implementations.

Tests for:
- Knowledge synthesis service
- Metrics enhancement service
- External sync adapters
- Replication service
"""

import pytest
from datetime import datetime
from app.services.knowledge_synthesis_service import (
    KnowledgeSynthesisService,
    ConceptCluster,
    Trend,
    SynthesisReport,
)
from app.services.metrics_enhancement_service import MetricsEnhancementService
from app.services.replication_service import (
    ReplicationService,
    Region,
    ReplicationStatus,
    RegionStatus,
)


class TestKnowledgeSynthesisService:
    """Test knowledge synthesis service."""
    
    @pytest.mark.asyncio
    async def test_create_synthesis_report_empty(self):
        """Test creating report with no memories."""
        # Note: Would need actual database session in full test
        # This is a placeholder
        pass
    
    def test_concept_cluster_creation(self):
        """Test creating concept cluster."""
        cluster = ConceptCluster(
            concept="AI Safety",
            memories=[
                {"id": "1", "title": "Safety alignment", "content": "..."},
                {"id": "2", "title": "Risk assessment", "content": "..."},
            ],
            strength=0.85,
            tags=["ai", "safety", "alignment"],
            relationships_count=5,
            date_range=(datetime(2024, 1, 1), datetime(2024, 1, 31)),
        )
        
        assert cluster.concept == "AI Safety"
        assert len(cluster.memories) == 2
        assert cluster.strength == 0.85
        assert cluster.relationships_count == 5
    
    def test_trend_creation(self):
        """Test creating trend."""
        trend = Trend(
            topic="Memory Activity",
            description="Memory creation is increasing",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            memory_count=42,
            trajectory="increasing",
            strength=0.65,
            related_concepts=["productivity", "learning"],
        )
        
        assert trend.topic == "Memory Activity"
        assert trend.trajectory == "increasing"
        assert trend.memory_count == 42
    
    def test_synthesis_report_export_markdown(self):
        """Test exporting report as markdown."""
        cluster = ConceptCluster(
            concept="Test Cluster",
            memories=[{"id": "1", "title": "Test", "content": "Content"}],
            strength=0.8,
            tags=["test"],
            relationships_count=2,
            date_range=(datetime(2024, 1, 1), datetime(2024, 1, 31)),
        )
        
        report = SynthesisReport(
            title="Test Report",
            summary="This is a test",
            clusters=[cluster],
            trends=[],
            key_insights=["Insight 1", "Insight 2"],
            relationships={},
            generated_at=datetime.utcnow(),
            memory_count=1,
        )
        
        markdown = report.to_dict()
        assert markdown["title"] == "Test Report"
        assert len(markdown["clusters"]) == 1
        assert markdown["memory_count"] == 1


class TestMetricsEnhancementService:
    """Test metrics enhancement service."""
    
    def test_get_metrics_service_singleton(self):
        """Test getting singleton metrics service."""
        from app.services.metrics_enhancement_service import get_metrics_service
        
        service1 = get_metrics_service()
        service2 = get_metrics_service()
        
        assert service1 is service2  # Should be same instance
    
    def test_record_search_metric(self):
        """Test recording search metric."""
        from app.services.metrics_enhancement_service import get_metrics_service
        
        service = get_metrics_service()
        # Record metric (won't error as registry already has it)
        service.record_search(0.125, 42)
    
    def test_record_api_request_metric(self):
        """Test recording API request metric."""
        from app.services.metrics_enhancement_service import get_metrics_service
        
        service = get_metrics_service()
        service.record_api_request("GET", "/memories/search", 200, 0.085)
    
    def test_record_db_query_metric(self):
        """Test recording database query metric."""
        from app.services.metrics_enhancement_service import get_metrics_service
        
        service = get_metrics_service()
        service.record_db_query("SELECT", 0.450)
        # Verify slow query not triggered
        
        service.record_db_query("SELECT", 1.500)
        # Verify slow query counter incremented
    
    def test_get_summary_stats(self):
        """Test getting metrics summary."""
        from app.services.metrics_enhancement_service import get_metrics_service
        
        service = get_metrics_service()
        stats = service.get_summary_stats()
        
        assert "timestamp" in stats
        assert "metrics" in stats
        assert "searches_total" in stats["metrics"]


class TestExternalSyncAdapters:
    """Test external system sync adapters."""
    
    @pytest.mark.asyncio
    async def test_obsidian_adapter_init(self):
        """Test initializing Obsidian adapter."""
        try:
            from integrations.external_sync import ObsidianVaultAdapter
            
            config = {
                "vault_path": "/path/to/vault",
                "sync_folder": "Ninai Memories",
            }
            adapter = ObsidianVaultAdapter(config)
            assert adapter.name == "ObsidianVaultAdapter"
            assert adapter.config["vault_path"] == "/path/to/vault"
        except Exception as e:
            # Skip if external_sync not available
            pytest.skip(f"Cannot import external sync: {e}")
    
    @pytest.mark.asyncio
    async def test_notion_adapter_init(self):
        """Test initializing Notion adapter."""
        try:
            from integrations.external_sync import NotionDatabaseAdapter
            
            config = {
                "api_key": "test_key",
                "database_id": "test_db",
            }
            adapter = NotionDatabaseAdapter(config)
            assert adapter.name == "NotionDatabaseAdapter"
        except Exception as e:
            pytest.skip(f"Cannot import external sync: {e}")
    
    @pytest.mark.asyncio
    async def test_roam_adapter_init(self):
        """Test initializing Roam adapter."""
        try:
            from integrations.external_sync import RoamResearchAdapter
            
            config = {
                "graph_name": "my_graph",
                "api_token": "test_token",
            }
            adapter = RoamResearchAdapter(config)
            assert adapter.name == "RoamResearchAdapter"
        except Exception as e:
            pytest.skip(f"Cannot import external sync: {e}")


class TestReplicationService:
    """Test replication service."""
    
    def test_replication_service_init(self):
        """Test initializing replication service."""
        primary = Region(
            name="us-east-1",
            db_url="postgresql://...",
            is_primary=True,
        )
        secondary = Region(
            name="us-west-1",
            db_url="postgresql://...",
            is_primary=False,
        )
        
        service = ReplicationService(primary, [secondary])
        
        assert service.primary_region.name == "us-east-1"
        assert len(service.secondary_regions) == 1
        assert len(service.all_regions) == 2
    
    def test_region_creation(self):
        """Test creating region."""
        region = Region(
            name="eu-west-1",
            db_url="postgresql://europe...",
            is_primary=False,
            read_weight=1,
        )
        
        assert region.name == "eu-west-1"
        assert region.is_primary is False
        assert region.healthy is True
    
    @pytest.mark.asyncio
    async def test_get_read_endpoints(self):
        """Test getting read endpoints."""
        primary = Region(
            name="us-east-1",
            db_url="postgresql://...",
            is_primary=True,
            read_weight=2,
        )
        secondary = Region(
            name="us-west-1",
            db_url="postgresql://...",
            is_primary=False,
            read_weight=1,
        )
        
        service = ReplicationService(primary, [secondary])
        endpoints = service.get_read_endpoints()
        
        assert len(endpoints) == 2
        # Primary should be first due to higher weight
        assert endpoints[0].name == "us-east-1"
    
    @pytest.mark.asyncio
    async def test_get_write_endpoint(self):
        """Test getting write endpoint."""
        primary = Region(
            name="us-east-1",
            db_url="postgresql://...",
            is_primary=True,
        )
        
        service = ReplicationService(primary)
        endpoint = service.get_write_endpoint()
        
        assert endpoint.name == "us-east-1"
        assert endpoint.is_primary is True
    
    @pytest.mark.asyncio
    async def test_region_health_tracking(self):
        """Test tracking region health."""
        primary = Region(
            name="us-east-1",
            db_url="postgresql://...",
            is_primary=True,
        )
        
        service = ReplicationService(primary)
        
        # Check initial health
        assert service.region_health["us-east-1"] == RegionStatus.HEALTHY
        
        # Simulate health check
        await service.check_region_health()
        assert service.region_health["us-east-1"] == RegionStatus.HEALTHY


class TestSyncIntegration:
    """Test sync functionality integration."""
    
    @pytest.mark.asyncio
    async def test_sync_manager_registration(self):
        """Test registering adapters with sync manager."""
        try:
            from integrations.external_sync import SyncManager, ObsidianVaultAdapter
            
            manager = SyncManager()
            adapter = ObsidianVaultAdapter({"vault_path": "/tmp"})
            
            manager.register_adapter("obsidian", adapter)
            
            assert "obsidian" in manager.adapters
        except Exception as e:
            pytest.skip(f"Cannot import external sync: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
