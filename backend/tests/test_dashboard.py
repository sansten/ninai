"""
Tests for Admin Dashboard Service
"""
import pytest
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import uuid4

from app.services.dashboard import DashboardService
from app.models.user import User
from app.models.memory import MemoryMetadata
from app.models.admin import AdminAuditLog
from app.models.organization import Organization


@pytest.fixture
async def test_dashboard_data(pg_db_session: AsyncSession):
    """Create test data for dashboard"""
    # Create test organization
    org = Organization(
        name="Test Organization",
        slug="test-org",
        is_active=True
    )
    pg_db_session.add(org)
    await pg_db_session.flush()
    
    # Create test users
    users = []
    for i in range(10):
        user = User(
            email=f"user{i}@test.com",
            full_name=f"Test User {i}",
            hashed_password="fake_hash",
            is_active=True,
            created_at=datetime.utcnow() - timedelta(days=i),
            last_login_at=datetime.utcnow() - timedelta(days=max(0, i-5))
        )
        pg_db_session.add(user)
        users.append(user)
    
    await pg_db_session.flush()
    
    await pg_db_session.flush()
    
    # Create test memories
    for i in range(50):
        memory = MemoryMetadata(
            organization_id=org.id,
            owner_id=users[i % len(users)].id,
            content_preview=f"Test memory {i}",
            content_hash=f"hash_{i}",
            vector_id=f"vec_{i}",
            embedding_model="test-model",
            created_at=datetime.utcnow() - timedelta(days=i // 10)
        )
        pg_db_session.add(memory)
    
    # Create audit logs
    for i in range(20):
        log = AdminAuditLog(
            id=str(uuid4()),
            admin_id=users[0].id,
            action="update" if i % 2 == 0 else "read",
            resource_type="user",
            resource_id=str(users[1].id),
            created_at=datetime.utcnow() - timedelta(hours=i)
        )
        pg_db_session.add(log)
    
    await pg_db_session.commit()
    return {"users": users, "memory_count": 50, "log_count": 20}


@pytest.mark.asyncio
class TestDashboardKPIs:
    """Test KPI calculation methods"""
    
    async def test_get_user_kpis(self, pg_db_session: AsyncSession, test_dashboard_data):
        """Test user KPI calculation"""
        kpis = await DashboardService.get_user_kpis(pg_db_session)
        
        assert len(kpis) == 3
        assert kpis[0].label == "Total Users"
        assert int(kpis[0].value.replace(",", "")) == 10
        
    async def test_get_memory_kpis(self, pg_db_session: AsyncSession, test_dashboard_data):
        """Test memory KPI calculation"""
        kpis = await DashboardService.get_memory_kpis(pg_db_session)
        
        assert len(kpis) == 3
        assert kpis[0].label == "Total Memories"
        assert int(kpis[0].value.replace(",", "")) == 50
        
    async def test_get_system_kpis(self, pg_db_session: AsyncSession, test_dashboard_data):
        """Test system KPI calculation"""
        kpis = await DashboardService.get_system_kpis(pg_db_session)
        
        assert len(kpis) == 3
        assert any(kpi.label == "API Requests (Today)" for kpi in kpis)


@pytest.mark.asyncio
class TestDashboardServices:
    """Test service health methods"""
    
    async def test_get_service_health(self, pg_db_session: AsyncSession):
        """Test service health status"""
        services = DashboardService.get_service_health(pg_db_session)
        
        assert len(services) >= 4
        assert all(svc.status in ["healthy", "degraded", "unhealthy"] for svc in services)
        assert all(hasattr(svc, "last_check") for svc in services)


@pytest.mark.asyncio
class TestDashboardActivities:
    """Test activity tracking methods"""
    
    async def test_get_recent_activities(self, pg_db_session: AsyncSession, test_dashboard_data):
        """Test recent activity retrieval"""
        activities = await DashboardService.get_recent_activities(pg_db_session, limit=10)
        
        assert len(activities) <= 10
        assert all("id" in act for act in activities)
        assert all("action" in act for act in activities)
        assert all("timestamp" in act for act in activities)
        
    async def test_recent_activities_limit(self, pg_db_session: AsyncSession, test_dashboard_data):
        """Test activity limit parameter"""
        activities_5 = await DashboardService.get_recent_activities(pg_db_session, limit=5)
        activities_all = await DashboardService.get_recent_activities(pg_db_session, limit=50)
        
        assert len(activities_5) <= 5
        assert len(activities_all) <= 50


@pytest.mark.asyncio
class TestDashboardComplete:
    """Test full dashboard data aggregation"""
    
    async def test_get_full_dashboard_data(self, pg_db_session: AsyncSession, test_dashboard_data):
        """Test complete dashboard data retrieval"""
        dashboard = await DashboardService.get_full_dashboard_data(pg_db_session)
        
        # Check structure
        assert "timestamp" in dashboard
        assert "kpis" in dashboard
        assert "services" in dashboard
        assert "recent_activities" in dashboard
        assert "alerts" in dashboard
        
        # Check KPI structure
        assert "users" in dashboard["kpis"]
        assert "memories" in dashboard["kpis"]
        assert "system" in dashboard["kpis"]
        
        assert len(dashboard["kpis"]["users"]) > 0
        assert len(dashboard["kpis"]["memories"]) > 0
        assert len(dashboard["kpis"]["system"]) > 0
        
        # Check services
        assert len(dashboard["services"]) > 0
        
        # Check activities
        assert isinstance(dashboard["recent_activities"], list)
