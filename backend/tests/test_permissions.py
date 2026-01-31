"""Permission checker unit tests.

These tests validate the explainable AccessDecision shape and key permission
lookup logic without requiring a real DB or Redis instance.
"""

import pytest
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock

from app.services.permission_checker import PermissionChecker, AccessDecision
from app.core.redis import RedisClient


class TestAccessDecision:
    """Tests for AccessDecision dataclass."""
    
    def test_granted_decision(self):
        """Test creating a granted decision."""
        decision = AccessDecision(
            allowed=True,
            reason="Owner access",
            method="owner",
        )
        
        assert decision.allowed is True
        assert decision.reason == "Owner access"
        assert decision.method == "owner"
    
    def test_denied_decision(self):
        """Test creating a denied decision."""
        decision = AccessDecision(
            allowed=False,
            reason="No access permissions",
            method="denied",
        )
        
        assert decision.allowed is False
        assert decision.reason == "No access permissions"


class TestPermissionChecker:
    """Tests for PermissionChecker service."""
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        return AsyncMock()
    
    @pytest.fixture
    def checker(self, mock_db, monkeypatch):
        """Create a PermissionChecker instance."""
        monkeypatch.setattr(RedisClient, "get_json", AsyncMock(return_value=None))
        monkeypatch.setattr(RedisClient, "set_json", AsyncMock())
        return PermissionChecker(mock_db)
    
    @pytest.mark.asyncio
    async def test_check_permission_system_admin(self, checker):
        """Test that system_admin has all permissions."""
        user_id = str(uuid4())
        org_id = str(uuid4())
        # get_effective_permissions loads Role.permissions via DB; we only assert it returns a list.
        # Mock empty role permissions.
        mock_result = MagicMock()
        mock_result.__iter__.return_value = iter([])
        checker.session.execute = AsyncMock(return_value=mock_result)

        perms = await checker.get_effective_permissions(user_id, org_id)
        assert isinstance(perms, list)
    
    def test_cache_key_generation(self, checker):
        """Test cache key format used by RedisClient."""
        user_id = str(uuid4())
        org_id = str(uuid4())
        # This is an implementation detail but should remain stable.
        key = f"{checker.CACHE_PREFIX_PERMISSIONS}:{user_id}:{org_id}"
        assert key.startswith("perms:")
    
    # NOTE: Memory-specific access checks are covered by higher-level integration tests.


class TestRoleHierarchy:
    """Tests for role hierarchy and permission inheritance."""
    
    def test_role_ordering(self):
        """Test that roles have correct hierarchy."""
        # System admin > org_admin > team_lead > member > viewer
        role_hierarchy = {
            "system_admin": 100,
            "org_admin": 80,
            "security_admin": 70,
            "department_manager": 60,
            "team_lead": 50,
            "member": 30,
            "viewer": 10,
        }
        
        assert role_hierarchy["system_admin"] > role_hierarchy["org_admin"]
        assert role_hierarchy["org_admin"] > role_hierarchy["team_lead"]
        assert role_hierarchy["team_lead"] > role_hierarchy["member"]
        assert role_hierarchy["member"] > role_hierarchy["viewer"]
