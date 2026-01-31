"""Admin UI tests"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from uuid import uuid4

from app.main import app
from app.database import get_db
from app.models.user import User
from app.models.admin import AdminRole, AdminSetting, AdminAuditLog
from app.services.admin import (
    AdminRoleService, AdminSettingService, AdminAuditService, AdminUserService
)
from app.schemas.admin import (
    AdminRoleCreate, AdminSettingCreate, AdminAuditLogFilter
)
from app.core.security import create_access_token


client = TestClient(app)

# Fixed org context for test tokens
ADMIN_TEST_ORG_ID = "00000000-0000-0000-0000-0000000000ad"


# ==================== FIXTURES ====================

@pytest.fixture
def db_session(db: Session):
    """Database session fixture"""
    yield db


@pytest.fixture
def admin_user(db_session: Session) -> User:
    """Create admin user fixture"""
    from app.core.security import get_password_hash
    
    user = User(
        id=str(uuid4()),
        email="admin@test.com",
        full_name="Admin User",
        hashed_password=get_password_hash("password123"),
        is_active=True,
        is_admin=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def admin_token(admin_user: User) -> str:
    """Create admin token"""
    return create_access_token(admin_user.id, ADMIN_TEST_ORG_ID)


@pytest.fixture
def regular_user(db_session: Session) -> User:
    """Create regular user fixture"""
    from app.core.security import get_password_hash
    
    user = User(
        id=str(uuid4()),
        email="user@test.com",
        full_name="Regular User",
        hashed_password=get_password_hash("password123"),
        is_active=True,
        is_admin=False,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def admin_role(db_session: Session, admin_user: User) -> AdminRole:
    """Create admin role fixture"""
    role = AdminRole(
        id=str(uuid4()),
        name="Admin",
        description="Administrator role",
        permissions=["users:read", "users:write", "settings:read"],
        created_by=admin_user.id,
    )
    db_session.add(role)
    db_session.commit()
    db_session.refresh(role)
    return role


# ==================== ROLE TESTS ====================

class TestAdminRoles:
    """Test admin role management"""
    
    def test_create_role(self, admin_token: str, db_session: Session):
        """Test creating admin role"""
        response = client.post(
            "/api/v1/admin/roles",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={
                "name": "Editor",
                "description": "Editor role",
                "permissions": ["users:read", "settings:read"],
            }
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Editor"
        assert "users:read" in data["permissions"]
    
    def test_list_roles(self, admin_token: str, admin_role: AdminRole):
        """Test listing roles"""
        response = client.get(
            "/api/v1/admin/roles",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) > 0
    
    def test_get_role(self, admin_token: str, admin_role: AdminRole):
        """Test getting specific role"""
        response = client.get(
            f"/api/v1/admin/roles/{admin_role.id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(admin_role.id)
        assert data["name"] == admin_role.name
    
    def test_update_role(self, admin_token: str, admin_role: AdminRole):
        """Test updating role"""
        response = client.put(
            f"/api/v1/admin/roles/{admin_role.id}",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={
                "permissions": ["users:read", "users:write", "settings:read", "settings:write"],
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "settings:write" in data["permissions"]
    
    def test_delete_role(self, admin_token: str, db_session: Session, admin_user: User):
        """Test deleting role"""
        # Create a role without users
        role = AdminRole(
            id=str(uuid4()),
            name="Temp Role",
            permissions=["users:read"],
            created_by=admin_user.id,
        )
        db_session.add(role)
        db_session.commit()
        
        response = client.delete(
            f"/api/v1/admin/roles/{role.id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert response.status_code == 204


# ==================== SETTINGS TESTS ====================

class TestAdminSettings:
    """Test admin settings management"""
    
    def test_create_setting(self, admin_token: str):
        """Test creating setting"""
        response = client.post(
            "/api/v1/admin/settings",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={
                "category": "general",
                "key": "app_name",
                "value": "NINAI",
                "type": "string",
                "description": "Application name",
            }
        )
        assert response.status_code == 201
        data = response.json()
        assert data["category"] == "general"
        assert data["key"] == "app_name"
    
    def test_list_settings(self, admin_token: str):
        """Test listing settings"""
        response = client.get(
            "/api/v1/admin/settings",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
    
    def test_get_setting(self, admin_token: str, db_session: Session, admin_user: User):
        """Test getting specific setting"""
        # Create setting
        setting = AdminSetting(
            id=str(uuid4()),
            category="general",
            key="app_name",
            value="NINAI",
            type="string",
            updated_by=admin_user.id,
        )
        db_session.add(setting)
        db_session.commit()
        
        response = client.get(
            f"/api/v1/admin/settings/{setting.id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(setting.id)
    
    def test_update_setting(self, admin_token: str, db_session: Session, admin_user: User):
        """Test updating setting"""
        setting = AdminSetting(
            id=str(uuid4()),
            category="general",
            key="app_name",
            value="NINAI",
            updated_by=admin_user.id,
        )
        db_session.add(setting)
        db_session.commit()
        
        response = client.put(
            f"/api/v1/admin/settings/{setting.id}",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"value": "NINAI Updated"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["value"] == "NINAI Updated"


# ==================== AUDIT LOG TESTS ====================

class TestAdminAuditLogs:
    """Test audit log management"""
    
    def test_list_audit_logs(self, admin_token: str, db_session: Session, admin_user: User):
        """Test listing audit logs"""
        # Create audit log
        log = AdminAuditLog(
            id=str(uuid4()),
            admin_id=admin_user.id,
            action="create",
            resource_type="user",
            resource_id=str(uuid4()),
            new_values={"email": "test@test.com"},
        )
        db_session.add(log)
        db_session.commit()
        
        response = client.get(
            "/api/v1/admin/audit-logs",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
    
    def test_get_audit_log(self, admin_token: str, db_session: Session, admin_user: User):
        """Test getting specific audit log"""
        log = AdminAuditLog(
            id=str(uuid4()),
            admin_id=admin_user.id,
            action="create",
            resource_type="user",
        )
        db_session.add(log)
        db_session.commit()
        
        response = client.get(
            f"/api/v1/admin/audit-logs/{log.id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(log.id)


# ==================== PERMISSION TESTS ====================

class TestAdminPermissions:
    """Test permission checking"""
    
    def test_get_permissions(self, admin_token: str):
        """Test getting available permissions"""
        response = client.get(
            "/api/v1/admin/permissions",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "permissions" in data
        assert len(data["permissions"]) > 0
    
    def test_permission_denied_without_permission(self, regular_user: User):
        """Test permission denied for non-admin"""
        token = create_access_token(regular_user.id, ADMIN_TEST_ORG_ID)
        response = client.get(
            "/api/v1/admin/roles",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code in [401, 403]


# ==================== SERVICE TESTS ====================

class TestAdminServices:
    """Test admin services"""
    
    def test_create_role_service(self, db_session: Session, admin_user: User):
        """Test AdminRoleService.create_role"""
        role_create = AdminRoleCreate(
            name="Test Role",
            permissions=["users:read"],
        )
        role = AdminRoleService.create_role(db_session, role_create, admin_user.id)
        assert role.name == "Test Role"
        assert "users:read" in role.permissions
    
    def test_create_setting_service(self, db_session: Session, admin_user: User):
        """Test AdminSettingService.create_setting"""
        setting_create = AdminSettingCreate(
            category="test",
            key="test_key",
            value="test_value",
        )
        setting = AdminSettingService.create_setting(db_session, setting_create, admin_user.id)
        assert setting.category == "test"
        assert setting.key == "test_key"
    
    def test_log_action_service(self, db_session: Session, admin_user: User):
        """Test AdminAuditService.log_action"""
        log = AdminAuditService.log_action(
            db_session,
            admin_id=admin_user.id,
            action="create",
            resource_type="user",
            new_values={"email": "test@test.com"}
        )
        assert log.action == "create"
        assert log.resource_type == "user"
    
    def test_list_audit_logs_service(self, db_session: Session, admin_user: User):
        """Test AdminAuditService.list_audit_logs"""
        # Create multiple logs
        for i in range(5):
            AdminAuditService.log_action(
                db_session,
                admin_id=admin_user.id,
                action="update",
                resource_type="role",
            )
        
        logs, total = AdminAuditService.list_audit_logs(
            db_session, AdminAuditLogFilter()
        )
        assert total >= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
