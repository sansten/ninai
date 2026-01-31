"""Admin business logic services"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import uuid4

from sqlalchemy import and_, desc
from sqlalchemy.orm import Session

from app.models.user import User
from app.models.admin import (
    AdminRole, AdminSetting, AdminAuditLog, AdminSession, AdminIPWhitelist
)
from app.schemas.admin import (
    AdminRoleCreate, AdminRoleUpdate, AdminSettingCreate, AdminSettingUpdate,
    AdminAuditLogFilter
)


class AdminRoleService:
    """Service for managing admin roles"""
    
    @staticmethod
    def create_role(
        db: Session,
        role_create: AdminRoleCreate,
        created_by: str
    ) -> AdminRole:
        """Create new admin role"""
        role = AdminRole(
            id=str(uuid4()),
            name=role_create.name,
            description=role_create.description,
            permissions=role_create.permissions,
            created_by=created_by
        )
        db.add(role)
        db.commit()
        db.refresh(role)
        return role
    
    @staticmethod
    def get_role(db: Session, role_id: str) -> Optional[AdminRole]:
        """Get role by ID"""
        return db.query(AdminRole).filter(AdminRole.id == role_id).first()
    
    @staticmethod
    def list_roles(db: Session, skip: int = 0, limit: int = 50) -> tuple[List[AdminRole], int]:
        """List all roles"""
        query = db.query(AdminRole).order_by(AdminRole.created_at)
        total = query.count()
        roles = query.offset(skip).limit(limit).all()
        return roles, total
    
    @staticmethod
    def update_role(
        db: Session,
        role_id: str,
        role_update: AdminRoleUpdate
    ) -> Optional[AdminRole]:
        """Update admin role"""
        role = AdminRoleService.get_role(db, role_id)
        if not role:
            return None
        
        if role_update.name:
            role.name = role_update.name
        if role_update.description is not None:
            role.description = role_update.description
        if role_update.permissions is not None:
            role.permissions = role_update.permissions
        
        db.commit()
        db.refresh(role)
        return role
    
    @staticmethod
    def delete_role(db: Session, role_id: str) -> bool:
        """Delete admin role"""
        role = AdminRoleService.get_role(db, role_id)
        if not role:
            return False
        
        # Check if role is assigned to users
        user_count = db.query(User).filter(User.admin_role_id == role_id).count()
        if user_count > 0:
            raise ValueError(f"Role is assigned to {user_count} users")
        
        db.delete(role)
        db.commit()
        return True


class AdminSettingService:
    """Service for managing admin settings"""
    
    @staticmethod
    def create_setting(
        db: Session,
        setting_create: AdminSettingCreate,
        updated_by: str
    ) -> AdminSetting:
        """Create new setting"""
        setting = AdminSetting(
            id=str(uuid4()),
            category=setting_create.category,
            key=setting_create.key,
            value=setting_create.value,
            type=setting_create.type,
            description=setting_create.description,
            is_secret=setting_create.is_secret,
            updated_by=updated_by
        )
        db.add(setting)
        db.commit()
        db.refresh(setting)
        return setting
    
    @staticmethod
    def get_setting(db: Session, category: str, key: str) -> Optional[AdminSetting]:
        """Get setting by category and key"""
        return db.query(AdminSetting).filter(
            and_(
                AdminSetting.category == category,
                AdminSetting.key == key
            )
        ).first()
    
    @staticmethod
    def get_setting_by_id(db: Session, setting_id: str) -> Optional[AdminSetting]:
        """Get setting by ID"""
        return db.query(AdminSetting).filter(AdminSetting.id == setting_id).first()
    
    @staticmethod
    def list_settings(
        db: Session,
        category: Optional[str] = None,
        skip: int = 0,
        limit: int = 50
    ) -> tuple[List[AdminSetting], int]:
        """List settings"""
        query = db.query(AdminSetting)
        
        if category:
            query = query.filter(AdminSetting.category == category)
        
        total = query.count()
        settings = query.offset(skip).limit(limit).all()
        return settings, total
    
    @staticmethod
    def update_setting(
        db: Session,
        setting_id: str,
        setting_update: AdminSettingUpdate,
        updated_by: str
    ) -> Optional[AdminSetting]:
        """Update setting"""
        setting = AdminSettingService.get_setting_by_id(db, setting_id)
        if not setting:
            return None
        
        if setting_update.value is not None:
            setting.value = setting_update.value
        if setting_update.description is not None:
            setting.description = setting_update.description
        if setting_update.is_secret is not None:
            setting.is_secret = setting_update.is_secret
        
        setting.updated_by = updated_by
        db.commit()
        db.refresh(setting)
        return setting
    
    @staticmethod
    def delete_setting(db: Session, setting_id: str) -> bool:
        """Delete setting"""
        setting = AdminSettingService.get_setting_by_id(db, setting_id)
        if not setting:
            return False
        
        db.delete(setting)
        db.commit()
        return True
    
    @staticmethod
    def get_category_settings(db: Session, category: str) -> Dict[str, Any]:
        """Get all settings in a category as dict"""
        settings = db.query(AdminSetting).filter(
            AdminSetting.category == category
        ).all()
        
        result = {}
        for setting in settings:
            value = "***REDACTED***" if setting.is_secret else setting.value
            result[setting.key] = value
        return result


class AdminAuditService:
    """Service for admin audit logging"""
    
    @staticmethod
    def log_action(
        db: Session,
        admin_id: str,
        action: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        old_values: Optional[Dict] = None,
        new_values: Optional[Dict] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> AdminAuditLog:
        """Log admin action"""
        audit = AdminAuditLog(
            id=str(uuid4()),
            admin_id=admin_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            old_values=old_values,
            new_values=new_values,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        db.add(audit)
        db.commit()
        db.refresh(audit)
        return audit
    
    @staticmethod
    def get_audit_log(db: Session, log_id: str) -> Optional[AdminAuditLog]:
        """Get audit log by ID"""
        return db.query(AdminAuditLog).filter(AdminAuditLog.id == log_id).first()
    
    @staticmethod
    def list_audit_logs(
        db: Session,
        filter_params: AdminAuditLogFilter,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[List[AdminAuditLog], int]:
        """List audit logs with filtering"""
        query = db.query(AdminAuditLog)
        
        if filter_params.admin_id:
            query = query.filter(AdminAuditLog.admin_id == filter_params.admin_id)
        
        if filter_params.action:
            query = query.filter(AdminAuditLog.action == filter_params.action)
        
        if filter_params.resource_type:
            query = query.filter(AdminAuditLog.resource_type == filter_params.resource_type)
        
        if filter_params.resource_id:
            query = query.filter(AdminAuditLog.resource_id == filter_params.resource_id)
        
        if filter_params.start_date:
            query = query.filter(AdminAuditLog.created_at >= filter_params.start_date)
        
        if filter_params.end_date:
            query = query.filter(AdminAuditLog.created_at <= filter_params.end_date)
        
        total = query.count()
        logs = query.order_by(desc(AdminAuditLog.created_at)).offset(skip).limit(limit).all()
        return logs, total
    
    @staticmethod
    def get_user_audit_logs(
        db: Session,
        user_id: str,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[List[AdminAuditLog], int]:
        """Get audit logs for specific user"""
        query = db.query(AdminAuditLog).filter(
            AdminAuditLog.resource_type == "user",
            AdminAuditLog.resource_id == user_id
        )
        total = query.count()
        logs = query.order_by(desc(AdminAuditLog.created_at)).offset(skip).limit(limit).all()
        return logs, total


class AdminUserService:
    """Service for managing users by admins"""
    
    @staticmethod
    def get_user(db: Session, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return db.query(User).filter(User.id == user_id).first()
    
    @staticmethod
    def list_users(
        db: Session,
        search: Optional[str] = None,
        role_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[List[User], int]:
        """List users"""
        query = db.query(User)
        
        if search:
            search_term = f"%{search}%"
            query = query.filter(
                (User.email.ilike(search_term)) |
                (User.full_name.ilike(search_term))
            )
        
        if role_id:
            query = query.filter(User.admin_role_id == role_id)
        
        total = query.count()
        users = query.order_by(User.created_at).offset(skip).limit(limit).all()
        return users, total
    
    @staticmethod
    def disable_user(db: Session, user_id: str) -> Optional[User]:
        """Disable user account"""
        user = AdminUserService.get_user(db, user_id)
        if not user:
            return None
        
        user.is_active = False
        db.commit()
        db.refresh(user)
        return user
    
    @staticmethod
    def enable_user(db: Session, user_id: str) -> Optional[User]:
        """Enable user account"""
        user = AdminUserService.get_user(db, user_id)
        if not user:
            return None
        
        user.is_active = True
        db.commit()
        db.refresh(user)
        return user
    
    @staticmethod
    def assign_role(db: Session, user_id: str, role_id: str) -> Optional[User]:
        """Assign role to user"""
        user = AdminUserService.get_user(db, user_id)
        if not user:
            return None
        
        # Verify role exists
        role = db.query(AdminRole).filter(AdminRole.id == role_id).first()
        if not role:
            raise ValueError(f"Role {role_id} not found")
        
        user.admin_role_id = role_id
        db.commit()
        db.refresh(user)
        return user


class AdminIPWhitelistService:
    """Service for IP whitelist management"""
    
    @staticmethod
    def add_ip(
        db: Session,
        ip_address: str,
        created_by: str,
        description: Optional[str] = None,
    ) -> AdminIPWhitelist:
        """Add IP to whitelist"""
        ip_entry = AdminIPWhitelist(
            id=str(uuid4()),
            ip_address=ip_address,
            description=description,
            created_by=created_by,
        )
        db.add(ip_entry)
        db.commit()
        db.refresh(ip_entry)
        return ip_entry
    
    @staticmethod
    def remove_ip(db: Session, ip_address: str) -> bool:
        """Remove IP from whitelist"""
        entry = db.query(AdminIPWhitelist).filter(
            AdminIPWhitelist.ip_address == ip_address
        ).first()
        
        if not entry:
            return False
        
        db.delete(entry)
        db.commit()
        return True
    
    @staticmethod
    def list_ips(db: Session) -> List[AdminIPWhitelist]:
        """List all whitelisted IPs"""
        return db.query(AdminIPWhitelist).order_by(AdminIPWhitelist.created_at).all()
    
    @staticmethod
    def is_ip_whitelisted(db: Session, ip_address: str) -> bool:
        """Check if IP is whitelisted"""
        entry = db.query(AdminIPWhitelist).filter(
            AdminIPWhitelist.ip_address == ip_address
        ).first()
        return entry is not None
