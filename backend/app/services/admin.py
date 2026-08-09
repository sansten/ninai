"""Admin business logic services"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import uuid4

from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

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
    async def create_role(
        db: AsyncSession,
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
        await db.commit()
        await db.refresh(role)
        return role

    @staticmethod
    async def get_role(db: AsyncSession, role_id: str) -> Optional[AdminRole]:
        """Get role by ID"""
        result = await db.execute(select(AdminRole).where(AdminRole.id == role_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def list_roles(db: AsyncSession, skip: int = 0, limit: int = 50) -> tuple[List[AdminRole], int]:
        """List all roles"""
        total = (await db.execute(select(func.count()).select_from(AdminRole))).scalar_one()
        result = await db.execute(
            select(AdminRole).order_by(AdminRole.created_at).offset(skip).limit(limit)
        )
        roles = list(result.scalars().all())
        return roles, total

    @staticmethod
    async def update_role(
        db: AsyncSession,
        role_id: str,
        role_update: AdminRoleUpdate
    ) -> Optional[AdminRole]:
        """Update admin role"""
        role = await AdminRoleService.get_role(db, role_id)
        if not role:
            return None

        if role_update.name:
            role.name = role_update.name
        if role_update.description is not None:
            role.description = role_update.description
        if role_update.permissions is not None:
            role.permissions = role_update.permissions

        await db.commit()
        await db.refresh(role)
        return role

    @staticmethod
    async def delete_role(db: AsyncSession, role_id: str) -> bool:
        """Delete admin role"""
        role = await AdminRoleService.get_role(db, role_id)
        if not role:
            return False

        # Check if role is assigned to users
        user_count = (
            await db.execute(
                select(func.count()).select_from(User).where(User.admin_role_id == role_id)
            )
        ).scalar_one()
        if user_count > 0:
            raise ValueError(f"Role is assigned to {user_count} users")

        await db.delete(role)
        await db.commit()
        return True


class AdminSettingService:
    """Service for managing admin settings"""

    @staticmethod
    async def create_setting(
        db: AsyncSession,
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
        await db.commit()
        await db.refresh(setting)
        return setting

    @staticmethod
    async def get_setting(db: AsyncSession, category: str, key: str) -> Optional[AdminSetting]:
        """Get setting by category and key"""
        result = await db.execute(
            select(AdminSetting).where(
                and_(
                    AdminSetting.category == category,
                    AdminSetting.key == key
                )
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_setting_by_id(db: AsyncSession, setting_id: str) -> Optional[AdminSetting]:
        """Get setting by ID"""
        result = await db.execute(select(AdminSetting).where(AdminSetting.id == setting_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def list_settings(
        db: AsyncSession,
        category: Optional[str] = None,
        skip: int = 0,
        limit: int = 50
    ) -> tuple[List[AdminSetting], int]:
        """List settings"""
        base = select(AdminSetting)
        count_base = select(func.count()).select_from(AdminSetting)

        if category:
            base = base.where(AdminSetting.category == category)
            count_base = count_base.where(AdminSetting.category == category)

        total = (await db.execute(count_base)).scalar_one()
        result = await db.execute(base.offset(skip).limit(limit))
        settings = list(result.scalars().all())
        return settings, total

    @staticmethod
    async def update_setting(
        db: AsyncSession,
        setting_id: str,
        setting_update: AdminSettingUpdate,
        updated_by: str
    ) -> Optional[AdminSetting]:
        """Update setting"""
        setting = await AdminSettingService.get_setting_by_id(db, setting_id)
        if not setting:
            return None

        if setting_update.value is not None:
            setting.value = setting_update.value
        if setting_update.description is not None:
            setting.description = setting_update.description
        if setting_update.is_secret is not None:
            setting.is_secret = setting_update.is_secret

        setting.updated_by = updated_by
        await db.commit()
        await db.refresh(setting)
        return setting

    @staticmethod
    async def delete_setting(db: AsyncSession, setting_id: str) -> bool:
        """Delete setting"""
        setting = await AdminSettingService.get_setting_by_id(db, setting_id)
        if not setting:
            return False

        await db.delete(setting)
        await db.commit()
        return True

    @staticmethod
    async def get_category_settings(db: AsyncSession, category: str) -> Dict[str, Any]:
        """Get all settings in a category as dict"""
        result = await db.execute(
            select(AdminSetting).where(AdminSetting.category == category)
        )
        settings = result.scalars().all()

        result_dict = {}
        for setting in settings:
            value = "***REDACTED***" if setting.is_secret else setting.value
            result_dict[setting.key] = value
        return result_dict


class AdminAuditService:
    """Service for admin audit logging"""

    @staticmethod
    async def log_action(
        db: AsyncSession,
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
        await db.commit()
        await db.refresh(audit)
        return audit

    @staticmethod
    async def get_audit_log(db: AsyncSession, log_id: str) -> Optional[AdminAuditLog]:
        """Get audit log by ID"""
        result = await db.execute(select(AdminAuditLog).where(AdminAuditLog.id == log_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def list_audit_logs(
        db: AsyncSession,
        filter_params: AdminAuditLogFilter,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[List[AdminAuditLog], int]:
        """List audit logs with filtering"""
        base = select(AdminAuditLog)
        count_base = select(func.count()).select_from(AdminAuditLog)

        def _apply(stmt):
            if filter_params.admin_id:
                stmt = stmt.where(AdminAuditLog.admin_id == filter_params.admin_id)
            if filter_params.action:
                stmt = stmt.where(AdminAuditLog.action == filter_params.action)
            if filter_params.resource_type:
                stmt = stmt.where(AdminAuditLog.resource_type == filter_params.resource_type)
            if filter_params.resource_id:
                stmt = stmt.where(AdminAuditLog.resource_id == filter_params.resource_id)
            if filter_params.start_date:
                stmt = stmt.where(AdminAuditLog.created_at >= filter_params.start_date)
            if filter_params.end_date:
                stmt = stmt.where(AdminAuditLog.created_at <= filter_params.end_date)
            return stmt

        total = (await db.execute(_apply(count_base))).scalar_one()
        result = await db.execute(
            _apply(base).order_by(desc(AdminAuditLog.created_at)).offset(skip).limit(limit)
        )
        logs = list(result.scalars().all())
        return logs, total

    @staticmethod
    async def get_user_audit_logs(
        db: AsyncSession,
        user_id: str,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[List[AdminAuditLog], int]:
        """Get audit logs for specific user"""
        conditions = (
            AdminAuditLog.resource_type == "user",
            AdminAuditLog.resource_id == user_id,
        )
        total = (
            await db.execute(select(func.count()).select_from(AdminAuditLog).where(*conditions))
        ).scalar_one()
        result = await db.execute(
            select(AdminAuditLog)
            .where(*conditions)
            .order_by(desc(AdminAuditLog.created_at))
            .offset(skip)
            .limit(limit)
        )
        logs = list(result.scalars().all())
        return logs, total


class AdminUserService:
    """Service for managing users by admins"""

    @staticmethod
    async def get_user(db: AsyncSession, user_id: str) -> Optional[User]:
        """Get user by ID"""
        result = await db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def list_users(
        db: AsyncSession,
        search: Optional[str] = None,
        role_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[List[User], int]:
        """List users"""
        base = select(User)
        count_base = select(func.count()).select_from(User)

        if search:
            search_term = f"%{search}%"
            cond = (User.email.ilike(search_term)) | (User.full_name.ilike(search_term))
            base = base.where(cond)
            count_base = count_base.where(cond)

        if role_id:
            base = base.where(User.admin_role_id == role_id)
            count_base = count_base.where(User.admin_role_id == role_id)

        total = (await db.execute(count_base)).scalar_one()
        result = await db.execute(base.order_by(User.created_at).offset(skip).limit(limit))
        users = list(result.scalars().all())
        return users, total

    @staticmethod
    async def disable_user(db: AsyncSession, user_id: str) -> Optional[User]:
        """Disable user account"""
        user = await AdminUserService.get_user(db, user_id)
        if not user:
            return None

        user.is_active = False
        await db.commit()
        await db.refresh(user)
        return user

    @staticmethod
    async def enable_user(db: AsyncSession, user_id: str) -> Optional[User]:
        """Enable user account"""
        user = await AdminUserService.get_user(db, user_id)
        if not user:
            return None

        user.is_active = True
        await db.commit()
        await db.refresh(user)
        return user

    @staticmethod
    async def assign_role(db: AsyncSession, user_id: str, role_id: str) -> Optional[User]:
        """Assign role to user"""
        user = await AdminUserService.get_user(db, user_id)
        if not user:
            return None

        # Verify role exists
        role_result = await db.execute(select(AdminRole).where(AdminRole.id == role_id))
        role = role_result.scalar_one_or_none()
        if not role:
            raise ValueError(f"Role {role_id} not found")

        user.admin_role_id = role_id
        await db.commit()
        await db.refresh(user)
        return user


class AdminIPWhitelistService:
    """Service for IP whitelist management"""

    @staticmethod
    async def add_ip(
        db: AsyncSession,
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
        await db.commit()
        await db.refresh(ip_entry)
        return ip_entry

    @staticmethod
    async def remove_ip(db: AsyncSession, ip_address: str) -> bool:
        """Remove IP from whitelist"""
        result = await db.execute(
            select(AdminIPWhitelist).where(AdminIPWhitelist.ip_address == ip_address)
        )
        entry = result.scalar_one_or_none()

        if not entry:
            return False

        await db.delete(entry)
        await db.commit()
        return True

    @staticmethod
    async def list_ips(db: AsyncSession) -> List[AdminIPWhitelist]:
        """List all whitelisted IPs"""
        result = await db.execute(
            select(AdminIPWhitelist).order_by(AdminIPWhitelist.created_at)
        )
        return list(result.scalars().all())

    @staticmethod
    async def is_ip_whitelisted(db: AsyncSession, ip_address: str) -> bool:
        """Check if IP is whitelisted"""
        result = await db.execute(
            select(AdminIPWhitelist).where(AdminIPWhitelist.ip_address == ip_address)
        )
        return result.scalar_one_or_none() is not None
