"""
Setup UserRole assignment for admin user
"""
import asyncio
import sys
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

# Add backend to path
sys.path.insert(0, '.')

from app.core.config import settings
from app.models.user import User, UserRole, Role
from app.models.organization import Organization


async def main():
    """Setup user roles for admin user"""
    
    # Create async engine
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as db:
        try:
            # Find admin user
            user_result = await db.execute(
                select(User).where(User.email == "admin@test.com")
            )
            user = user_result.scalar_one_or_none()
            
            if not user:
                print("ERROR: admin@test.com user not found")
                return
            
            print(f"Found user: {user.email} (id={user.id})")
            
            # Find or create default organization
            org_result = await db.execute(
                select(Organization).where(Organization.slug == "default")
            )
            org = org_result.scalar_one_or_none()
            
            if not org:
                # Create default organization
                org = Organization(
                    name="Default Organization",
                    slug="default",
                    is_active=True,
                    settings={},
                )
                db.add(org)
                await db.flush()
                print(f"Created organization: {org.name} (id={org.id})")
            else:
                print(f"Found organization: {org.name} (id={org.id})")
            
            # Find or create admin Role (not AdminRole - this is the RBAC role in "roles" table)
            admin_permissions = [
                "system:read",
                "system:write",
                "users:read",
                "users:write",
                "roles:read",
                "roles:write",
                "settings:read",
                "settings:write",
                "audit:read",
            ]
            
            role_result = await db.execute(
                select(Role).where(
                    Role.name == "org_admin",
                    Role.is_system == True,
                )
            )
            admin_role = role_result.scalar_one_or_none()
            
            if not admin_role:
                # Create system admin role
                admin_role = Role(
                    name="org_admin",
                    display_name="System Administrator",
                    description="Full system administration access",
                    permissions=admin_permissions,
                    is_system=True,
                    organization_id=None,  # System role (org-agnostic)
                )
                db.add(admin_role)
                await db.flush()
                print(f"Created system Role: {admin_role.name} (id={admin_role.id})")
            else:
                print(f"Found system Role: {admin_role.name} (id={admin_role.id})")
            
            # Check if UserRole already exists
            user_role_result = await db.execute(
                select(UserRole).where(
                    UserRole.user_id == user.id,
                    UserRole.organization_id == org.id,
                    UserRole.role_id == admin_role.id,
                )
            )
            existing_user_role = user_role_result.scalar_one_or_none()
            
            if existing_user_role:
                print(f"UserRole already exists: {existing_user_role}")
                return
            
            # Create UserRole assignment
            user_role = UserRole(
                user_id=user.id,
                role_id=admin_role.id,
                organization_id=org.id,
                grant_reason="System admin setup",
            )
            db.add(user_role)
            await db.commit()
            
            print(f"Created UserRole: user={user.id}, role={admin_role.id}, org={org.id}")
            print("SUCCESS: Admin user role assignment complete")
            
        finally:
            await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
