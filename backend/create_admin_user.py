import asyncio
import uuid
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from app.models import User, AdminRole
from passlib.context import CryptContext
from app.core.config import settings
from sqlalchemy import select

async def create_admin():
    # Create engine and session
    engine = create_async_engine(settings.DATABASE_URL)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')
        
        # Create admin role if it doesn't exist
        result = await session.execute(select(AdminRole).filter(AdminRole.name == 'Admin'))
        admin_role = result.scalar()
        
        if not admin_role:
            admin_role = AdminRole(
                id=str(uuid.uuid4()),
                name='Admin',
                description='System Administrator',
                permissions=['system:read', 'system:write', 'users:read', 'users:write', 'roles:read', 'roles:write', 'settings:read', 'settings:write', 'audit:read'],
                is_system=True
            )
            session.add(admin_role)
            await session.commit()
        
        # Check if user exists
        result = await session.execute(select(User).filter(User.email == 'admin@test.com'))
        existing = result.scalar()
        
        if existing:
            print(f'Admin user already exists (id: {existing.id})')
        else:
            user = User(
                email='admin@test.com',
                hashed_password=pwd_context.hash('admin123'),
                full_name='Admin User',
                is_admin=True,
                is_active=True,
                is_superuser=False,
                role='admin',
                admin_role_id=admin_role.id
            )
            session.add(user)
            await session.commit()
            print(f'Created admin user: admin@test.com / admin123 (id: {user.id})')
    
    await engine.dispose()

if __name__ == '__main__':
    asyncio.run(create_admin())
