import asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from app.models import User, AdminRole
from app.core.config import settings
from sqlalchemy import select

async def check():
    engine = create_async_engine(settings.DATABASE_URL)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        # Check admin user
        result = await session.execute(select(User).filter(User.email == 'admin@test.com'))
        user = result.scalar()
        if user:
            print(f"User found: {user.email}, admin_role_id={user.admin_role_id}, is_admin={user.is_admin}")
        else:
            print("No admin user found")
        
        # Check admin roles
        result = await session.execute(select(AdminRole))
        roles = result.scalars().all()
        print(f"AdminRoles in DB: {len(roles)}")
        for role in roles:
            print(f"  - {role.name} (id={role.id}, permissions={role.permissions})")
    
    await engine.dispose()

asyncio.run(check())
