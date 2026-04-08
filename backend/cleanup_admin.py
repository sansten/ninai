import asyncio
import uuid
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from app.models import User, AdminRole
from app.core.config import settings
from sqlalchemy import select, delete

async def reset_and_create():
    engine = create_async_engine(settings.DATABASE_URL)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        # Delete existing admin user
        await session.execute(delete(User).filter(User.email == 'admin@test.com'))
        await session.commit()
        
        # Delete existing admin role
        await session.execute(delete(AdminRole).filter(AdminRole.name == 'Admin'))
        await session.commit()
        
        print("Cleaned up existing admin data")
    
    await engine.dispose()

asyncio.run(reset_and_create())
