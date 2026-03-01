import asyncio
import os
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from passlib.context import CryptContext

async def fix_password():
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    hashed = pwd_context.hash("admin123")
    print("Generated hash:", hashed)
    
    postgres_url = f"postgresql+asyncpg://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
    engine = create_async_engine(postgres_url, echo=False)
    
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        await session.execute(
            text("UPDATE users SET hashed_password = :hash WHERE email = 'admin@ninai.dev'"),
            {"hash": hashed}
        )
        await session.commit()
        print("Password updated successfully")
    
    await engine.dispose()

asyncio.run(fix_password())
