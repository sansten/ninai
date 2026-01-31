import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

async def check():
    engine = create_async_engine('postgresql+asyncpg://ninai:ninai_dev_password@localhost:5432/ninai')
    async with engine.begin() as conn:
        result = await conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'agent_processes'"))
        row = result.fetchone()
        print('✓ agent_processes table exists!' if row else '✗ agent_processes table NOT found')
        
        # Show table structure
        result = await conn.execute(text("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'agent_processes' ORDER BY ordinal_position"))
        print("\nTable columns:")
        for col_name, data_type in result.fetchall():
            print(f"  {col_name}: {data_type}")
        
    await engine.dispose()

asyncio.run(check())
