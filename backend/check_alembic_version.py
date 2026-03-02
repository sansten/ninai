import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
engine = create_engine(os.getenv('DATABASE_URL'))

with engine.connect() as conn:
    result = conn.execute(text('SELECT * FROM alembic_version'))
    for row in result:
        print(f"Current version: {row[0]}")
