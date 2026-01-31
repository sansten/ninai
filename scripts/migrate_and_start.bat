@echo off
REM Quick start: bring up Docker and run migrations (Windows PowerShell)

echo Starting Docker Compose services (Postgres, Redis, etc.)...
docker-compose up -d postgres redis qdrant

echo Waiting for Postgres to be healthy (30 seconds)...
timeout /t 30 /nobreak

echo Running Alembic migrations...
cd backend
d:/Sansten/Projects/Ninai2/.venv/Scripts/python.exe -m alembic upgrade head

echo.
echo Database migration complete!
echo.
echo Next steps:
echo   - Start backend: docker-compose up -d backend
echo   - View logs: docker-compose logs -f backend
echo   - Stop: docker-compose down
