#!/bin/bash
# Quick start: bring up Docker and run migrations

set -e

echo "Starting Docker Compose services (Postgres, Redis, etc.)..."
docker-compose up -d postgres redis qdrant

echo "Waiting for Postgres to be healthy..."
sleep 10

echo "Running Alembic migrations..."
cd backend
python -m alembic upgrade head

echo "✓ Database migration complete!"
echo ""
echo "Next steps:"
echo "  - Start backend: docker-compose up -d backend"
echo "  - View logs: docker-compose logs -f backend"
echo "  - Stop: docker-compose down"
