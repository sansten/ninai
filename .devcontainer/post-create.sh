#!/bin/bash
# Post-create script for dev container
# Runs after container is initialized

set -e

echo "=== Ninai2 Dev Container Setup ==="
echo ""

# Update Python package manager
echo "Updating pip and poetry..."
pip install --user --upgrade pip poetry

# Setup backend
echo "Setting up backend environment..."
cd /workspace/repos/ninai/backend

# Create and activate virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing backend dependencies..."
poetry install

# Setup database
echo "Waiting for PostgreSQL to be ready..."
for i in {1..30}; do
  if pg_isready -h localhost -U ninai &> /dev/null; then
    echo "PostgreSQL is ready!"
    break
  fi
  echo "Waiting... ($i/30)"
  sleep 1
done

# Run migrations
echo "Running database migrations..."
alembic upgrade head

# Setup frontend
echo "Setting up frontend environment..."
cd /workspace/frontend
npm install

# Create dev user and org
echo "Creating development user and organization..."
cd /workspace/repos/ninai/backend
source venv/bin/activate
python -c "
import asyncio
from app.models import User, Organization, OrgUser
from app.services.auth import hash_password
from sqlalchemy import text

async def setup_dev():
    from app.db import AsyncSessionLocal
    async with AsyncSessionLocal() as session:
        try:
            # Create dev org if not exists
            result = await session.execute(
                text('SELECT id FROM organizations WHERE name = :name'),
                {'name': 'Ninai Development'}
            )
            org_id = result.scalar()
            
            if not org_id:
                result = await session.execute(
                    text('''INSERT INTO organizations (name, tier) 
                           VALUES (:name, :tier) RETURNING id'''),
                    {'name': 'Ninai Development', 'tier': 'pro'}
                )
                org_id = result.scalar()
            
            # Create dev user if not exists
            result = await session.execute(
                text('SELECT id FROM users WHERE email = :email'),
                {'email': 'dev@ninai.local'}
            )
            user_id = result.scalar()
            
            if not user_id:
                hashed_pwd = hash_password('dev123456')
                result = await session.execute(
                    text('''INSERT INTO users (email, hashed_password) 
                           VALUES (:email, :password) RETURNING id'''),
                    {'email': 'dev@ninai.local', 'password': hashed_pwd}
                )
                user_id = result.scalar()
            
            # Add user to org as admin
            result = await session.execute(
                text('SELECT id FROM org_users WHERE user_id = :user_id AND org_id = :org_id'),
                {'user_id': user_id, 'org_id': org_id}
            )
            org_user_id = result.scalar()
            
            if not org_user_id:
                await session.execute(
                    text('''INSERT INTO org_users (user_id, org_id, role) 
                           VALUES (:user_id, :org_id, :role)'''),
                    {'user_id': user_id, 'org_id': org_id, 'role': 'org_admin'}
                )
            
            await session.commit()
            print('✓ Dev user and org created')
        except Exception as e:
            print(f'✓ Dev user already exists or error: {e}')

asyncio.run(setup_dev())
" || true

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Dev credentials:"
echo "  Email: dev@ninai.local"
echo "  Password: dev123456"
echo ""
echo "Next steps:"
echo "  1. Start backend: cd repos/ninai/backend && source venv/bin/activate && python -m uvicorn app.main:app --reload"
echo "  2. Start frontend: cd frontend && npm start"
echo "  3. Run tests: cd repos/ninai/backend && python -m pytest tests/ -v"
echo ""
echo "Services running on:"
echo "  Backend: http://localhost:8000"
echo "  Frontend: http://localhost:3000"
echo "  Prometheus: http://localhost:9090"
echo "  Grafana: http://localhost:3100"
echo ""
