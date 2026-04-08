# Ninai2 Development Environments

This directory contains configuration for cloud and container-based development environments.

## Quick Start

### GitHub Codespaces (Recommended for Cloud Development)

1. Click the **Code** button on GitHub
2. Select **Codespaces** tab
3. Click **Create codespace on main**
4. Wait for environment to initialize (~3 minutes)
5. Services will automatically start:
   - Backend: http://localhost:8000
   - Frontend: http://localhost:3000

No installation needed! Full development environment in your browser.

### Gitpod (Alternative Cloud IDE)

1. Visit https://gitpod.io#https://github.com/sansten/ninai
2. Or click the **Gitpod button** in the README
3. VS Code will open in your browser
4. Services auto-start (see "Backend", "Frontend", "Tests" tabs)

### VS Code Dev Containers (Local Docker)

**Prerequisites:**
- Docker Desktop (with WSL2 on Windows)
- VS Code with Remote Containers extension

**Setup:**
1. Open Ninai2 repository in VS Code
2. Click **><** in bottom-left corner
3. Select **Reopen in Container**
4. VS Code will build and start the container
5. Wait for `post-create.sh` to complete
6. Services ready in terminal

## Environment Services

| Service | Port | Purpose | Auto-Start |
|---------|------|---------|-----------|
| FastAPI Backend | 8000 | REST API | ✓ |
| React Frontend | 3000 | Web UI | ✓ |
| PostgreSQL | 5432 | Database | ✓ |
| Redis | 6379 | Cache/Broker | ✓ |
| Qdrant | 6333 | Vector DB | ✓ |
| Ollama | 11434 | Local LLM | ✓ |
| Prometheus | 9090 | Metrics | ✓ |
| Grafana | 3100 | Dashboards | ✓ |

## Development Workflow

### Running the Backend

```bash
cd repos/ninai/backend
source venv/bin/activate
python -m uvicorn app.main:app --reload --host 0.0.0.0
```

API available at http://localhost:8000
Docs at http://localhost:8000/docs (Swagger UI)

### Running the Frontend

```bash
cd frontend
npm start
```

Frontend available at http://localhost:3000

### Running Tests

```bash
cd repos/ninai/backend
source venv/bin/activate
python -m pytest tests/ -v
```

### Database Migrations

```bash
cd repos/ninai/backend
source venv/bin/activate
alembic upgrade head  # Apply pending migrations
alembic downgrade -1  # Revert one migration
alembic revision -m "description"  # Create new migration
```

### Creating a Database Commit

```bash
cd repos/ninai/backend
source venv/bin/activate
alembic revision --autogenerate -m "add new_column to users table"
# Review the migration in alembic/versions/
alembic upgrade head
```

## Development Credentials

**Default Dev User:**
- Email: `dev@ninai.local`
- Password: `dev123456`
- Role: `org_admin`
- Org: `Ninai Development`

**API Access:**
1. Login via UI or API:
   ```bash
   curl -X POST http://localhost:8000/auth/login \
     -H "Content-Type: application/json" \
     -d '{"username":"dev@ninai.local","password":"dev123456"}'
   ```

2. Use returned token:
   ```bash
   curl http://localhost:8000/api/v1/org \
     -H "Authorization: Bearer <token>"
   ```

## Useful Commands

### View Logs

```bash
# Codespaces/Gitpod
docker-compose logs -f backend

# Dev Container
# Check VS Code terminals
```

### Database Access

```bash
# PostgreSQL
psql postgresql://ninai:ninai@localhost:5432/ninai_dev

# Redis CLI
redis-cli -h localhost -p 6379

# Qdrant Web UI
http://localhost:6333/dashboard
```

### Code Quality

```bash
# Format code with Black
black repos/ninai/backend/app

# Lint with Ruff
ruff check repos/ninai/backend/app

# Type check with Pylance
cd repos/ninai/backend && python -m pyright app

# Run pytest with coverage
pytest tests/ --cov=app --cov-report=html
```

## Troubleshooting

### Services Not Starting

```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs service_name

# Restart all services
docker-compose restart
```

### Database Connection Refused

```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Verify credentials
psql postgres://ninai:ninai@localhost:5432/ninai_dev -c "SELECT version();"
```

### Backend Startup Issues

```bash
# Check Python environment
source venv/bin/activate
python --version

# Verify dependencies
pip list | grep ninai

# Check database migrations
cd repos/ninai/backend
alembic current
```

### Frontend Build Issues

```bash
# Clear npm cache
npm cache clean --force

# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install

# Check Node version
node --version  # Should be 18+
```

### Out of Disk Space (Codespaces only)

```bash
# Clean up unused images and containers
docker system prune -a

# Remove node_modules temporarily
cd frontend && rm -rf node_modules
npm install || npm cache clean --force && npm install
```

## Configuration Files

### `.gitpod.yml`
- Gitpod workspace configuration
- Defines init/onStart phases
- VS Code extension/port declarations
- Startup tasks

### `.devcontainer/devcontainer.json`
- VS Code Dev Container configuration
- Docker image and features
- VS Code extensions and settings
- Environment variables

### `.devcontainer/Dockerfile`
- Extended dev container image
- System dependencies
- Docker client integration

### `.devcontainer/post-create.sh`
- Post-initialization script
- Database setup
- User creation
- Dependency installation

## CI/CD Integration

**Local Testing Before Push:**
```bash
# Run full test suite locally
cd repos/ninai/backend && python -m pytest tests/ -x -q

# Check code quality
ruff check app/

# Format code
black app/
```

**GitHub Actions**: These run on every push (see `.github/workflows/`)

## Best Practices

1. **Commit credentials**: Never commit `.env` files or API keys
   - Dev credentials in comments are safe
   - Use environment variables for secrets

2. **Database**: Use migrations for schema changes
   - Always create alembic migrations
   - Never modify schema directly

3. **Testing**: Run tests before committing
   - Local: `pytest tests/ -x -q`
   - Fast subset: `pytest tests/test_specific.py -v`

4. **Code style**: Use formatters before committing
   - Black: `black app/`
   - Ruff: `ruff check app/ --fix`

5. **Branches**: Use feature branches
   ```bash
   git checkout -b feat/my-feature
   # Make changes, test, commit
   git push origin feat/my-feature
   # Create PR on GitHub
   ```

## Performance Tips

### Speed Up Codespaces
- Use 8+ CPU machines (in Codespaces settings)
- Close unused terminals/ports
- Disable extensions you don't need

### Speed Up Docker Builds
- Use image caching: never rebuild from scratch
- Install dependencies in Dockerfile (not container)
- Minimize layer count

## Advanced: Custom Setup

To customize the development environment:

1. **Edit `.gitpod.yml`** (for Gitpod)
   - Add apt packages under `init` phase
   - Add startup commands under `onStart` phase

2. **Edit `.devcontainer/devcontainer.json`** (for VS Code/Codespaces)
   - Add features from https://containers.dev
   - Modify VS Code extensions and settings

3. **Edit `.devcontainer/Dockerfile`** (for custom base image)
   - Install system-level dependencies
   - Configure environment variables

4. **Edit `.devcontainer/post-create.sh`** (for setup automation)
   - Initialize custom data
   - Configure services

Then commit and push. Gitpod/Codespaces will rebuild with your changes.

## Getting Help

- **Gitpod Docs**: https://www.gitpod.io/docs
- **Codespaces Docs**: https://docs.github.com/en/codespaces
- **Dev Containers Docs**: https://containers.dev
- **Ninai2 Issues**: https://github.com/sansten/ninai/issues
