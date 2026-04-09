# Ninai Quick Start Guide

## Starting Ninai

Simply run:

```bash
docker compose up -d
```

That's it! The system will:
1. ✅ Start all required services (backend, frontend, databases)
2. ✅ Create all database tables automatically
3. ✅ Seed default organization, roles, and admin user
4. ✅ Be ready to use in ~30 seconds

## Login and Security

Default seed users are intended for local development only.

Credentials are defined in `.env.example`.

**Before exposing the environment to any shared network:**
- change all seeded passwords immediately
- remove or disable demo users you do not need
- enforce your production authentication policy (SSO and/or strong password controls)

> **Security:** These credentials are for local development only.

## Accessing the Application

- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

## Adding New Users

1. Login as admin@ninai.dev
2. Navigate to Users/Settings
3. Add new users with corporate emails
4. Assign appropriate roles (admin, member, viewer)

## Stopping Ninai

```bash
docker compose down
```

## Fresh Start (Reset Everything)

```bash
# Stop and remove all data
docker compose down -v

# Start fresh
docker compose up -d
```

The database will be automatically re-initialized with default data.

## Troubleshooting

### Backend not starting?
```bash
docker logs ninai-backend
```

### Database issues?
```bash
docker logs ninai-postgres
```

### Frontend not loading?
```bash
docker logs ninai-frontend
```

## System Architecture

- **Backend:** FastAPI (Python 3.11)
- **Frontend:** React + TypeScript + Vite
- **Database:** PostgreSQL 15
- **Vector Store:** Qdrant
- **Search:** Elasticsearch
- **Cache/Queue:** Redis (FalkorDB)
- **LLM:** Ollama (local-first)
- **OCR:** Tesseract

## Production Deployment

For production deployment, see:
- [DEPLOYMENT_RUNBOOK.md](DEPLOYMENT_RUNBOOK.md)
- [Ninai_Three_Tier_Architecture_FINAL.md](Ninai_Three_Tier_Architecture_FINAL.md)

## Support

- Documentation: `./docs/`
- Issues: Check logs using `docker compose logs -f`
- Architecture: See `ARCHITECTURE.md`
