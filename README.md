# Ninai - Enterprise Agentic AI Memory OS

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue.svg)](https://www.typescriptlang.org/)

An enterprise-grade, multi-tenant agentic AI memory operating system with hierarchical RBAC, 
Row-Level Security (RLS), and vector search capabilities.

## 📌 Project Overview

Ninai is a governed “Memory OS” for enterprise AI agents: capture knowledge, route it through human review, and promote it into durable long-term memory—while enforcing tenant isolation and least-privilege access at the database layer.

- **Security by design**: Postgres RLS-first multi-tenancy + scoped RBAC (org/team) so data access is enforced even if application code makes a mistake
- **Governed knowledge lifecycle**: submit → review queue → approve/reject with comments → publish an immutable version (optionally promote to memory with tags/topics)
- **Operational traceability**: request-level `X-Trace-ID` plumbing + provenance-ready models for end-to-end audit trails
- **Search you can trust**: vector retrieval with Postgres re-verification to preserve “vector + SQL parity” under real access controls
- **LLM-flexible**: designed for hosted providers or local inference (e.g., Ollama) behind organization controls

## 🌟 Features

- **RLS-First Multi-Tenancy**: Postgres Row-Level Security enforces org isolation at the data layer (not just app logic)
- **Hierarchical RBAC (Org → Team)**: Scoped access control for users/teams with least-privilege defaults
- **Governed Knowledge Review (HITL)**: Submit knowledge for approval, route to a dedicated reviewer queue, approve/reject with comments
- **Non-Admin Reviewers**: Admins assign `knowledge_reviewer` capability; reviewers can approve via `/review` without admin portal access
- **Immutable Versions + Traceability**: Knowledge is versioned; approvals publish an explicit version; requests can carry `X-Trace-ID`
- **Promotion to Long-Term Memory (Optional)**: On approval, reviewers can promote items into durable memory with tags/topics mapping
- **Vector Search With SQL Verification**: Qdrant retrieval + Postgres RLS re-verification for secure “vector + SQL parity”
- **Audit & Compliance Foundations**: Central audit logging and security-oriented middleware hooks for regulated environments
- **LLM / Local Model Ready**: Designed to run with hosted LLMs or local inference (e.g., Ollama) behind org controls

## 📦 Editions

Ninai can be adopted in three ways:

1. **Open Source (Community)** — MIT core.
2. **Enterprise (Managed by Sansten AI on Google Cloud)** — Enterprise features + SLAs.
3. **Enterprise (Self-Managed by Client)** — Enterprise features + client-operated.

See [docs/EDITIONS.md](docs/EDITIONS.md) for the one-page matrix.

## 🧭 This Repo (3-repo Setup)

This repo is the **OSS (Community) codebase** and the runtime for the API/UI.

- Enterprise add-on package (private): `../ninai-enterprise`
- Deployment/infra (private): `../ninai-deploy`

### Run OSS only

```bash
docker compose up -d --build
```

### Enable Enterprise (local dev)

1) Install the enterprise add-on in the same Python environment as the backend:

```bash
pip install -e ../ninai-enterprise
```

2) Provide a license token (or you will see `403 feature_not_enabled` for `enterprise.*`):

PowerShell:

```powershell
$env:NINAI_LICENSE_TOKEN = "ninai1.<payload>.<sig>"
```

## 📁 Project Structure

```
ninai/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/            # API routes (v1)
│   │   ├── core/           # Config, security, database
│   │   ├── middleware/     # Tenant context, audit logging
│   │   ├── models/         # SQLAlchemy models
│   │   ├── schemas/        # Pydantic schemas
│   │   ├── services/       # Business logic
│   │   └── main.py         # Application entry
│   ├── alembic/            # Database migrations
│   ├── tests/              # Backend tests
│   └── scripts/            # Utility scripts
├── frontend/               # React + TypeScript frontend
│   ├── src/
│   │   ├── components/     # Reusable components
│   │   ├── contexts/       # React contexts
│   │   ├── hooks/          # Custom hooks
│   │   ├── pages/          # Page components
│   │   ├── services/       # API client
│   │   └── types/          # TypeScript types
│   └── ...
├── docker/                 # Docker configurations
├── docs/                   # Documentation
└── docker-compose.yml      # Development environment
```

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- Node.js 18+
- PostgreSQL 15+ (via Docker)
- Redis 7+ (via Docker)
- Qdrant (via Docker)

### Development Setup (Docker Compose)

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/ninai.git
   cd ninai
   ```

2. **Start the full stack**
   ```bash
   docker compose up -d --build
   ```

3. **Run database migrations**
   ```bash
   docker compose exec backend alembic upgrade head
   ```

4. **Seed demo data (recommended for first run)**
   This creates a demo organization and demo users.
   ```bash
   docker compose exec backend python -m scripts.seed_data
   ```

5. **Open the app**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - Swagger: http://localhost:8000/docs

### Demo Login Credentials

After seeding, you can sign in with:

- Admin: `admin@ninai.dev` / `admin1234`
- Demo user: `demo@ninai.dev` / `demo1234`
- Developer: `dev@ninai.dev` / `dev12345`
- Reviewer: `reviewer@ninai.dev` / `review1234`

### Resetting Your Dev Database

If you want a completely fresh start:

```bash
docker compose down -v
docker compose up -d --build
docker compose exec backend alembic upgrade head
docker compose exec backend python -m scripts.seed_data
```

## 🧰 Development Notes

## ✅ Running Backend Tests

### Fast unit tests (DB-less)

From the repo root:

```bash
python -m pytest -q
```

### Postgres-backed integration playbook (Section 6)

These tests run against a real Postgres database and apply Alembic migrations (required for RLS policies).

1) Start Postgres (Docker Desktop must be running on Windows):

```bash
docker compose up -d postgres
```

2) Run the playbook suite:

PowerShell:

```powershell
$env:RUN_POSTGRES_TESTS = "1"
python -m pytest -q backend/tests/test_requirements_playbook_postgres.py
```

If Postgres is not reachable and `RUN_POSTGRES_TESTS=1` is set, the suite will fail (not skip) to avoid silently missing mandatory coverage.

Optional overrides:

- `POSTGRES_TEST_HOST`, `POSTGRES_TEST_PORT`, `POSTGRES_TEST_USER`, `POSTGRES_TEST_PASSWORD`
- `POSTGRES_TEST_BASE_DB` (defaults to `ninai`) and `POSTGRES_TEST_DB` (defaults to `<base>_test`)

### Optional performance smoke checks

```powershell
$env:RUN_POSTGRES_TESTS = "1"
$env:RUN_PERF_TESTS = "1"
python -m pytest -q backend/tests/test_requirements_playbook_postgres.py -k perf
```

### Frontend ↔ Backend API Routing

- The Dockerized frontend is configured to call the backend through a relative base path: `/api/v1`.
- Vite proxies `/api/v1` to the backend container; if you change proxy settings, remember that `localhost` inside a container is the container itself.

### React DevTools

Install the official React DevTools browser extension for a better development experience:
- https://reactjs.org/link/react-devtools

### React Router Warnings

You may see React Router v6 “future flag” warnings in the console. This repo opts into the recommended v7 future flags to reduce noise.

## 🔐 Authentication (Password + OIDC SSO)

Ninai supports **local email/password** authentication and **OIDC SSO (Option A)**. Corporates can choose:

- `AUTH_MODE=password` (default): only local email/password
- `AUTH_MODE=oidc`: only SSO
- `AUTH_MODE=both`: show both options on the login page

The frontend automatically calls `GET /api/v1/auth/methods` to decide what to show.

### Enable OIDC SSO (Option A)

1) Configure the backend (environment variables on the `backend` service):

- `AUTH_MODE=both`
- `OIDC_ISSUER=<issuer-url>`
- `OIDC_CLIENT_ID=<client-id>`
- `OIDC_DEFAULT_ORG_SLUG=ninai-demo` (or set `OIDC_DEFAULT_ORG_ID=<uuid>`)
- Optional hardening:
   - `OIDC_ALLOWED_EMAIL_DOMAINS=example.com,example.org`
   - `OIDC_DEFAULT_ROLE=member`
   - `OIDC_GROUPS_CLAIM=groups` (provider-specific)
   - `OIDC_GROUP_TO_ROLE_JSON={"Ninai-Org-Admins":"org_admin","Ninai-Members":"member"}`

2) Configure the frontend (environment variables on the `frontend` service):

- `VITE_OIDC_AUTHORITY=<same-as-OIDC_ISSUER>`
- `VITE_OIDC_CLIENT_ID=<same-as-OIDC_CLIENT_ID>`
- Optional:
   - `VITE_OIDC_SCOPE=openid profile email`
   - `VITE_OIDC_REDIRECT_URI=http://localhost:3000/auth/oidc/callback`

3) Add this redirect URL to your identity provider:

- `http://localhost:3000/auth/oidc/callback`

### Common Issuer Examples

- Microsoft Entra ID (Azure AD v2): `https://login.microsoftonline.com/<tenant-id>/v2.0`
- Keycloak (realm issuer): `https://<keycloak-host>/realms/<realm-name>`

## 🔍 Troubleshooting

### Dashboard shows `/api/v1/audit/stats` 500

- Confirm you seeded data and can log in.
- Check backend logs: `docker compose logs -f backend`
- If you recently pulled changes, rebuild containers: `docker compose up -d --build`

## 🔒 Security Model

### Row-Level Security (RLS)

All critical tables enforce RLS with organization isolation:

```sql
-- Example: org_isolation policy
CREATE POLICY org_isolation ON memory_metadata
  USING (organization_id = current_setting('app.current_org_id')::uuid);
```

### Tenant Context

Every request sets database session variables for RLS:

```python
async with session.begin():
    await session.execute(text("SET LOCAL app.current_org_id = :org_id"), {"org_id": org_id})
    await session.execute(text("SET LOCAL app.current_user_id = :user_id"), {"user_id": user_id})
```

### Vector Search Parity

Qdrant searches always include org filter, with Postgres RLS re-verification:

```python
# Qdrant filter always includes organization_id
filter = Filter(must=[FieldCondition(key="organization_id", match=MatchValue(value=org_id))])

# Results are re-verified against Postgres RLS
verified_ids = await verify_access(qdrant_ids, session)
```

## 🧪 Testing

```bash
# Backend tests
cd backend
pytest

# Frontend lint + build
cd frontend
npm run lint
npm run build
```

## 📝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- **Python**: Black + isort + ruff
- **TypeScript**: ESLint + Prettier
- **Commits**: Conventional Commits

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with:
- [FastAPI](https://fastapi.tiangolo.com/)
- [SQLAlchemy](https://www.sqlalchemy.org/)
- [Qdrant](https://qdrant.tech/)
- [React](https://react.dev/)
- [Tailwind CSS](https://tailwindcss.com/)
