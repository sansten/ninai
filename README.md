# Ninai - The Secure Memory Layer for AI Agents

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue.svg)](https://www.typescriptlang.org/)

A multi-tenant agentic memory system for RAG and agent builders. Store, govern, and retrieve agent knowledge securely with Postgres Row-Level Security (RLS) and explainable vector search—no cross-tenant data leaks.

**Get working in 30 minutes**: `docker compose up` with Postgres + Qdrant, seed demo data, and start building.

## What is Ninai?

Ninai is an **open-source memory layer** for AI agents and RAG systems. It solves the **"secure multi-tenant memory"** problem:

- **Store agent knowledge** as immutable versions with audit trails
- **Enforce data isolation** via Postgres RLS at the database layer (not just app logic)  
- **Retrieve with explainability** — vector search + SQL verification + scoring breakdown logs
- **Optional human review** — route new knowledge through approval queues before promotion to long-term memory
- **Simple to adopt** — single docker-compose, no Kubernetes, optional Grafana/Redis/Elastic

**Ideal for**: agent builders, RAG teams, and platforms needing provably secure multi-tenant memory.

### Why NOT enterprise-bloat?

Ninai's core is OSS. Enterprise add-ons (SCIM, managed SLAs, advanced eval drift) are **optional** and **clearly separated**. The OSS version works right now with just Postgres + Qdrant.

##  Core Features (OSS)

- **RLS-First Multi-Tenancy**: Postgres Row-Level Security enforces org isolation at the data layer—attackers cannot query across organizations even if app code has a vulnerability
- **Hierarchical RBAC (Org → Team)**: Scoped access control with least-privilege defaults
- **Explainable Vector Retrieval**: Qdrant search + Postgres RLS re-verification + scoring breakdown logs
- **Governed Knowledge Lifecycle**: Submit → review queue → approve/reject → immutable version → optional promotion to memory
- **Audit-Ready Traceability**: Request-level `X-Trace-ID`, versioned knowledge, approval comments, change history
- **Non-Admin Reviewers**: Delegate knowledge approvals without giving out admin passwords
- **Works Today with Minimal Stack**: Postgres + Qdrant; optional Redis/Elastic/Grafana for scale
- Interested to contribute to opensource send out an email : opensource@sansten.com 

##  Editions

Ninai comes in three flavors:

| Feature | OSS | Enterprise Managed | Enterprise Self-Managed |
|---------|-----|-------------------|------------------------|
| **Multi-tenant RLS** | ✅ | ✅ | ✅ |
| **RBAC / Knowledge Review** | ✅ | ✅ | ✅ |
| **Vector Search** | ✅ | ✅ | ✅ |
| **OIDC SSO** | ✅ | ✅ | ✅ |
| **Grafana / Audit** | ✅ | ✅ | ✅ |
| **SCIM Provisioning** | ❌ | ✅ | ✅ |
| **SLA / Managed Hosting** | ❌ | ✅ | ❌ |
| **Advanced Eval + Drift** | ❌ | ✅ | ✅ |
| **DLQ / Dead Letter Handling** | ❌ | ✅ | ✅ |

See [docs/EDITIONS.md](docs/EDITIONS.md) for the full comparison.

## Quick Start (30 Minutes)

### Prerequisites

- Docker & Docker Compose
- Python 3.11+ (for local dev)
- Node.js 18+ (for frontend dev)

### 1. Clone & Start the Stack

```bash
git clone https://github.com/your-org/ninai.git
cd ninai

# Start PostgreSQL, Qdrant, Redis, and the app
docker compose up -d --build

# Run migrations
docker compose exec backend alembic upgrade head

# Seed demo data (org, users, sample knowledge)
docker compose exec backend python -m scripts.seed_data
```

### 2. Open the App

- **Frontend**: http://localhost:3000  
- **API Docs**: http://localhost:8000/docs

### 3. Log In

After seeding, use demo credentials:

| User | Email | Password | Role |
|------|-------|----------|------|
| Admin | `admin@ninai.dev` | `admin1234` | Org Admin |
| Reviewer | `reviewer@ninai.dev` | `review1234` | Knowledge Reviewer |
| Agent Builder | `dev@ninai.dev` | `dev12345` | Team Member |

### 4. Try It

1. Log in as the agent builder
2. Go to **Knowledge** → **Submit** → add a sample memory item
3. Switch to reviewer account and **approve** in the review queue
4. Query via the **Search** tab or API

##  Architecture

### Data Plane vs. Control Plane

**Data Plane** (OSS core):
- Memory read/write/retrieve
- Vector search (Qdrant) + RLS re-verification (Postgres)
- Knowledge versioning and audit logs

**Control Plane** (OSS + Enterprise):
- Policy configuration (RBAC, RLS policies)
- Knowledge review and approval workflows
- Admin operations and user lifecycle

### Directory Structure

```
ninai/
├── backend/                 # FastAPI + SQLAlchemy
│   ├── app/
│   │   ├── api/            # v1 routes
│   │   ├── core/           # Config, security, DB
│   │   ├── middleware/     # Tenant context, audit logging
│   │   ├── models/         # SQLAlchemy ORM
│   │   ├── schemas/        # Pydantic + API responses
│   │   ├── services/       # Business logic
│   │   └── main.py
│   ├── alembic/            # Database migrations
│   ├── tests/              # Unit + integration tests
│   └── scripts/            # Utils (seed, setup, etc.)
├── frontend/               # React 18 + TypeScript
│   ├── src/
│   │   ├── components/     # Reusable UI
│   │   ├── contexts/       # Auth, tenant, settings
│   │   ├── hooks/          # Custom React hooks
│   │   ├── pages/          # Full pages (Memory, Review, etc.)
│   │   ├── services/       # API client
│   │   └── types/          # TypeScript interfaces
│   └── package.json
├── docker/                 # Postgres, Qdrant, Redis configs
├── docs/                   # User & operator guides
├── docker-compose.yml      # Dev environment
└── docker-compose.prod.yml # (Optional) prod baseline
```

##  Security Guarantees

### Multi-Tenancy Enforcement

**"No query without a tenant filter"** is enforced at the database layer:

```python
# Before ANY database operation, set the tenant context:
async with session.begin():
    await session.execute(
        text("SET LOCAL app.current_org_id = :org_id"), 
        {"org_id": org_id}
    )
    # Now all queries inherit org_id via Postgres RLS
```

**Postgres RLS Policies** on critical tables (memory, knowledge, audit):

```sql
CREATE POLICY org_isolation ON memory_metadata
  USING (organization_id = current_setting('app.current_org_id')::uuid);
```

This means:
- ✅ App code mistakes cannot leak data across orgs
- ✅ SQL injection cannot escape the org
- ✅ Even a stolen admin account can only see their org

### Vector Search Parity

Qdrant searches always include the org filter, with Postgres re-verification:

```python
# 1. Qdrant filter always includes org
filter = Filter(must=[FieldCondition(key="organization_id", match=MatchValue(value=org_id))])

# 2. Results are verified against Postgres RLS
verified_ids = await verify_access(qdrant_ids, session)
```

### Explainable Retrieval

Every vector search returns a scoring breakdown so you can audit why a result was ranked:

```json
{
  "id": "...",
  "text": "...",
  "score": 0.92,
  "explanation": {
    "semantic_score": 0.85,
    "recency_boost": 0.07,
    "relevance_filters": ["tag:agent-behavior"]
  }
}
```

##  Key Concepts

### Knowledge Lifecycle

1. **Submit**: Agent/developer submits knowledge → stored with `status=pending_review`
2. **Review Queue**: Reviewer sees pending items, reads metadata, approves/rejects
3. **Approval**: On approval, knowledge gets `version=1`, timestamp, approver audit trail
4. **Memory Promotion** (optional): Reviewer optionally tags knowledge for long-term memory with topics/tags
5. **Retrieval**: Agents retrieve via search; results include version info and audit trail

### Policy-as-Code

RBAC and RLS policies are versioned and auditable:

```python
# Define org-level policies
policies = {
    "org_isolation": "SELECT * FROM memory WHERE org_id = current_org",
    "team_isolation": "SELECT * FROM knowledge WHERE team_id IN (current_teams)",
    "review_gate": "knowledge must pass review before promotion"
}
```

### Tenant Isolation Story

**Strict guarantees**:
- Postgres enforces RLS at the session layer (no app-level bugs can escape)
- Qdrant filters + Postgres re-verification (defense in depth)
- No shared indexes or caches across orgs
- Audit logs prove that every query respects org boundaries

**Test suite proves no leakage**:
- Cross-tenant query tests fail loudly
- RLS policy tests verify org isolation is airtight

##  Testing

### Fast Unit Tests (No DB)

```bash
python -m pytest -q
```

### Integration Tests (Postgres Required)

Start Postgres and set the flag:

```bash
docker compose up -d postgres
RUN_POSTGRES_TESTS=1 python -m pytest -q backend/tests/
```

### Cross-Tenant Leakage Tests

```bash
RUN_POSTGRES_TESTS=1 python -m pytest -q -k "test_no_cross_tenant_leak"
```

Confirms:
- User A cannot read org B's memory
- Team A cannot read team B's knowledge
- Admin account still respects org boundaries

## 🔧 Configuration

### Kill-Switch Simple Mode

For dev/testing, run with just **Postgres + Qdrant**:

```bash
docker compose --profile lite up
```

This disables Redis, Elastic, Grafana. Ninai still works; just slower under load.

### Schema Versioning & Migrations

Ninai uses **Alembic** for database evolution:

```bash
# Create a migration
alembic revision --autogenerate -m "add new column"

# Apply
alembic upgrade head

# Rollback
alembic downgrade -1
```

**Migration strategy**:
- Migrations are idempotent
- Backward-compatible schema changes only (for rolling deployments)
- Qdrant payload schema evolved via versioned fields; old versions still readable

### Latency & Cost Budgets

**Expected p95 latencies**:
- Memory write: < 100ms (Postgres + Qdrant)
- Vector search: < 200ms (Qdrant + RLS re-verification)
- Knowledge approval: < 50ms (Postgres only)

**Cost control**:
- Query result caching (optional Redis)
- Token rate limiting per org/user
- Qdrant payload size limits

**Defaults**:
- Rate limit: 1000 req/min per org
- Retention: 90 days (configurable)
- Cache TTL: 5 minutes (configurable)

### Data Lifecycle

**Right to Delete (GDPR)**:

```python
DELETE FROM memory WHERE organization_id = org_id AND user_id = user_id
# Triggers cascade: memory_metadata, audit_logs, versions
```

**Retention Policies**:

```python
# Archival after 90 days
DELETE FROM memory WHERE created_at < NOW() - INTERVAL '90 days' AND status = 'archived'
```

**Export & Portability**:

```bash
# Export all org data as JSON
GET /api/v1/orgs/{org_id}/export
```

## 🛡️ Threat Model

### What Ninai Protects Against

✅ **Cross-tenant data leakage** via query mistakes  
✅ **SQL injection** across org boundaries (RLS blocks it)  
✅ **Unauthorized knowledge approval** (strict role-based gates)  
✅ **Tampering with audit logs** (immutable append-only)  
✅ **Prompt injection in retrieved knowledge** (PolicyGuard validates retrieval context)

### What Ninai Does NOT Protect Against

❌ **Denial of service** (rate limiting is basic; bring your own DDoS protection)  
❌ **Network eavesdropping** (use TLS/mTLS in production)  
❌ **Compromised application server** (if your backend is hacked, all bets are off)  
❌ **Insider threats** (org admins can see all org data by design)  
❌ **Supply chain attacks** (keep dependencies updated)

### PolicyGuard (Tool Misuse Prevention)

Ninai includes **PolicyGuard**: a policy engine that validates knowledge before retrieval to prevent:
- Prompt injection attacks
- Out-of-context retrieval
- Unauthorized tool calls

```python
# PolicyGuard validation before returning results
policy_check = await policeguard.validate(
    knowledge_id=...,
    query_context=...,
    retrieved_context=...
)
if not policy_check.allowed:
    return []  # Fail closed
```

##  Observability & SLOs

### Key Metrics

- **Retrieval Latency** (p50, p95, p99)
- **RLS Re-verification Latency**
- **Knowledge Review Queue Depth** (SLO: < 10 min for approval)
- **Cross-Tenant Audit Events** (SLO: 0)
- **Token Usage** (per org, capped)

### Grafana Dashboards (Included)

- Memory I/O dashboard
- Review queue SLA tracking
- Tenant isolation audit board
- Error rate + SLO burn-down

### Recommended SLOs

| Metric | OSS Target | Enterprise Target |
|--------|-----------|------------------|
| Retrieval latency (p95) | 500ms | 200ms |
| Approval SLA | 4 hours | 1 hour |
| Audit log durability | 24 hours | 7 days |
| Availability | 99.5% | 99.99% |

##  Deployment

### Development (Docker Compose)

```bash
docker compose up -d --build
```

### Production (Self-Managed)

See [../ninai-deploy/self-managed/](../ninai-deploy/self-managed/) for:
- Kubernetes YAML manifests
- Helm charts with values overlays
- Terraform for GCP infrastructure
- Runbooks for ops (backup, restore, scaling)

### Enterprise Managed (Sansten AI)

Hosted on Google Cloud with:
- Managed database + backups
- SLA monitoring + incident response
- Auto-scaling based on load
- Advanced eval + drift detection

Contact: [sales@sanstenaix.dev](mailto:sales@sanstenaix.dev)

##  Authentication

### Local Email/Password

Default. No external dependencies.

```bash
AUTH_MODE=password  # Only email/password login
```

### OIDC SSO (Keycloak, Azure AD, Google, etc.)

```bash
AUTH_MODE=both
OIDC_ISSUER=https://login.microsoftonline.com/tenant-id/v2.0
OIDC_CLIENT_ID=<client-id>
OIDC_ALLOWED_EMAIL_DOMAINS=example.com
OIDC_DEFAULT_ORG_SLUG=ninai-demo
```

**Frontend env vars**:

```bash
VITE_OIDC_AUTHORITY=https://login.microsoftonline.com/tenant-id/v2.0
VITE_OIDC_CLIENT_ID=<client-id>
VITE_OIDC_REDIRECT_URI=http://localhost:3000/auth/oidc/callback
```

## Development

### Backend Tests

```bash
# Fast (no DB)
python -m pytest -q

# With Postgres
RUN_POSTGRES_TESTS=1 python -m pytest -q

# With performance tests
RUN_POSTGRES_TESTS=1 RUN_PERF_TESTS=1 python -m pytest -k perf
```

### Frontend Lint + Build

```bash
cd frontend
npm run lint
npm run build
```

### Code Style

- **Python**: Black, isort, ruff
- **TypeScript**: ESLint, Prettier
- **Commits**: Conventional Commits

## Documentation

- [docs/EDITIONS.md](docs/EDITIONS.md) — OSS vs Enterprise feature matrix
- [docs/SECURITY.md](docs/SECURITY.md) — RLS policies, audit logging, threat model
- [docs/API.md](docs/API.md) — REST API reference
- [../ninai-deploy/](../ninai-deploy/) — Deployment runbooks & infrastructure

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Built With

- [FastAPI](https://fastapi.tiangolo.com/)
- [SQLAlchemy](https://www.sqlalchemy.org/)
- [Qdrant](https://qdrant.tech/)
- [React 18](https://react.dev/)
- [Tailwind CSS](https://tailwindcss.com/)
- [PostgreSQL 15+](https://www.postgresql.org/)

---

**Questions?** Open an issue on GitHub or email [support@sansten.com](mailto:support@sansten.com).

**Want to use Enterprise?** See [../ninai-enterprise/](../ninai-enterprise/) or contact sales at [sales@sanstenaix.dev](mailto:sales@sansten.com).
