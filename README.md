# Ninai Memory OS for AI Agents

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue.svg)](https://www.typescriptlang.org/)

**Multi-tenant agent memory, built to not leak data. Postgres RLS, explainable retrieval, full audit trail.**

## The Problem We Solved

We were building a platform where agents needed to remember things—user context, API patterns, decision history. Standard RAG solutions exist, but they all have the same flaw: **they treat memory like a library**, not like a conversation.

Standard RAG:
- Indexing is designed for **huge, diverse corpora** (Wikipedia, documents, web)
- Returns top-10 similar chunks
- Collapses into dense regions when retrieving from bounded dialogue streams

Agent memory is **fundamentally different**:
- Small, coherent, temporally-linked conversations
- Often redundant across sessions (same user, same questions)
- Needs strict isolation (Company A can't read Company B's memories)
- Needs governance (don't let bad information into agent decisions)

So we built Ninai: a memory system designed **for agents, not documents**.

```python
from ninai import NinaiClient

client = NinaiClient(api_key="nai_...", api_base="http://localhost:8000/api/v1")

# Store a memory (goes to review queue)
memory = client.memories.create(
    content="Deploy via: kubectl apply -f manifest.yaml",
    tags=["devops", "kubernetes"],
    classification="internal"
)

# Retrieve with full explanation
results = client.memories.search(
    query="How do I deploy?",
    tags=["devops"],
    limit=5
)
# Includes scoring breakdown, why each result matched, audit trail
```

---

## What Makes Ninai Different

We didn't build another vector database. We built a **memory operating system** with:

- **RLS-first security** — Postgres Row-Level Security enforces org isolation at the database layer. App bugs cannot leak data across tenants. Period.
- **Governed knowledge** — Submit → Review Queue → Approval → Immutable Version. Bad information is caught before agents use it.
- **Explainable retrieval** — Every search result shows you the scoring breakdown. Why did this match? Relevance 0.8, Recency 0.3, Context gate passed? You can see it all.
- **Hierarchical memory** — Messages → Episodes → Semantic Facts → Topics. Prevents dense-region collapse where everything similar gets returned.
- **Uncertainty-gated expansion** — Only include deeper context (raw messages) if they actually reduce the LLM's prediction uncertainty. Saves tokens.
- **No vendor lock-in** — Core is pure open-source (MIT). Enterprise add-ons (SCIM, managed SLAs) are separate and optional.

Most memory systems treat security as a feature. **We made it the foundation.** Everything else builds on that.

---

## What You Get

| What | Why | Trade-off |
|------|-----|-----------|
| **RLS-backed isolation** | No cross-tenant data leaks, even if your app code has bugs | Requires Postgres 13+, slightly higher CPU on complex queries |
| **Knowledge approval workflows** | Bad/outdated info never contaminates agent behavior | Manual process—can bottleneck if you get lots of submissions |
| **8-component activation scoring** | Captures relevance, recency, frequency, importance, confidence, context, provenance, risk | More complex than simple vector similarity, harder to tune |
| **Explicit versioning + audit logs** | Regulatory compliance, ability to understand why an agent made a decision | Adds storage overhead (immutable logs) |
| **Hierarchical memory (4-level)** | Prevents "dense region collapse" in bounded dialogue—better quality retrieval | Overhead: requires semantic distillation, topic clustering |
| **Works today with Postgres + Qdrant** | No Kubernetes, no managed services required | You manage the databases; scale horizontally yourself |

**What we're NOT:**
- A real-time knowledge base with sub-100ms latency (we target p95 <500ms)
- A replacement for full-text search (we focus on semantic + metadata)
- A magic solution to prompt injection (we have PolicyGuard, but still apply defense-in-depth)
- Enterprise-ready with SLAs (OSS—bring your own monitoring/alerting)

---

## Quick Start (5 Minutes)

Honestly, you just need Docker, Postgres, and Qdrant. Everything else is optional.

### Get It Running

```bash
git clone https://github.com/sansten/ninai.git
cd ninai
docker compose up -d --build
```

Open **http://localhost:3000**. You'll see a clean dashboard.

Demo credentials (change these immediately in production):
- **dev@ninai.dev** / `dev12345` — Can submit, search, view
- **reviewer@ninai.dev** / `review1234` — Can approve/reject submissions
- **admin@ninai.dev** / `admin1234` — Full access

### Try It

**Via Python SDK:**
```python
from ninai import NinaiClient

c = NinaiClient(api_base_url="http://localhost:8000/api/v1", api_key="dev-token")

# Create (goes to review queue)
c.memories.create(content="K8s deploy: kubectl apply -f manifest.yaml", tags=["devops"])

# Search (after approval)
results = c.memories.search("kubernetes deployment", limit=5)
for r in results:
    print(f"Score: {r.score:.2f} | {r.content}")
```

**Via REST:**
```bash
curl -X POST http://localhost:8000/api/v1/memories/search \
  -H "Authorization: Bearer dev-token" \
  -H "Content-Type: application/json" \
  -d '{"query_text": "kubernetes", "limit": 5}'
```

Done. You now have multi-tenant memory with RLS enforcement running locally.

---

## How It Works

Think of it like a **post office system for agent memory**.

**The Control Plane** (your local postmaster):
- Routes all submissions through a review queue
- Prevents junk mail (bad memories) from entering the system
- Maintains an audit log of every sorting decision

**The Data Plane** (the actual mail sorting):
- Sorts memories by topic (devops, customer-service, api-design)
- Retrieves what's most relevant to a query
- Enforces that Company A can't access Company B's mail

```
User Story → Submit Memory → Review Queue → Approved Memories
                                ↓ (rejected)
                            Audit Log (why?)

Search Query → Vector Similarity → RLS Filter → Scoring → Explanation
```

Under the hood:
- **Postgres** stores memories with RLS policies (database-level tenant isolation)
- **Qdrant** does semantic search (vector similarity)
- **FastAPI** orchestrates and enforces approval workflows
- **React** gives you the dashboard

### Directory Structure

The codebase is organized by concern:

```
backend/
├── api/              # REST endpoints users call
├── services/         # Business logic (search, approval, scoring)
├── models/           # Database schema + RLS policies
├── middleware/       # Authentication + tenant context
└── core/             # Configuration + database setup

frontend/
├── pages/            # Dashboard, search, review queue
├── components/       # Reusable UI elements
├── hooks/            # React custom hooks (useAuth, useMemory)
└── services/         # API client library

docker/
├── postgres/         # RLS policy scripts
└── qdrant/           # Vector DB configuration
```

Full architecture details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## Security: Why Database-Level Isolation Matters

The problem most memory systems ignore: **your app code will have bugs**. 

When bugs happen, standard setups like this break:

```
Client 1 (Company A)
    ↓
App → "SELECT * FROM memories WHERE owner_id = current_user"
    ↓
Oh wait. A bug set owner_id = NULL
    ↓
Returns Company B's memories
```

Ninai does it differently:

```
Client 1 (Company A) sets org_id = "org_abc"
    ↓
Database executes: "SET LOCAL app.current_org_id = 'org_abc'"
    ↓
App queries "SELECT * FROM memories" (no WHERE clause needed)
    ↓
Database RLS policy silently filters: WHERE org_id = current_setting('app.current_org_id')
    ↓
Returns only Company A's memories, even if app code is buggy
```

**The key difference:** RLS happens at the database layer, before your app code runs. A bug in your approval logic can't leak data across tenants. A SQL injection can't escape the org boundary.

### What We Protect

✅ **Cross-tenant data leakage** — Even with app bugs, you can't see other orgs' memories  
✅ **SQL injection** — RLS applies regardless of query shape  
✅ **Unauthorized approval** — Only designated reviewers can move memories forward  
✅ **Audit trail tampering** — All operations logged immutably  
✅ **Prompt injection** (PolicyGuard) — Bad knowledge blocked before reaching agents

### Your Responsibilities

❌ Network TLS (configure your reverse proxy)  
❌ DDoS protection (Cloudflare, AWS WAF, etc.)  
❌ Compromised credentials (use strong passwords + SSO)  
❌ Insider threats (org admins see all org data—this is expected)  

See [docs/SECURITY.md](docs/SECURITY.md) for threat modeling details.

---

## Testing

### Unit Tests (No Database Required)

```bash
python -m pytest -q
```

### Integration Tests (Postgres Required)

```bash
docker compose up -d postgres
TEST_DB_URL=postgresql://... python -m pytest -q backend/tests/
```

### Cross-Tenant Isolation Tests

```bash
# Verify no data leakage between orgs
python -m pytest -q -k "test_cross_tenant"
```

---

## Deployment

### Development

```bash
docker compose up -d --build
```

### Production

For production-grade deployment, see:
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) — Self-managed runbook  
- [ninai-deploy/](../ninai-deploy/) — Kubernetes manifests & Helm charts
- [ninai-enterprise/](../ninai-enterprise/) — Enterprise editions with professional support

### Environment Variables

**Backend**:
```bash
DATABASE_URL=postgresql://ninai:password@postgres:5432/ninai
QDRANT_URL=http://qdrant:6333
REDIS_URL=redis://redis:6379  # Optional
LOG_LEVEL=INFO
```

**Frontend**:
```bash
VITE_API_BASE=http://localhost:8000/api/v1
VITE_OIDC_CLIENT_ID=...  # Optional SSO
VITE_OIDC_AUTHORITY=...
```

---

## Authentication

### Email/Password (Default)

No external dependencies required.

### OIDC SSO (Optional)

Supports Azure AD, Keycloak, Google, Okta, etc.:

```bash
AUTH_MODE=both
OIDC_ISSUER=https://login.microsoftonline.com/tenant-id/v2.0  
OIDC_CLIENT_ID=your-client-id
OIDC_ALLOWED_EMAIL_DOMAINS=example.com
```

---

## Contributing

We actively welcome contributions!

1. **Read** [CONTRIBUTING.md](CONTRIBUTING.md)
2. **Fork & clone**
3. **Create a feature branch** (`git checkout -b feature/amazing-thing`)
4. **Commit** with [Conventional Commits](https://www.conventionalcommits.org/)
5. **Push & open a PR**

### Code Style

- **Python**: Black, isort, ruff
- **TypeScript**: ESLint, Prettier
- **Docs**: Markdown with working code examples

### Getting Help

- 💬 [GitHub Discussions](https://github.com/sansten/ninai/discussions)
- 📧 opensource@sansten.com
- 🐛 [Report bugs](https://github.com/sansten/ninai/issues)

---

## Roadmap

| Phase | Status | Focus |
|-------|--------|-------|
| 1 | ✅ Done | Core memory CRUD, RLS multi-tenancy, vector search |
| 2 | ✅ Done | Knowledge review queue, approval workflows, audit logging |
| 3 | ✅ Done | Explainable retrieval, co-activation graphs, uncertainty gating |
| 4 | 🚧 In Progress | Benchmarking (LoCoMo/PerLTQA), publication, researcher outreach |

See [docs/ROADMAP.md](docs/ROADMAP.md) for more details.

---

## Support & Editions

| Edition | Support | SLA | Cost |
|---------|---------|-----|------|
| **OSS** | Community | None | Free |
| **Enterprise Managed** | Sansten | 99.99% uptime | Contact sales |
| **Enterprise Self-Managed** | Commercial | Custom | Contact sales |

Interested in enterprise? Email **sales@sansten.com** or see [docs/EDITIONS.md](docs/EDITIONS.md).

---

## Built With

- [FastAPI](https://fastapi.tiangolo.com/) — Modern Python framework
- [SQLAlchemy](https://www.sqlalchemy.org/) — ORM + SQL toolkit
- [Qdrant](https://qdrant.tech/) — Vector database
- [React 18](https://react.dev/) — UI library
- [Tailwind CSS](https://tailwindcss.com/) — Styling
- [PostgreSQL 15+](https://www.postgresql.org/) — Relational DB
- [Alembic](https://alembic.sqlalchemy.org/) — Schema migrations

---

## License

**MIT** — See [LICENSE](LICENSE) for details.

---

## Questions?

- ⭐ Star the repo if it helped!
- 🐛 [Report bugs](https://github.com/sansten/ninai/issues/new)
- 💡 [Share ideas](https://github.com/sansten/ninai/discussions)
- 🤝 [Contribute code](CONTRIBUTING.md)

**Email**: [support@sansten.com](mailto:support@sansten.com) | **Discussions**: [GitHub](https://github.com/sansten/ninai/discussions)
