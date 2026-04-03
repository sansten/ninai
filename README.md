# Ninai — Cognitive OS for Enterprise AI Agents

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)

**Apps write raw data. Ninai makes sense of it.**

Ninai is not a memory cache or a vector search wrapper. It is a Cognitive OS: a reasoning layer that sits between your enterprise AI agents and their stored knowledge, continuously enriching what comes in, linking entities across silos, assembling full context on read, and reasoning proactively — without any of that logic living in your agent or application code.

---

## The Core Problem

Enterprise AI agents accumulate knowledge in silos. A CRM agent knows about customers. A ticketing agent knows about incidents. A document agent knows about procedures. None of them know what the others know, and none of them can reason across boundaries.

Dropping a vector database in front of each agent doesn't fix this. You still get isolated retrieval with no understanding of causality, conflict, temporal change, or entity identity across systems.

Ninai approaches this differently:

- Every piece of data written to Ninai is **automatically enriched**: entities extracted, normalized, linked to the world model, scored for credibility, checked for conflicts with existing knowledge
- Every read assembles **full context**: causal chains, related episodes, peer signals, anomalies, attention-weighted relevance across silos
- The system **reasons proactively**: generating goals it should pursue, monitoring for anomalies, consolidating redundant knowledge, flagging inconsistencies before they surface in agent responses
- **Multi-tenant isolation** is enforced at the database layer via PostgreSQL RLS — application bugs cannot leak data across org boundaries

---

## What Ninai Actually Does

### Data Enrichment on Write

When an application writes a memory, Ninai runs a pipeline of agents before storing it:

- **Semantic normalization** — synonyms resolved, units standardized, terminology canonicalized across the org vocabulary
- **Entity resolution** — "AWS" and "Amazon Web Services" in the same context become the same entity; cross-silo duplicates are merged
- **World model integration** — new facts are linked into a graph of entities, relationships, and causal edges
- **Credibility scoring** — source reliability, consensus across silos, and confidence are factored into a credibility signal
- **Conflict detection** — incoming knowledge is checked against existing facts; conflicts are flagged for resolution or human review
- **Silo propagation** — facts relevant to other silos are forwarded, so finance learns what ops knows when it matters

### Context Assembly on Read

When an agent queries Ninai, it doesn't get raw search results. It gets assembled context:

- Semantically matched memories ranked by an 8-component activation score (relevance, recency, frequency, importance, confidence, context match, provenance, risk)
- Causal chains explaining why things happened
- Episodic groupings that reconstruct what was happening around the time of relevant events
- Peer signals from agents and silos that bear on the query
- Uncertainty estimates so the agent knows what it doesn't know
- A narrative synthesis — a structured summary of what the system believes, with confidence

### Autonomous Reasoning

Ninai runs background processes that don't wait to be asked:

- **Predictive monitoring** — tracks world state, detects when predictions diverge from reality
- **Anomaly detection** — identifies outliers across entity behavior and system metrics
- **Memory consolidation** — merges redundant facts, strengthens weak memories, prunes stale knowledge
- **Memory decay** — relevance ages over time so old, unconfirmed facts don't crowd out recent signal
- **Proactive memory push** — pushes relevant context to agents before they ask for it
- **Autonomous goal generation** — identifies knowledge gaps, low-reliability tools, and high-activity domains; generates goals the system should pursue on its own

### Human-in-the-Loop

Not every decision should be fully automated:

- **Human review queue** — ambiguous facts, high-stakes conflicts, and low-confidence outputs are routed to reviewers
- **Audit and explainability trail** — every agent decision is logged with inputs, reasoning, and outputs; fully queryable
- **Adaptive conflict resolution** — conflicts that can be resolved heuristically are; those that can't are escalated

### Multi-Agent Coordination

Ninai treats agents as first-class entities:

- **Orchestration bus** — agents register capabilities and emit signals; others subscribe and react
- **Theory of mind modeling** — the system builds belief models of users and peer agents: what does this user know? what is this agent weak at? adjusts tone and collaboration strategy accordingly
- **Playbook execution tracking** — multi-step procedures are tracked across turns; partial progress is preserved
- **Multi-turn goal tracking** — goals decomposed across many interactions stay coherent

---

## Architecture

```
Applications
    │  write raw data
    ▼
Enrichment Pipeline (per write)
    ├── SemanticNormalizationAgent
    ├── EntityResolutionAgent
    ├── WorldModelAgent          → World Model Graph (Qdrant + Postgres)
    ├── CredibilityAgent
    ├── ConflictDetectionAgent
    ├── AdaptiveConflictResolutionAgent
    └── SiloPropagationAgent     → other silos

Storage Layer
    ├── PostgreSQL (RLS-enforced, per-tenant)   — facts, entities, audit, review queue
    ├── Qdrant                                  — vector embeddings
    └── Redis                                   — cache, Celery broker

Background Processes (Celery)
    ├── PredictiveMonitorAgent
    ├── AnomalyDetectionAgent
    ├── MemoryDecayAgent
    ├── MemoryConsolidationAgent
    ├── ProactiveMemoryPushAgent
    └── AutonomousGoalGenerationAgent

Query Path (per read)
    ├── QueryIntelligenceAgent   — intent classification, query expansion
    ├── ContextAmplifierAgent    — expert context from silos
    ├── OrgAttentionAgent        — attention-weighted relevance
    ├── CausalReasoningAgent     — causal chain assembly
    ├── EpisodicGroupingAgent    — episode reconstruction
    ├── TemporalReasoningAgent   — time-aware context ordering
    ├── NarrativeSynthesisAgent  — structured summary with confidence
    └── UncertaintyReportingAgent — what we don't know

Cognitive Layer
    ├── GoalDecompositionAgent
    ├── MetaCognitivePlanningAgent
    ├── TheoryOfMindAgent        — user + peer agent belief models
    ├── PlaybookAgent + PlaybookExecutionTrackerAgent
    ├── MultiTurnGoalTrackingAgent
    ├── AuditTrailAgent
    ├── HumanReviewQueueAgent
    └── OrchestrationBusAgent

LLM Backend
    └── Ollama (local, default: qwen2.5:0.5b)
        └── heuristic fallback on every agent (AGENT_STRATEGY=heuristic|llm)
```

**Key architectural rules across all agents:**
- Every agent has a heuristic path that requires no LLM — the system degrades gracefully when Ollama is unavailable
- Every service call is wrapped in try/except with fault isolation — one agent failing does not cascade
- All database queries are scoped via `set_tenant_context()` before execution — RLS enforces the rest
- EMA-based learning (`new = 0.75 * prev + 0.25 * outcome`) for strategy and tool reliability tracking

---

## Stack

| Component | Technology |
|-----------|-----------|
| API | FastAPI (async) |
| ORM | SQLAlchemy async |
| Database | PostgreSQL 15+ with RLS |
| Vector store | Qdrant |
| Cache / broker | Redis |
| Task queue | Celery |
| Local LLM | Ollama (qwen2.5:0.5b default) |
| Runtime | Python 3.12+ |

---

## Quick Start

```bash
git clone https://github.com/sansten/ninai.git
cd ninai
docker compose up -d --build
```

Open **http://localhost:3000**.

Demo credentials (change in production):
- `dev@ninai.dev` / `dev12345` — submit, search, view
- `reviewer@ninai.dev` / `review1234` — approve/reject submissions
- `admin@ninai.dev` / `admin1234` — full access

### Write a memory

```python
from ninai import NinaiClient

client = NinaiClient(api_base_url="http://localhost:8000/api/v1", api_key="dev-token")

# Ninai enriches this automatically: entity extraction, credibility scoring,
# conflict checking, world model linking, silo propagation
client.memories.create(
    content="Deploy to production via: kubectl apply -f manifests/prod.yaml",
    tags=["devops", "kubernetes"],
    classification="internal"
)
```

### Query with assembled context

```python
results = client.memories.search(
    query="How do we deploy to production?",
    tags=["devops"],
    limit=5
)
# Returns: scored results + causal context + episode summary +
#          uncertainty estimate + narrative synthesis
```

### REST

```bash
curl -X POST http://localhost:8000/api/v1/memories/search \
  -H "Authorization: Bearer dev-token" \
  -H "Content-Type: application/json" \
  -d '{"query_text": "kubernetes deployment", "limit": 5}'
```

---

## Security

Ninai makes multi-tenant isolation a foundation, not a feature.

**How it works:** Every request sets a Postgres session variable (`SET LOCAL app.current_org_id = 'org_abc'`). RLS policies attached to every table enforce this at the database layer — your application code never needs a WHERE clause for tenant filtering. A bug in application logic cannot return another org's data.

**What is protected:**
- Cross-tenant data leakage — enforced at database layer, survives app bugs
- SQL injection — RLS applies regardless of query shape
- Unauthorized knowledge promotion — review queue gates all submissions
- Decision audit trail — every agent action is logged immutably

**What you manage:**
- Network TLS (configure your reverse proxy or load balancer)
- DDoS protection
- Credential security and SSO integration

See [docs/SECURITY.md](docs/SECURITY.md) for full threat model.

---

## Cognitive Capabilities (Phase Status)

| Phase | Capability | Status |
|-------|-----------|--------|
| 1–3 | Uncertainty gating, adaptive strategy, cognitive context bus | Done |
| 4 | Unified World Model Graph | Done |
| 5 | Predictive World State Monitor | Done |
| 6 | Semantic Normalization | Done |
| 7 | Entity Resolution | Done |
| 8 | Expert Context Amplifier | Done |
| 9 | Cross-Silo Signal Propagation | Done |
| 10 | Organizational Attention Model | Done |
| 11 | Proactive Memory Push | Done |
| 12 | Causal Reasoning Engine | Done |
| 13 | Cross-Silo Conflict Detection | Done |
| 14 | Adaptive Conflict Resolution | Done |
| 15 | Memory Decay & Relevance Aging | Done |
| 16 | Memory Consolidation Engine | Done |
| 17 | Temporal Reasoning Engine | Done |
| 18 | Episodic Memory Grouping | Done |
| 19 | Credibility Scoring | Done |
| 20 | Playbook / Skill Memory | Done |
| 21 | Goal Decomposition | Done |
| 22 | Uncertainty Reporting | Done |
| 23 | Narrative Synthesis | Done |
| 24 | Feedback Integration & Self-Correction | Done |
| 25 | Anomaly Detection | Done |
| 26 | Enrichment API Surface | Done |
| 27 | Query Intelligence Layer | Done |
| 28 | Feature Readiness Engine | Done |
| 29 | Cross-Agent Orchestration Bus | Done |
| 30 | Tenant-Scoped Knowledge Graph API | Done |
| 31 | Audit & Explainability Trail | Done |
| 32 | Human Review Queue | Done |
| 33 | Playbook Execution Tracker | Done |
| 34 | Multi-Turn Goal Tracking | Done |
| 35 | Adaptive Enrichment Budget | Done |
| 36 | Meta-Cognitive Planning | Done |
| 37 | Autonomous Goal Generation & Intrinsic Motivation | Done |
| 38 | Theory of Mind & Multi-Agent Modeling | Done |
| 39 | Causal API Surface | Planned |
| 40 | Memory Sleep Cycle | Planned |
| 41 | Compositional Generalization Engine | Planned |
| 42 | Emotional & Affective Memory | Planned |
| 43 | Multimodal Deep Memory | Planned |
| 44 | Federated Memory & Collective Intelligence | Planned |

**3,252 tests passing** across the full agent suite.

---

## Testing

```bash
# Unit tests (agents, services — no database required for most)
cd backend
python -m pytest tests/ -x -q

# With Postgres (full integration suite)
docker compose up -d postgres redis qdrant
python -m pytest tests/ -q
```

All agents test both the heuristic path (no LLM) and the LLM path (with Ollama mock), plus fallback behavior when the LLM fails.

---

## Benchmark Results and Industry Comparison

Last measured on Kaggle-backed unit benchmark suite:

```bash
cd backend
python -m tests.benchmarks.run_all --mode unit --strategy heuristic --dataset kaggle --json
python -m tests.benchmarks.run_all --mode unit --strategy llm --dataset kaggle --ollama-model qwen2.5:7b --json
python -m tests.benchmarks.run_all --mode unit --strategy heuristic --dataset kaggle --runs 3 --json
```

### Ninai Results (Current)

| Metric | Heuristic | LLM (qwen2.5:7b) | Heuristic (3-run mean) |
|---|---:|---:|---:|
| Duration (seconds) | 1.188 | 239.516 | 0.837 |
| Composite score (quality x reliability) | 0.7952 | n/a | 0.7952 |
| Conflict F1 | 0.6667 | 0.7273 | 0.6667 |
| Goal accuracy | 0.8438 | 0.6875 | 0.8438 |
| Goal scored accuracy | 0.8438 | 0.6825 | 0.8438 |
| LLM success rate | 0.0 | 0.4922 | 0.0 |
| Fallback rate | 0.0 | 0.5078 | 0.0 |
| Recall@10 | 0.875 | 0.875 | 0.875 |

### LLM Model Comparison (Kaggle, unit mode)

The table below compares available Ollama models on the same benchmark runner. This is the model-to-model view for Ninai's LLM path.

| Metric | llama3.2:latest | qwen2.5:7b | deepseek-coder-v2:16b |
|---|---:|---:|---:|
| Duration (seconds) | 163.000 | 239.516 | n/a (latest run failed) |
| Conflict F1 | 0.7273 | 0.7273 | n/a (latest run failed) |
| Goal accuracy | 0.7188 | 0.6875 | n/a (latest run failed) |
| Goal scored accuracy | 0.7188 | 0.6825 | n/a (latest run failed) |
| LLM success rate | 0.5000 | 0.4922 | n/a (latest run failed) |
| Fallback rate | 0.5000 | 0.5078 | n/a (latest run failed) |
| Recall@10 | 0.875 | 0.875 | n/a (latest run failed) |

Re-run model comparison:

```bash
cd backend
python -m tests.benchmarks.run_all --mode unit --strategy llm --dataset kaggle --ollama-model llama3.2:latest --json
python -m tests.benchmarks.run_all --mode unit --strategy llm --dataset kaggle --ollama-model qwen2.5:7b --json
python -m tests.benchmarks.run_all --mode unit --strategy llm --dataset kaggle --ollama-model deepseek-coder-v2:16b --json
```

### How This Compares to Industry Benchmarks

Ninai's benchmark is product-specific (enterprise memory + reliability pipeline), while industry AGI benchmarks are capability-specific and cross-model comparable.

| Benchmark | URL | Primary capability measured | Directly comparable to Ninai score? |
|---|---|---|---|
| MMLU | https://github.com/hendrycks/test | Broad academic/professional knowledge QA | No (different task format and labels) |
| BIG-bench Hard (BBH) | https://github.com/suzgunmirac/BIG-Bench-Hard | Hard reasoning tasks across diverse categories | No (reasoning set, not enterprise memory workflow) |
| GSM8K | https://github.com/openai/grade-school-math | Multi-step math reasoning | No (math-focused benchmark) |
| HumanEval | https://github.com/openai/human-eval | Code generation correctness | No (coding benchmark) |
| SWE-bench | https://www.swebench.com/ | Real-world software issue resolution | No (software engineering agent benchmark) |
| MMMU | https://mmmu-benchmark.github.io/ | Multimodal reasoning across university-level tasks | No (multimodal benchmark) |

### Practical Interpretation

- Ninai benchmark: strong for deployment readiness (fallback behavior, tool reliability, domain conflict detection, retrieval quality).
- Industry benchmarks: strong for external model capability comparison.
- Best practice: report both.

Recommended reporting split:
1. External panel (for comparability): MMLU/BBH + HumanEval/SWE-bench (+ MMMU if multimodal scope).
2. Internal panel (for product readiness): Ninai Kaggle benchmark with composite score, LLM success rate, and domain confusion matrices.

---

## Configuration

**Backend** (`backend/.env`):
```bash
POSTGRES_HOST=localhost
POSTGRES_USER=ninai
POSTGRES_PASSWORD=ninai_dev_password
POSTGRES_DB=ninai

QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:0.5b
OLLAMA_TIMEOUT_SECONDS=5.0

AGENT_STRATEGY=llm      # or: heuristic (no LLM, faster, fully deterministic)
APP_ENV=development
```

**Frontend** (`frontend/.env`):
```bash
VITE_API_BASE=http://localhost:8000/api/v1
VITE_OIDC_CLIENT_ID=...   # optional SSO
VITE_OIDC_AUTHORITY=...
```

---

## Authentication

**Email/password** works out of the box, no external dependencies.

**OIDC SSO** (Azure AD, Keycloak, Google, Okta, etc.):
```bash
AUTH_MODE=both
OIDC_ISSUER=https://login.microsoftonline.com/tenant-id/v2.0
OIDC_CLIENT_ID=your-client-id
OIDC_ALLOWED_EMAIL_DOMAINS=example.com
```

---

## Deployment

### Development
```bash
docker compose up -d --build
```

### Production
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) — self-managed runbook
- [ninai-deploy/](../ninai-deploy/) — Kubernetes manifests and Helm charts
- [ninai-enterprise/](../ninai-enterprise/) — enterprise editions

---

## Directory Structure

```
backend/
├── app/
│   ├── agents/          # 38+ cognitive agents (one file per agent)
│   ├── api/             # REST endpoints
│   ├── services/        # Cognitive loop, strategy learning, world model
│   ├── models/          # SQLAlchemy models + RLS
│   ├── middleware/       # Auth, tenant context, rate limiting
│   └── core/            # Config, database, bootstrap
├── tests/               # 3,252 passing tests
└── alembic/             # Schema migrations
```

---

## Contributing

1. Fork and clone
2. `cd backend && python -m pytest tests/ -x -q` — make sure everything passes
3. Create a feature branch
4. Keep commits concise and imperative (`feat: add X`, `fix: Y in Z`)
5. Open a PR with a plain description and test plan

### Code style
- Python: Black, isort, ruff
- Commits: [Conventional Commits](https://www.conventionalcommits.org/)

### Getting help
- [GitHub Discussions](https://github.com/sansten/ninai/discussions)
- [Report bugs](https://github.com/sansten/ninai/issues)
- Email: opensource@sansten.com

---

## Support

| Edition | Support | Cost |
|---------|---------|------|
| OSS | Community | Free (MIT) |
| Enterprise Managed | Sansten, SLA 99.99% | Contact sales |
| Enterprise Self-Managed | Commercial license | Contact sales |

**sales@sansten.com** — [docs/EDITIONS.md](docs/EDITIONS.md)

---

## License

MIT — see [LICENSE](LICENSE).
