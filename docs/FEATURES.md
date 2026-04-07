# Ninai: Community vs Enterprise Feature Comparison

## Overview

**Ninai** is built on an open-core model:
- **Community Edition** (MIT Licensed): Personal agent deployments with core memory & reasoning
- **Enterprise Edition** (Closed-source plugin): Advanced operations, governance, and observability for teams

---

## Feature Matrix

| Feature | Community | Enterprise | Target User |
|---------|-----------|-----------|-------------|
| **Core Memory System** | ✅ Full | ✅ Full | Everyone |
| **Agent Framework** | ✅ Full | ✅ Full | Everyone |
| **Personal Agents** | ✅ Full | ✅ Full | Individuals, solo builders |
| **Authentication (Password/OIDC)** | ✅ Full | ✅ Full | Everyone |
| **Basic Backups** | ✅ Full | ✅ Full | Everyone |
| **Audit Logging** | ✅ Full | ✅ Full | Everyone |
| **Multi-org Support** | ✅ Full | ✅ Full | Small teams |
| **Admin Settings (Auth Config)** | ✅ Full | ✅ Full | Everyone |
| **SDK & API** | ✅ Full | ✅ Full | Developers |
| **Knowledge Review** | ✅ Full | ✅ Full | Everyone |
| **Data Export** | ✅ Full | ✅ Full | Everyone |

---

## Enterprise-Only Features

### 1. Admin Operations (`admin_operations`)
**What it is:** High-risk operational controls for managing system health and resource usage.

**Components:**
- Policy version management & enforcement
- Memory snapshot creation/restore/verification
- Resource budget throttling
- Alert rules & notifications
- Queue management (pause/resume/drain)
- Pipeline monitoring

**UI Location:** Settings → Admin → Ops & Monitoring

**Target Users:**
- **DevOps/Platform Teams**: Manage queue depths, resource limits, and system health
- **Site Reliability Engineers**: Snapshot/restore memory state, verify backups
- **Ops Teams**: Configure alerts and monitoring for production deployments

**Use Case Example:**
```
Production memory system is consuming too much compute.
Ops team accesses Admin Ops dashboard to:
- View resource utilization metrics
- Set throttle_rate to 0.5 (50% reduction)
- Create memory snapshot as backup
- Monitor queue depth trending
```

**Endpoints:**
- `GET /api/v1/admin/ops/policies` - List policy versions
- `POST /api/v1/admin/ops/resources/throttle` - Set resource limits
- `POST /api/v1/admin/ops/backups/snapshots` - Create snapshot
- `GET /api/v1/admin/ops/alerts` - List alert rules
- `GET /api/v1/admin/ops/queues` - Queue management

---

### 2. Drift Detection (`drift_detection`)
**What it is:** Automated detection of memory quality degradation and anomaly tracking.

**Components:**
- Memory coherence analysis (promotion reversals, consolidation patterns)
- Temporal metric tracking (7-day lookback window)
- Severity classification (low/medium/high)
- Drift report generation with delta tracking

**UI Location:** Settings → Admin → Ops & Monitoring (separate tab)

**Database:** `drift_reports` table (enterprise-only migration)
- Tracks metric deltas, severity, timestamps
- Row-level security (org isolation)

**Target Users:**
- **Memory System Operators**: Monitor memory quality trends
- **ML/Quality Teams**: Detect when memory models are degrading
- **Research Teams**: Study agent behavior changes over time

**Use Case Example:**
```
Memory team sets up drift detection scheduled task (weekly).
System monitors:
- Memory promotion reversal rate (increased from 5% → 8%)
- Knowledge consolidation failures (increased from 2% → 4%)
- Long-term memory activation degradation

Generated report shows delta=0.03, severity=medium.
Team investigates → found LLM prompt changed → rolls back.
```

**Endpoints:**
- `POST /api/v1/meta/drift/run` - Trigger drift detection (task)
- `GET /api/v1/meta/drift/latest` - Fetch latest drift report
- `GET /api/v1/meta/drift/latest?metric_name=...` - Filter by metric

**Tasks:**
- `app.tasks.meta_agent.drift_detection_task` - Scheduled celery task

---

### 3. Auto-Eval Benchmarks (`auto_eval_benchmarks`)
**What it is:** Automated evaluation of retrieval quality and agent reasoning.

**Components:**
- Retrieval explanation analysis
- Candidate ranking & scoring
- Performance benchmarking over time
- Co-activation pattern detection

**UI Location:** Settings → Admin → Ops & Monitoring (Advanced Eval tab)

**Target Users:**
- **Data Science Teams**: Evaluate retrieval quality metrics
- **ML Researchers**: Benchmark agent reasoning improvements
- **QA Teams**: Track evaluation metrics in production

**Use Case Example:**
```
After rolling out an improved embedding model, team runs AutoEvalBench:
- Measures retrieval precision@5: 0.87 → 0.91
- Scores reasoning quality by comparing agent outputs to golden set
- Detects co-activation patterns (what memories trigger together)
- Shows improvement across benchmark suite
```

**Endpoints:**
- `POST /api/v1/memory-activation/admin/autoevalbench/run` - Run evaluation

**Monitoring:**
- `GET /api/v1/memory-activation/admin/observability/coactivation/top-edges` - View co-activation insights

---

### 4. Advanced Memory Observability (`memory_observability`)
**What it is:** Deep instrumentation of memory system operations for monitoring and debugging.

**Components:**
- Memory activation tracing
- Consolidation performance metrics
- Promotion reversal tracking
- Token utilization analysis
- Custom observable metrics

**UI Location:** Settings → Admin → Ops & Monitoring (Observability tab)

**Prometheus Metrics:**
- `memory_operations_total` - Count by operation type (read/write/consolidate)
- `memory_operation_duration_ms` - Latency distribution
- `agent_execution_duration_ms` - Agent performance
- `agent_tokens_consumed` - Token tracking per agent
- `resource_utilization` - Per-org resource consumption (CPU, tokens, etc.)

**Target Users:**
- **Platform Engineers**: Monitor memory system SLOs
- **Operations Teams**: Debug performance issues
- **Capacity Planning**: Track resource trends
- **Finance/Billing**: Tokenization for cost allocation

**Use Case Example:**
```
Memory read latency exceeds SLO (p99 > 500ms).
Platform engineer checks observability dashboard:
- Memory reads trending up 20% this week
- Token consumption per read: 2.5 (was 1.8)
- Consolidation churn rate: 8% (was 3%)
→ Root cause: Consolidation algorithm change
```

**Integration:**
- Prometheus scrape endpoint exports all metrics
- K8s deployments auto-scrape via ServiceMonitor (Prometheus operator)
- Grafana dashboards can query metrics

---

## Community Edition (Always Included)

### Core Memory System
- **Memory models**: Long-term, short-term, episodic memory
- **Consolidation**: Automated knowledge merging
- **Promotion/Demotion**: Dynamic memory relevance management
- **Fuzzy matching**: Approximate memory retrieval
- **Deduplication**: Redundancy elimination

### Agent Framework
- **Agent runtime**: Personal agent scheduling & execution
- **Tool binding**: Extensible tool/skill loading
- **Reasoning loops**: Plan → Execute → Reflect
- **Context management**: Tenant isolation, RLS

### Storage & Persistence
- **PostgreSQL backend**: JSONB memory, RLS policies
- **Asyncpg driver**: High-performance database connection
- **Migrations**: Alembic versioning (OSS-only tables)
- **Backups**: Basic snapshot/restore of entire database

### Identity & Access Control
- **Authentication**: Password + OIDC SSO
- **JWT tokens**: Access & refresh tokens
- **RBAC**: Org admin, member, guest roles
- **Tenant isolation**: Multi-org RLS enforcement

### Developer APIs
- **REST API**: Full v1 endpoint coverage
- **Python SDK**: `ninai` package with client library
- **Webhooks**: Event streaming for integrations
- **API Keys**: Long-lived credentials for automation
- **Proof Productization Playbook**: See `docs/PHASE54_PROOF_PRODUCTIZATION_PLAYBOOK.md` for tiering, rollout checklist, and Python/Node/Java reference integrations

### Observability (Basic)
- **Request logging**: HTTP method, path, status, duration
- **Audit trail**: User actions logged
- **Error tracking**: Stack traces in logs
- **Health check**: `/health` endpoint

---

## How Target Users Deploy

### Solo Builders / Students
```
Deployment: Community Edition (Docker or local)
License: MIT (free, open-source)
Setup: 30 min (docker-compose up)
Cost: $0

Use:
- Build personal AI agents
- Experiment with memory models
- Research agent architectures
- Learn agentic AI concepts
```

### Small Teams (2-10 people)
```
Deployment: Community Edition or Enterprise (self-hosted)
License: Community = MIT | Enterprise = Proprietary
Setup: 1-2 hours (K8s or Docker Compose)
Cost: Community = $0 | Enterprise = Custom

Use:
- Shared agent development
- Multi-user memory systems
- Basic operational monitoring
- Knowledge base management
```

### Production Enterprise (50+ users)
```
Deployment: Enterprise Edition (K8s, managed)
License: Proprietary + support contract
Setup: 1-2 days (K8s + Terraform + observability stack)
Cost: $50k-$500k+/year (depends on scale, support)

Use:
- Multi-tenant SaaS offering
- Production memory governance
- Resource budgeting & cost tracking
- Advanced observability & compliance
- On-prem or managed hosting
- 24/7 support + custom features
```

---

## Feature Maturity & Roadmap

### Stable (Production Ready)
- ✅ Core memory system
- ✅ Agent framework
- ✅ Multi-org RBAC
- ✅ Backups & recovery

### Evolving (Community Focus)
- 🔄 Knowledge consolidation algorithms (improving)
- 🔄 SDK language bindings (Go, JS in progress)
- 🔄 Tool ecosystem (more integrations)

### Advanced (Enterprise)
- ✅ Drift detection
- ✅ Admin operations
- 🔄 Advanced observability (metric dashboards improving)
- 🔄 Auto-eval benchmarks (framework stabilizing)
- 📅 Enterprise identity (SCIM, advanced SSO planned)

---

## License & Support Model

| Aspect | Community | Enterprise |
|--------|-----------|-----------|
| **License** | MIT | Proprietary |
| **Source** | Public GitHub | Private repo |
| **Cost** | $0 | Custom pricing |
| **Support** | Community forums | Dedicated SLA |
| **SLA** | None | 99.5% / 99.9% options |
| **Features locked behind** | None | License token (time-based + claims) |
| **Self-hosted** | ✅ Yes | ✅ Yes |
| **Managed hosting** | Not offered | ✅ Available |

---

## Migration Paths

### Community → Enterprise
1. Install `ninai-enterprise` package
2. Set `NINAI_LICENSE_TOKEN` environment variable
3. Run enterprise migrations: `python -m alembic -c alembic_enterprise.ini upgrade head`
4. Restart application
5. New features appear in UI automatically via feature detection

**Data Impact:** None - enterprise migrations add tables, preserve all existing data

### Enterprise → Community
1. Optional: Export enterprise data (drift reports, snapshots)
2. Downgrade enterprise schema: `python -m alembic -c alembic_enterprise.ini downgrade base`
3. Remove `NINAI_LICENSE_TOKEN`
4. Uninstall `ninai-enterprise`
5. Restart application
6. Enterprise UI tabs hidden, system continues running

**Data Impact:** Enterprise tables dropped (unless you preserve them)

---

## Frequently Asked Questions
## CognitiveOS: Autonomous Intelligence Platform

Ninai operates as a **CognitiveOS** — an autonomous cognitive layer that continuously monitors organizational context, plans ahead, and executes bounded actions on behalf of users. This section documents what is implemented, what the system commits to, and where human oversight is always maintained.

### What is Implemented (H1 — Phase Status)

| Capability | Status | Evidence |
|------------|--------|----------|
| Autonomous cognitive heartbeat (spawn sessions without user prompt) | ✅ Implemented | `cognitive_heartbeat_task`, `test_gate_a1_a4_autonomous_trigger.py` |
| Per-org system cognition state (focus, load, next action) | ✅ Implemented | `SystemCognitionState`, `GET /api/v1/cognitive/gateway/state`, `test_gate_a2_cognition_state.py` |
| Context chaining across cognitive verbs (read/decide/plan/write) | ✅ Implemented | `GatewayContextSession`, `test_cognitive_context_aggregator.py` |
| Closed-loop output to action path (tracked action records) | ✅ Implemented | `AutonomousActionAgent`, `CognitiveSession.is_autonomous`, `test_gate_a1_a4_autonomous_trigger.py` |
| Policy guardrails on autonomous actions | ✅ Implemented | Audit events `policy.autonomous_action.*`, `test_autonomous_action_agent.py` |
| Human-in-the-loop review queue | ✅ Implemented | `/api/v1/review/queue`, `test_human_review_queue_agent.py` |
| Kill switch (per-org and global) | ✅ Implemented | `PUT /api/v1/admin/cognitive-autonomy`, `test_cognitive_autonomy_control.py` |
| Heartbeat freshness and queue lag SLOs | ✅ Implemented | `test_heartbeat_freshness_slo.py`, `test_session_autonomy_slo.py` |
| Context loss safe fallback | ✅ Implemented | `load_gateway_context` → None on Redis unavail, `test_gate_c4_context_resilience.py` |
| Streaming SSE contract stability | ✅ Implemented | `test_gate_f1_f3_sse_contract.py` |
| External framework adapters (ADK, LangChain) | ✅ Implemented | `test_framework_integrations_new.py` |
| Multi-tenant isolation at cognitive layer | ✅ Implemented | RLS + org-scoped heartbeat, `test_cognitive_autonomy_control.py` |

**Not yet available** (on roadmap, not committed):
- Formal decision quality benchmarks (E1/E2)
- Explanation fidelity audit suite (E3)
- Uncertainty hypothesis closure (E4)
- SCIM / advanced SSO (Enterprise identity)

---

### What Ninai as a CognitiveOS Does and Does Not Do (H2 — Bounded Claims)

**Ninai autonomously:**
- Monitors your organization's cognitive state via a scheduled heartbeat (every ~5 minutes)
- Spawns cognitive sessions to make decisions, plans, and write memory when autonomy is enabled
- Routes high-risk or ambiguous actions to a human review queue before proceeding
- Evaluates each action through a configurable policy before executing

**Ninai does NOT:**
- Execute actions outside your configured tool scope and capability tokens
- Retain or use data from other organizations (strict RLS isolation)
- Bypass human review for actions classified as `human_review_required`
- Continue autonomous operation if the global or org-level kill switch is disabled
- Guarantee specific decision quality on all tasks — this is a probabilistic system; quality depends on your LLM configuration and data quality

**Capability boundaries:**
- All autonomous actions are bounded by the capabilities you grant via `POST /api/v1/admin/capability-tokens`
- Actions that would normally require `syscall` scope are blocked unless explicitly granted
- There is no self-modification of policy or capability grants — configuration changes require human admin action

---

### Autonomy Transparency and Trust (H3 — How It Works for Customers)

**Where humans are always in the loop:**
1. **Policy decisions marked `human_review_required`** — placed in queue at `/api/v1/review/queue`, not executed until a human claims and approves.
2. **First-time capability grant** — no autonomous action can use a capability that has not been explicitly granted by an org admin.
3. **Emergency halt** — any org admin can disable autonomous mode; a global kill switch is available to platform admins.
4. **Audit trail** — every autonomous action emits a `policy.autonomous_action.*` audit event linkable back to the triggering heartbeat session.

**How tenant safety is enforced:**
- Every cognitive session, heartbeat run, and context write is scoped to a single `organization_id`
- PostgreSQL row-level security (RLS) prevents cross-org reads/writes at the database layer
- SSE streams filter events by `organization_id` before emitting — defense in depth above RLS
- Capability tokens are org-scoped; a token from one org cannot authorize actions in another

**How to audit autonomous activity:**
```bash
# All autonomous decisions in the last hour
GET /api/v1/audit?event_type=policy.autonomous_action.approved&limit=50

# Sessions spawned by heartbeat (not by user)
GET /api/v1/cognitive/sessions?is_autonomous=true&limit=20

# Current cognition state
GET /api/v1/cognitive/gateway/state
```

**How to limit or disable autonomy:**
```bash
# Disable for your org
PUT /api/v1/admin/cognitive-autonomy
{"enabled": false}

# Set a lower heartbeat spawn rate (reduce autonomous activity)
PUT /api/v1/admin/cognitive-autonomy
{"enabled": true, "max_sessions_per_heartbeat": 1}
```

---

## Frequently Asked Questions


**Q: Can I use Community edition in production?**
A: Yes! Community is MIT-licensed and suitable for production. Choose Enterprise if you need advanced ops/observability.

**Q: What if I start with Community and want to upgrade?**
A: Seamless upgrade path - enterprise migrations layer on top of OSS schema without touching existing data.

**Q: Are enterprise features available as optional packages?**
A: No. Enterprise is an all-or-nothing plugin. Can't cherry-pick features.

**Q: Can I run both OSS and Enterprise in production simultaneously?**
A: Yes, but they'd be separate deployments (separate databases). Not intended for same instance.

**Q: What about compliance/security in Community?**
A: Community includes RBAC, RLS, audit logging, OIDC SSO. No licensing or observability differences.

**Q: If I lose my license token, what happens?**
A: Enterprise routes return 404. Application continues. You can downgrade to Community or renew license.
