# Ninai Community Edition: Tier Structure

**Edition**: Community (MIT License)  
**Price**: Free  
**Self-Hosted**: Yes  
**Support**: Community forums

---

## What's Included

This is the **complete, production-ready memory OS for AI agents**. No limitations or demos—just pure memory infrastructure.

### Core Capabilities

✅ **Multi-Tenant Memory System**
- Short-term memory (Redis) + Long-term memory (PostgreSQL + Qdrant)
- Self-model for calibration tracking
- Memory consolidation & deduplication
- Graduated memory promotion (hot → warm → cold)

✅ **Security & Multi-Tenancy**
- Row-Level Security (RLS) enforced at PostgreSQL level
- Hierarchical RBAC (Org → Team → User)
- Audit logging for all operations
- OIDC SSO + MFA support
- API key management

✅ **Agent Framework**
- Planner, Executor, Critic, Meta-Agent
- PolicyGuard safety constraints
- Calibration profiles for confidence tracking
- Belief store for memory quality signals

✅ **Knowledge Management**
- Human-in-the-loop review workflow
- Knowledge versioning & immutable history
- Semantic tagging & topics
- Non-admin reviewer approval

✅ **Search & Retrieval**
- Vector search (Qdrant)
- Hybrid search (BM25 + semantic)
- Advanced filtering (tag, date, scope)
- Cross-tenant isolation

✅ **Integrations**
- LangChain memory adapter
- LlamaIndex integration
- CrewAI compatibility
- Basic webhooks (event streaming)
- Python SDK

✅ **Operations & Observability**
- Docker Compose for local dev
- Kubernetes manifests for production
- Prometheus metrics + Grafana dashboards
- Manual backups via pg_dump
- Health checks & admission control

✅ **642 Production Tests**
- Memory lifecycle tests
- Agent framework tests
- RBAC enforcement tests
- Cross-tenant security tests
- Integration tests
- API endpoint tests

---

## What's NOT Included (Enterprise Only)

❌ **Policy Simulation** — Deploy memory changes safely before production  
❌ **AutoEvalBench** — Automated evaluation analytics  
❌ **Drift Detection** — Memory quality monitoring  
❌ **Resource Control** — Throttling & admission policies  
❌ **SCIM 2.0** — Identity provider sync  
❌ **Governance Dashboard** — Compliance & audit reporting  
❌ **Meta-Agent Monitoring** — Advanced calibration tracking  
❌ **Managed Infrastructure** — SaaS operations by Sansten AI  
❌ **99.9% SLA** — Uptime guarantee  
❌ **24/7 Support** — Phone + Slack support  

---

## No Feature Gates in Community

Community Edition has **zero enterprise dependencies**. All code is self-contained.

```python
# Community code never checks for enterprise licenses
# Enterprise features are optional add-ons
# Upgrade is zero-downtime
```

If you want operational controls, upgrade to Enterprise Self-Managed or Enterprise Managed.

---

## Architecture

```
Community Edition
├─ Memory System (PostgreSQL + Redis + Qdrant)
├─ Agent Framework (Planner, Executor, Critic, Meta-Agent)
├─ Security (RLS, RBAC, OIDC, MFA, audit logging)
├─ Knowledge Management (submission, review, versioning)
├─ Search & Retrieval (vector + hybrid + filtering)
├─ Integrations (LangChain, LlamaIndex, CrewAI, webhooks)
└─ Operations (Docker, Kubernetes, Prometheus, backups)

Enterprise Add-ons (NOT included)
├─ Policy Simulation
├─ AutoEvalBench
├─ Drift Detection
├─ Resource Control
├─ SCIM Identity Lifecycle
├─ Governance Dashboard
├─ Meta-Agent Monitoring
└─ Managed Infrastructure

```

---

## Deployment

### Local Development
```bash
docker-compose up
# Starts: PostgreSQL, Redis, Qdrant, Ninai API, Celery
```

### Production (Self-Managed)
```bash
# Use provided Kubernetes manifests in ./k8s/
kubectl apply -f k8s/
```

---

## Who Should Use Community

✅ **Solo developers** — Build personal agents with persistent memory  
✅ **Research teams** — Non-commercial memory experimentation  
✅ **Open-source projects** — Need memory infrastructure  
✅ **Startups < 20 users** — Proof-of-concept phase  
✅ **DevOps teams** — Full infrastructure ownership  

---

## Upgrade to Enterprise (Zero-Downtime)

Community Edition is not a trial—it's a complete product. When you're ready for operational controls:

```bash
# 1. Install enterprise package (5 min)
pip install ninai-enterprise --index-url https://private-registry.sansten.ai

# 2. Run migrations (30 sec, additive only)
alembic -c alembic_enterprise.ini upgrade head

# 3. Set license token
export NINAI_LICENSE_TOKEN="ninai1.<payload>.<sig>"

# 4. Restart (rolling update, zero downtime)
docker-compose up -d

# Result: Enterprise features now available
# All existing data preserved
```

### Enterprise Self-Managed ($50/user/mo)
You manage Kubernetes, we provide the features.

### Enterprise Managed ($75/user/mo)
We manage everything, 99.9% SLA included.

---

## Support

- **Documentation**: https://docs.sansten.ai
- **GitHub Issues**: https://github.com/sansten/ninai/issues
- **Community Forum**: https://community.sansten.ai
- **Commercial Support**: Email sales@sansten.ai for Enterprise SLAs

---

## License

MIT License — Modify, distribute, use commercially. No restrictions.

```
Copyright (c) 2026 Sansten AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

[Full MIT license text...]
```

---

## FAQ

**Q: Is Community Edition suitable for production?**  
A: Yes. It includes 642 tests, RLS enforcement, audit logging, and complete observability. It's production-grade.

**Q: Can I use Community Edition commercially?**  
A: Yes. MIT license allows commercial use, modification, and redistribution.

**Q: How do I upgrade to Enterprise without downtime?**  
A: Zero-downtime upgrade is built-in. See "Upgrade to Enterprise" section above.

**Q: What happens to my data if I don't pay for Enterprise?**  
A: Your data stays in Community Edition forever. Enterprise features are purely additive—nothing is removed.

**Q: Can I run Community Edition on my infrastructure?**  
A: Yes. Community is fully self-hosted. Docker Compose for dev, Kubernetes for production.

**Q: Do I need Kubernetes expertise?**  
A: No. Docker Compose works for single-machine deployments. For scale, we provide Kubernetes manifests.
