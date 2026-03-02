
# Ninai Open-Core Architecture — Three-Tier Separation Master Specification
Generated: 2026-01-31T02:46:43.567141Z

This document defines the architectural, feature, and operational separation between:

1. Community Edition (OSS, MIT) — Data Plane
2. Enterprise Self-Managed — Control Plane Add-on
3. Enterprise Managed (Sansten Hosted) — Fully Operated Platform

This specification is designed to be fed directly into GitHub Copilot for structured implementation and refactoring.

Core Principle:
Community = Data Plane
Enterprise = Control Plane
Managed = Operational Envelope

------------------------------------------------------------
SECTION 1 — DATA PLANE (Community OSS)
------------------------------------------------------------

Must Always Include:
- Multi-tenant RLS enforcement
- Hierarchical RBAC
- Memory lifecycle (STM, LTM, activation scoring)
- Vector search (Qdrant) with tenant filter enforcement
- Agent runtime (planner-executor-critic)
- Knowledge review workflow
- Tool invocation framework
- Basic audit logging
- Webhooks (basic)
- Docker Compose deployment
- Basic Prometheus metrics

Rules:
- System must run fully without enterprise package installed.
- Activation scoring must remain deterministic and auditable.
- RLS must enforce tenant isolation at DB level.
- No enterprise feature may replace core memory behavior.

------------------------------------------------------------
SECTION 2 — CONTROL PLANE (Enterprise Self-Managed)
------------------------------------------------------------

Additive modules only. Must not modify core tables.

Required Modules:

1. Policy Simulation Engine
   - policy_versions table
   - policy_rollout_jobs table
   - historical retrieval simulation
   - canary rollout endpoints
   - rollback endpoint

2. AutoEvalBench
   - baseline metrics storage
   - regression comparison engine
   - scheduled benchmark runner
   - evaluation report exporter

3. Drift Detection
   - drift_reports table
   - severity classification logic
   - alert threshold configuration

4. Resource Control
   - org_memory_budget table
   - per-user write caps
   - queue pause/resume control
   - admission control middleware

5. Identity Lifecycle
   - SCIM 2.0 endpoints
   - automated role mapping
   - delegated admin configuration

6. Governance Dashboard
   - advanced audit search
   - retention violation viewer
   - compliance report export

7. Meta-Agent Monitoring
   - calibration drift aggregation
   - belief instability metrics
   - performance dashboards

Feature flags required:
enterprise.policy_simulation
enterprise.autoeval
enterprise.drift
enterprise.resource_control
enterprise.identity_lifecycle
enterprise.governance_dashboard
enterprise.meta_monitoring

------------------------------------------------------------
SECTION 3 — MANAGED OPERATIONS (Enterprise Managed)
------------------------------------------------------------

Separate repository: ninai-managed-ops

Required components:
- Helm production chart
- Terraform infra modules
- Upgrade orchestrator
- Failover controller
- SLA monitor
- Backup replication validator
- Preflight system validator

Managed edition must not alter application logic.
Infrastructure automation only.

------------------------------------------------------------
SECTION 4 — LICENSE ENFORCEMENT
------------------------------------------------------------

License must:
- Validate Ed25519 signature
- Validate org_id
- Validate expiration
- Enable enterprise feature flags

If invalid:
- Enterprise endpoints return 403
- Core memory remains functional
- No data deletion

------------------------------------------------------------
SECTION 5 — MIGRATION RULES
------------------------------------------------------------

Community → Enterprise:
- Additive migrations only
- Zero destructive schema changes
- No data transformation required

Enterprise → Community (license expired):
- Enterprise endpoints disabled
- Core memory continues working

Self-Managed → Managed:
- Infrastructure migration only
- Application schema unchanged

------------------------------------------------------------
SECTION 6 — TEST REQUIREMENTS
------------------------------------------------------------

Community must include:
- Cross-tenant leakage tests
- Activation monotonicity tests
- Agent loop stability tests
- RLS enforcement tests

Enterprise must include:
- Policy simulation regression tests
- Drift detection threshold tests
- Resource throttling tests
- License tamper validation tests

Managed must include:
- Failover simulation tests
- Backup verification tests
- Upgrade orchestration tests

------------------------------------------------------------
SECTION 7 — COPILOT IMPLEMENTATION PROMPT
------------------------------------------------------------

Implement Ninai Three-Tier Architecture Separation as specified.

Constraints:
- Preserve strict separation between data plane and control plane.
- Do not move core memory logic into enterprise.
- Add enterprise modules as additive packages.
- Implement feature gates for all enterprise modules.
- Ensure community edition remains production-grade.
- Ensure zero-downtime upgrade path.

Stop only when:
- All tests pass.
- Feature gates correctly isolate enterprise functionality.
- No core memory logic depends on license validation.

END OF SPECIFICATION
