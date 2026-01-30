# Ninai Editions & Hosting Options

This project is offered in three “packaging” options:

1. **Open Source (Community)** — MIT-licensed core.
2. **Enterprise (Managed by Sansten AI on Google Cloud)** — Enterprise features + we run it with SLAs.
3. **Enterprise (Self-Managed by Client)** — Enterprise features + you run it in your environment.

The goal is to keep **(2)** and **(3)** on the **same Enterprise codebase** (same features, same schema), while the difference is **who operates it**.

## Summary Matrix

| Category | Open Source (Community) | Enterprise (Self-Managed by Client) | Enterprise (Managed by Sansten AI on Google Cloud) |
|---|---|---|---|
| License | MIT | Commercial license | Commercial subscription (includes hosting) |
| Where it runs | Anywhere | Client infrastructure (on-prem / cloud) | Sansten AI GCP (single-tenant or dedicated, as agreed) |
| Who operates | You | Client SRE/IT | Sansten AI |
| Updates & patches | Community cadence | Contracted release channel + upgrade guidance | Managed rolling upgrades (maintenance windows / zero-downtime plan) |
| Support | Community / best-effort | Paid support (optional tiers) | Included support with SLA |
| SLA | None | Optional | Yes (tiered) |
| Backups & DR | DIY | Client-run (guidance + runbooks) | Included (automated backups + DR posture) |
| Monitoring/Observability | Basic | Advanced (Enterprise) | Advanced (Enterprise) + managed dashboards/alerting |
| Security/Compliance extras | Baseline | Enterprise controls (e.g., SSO/SCIM, advanced audit/export, policy controls) | Same Enterprise controls + managed compliance artifacts/process |
| Data residency | DIY | Client-controlled | Region choices within GCP (per contract) |
| Customization | Code changes / forks | Supported extensions + professional services | Supported extensions + professional services |

## What typically differs between Community vs Enterprise

This repo already has explicit **feature flags** for Enterprise capabilities (see `EnterpriseFeatures` in the backend).

- **Community (MIT)**: core API, multi-tenancy/RLS, standard RBAC, baseline audit events, and core memory/knowledge flows.
- **Enterprise (licensed)**: “operate at scale” and compliance/identity features enabled via `enterprise.*` entitlements.

### Enterprise feature flags (defined)

- `enterprise.autoevalbench` — evaluation tooling over retrieval explanations
- `enterprise.drift_detection` — drift detection pipeline & APIs
- `enterprise.observability` — advanced observability controls (config, webhooks, metrics pipelines)
- `enterprise.admin_ops` — higher-risk operational endpoints (ops/queues, replication controls, etc.)
- `enterprise.license_management` — license/entitlement administration
- `enterprise.sso_advanced` / `enterprise.scim` — enterprise identity

### Enterprise gates currently enforced (today)

These routes return **403** in Community builds (feature not enabled):

- `enterprise.autoevalbench`
  - `POST /api/v1/memory-activation/admin/autoevalbench/run`
- `enterprise.drift_detection`
  - `GET /api/v1/meta/drift/latest`
  - `POST /api/v1/meta/drift/run`

As we expand Enterprise, we’ll apply the same pattern to the `enterprise.observability` and `enterprise.admin_ops` surfaces (keeping Community stable).

## Recommended go-to-market positioning

- **Open Source (Community)**
  - For developers, prototyping, internal tools, small deployments.
  - Clear “upgrade hooks”: enterprise features exist behind gates, so you can adopt Enterprise without changing your data model.

- **Enterprise (Self-Managed)**
  - For regulated customers who must control infrastructure (finance/healthcare/public sector).
  - Provide: offline license option, hardened deployment docs, upgrade runbooks, and support tiers.

- **Enterprise (Managed on GCP by Sansten AI)**
  - For customers who want outcomes, not infrastructure.
  - Differentiate with: SLAs, onboarding, operational dashboards, incident response, and upgrade guarantees.

## Upgrade path (Community → Enterprise)

Best practice is an **additive Enterprise package** that plugs into the OSS core:

- Community installs and runs without any enterprise dependencies.
- Enterprise installs an additional package that:
  - registers extra routes/UI,
  - swaps in a license-backed feature gate,
  - optionally adds additive migrations.

This keeps the upgrade path mostly “install + configure license”, not “fork + rewrite”.

## “Tamper-proof” licensing (practical definition)

- **Tamper-proof token**: the Enterprise license is a cryptographically signed token (Ed25519). If anyone edits even one byte of the payload (features, expiry, org id), signature verification fails.
- **Tamper-resistant enforcement**: in *self-managed* deployments, a customer who controls the servers can always modify source/binaries to bypass checks. No offline licensing scheme can prevent that 100%.

What we can do (and are implementing):

- Verify a signed license token at runtime and drive entitlements from it.
- Enforce gates in multiple places (API endpoints + background tasks) so bypass requires deliberate code changes.
- Add time-claim validation (`nbf`/`exp`) with clock-skew handling, plus optional “anti-clock-rollback” tracking (persist last-seen validation in DB).
