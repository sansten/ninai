# Security Policy

Ninai2 is a production-grade autonomous intelligence platform with built-in multi-tenancy, encryption, and compliance primitives. This document defines security practices, vulnerability reporting, deployment guidelines, and best practices for contributors and operators.

## Supported Versions

| Version | Status    | Support Until |
|---------|-----------|---------------|
| 1.x     | Current   | 6 months      |
| 0.x     | End-of-Life| N/A           |

## Reporting a Vulnerability

If you discover a security issue, **do not open a public issue**. Submit a private report immediately.

### GitHub Security Advisories (Preferred)
1. Go to the repository **Security** tab
2. Click **"Report a vulnerability"** in the left sidebar
3. Provide:
   - **Vulnerability type** (e.g., SQL injection, auth bypass, data exposure)
   - **Affected component** (e.g., `backend/app/services/auth.py`)
   - **Reproduction steps** with minimal example
   - **Expected vs. actual behavior**
   - **Impact assessment** (data loss, privilege escalation, DoS, etc.)

### Alternative: Direct Contact
If you cannot use Security Advisories, email the maintainers via GitHub (@sansten) with:
- Subject: `[SECURITY] <brief title>`
- Body: vulnerability details from above

### Response Timeline
- **Acknowledgment**: Within 24 hours
- **Triage**: Within 48 hours (severity assessment)
- **Fix and CVE**: 7-30 days depending on severity
- **Disclosure**: Coordinated after patch is available

## Security Architecture

### Multi-Tenancy & Data Isolation
- **RLS (Row-Level Security)**: PostgreSQL policies enforce per-org data isolation
- **Tenant Context**: All queries scoped via `TenantMixin` base class
- **API Guards**: Org membership verified before all operations

### Authentication & Authorization
- **Session tokens**: 30-minute expiration, refresh token rotation
- **Role-based access**: `org_admin`, `org_member`, `guest` enforced via middleware
- **API keys**: Org-scoped, revocable, hashed in database (bcrypt + salt)

### Encryption
- **In-transit**: TLS 1.2+ required (enforced by ingress controller)
- **At-rest**: Database encryption via PostgreSQL pgcrypto; Redis data via TLS
- **Secrets**: Environment variables (not committed); Kubernetes secrets mounted
- **LLM API keys**: Stored in org_llm_config with reference indirection (no plaintext)

### Audit & Compliance
- **Audit logs**: All data mutations recorded in `audit_events` table with user/org/timestamp
- **GDPR support**: Data export (Article 20) and right-to-erasure (Article 17) via `TenantOffboardingService`
- **Log retention**: Audit logs retained for 90 days (configurable via env)

## Secure Configuration

### Environment Variables (Required in Production)
```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/ninai?sslmode=require

# Redis
REDIS_URL=redis://user:pass@host:6379?ssl=true

# Qdrant (Vector DB)
QDRANT_URL=https://host:6333
QDRANT_API_KEY=<secure-key>

# vLLM or External LLM
VLLM_BASE_URL=http://localhost:11434
VLLM_MODEL=qwen2.5:7b
# OR
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Session & JWT
SECRET_KEY=<random-64-char-key>  # Use: openssl rand -hex 32
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# TLS / Certificates
TLS_CERT_PATH=/etc/certs/cert.pem
TLS_KEY_PATH=/etc/certs/key.pem

# Logging
LOG_LEVEL=info  # Avoid 'debug' in production
```

### Kubernetes Deployment Checklist
- ✅ Use sealed secrets or external secret management (Vault, Sealed Secrets)
- ✅ Enable Pod Security Policies or Pod Security Standards
- ✅ Set resource limits and requests per container
- ✅ Use read-only root filesystem where possible
- ✅ Drop unnecessary Linux capabilities
- ✅ Enable network policies to restrict pod-to-pod traffic
- ✅ Use RBAC to limit service account permissions

## Security Best Practices for Contributors

### Code Review
- **All PRs require review**: 2+ approvals before merge
- **Security-focused review**: Look for: SQL injection, XXE, SSRF, auth bypasses, race conditions
- **Dependency audit**: Check `pip-audit` and `poetry audit` in CI before merge
- **Test coverage**: Minimum 80% for security-critical paths (auth, encryption, RLS)

### Dependency Management
- **Pinned versions**: Use exact versions in `requirements.txt` (not `>=`)
- **Vulnerability scanning**: GitHub Dependabot scans on push; act within 7 days for critical
- **Deprecated library policy**: Replace within 90 days of deprecation notice

### Logging & Error Handling
- ❌ **Never log**: Passwords, API keys, PII, auth tokens, database credentials
- ✅ **Always log**: Who (user_id), what (action), when (timestamp), where (org_id/tenant), why (error code)
- ✅ **Sanitize errors**: Return generic "An error occurred" to clients; log full details server-side

### Local Development
```bash
# Use HTTPS for local testing (requires self-signed cert)
openssl req -x509 -newkey rsa:4096 -out cert.pem -keyout key.pem -days 365 -nodes

# Set up pre-commit hooks for secret detection
pip install pre-commit detect-secrets
pre-commit install

# Scan for secrets before committing
detect-secrets scan --baseline .secrets.baseline
```

## Testing Security

Run security test suite (included in CI):
```bash
# Syntax and import safety
python -m bandit -r backend/app/ -f json -o bandit-report.json

# Dependency vulnerabilities
pip-audit --skip-editable

# OWASP ZAP scan (if deployed)
docker run -t owasp/zap2docker-stable zap-baseline.py -t http://localhost:8000
```

## Incident Response

If a vulnerability is discovered post-deployment:

1. **Immediate**: Take affected service offline if exploitable remotely
2. **Triage**: Assess scope (data exposure? privilege escalation? availability?)
3. **Fix**: Create patch; test against vulnerability; backport to supported versions
4. **Deploy**: Release patch; notify users via security advisory
5. **Post-mortem**: Document root cause and prevention measures (add test, update code review checklist)

## Third-Party Security Audits

Ninai2 is subject to periodic security audits. For access to audit reports or attestations, contact maintainers. HIPAA, SOC 2, and ISO 27001 compliance roadmaps are available upon request.

## Security Headers (Recommended for Production)

Configure your ingress controller to include:
```yaml
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains
Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline'
Referrer-Policy: strict-origin-when-cross-origin
```

## Questions?

Contact the maintainers or open a discussion in the GitHub repository under "Security" category.
