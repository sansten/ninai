"""DEPLOYMENT RUNBOOK: Three-Tier Architecture Implementation

Complete step-by-step guide for deploying Ninai three-tier architecture.
Covers: OSS → Enterprise, Enterprise setup, monitoring, and rollback procedures.
"""

# ============================================================================
# SECTION 1: PRE-DEPLOYMENT CHECKLIST (48 Hours Before)
# ============================================================================

PRE_DEPLOYMENT = """
□ Stakeholder Approval
  □ Product manager approved feature set
  □ Security team reviewed license validation
  □ Operations team reviewed deployment plan
  □ Compliance team reviewed data handling

□ Infrastructure Ready
  □ Database backups created
  □ Alembic migrations tested on staging
  □ Environment variables configured (NINAI_LICENSE_TOKEN, NINAI_LICENSE_PUBLIC_KEY_PATH)
  □ Monitoring and alerting set up
  □ Log aggregation active (Loki/ELK)

□ Code Ready
  □ All tests passing locally
  □ Staging deployment successful
  □ No regressions in community features (642+ tests)
  □ Feature gate decorators applied to all 56 enterprise endpoints
  □ Deployment docker images built and tested

□ License Ready
  □ License token generated (using license_issuer.py)
  □ License token validates (test with verify command)
  □ Token includes all required features
  □ Token expiration date set correctly (1+ year out)
  □ License backed up securely

□ Communication Plan
  □ User notification scheduled (if applicable)
  □ Support team briefed on new features
  □ Documentation updated and reviewed
  □ Runbook reviewed by operations team
"""

# ============================================================================
# SECTION 2: DEPLOYMENT FOR COMMUNITY EDITION → ENTERPRISE SELF-MANAGED
# ============================================================================

UPGRADE_DEPLOYMENT = """
ESTIMATED TIME: 30-45 minutes (5-minute maintenance window)

STEP 1: PRE-UPGRADE VALIDATION (NO DOWNTIME)
================================
1. Verify current state:
   $ docker-compose ps                          # All services running
   $ curl http://localhost:8000/health          # Health check passing
   $ pytest ninai/backend/tests -x --tb=short   # Community tests passing

2. Backup database:
   $ pg_dump $DATABASE_URL > backup_$(date +%Y%m%d_%H%M%S).sql
   $ aws s3 cp backup_*.sql s3://backups/ninai/
   
3. Tag current deployment:
   $ git tag deployment/pre-enterprise-$(date +%Y%m%d)
   $ git push origin deployment/pre-enterprise-*

STEP 2: DEPLOY NEW CODE (5-MINUTE WINDOW)
=========================================
1. Stop services:
   $ docker-compose down
   
2. Pull new code:
   $ git pull origin main
   
3. Update dependencies:
   $ pip install -r ninai/backend/requirements.txt
   $ pip install -r ninai-enterprise/requirements.txt  # New!
   
4. Verify no breaking changes:
   $ python -m mypy ninai/backend/app --no-error-summary 2>&1 | tail -20
   
5. Set environment variables:
   $ export NINAI_LICENSE_TOKEN="ninai1.eyJ..."
   $ export NINAI_LICENSE_PUBLIC_KEY_PATH="/etc/ninai/license_public.pem"
   
6. Start services:
   $ docker-compose up -d
   
7. Wait for startup (30-60 seconds):
   $ sleep 60
   $ docker-compose ps  # Verify all running

STEP 3: RUN MIGRATIONS (5 MINUTE WINDOW)
========================================
1. Verify database connection:
   $ alembic -c alembic.ini current  # Should show latest community migration
   
2. Run enterprise migrations:
   $ alembic -c alembic_enterprise.ini upgrade head
   
3. Verify tables created:
   $ psql $DATABASE_URL -c "SELECT tablename FROM pg_tables WHERE tableowner='ninai' ORDER BY tablename LIMIT 5;"
   Expected tables: policy_versions, drift_reports, org_memory_budget, policy_rollout_jobs
   
4. Verify RLS policies:
   $ psql $DATABASE_URL -c "SELECT policyname FROM pg_policies WHERE tablename='drift_reports';"
   Expected: drift_reports_tenant_isolation

STEP 4: POST-UPGRADE VALIDATION (NO DOWNTIME)
==============================================
1. Verify services healthy:
   $ curl http://localhost:8000/health
   $ docker-compose logs -n 50 | grep -i error
   
2. Verify community features work:
   $ pytest ninai/backend/tests -k "not enterprise" -x --tb=short
   Expected: 642+ tests pass
   
3. Verify enterprise endpoints protected:
   $ curl -X POST http://localhost:8000/api/v1/admin/ops/policies \
     -H "Content-Type: application/json" -d '{}'
   Expected: 403 Forbidden (no license token in request)
   
4. Verify enterprise features accessible:
   $ curl -X POST http://localhost:8000/api/v1/admin/ops/policies \
     -H "Authorization: Bearer $TEST_TOKEN" \
     -H "NINAI-License-Token: $NINAI_LICENSE_TOKEN" \
     -H "Content-Type: application/json" -d '{"policy_name": "test"}'
   Expected: 200 OK or 400 (invalid data, not auth error)
   
5. Check logs for license validation:
   $ docker-compose logs ninai-backend | grep -i license | tail -20
   Expected: "License valid for org ..., features: [...]"
   
6. Monitor metrics:
   $ curl http://localhost:8000/api/v1/admin/ops/metrics/prometheus | grep -i license
   Expected: license_validation_success_total{org="test-org"} 1

STEP 5: COMMUNICATION
====================
1. Notify stakeholders:
   - "Enterprise deployment complete"
   - "All features tested and validated"
   - "No impact on community tier"
   
2. Enable license-based feature access:
   - Admin dashboard now shows enterprise features
   - Feature gates enforce license entitlements
   - Support team can help with license renewal
"""

# ============================================================================
# SECTION 3: ROLLBACK PROCEDURE (IF NEEDED)
# ============================================================================

ROLLBACK = """
ESTIMATED TIME: 10-15 minutes

STEP 1: QUICK HEALTH CHECK
==========================
If experiencing issues within first 5 minutes:

1. Check error logs:
   $ docker-compose logs --tail=100 | grep -i error

2. Common issues and fixes:
   a) License token not found:
      - Verify NINAI_LICENSE_TOKEN is set: echo $NINAI_LICENSE_TOKEN
      - Check /etc/ninai/license_public.pem exists
      - Fix: export NINAI_LICENSE_TOKEN="..." && docker-compose restart
      
   b) Database migration failed:
      - Check alembic history: alembic -c alembic_enterprise.ini history
      - Rollback: alembic -c alembic_enterprise.ini downgrade base
      - Fix database issue, then retry: alembic -c alembic_enterprise.ini upgrade head
      
   c) Community tests failing:
      - This indicates breaking change, need full rollback
      
STEP 2: FULL ROLLBACK (If needed)
=================================
1. Stop services:
   $ docker-compose down
   
2. Downgrade enterprise migrations:
   $ alembic -c alembic_enterprise.ini downgrade base
   
3. Restore previous code:
   $ git checkout deployment/pre-enterprise-YYYYMMDD
   
4. Restore environment (remove enterprise vars):
   $ unset NINAI_LICENSE_TOKEN
   $ unset NINAI_LICENSE_PUBLIC_KEY_PATH
   
5. Start services:
   $ docker-compose up -d
   
6. Verify community tests:
   $ pytest ninai/backend/tests -x --tb=short
   Expected: 642+ tests pass
   
7. Notify stakeholders:
   - "Rolled back to previous version"
   - "Investigating issue"
   - "No data loss"

STEP 3: POST-ROLLBACK ANALYSIS
==============================
1. Review logs:
   $ docker-compose logs --since 10m > rollback_analysis.log
   
2. Check database state:
   $ psql $DATABASE_URL -c "SELECT * FROM alembic_version;"
   
3. Schedule postmortem:
   - Review root cause
   - Identify fixes needed
   - Plan retry
"""

# ============================================================================
# SECTION 4: LICENSE RENEWAL WORKFLOW
# ============================================================================

LICENSE_RENEWAL = """
QUARTERLY (90 Days Before Expiration)

STEP 1: LICENSE RENEWAL REQUEST (Week 1)
========================================
1. Check current license expiration:
   $ curl http://localhost:8000/api/v1/license
   Expected: { "expires_at": "2026-06-30", "org_id": "...", "features": [...] }
   
2. Request new license from Sansten AI:
   - Email: licenses@sansten.ai
   - Include: org_id, current plan, desired features
   - Include: number of users/seats
   
3. Receive new license token in response

STEP 2: LICENSE UPDATE (Week 2-3)
=================================
1. Test new license locally:
   $ export NINAI_LICENSE_TOKEN="ninai1.eyJ..." (new token)
   $ curl http://localhost:8000/api/v1/license
   Expected: New expiration date, same org_id
   
2. Update license in staging:
   $ export NINAI_LICENSE_TOKEN="..." (new token)
   $ docker-compose restart
   $ curl http://localhost:8000/api/v1/license
   
3. Verify all enterprise endpoints still accessible:
   $ pytest ninai-enterprise/tests/test_license_validation.py -v

STEP 3: LICENSE DEPLOYMENT (Week 4)
===================================
1. Schedule maintenance window (5 minutes):
   - 2:00 AM UTC (minimize user impact)
   
2. Update production:
   $ kubectl set env deployment/ninai-backend NINAI_LICENSE_TOKEN="ninai1.eyJ..."
   $ kubectl rollout status deployment/ninai-backend
   
3. Verify license updated:
   $ curl -H "Authorization: Bearer $ADMIN_TOKEN" \
     http://api.example.com/api/v1/license
   Expected: New expiration date
   
4. Set reminder for next renewal:
   - Email reminder 90 days before new exp date
   - Create calendar event

STEP 4: MONITORING
==================
1. Alert if license expires in 7 days:
   ```yaml
   - alert: LicenseExpiringWarning
     expr: (license_exp_timestamp - time()) < 604800  # 7 days
     annotations:
       summary: "Ninai license expiring in 7 days"
   ```

2. Alert if license is invalid:
   ```yaml
   - alert: LicenseInvalid
     expr: license_validation_failures_total > 0
     annotations:
       summary: "Ninai license validation failing"
   ```
"""

# ============================================================================
# SECTION 5: MONITORING & OBSERVABILITY
# ============================================================================

MONITORING = """
Key Metrics to Track:

1. License Validation Metrics
   - license_validation_success_total[org]: Counter of successful validations
   - license_validation_failures_total[reason]: Counter by reason (expired, invalid_sig, missing)
   - license_features_accessed[feature]: Counter per feature gate
   - license_features_denied[feature]: Counter of 403s per feature

2. Enterprise Endpoint Metrics
   - enterprise_endpoint_requests_total[endpoint]: All requests
   - enterprise_endpoint_errors_total[endpoint]: Errors (403, 500, etc)
   - enterprise_endpoint_latency_seconds[endpoint]: Response times

3. Feature Usage Metrics
   - enterprise.admin_ops requests
   - enterprise.drift_detection requests
   - enterprise.autoeval requests
   - enterprise.resource_control requests

Key Logs to Monitor:

1. Info
   "License valid for org test-org-123, features: [enterprise.admin_ops, ...]"

2. Warnings
   "License validation failed: NINAI_LICENSE_TOKEN is not set"
   "Access denied: feature enterprise.admin_ops not in license for org test-org-123"

3. Errors
   "Invalid token signature"
   "Token expired"
   "Failed to load license public key"

Dashboards to Create:

1. License Health
   - License expiration date
   - Validation success rate
   - Features in use
   - Orgs with licenses

2. Enterprise Features
   - Requests per feature
   - 403 rate per feature
   - Latency by feature
   - Errors by feature

3. Deployment Status
   - Service health
   - Database migration status
   - Enterprise tables row counts
"""

# ============================================================================
# SECTION 6: TROUBLESHOOTING
# ============================================================================

TROUBLESHOOTING = """
Problem: "NINAI_LICENSE_TOKEN is not set"
Solution:
  1. Check env var: echo $NINAI_LICENSE_TOKEN
  2. If empty, set it: export NINAI_LICENSE_TOKEN="ninai1...."
  3. Restart services: docker-compose restart
  4. Verify: docker exec ninai-backend env | grep NINAI_LICENSE

Problem: "Invalid token signature"
Solution:
  1. Verify token format starts with "ninai1."
  2. Verify public key is correct: cat $NINAI_LICENSE_PUBLIC_KEY_PATH
  3. Regenerate token with license_issuer.py if needed
  4. Check logs: docker-compose logs | grep -i "signature"

Problem: "Token expired"
Solution:
  1. Check token expiration: 
     python -c "import base64, json; \
     payload = base64.urlsafe_b64decode(token.split('.')[1] + '=='); \
     print(json.loads(payload))"
  2. If expired, request new license from Sansten AI
  3. Update NINAI_LICENSE_TOKEN with new token
  4. Restart services

Problem: "Feature not licensed"
Solution:
  1. Check licensed features in token:
     python -c "... print(json.loads(payload)['features'])"
  2. Contact Sansten AI to add feature to license
  3. Receive new token with additional features
  4. Update NINAI_LICENSE_TOKEN
  5. Restart services

Problem: "Database migration failed"
Solution:
  1. Check migration status: alembic -c alembic_enterprise.ini history
  2. Rollback if needed: alembic -c alembic_enterprise.ini downgrade -1
  3. Check database logs for errors
  4. Retry migration: alembic -c alembic_enterprise.ini upgrade head

Problem: "Community tests failing after upgrade"
Solution:
  1. This indicates breaking change - ROLLBACK immediately
  2. Run rollback procedure from SECTION 3
  3. Investigate root cause
  4. Do NOT continue deployment
"""

# ============================================================================
# SECTION 7: QUICK REFERENCE
# ============================================================================

QUICK_COMMANDS = """
# View current license
curl http://localhost:8000/api/v1/license

# Check service health
curl http://localhost:8000/health

# Run community tests
pytest ninai/backend/tests -x

# Run enterprise tests
pytest ninai-enterprise/tests/ -x

# Run license validation tests
pytest ninai-enterprise/tests/test_license_validation.py -v

# Check migrations
alembic -c alembic_enterprise.ini current

# Run migration
alembic -c alembic_enterprise.ini upgrade head

# Rollback migration
alembic -c alembic_enterprise.ini downgrade base

# View logs
docker-compose logs -f --tail=100

# Restart services
docker-compose restart

# Check environment variables
docker exec ninai-backend env | grep NINAI

# Test license validation
python repos/ninai-enterprise/verify_implementation.py
"""
