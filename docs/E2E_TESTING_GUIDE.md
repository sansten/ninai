"""End-to-End Testing Guide for Three-Tier Architecture

This document describes how to test the complete license validation flow.
"""

# 1. GENERATE TEST LICENSE TOKENS

# Option A: Use license_issuer.py (Python CLI)
"""
cd repos/license-issuer

# Generate Ed25519 keypair
python license_issuer.py gen-keys --out-dir ./test_keys

# Issue test license (1 year validity)
python license_issuer.py issue \
  --private-key ./test_keys/license_private.pem \
  --org-id "test-org-123" \
  --features enterprise.admin_ops,enterprise.drift_detection,enterprise.autoeval \
  --exp-days 365 \
  --license-id "TEST-001" \
  --plan enterprise-self-managed

# Verify token signature
python license_issuer.py verify \
  --public-key ./test_keys/license_public.pem \
  --token "ninai1...."

Output:
✓ Token signature verified
{
  "exp": 1756204600,
  "features": ["enterprise.admin_ops", "enterprise.autoeval", "enterprise.drift_detection"],
  "iat": 1704668600,
  "license_id": "TEST-001",
  "nbf": null,
  "org_id": "test-org-123",
  "plan": "enterprise-self-managed"
}
"""

# 2. DEPLOYMENT SETUP

"""
# Set license token in environment
export NINAI_LICENSE_TOKEN="ninai1.eyJvcmdfaWQiOiAi..."

# Or in .env file for local testing
echo 'NINAI_LICENSE_TOKEN=ninai1.eyJvcmdfaWQiOiAi...' >> .env
"""

# 3. TEST SCENARIOS

TEST_SCENARIOS = [
    {
        "name": "Valid License - Feature Accessible",
        "description": "Request endpoint with valid license containing required feature",
        "endpoint": "POST /api/v1/admin/ops/policies",
        "license_features": ["enterprise.admin_ops"],
        "expected_status": 200,
        "expected_behavior": "Endpoint executes successfully",
    },
    {
        "name": "Invalid License - Missing Feature",
        "description": "Request endpoint with valid license but missing required feature",
        "endpoint": "POST /api/v1/admin/ops/policies",
        "license_features": ["enterprise.autoeval"],
        "expected_status": 403,
        "expected_error": "Feature 'enterprise.admin_ops' not licensed",
    },
    {
        "name": "Invalid Signature - Token Rejected",
        "description": "Request with tampered token (signature doesn't match)",
        "endpoint": "POST /api/v1/admin/ops/policies",
        "license_token": "ninai1.eyJvcmdfaWQiOiAi...TAMPERED...eA==.aW52YWxpZF9zaWduYXR1cmU=",
        "expected_status": 403,
        "expected_error": "Invalid token signature",
    },
    {
        "name": "Expired License - Token Rejected",
        "description": "Request with expired license token (exp < now)",
        "endpoint": "POST /api/v1/admin/ops/policies",
        "license_exp": "2020-01-01",
        "expected_status": 403,
        "expected_error": "Token expired",
    },
    {
        "name": "Missing License - Feature Unavailable",
        "description": "Request without NINAI_LICENSE_TOKEN environment variable",
        "endpoint": "POST /api/v1/admin/ops/policies",
        "env_vars": {"NINAI_LICENSE_TOKEN": ""},
        "expected_status": 403,
        "expected_error": "Enterprise license required",
    },
    {
        "name": "Community Edition Independence",
        "description": "Verify community edition works without ninai-enterprise",
        "endpoints": [
            "POST /api/v1/memories",
            "GET /api/v1/agents",
            "POST /api/v1/cognitive_loop",
        ],
        "expected_status": 200,
        "expected_behavior": "All community endpoints work without enterprise package",
    },
]

# 4. PYTHON TEST SCRIPT

TEST_SCRIPT = """
import httpx
import os
import pytest

# Test setup
BASE_URL = "http://localhost:8000"
VALID_LICENSE = "ninai1.eyJvcmdfaWQiOiAidGVzdC1vcmctMTIzIiwgImZlYXR1cmVzIjogWyJlbnRlcnByaXNlLmFkbWluX29wcyJdLCAiaWF0IjogMTcwNDY2ODYwMCwgImV4cCI6IDE3MzYyMDQ2MDB9..."

async def test_valid_license_grant_access():
    \"\"\"Test that valid license with feature grants endpoint access.\"\"\"
    os.environ["NINAI_LICENSE_TOKEN"] = VALID_LICENSE
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/api/v1/admin/ops/policies",
            headers={"Authorization": "Bearer test-token"},
            json={"policy_name": "test", "rules": []},
        )
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert "policy_id" in response.json()

async def test_missing_feature_deny_access():
    \"\"\"Test that license without feature returns 403.\"\"\"
    # Issue token with only autoeval feature
    license_token = issue_license(features=["enterprise.autoeval"])
    os.environ["NINAI_LICENSE_TOKEN"] = license_token
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/api/v1/admin/ops/policies",
            json={},
        )
    
    assert response.status_code == 403
    assert "not licensed" in response.text.lower()

async def test_invalid_signature_rejected():
    \"\"\"Test that tampered token is rejected.\"\"\"
    tampered = VALID_LICENSE[:-10] + "0000000000"  # Corrupt last 10 chars
    os.environ["NINAI_LICENSE_TOKEN"] = tampered
    
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{BASE_URL}/api/v1/admin/ops/policies", json={})
    
    assert response.status_code == 403
    assert "Invalid token signature" in response.text or "not licensed" in response.text

async def test_expired_token_rejected():
    \"\"\"Test that expired token is rejected.\"\"\"
    # Create token with exp in past
    expired_license = create_license(exp_days=-1)
    os.environ["NINAI_LICENSE_TOKEN"] = expired_license
    
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{BASE_URL}/api/v1/admin/ops/policies", json={})
    
    assert response.status_code == 403
    assert "expired" in response.text.lower() or "not licensed" in response.text

async def test_missing_license_denied():
    \"\"\"Test that missing license denies access.\"\"\"
    os.environ.pop("NINAI_LICENSE_TOKEN", None)
    
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{BASE_URL}/api/v1/admin/ops/policies", json={})
    
    assert response.status_code == 403

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
\"\"\"

# 5. DEPLOYMENT VALIDATION CHECKLIST

DEPLOYMENT_CHECKLIST = [
    "✅ License token generated with correct features",
    "✅ NINAI_LICENSE_TOKEN environment variable set",
    "✅ NINAI_LICENSE_PUBLIC_KEY_PATH configured or key in config/",
    "✅ Alembic migrations run: alembic upgrade head",
    "✅ All enterprise endpoints decorated with @require_license_feature",
    "✅ Run: pytest tests/test_license_validation.py -v",
    "✅ Test valid license grants access (200 OK)",
    "✅ Test missing feature denies access (403)",
    "✅ Test invalid signature rejected (403)",
    "✅ Test expired token rejected (403)",
    "✅ Test community edition works independently",
    "✅ Monitor logs for license validation messages",
    "✅ Verify 403 responses for unlicensed features",
    "✅ Create runbook for license renewal",
]

# 6. MANUAL TESTING WITH CURL

"""
# Get license token
TOKEN="ninai1.eyJvcmdfaWQiOiAidGVzdC1vcmctMTIzIiwgImZlYXR1cmVzIjogWyJlbnRlcnByaXNlLmFkbWluX29wcyJdLCAiaWF0IjogMTcwNDY2ODYwMCwgImV4cCI6IDE3MzYyMDQ2MDB9..."

# Test valid license - should work
curl -X POST http://localhost:8000/api/v1/admin/ops/policies \
  -H "Authorization: Bearer test-token" \
  -H "Content-Type: application/json" \
  -d '{"policy_name": "test"}' \
  -H "NINAI_LICENSE_TOKEN: $TOKEN"
# Expected: 200 OK

# Test missing license - should return 403
curl -X POST http://localhost:8000/api/v1/admin/ops/policies \
  -H "Authorization: Bearer test-token" \
  -H "Content-Type: application/json" \
  -d '{"policy_name": "test"}'
# Expected: 403 Forbidden - "Enterprise license required"

# Test community endpoint - should work regardless of license
curl -X POST http://localhost:8000/api/v1/memories \
  -H "Authorization: Bearer test-token" \
  -H "Content-Type: application/json" \
  -d '{"memory_text": "test"}'
# Expected: 200 OK (community feature, no license needed)
"""

# 7. EXPECTED LOGS

"""
License validation logs to expect:

INFO: License valid for org test-org-123, features: ['enterprise.admin_ops', 'enterprise.drift_detection']
WARNING: License validation failed: NINAI_LICENSE_TOKEN is not set
WARNING: Access denied: feature enterprise.admin_ops not in license for org test-org-123
ERROR: Invalid token signature
ERROR: Token expired
"""
