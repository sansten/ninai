#!/usr/bin/env python3
"""
E2E Test Helper - Generate Auth Token for E2E Tests

Generates a valid JWT token with org/user context for use in Playwright tests.
Run this to get a token, then set E2E_AUTH_TOKEN environment variable.

Usage:
  python backend/scripts/generate_e2e_token.py --user-id <uuid> --org-id <uuid>
  
  # Output: eyJ0eXAiOiJKV1QiLCJhbGc...
  # Use: $env:E2E_AUTH_TOKEN="<token>"
"""

import sys
import argparse
from datetime import datetime, timedelta, timezone
import uuid

# Add backend to path
sys.path.insert(0, '/d/Sansten/Projects/Ninai2/backend')

from jose import jwt
from app.core.config import settings


def generate_test_token(user_id: str, org_id: str, roles: list = None):
    """
    Generate a JWT token for E2E testing.
    
    Args:
        user_id: UUID string for test user
        org_id: UUID string for test org
        roles: List of role names (default: ['user'])
    
    Returns:
        JWT token string
    """
    if roles is None:
        roles = ['user', 'org_admin']
    
    now = datetime.now(timezone.utc)
    expires = now + timedelta(hours=24)
    
    payload = {
        'sub': user_id,
        'org_id': org_id,
        'exp': expires,
        'iat': now,
        'type': 'access',
        'roles': roles,
    }
    
    token = jwt.encode(
        payload,
        settings.SECRET_KEY,
        algorithm='HS256'
    )
    
    return token


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate E2E test JWT token')
    parser.add_argument('--user-id', default=str(uuid.uuid4()), help='User UUID')
    parser.add_argument('--org-id', default=str(uuid.uuid4()), help='Organization UUID')
    parser.add_argument('--roles', default='user,org_admin', help='Comma-separated roles')
    
    args = parser.parse_args()
    roles = [r.strip() for r in args.roles.split(',')]
    
    token = generate_test_token(args.user_id, args.org_id, roles)
    
    print(f'# Test User ID: {args.user_id}')
    print(f'# Test Org ID: {args.org_id}')
    print(f'# Roles: {", ".join(roles)}')
    print(f'# Token (valid 24 hours):')
    print(token)
    print()
    print('# PowerShell:')
    print(f'$env:E2E_AUTH_TOKEN="{token}"')
    print()
    print('# Bash:')
    print(f'export E2E_AUTH_TOKEN="{token}"')
