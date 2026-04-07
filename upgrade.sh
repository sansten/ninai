#!/usr/bin/env bash
# Ninai Community -> Enterprise upgrade script
# Usage: bash upgrade.sh [--dry-run]

set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

log() { echo "[ninai-upgrade] $*"; }
die() { echo "[ninai-upgrade] ERROR: $*" >&2; exit 1; }

# 1. Check current version
log "Checking current Ninai version..."
CURRENT=$(python -c "import app; print(getattr(app, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
log "Current version: $CURRENT"

# 2. Check enterprise package
log "Checking for ninai-enterprise package..."
if ! python -c "import ninai_enterprise" 2>/dev/null; then
    die "ninai-enterprise package not found. Run: pip install ninai-enterprise"
fi
ENT_VER=$(python -c "import ninai_enterprise; print(ninai_enterprise.__version__)")
log "Enterprise package version: $ENT_VER"

# 3. Check env vars
[[ -z "${POSTGRES_URL:-}" ]] && die "POSTGRES_URL not set"
[[ -z "${LICENSE_TOKEN:-}" ]] && die "LICENSE_TOKEN not set"

# 4. Run alembic migrations
log "Running database migrations..."
if $DRY_RUN; then
    log "[DRY RUN] Would run: alembic upgrade head"
else
    alembic upgrade head
fi

# 5. Validate feature gates
log "Validating feature gates..."
python -c "
from app.core.feature_gate import get_feature_gate
gate = get_feature_gate()
print(f'Feature gate type: {type(gate).__name__}')
"

# 6. Health check
log "Running health check..."
if ! $DRY_RUN; then
    python -c "
import asyncio
import httpx

async def check():
    async with httpx.AsyncClient() as c:
        r = await c.get('http://localhost:8000/api/v1/health', timeout=10)
        assert r.status_code == 200, f'Health check failed: {r.status_code}'
        print('Health check passed:', r.json().get('status'))

asyncio.run(check())
"
fi

log "Upgrade complete. Enterprise features are now active."