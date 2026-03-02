# Migration Guide: OSS ↔ Enterprise

## Current Migration Safety Status

### ✅ OSS → Enterprise (Upgrade)

**Status: SAFE with documented steps**

The upgrade path is clean because:
- Enterprise adds only new tables (currently: `drift_reports`)
- No modifications to existing OSS tables
- Enterprise migrations use a separate version table (`alembic_version_enterprise`)
- All OSS data remains intact and accessible

**Steps:**
1. Install enterprise package: `pip install ninai-enterprise`
2. Set license token: `$env:NINAI_LICENSE_TOKEN = "ninai1.<payload>.<sig>"`
3. Run enterprise migrations:
   ```powershell
   cd repos/ninai-enterprise
   $env:ALEMBIC_DATABASE_URL_SYNC = "postgresql://USER:PASS@HOST:5432/DB"
   python -m alembic -c alembic_enterprise.ini upgrade head
   ```
4. Restart application (enterprise features now active)

**Data Impact:** None - all existing data preserved, enterprise adds new capabilities on top.

---

### ⚠️ Enterprise → Community (Downgrade)

**Status: REQUIRES MANUAL CLEANUP**

**Current Issues:**

1. **Orphaned Enterprise Tables**
   - Enterprise tables (`drift_reports`) remain in database after uninstalling plugin
   - OSS code doesn't reference them, so no crashes
   - Data is inaccessible but preserved (recovery possible if you re-upgrade)

2. **Separate Version Tracking**
   - `alembic_version_enterprise` table remains
   - Harmless but indicates enterprise was previously installed

3. **No Automated Downgrade Path**
   - Enterprise migrations have `downgrade()` but require manual execution
   - Risk of forgetting to run downgrades before uninstalling

**Recommended Downgrade Steps:**

**Option A: Clean Downgrade (data loss acceptable)**
```powershell
# 1. Export any enterprise data you want to keep
#    (drift reports, enterprise logs, etc.)

# 2. Downgrade enterprise schema
cd repos/ninai-enterprise
$env:ALEMBIC_DATABASE_URL_SYNC = "postgresql://USER:PASS@HOST:5432/DB"
python -m alembic -c alembic_enterprise.ini downgrade base

# 3. Uninstall enterprise package
pip uninstall ninai-enterprise

# 4. Remove license token
$env:NINAI_LICENSE_TOKEN = ""

# 5. Restart application (now running Community edition)
```

**Option B: Preserve Enterprise Data (keep tables)**
```powershell
# 1. Uninstall enterprise package
pip uninstall ninai-enterprise

# 2. Remove license token
$env:NINAI_LICENSE_TOKEN = ""

# 3. Restart (OSS ignores enterprise tables)
# Enterprise data preserved for potential future re-upgrade
```

**Data Impact:**
- **Option A:** Enterprise data deleted (drift reports lost)
- **Option B:** Enterprise data preserved but inaccessible until re-upgrade

---

## Recommended Improvements

### High Priority

1. **Add Migration Health Check Endpoint**
   ```python
   GET /api/v1/admin/migration-status
   Returns:
   - OSS schema version
   - Enterprise schema version (if installed)
   - Orphaned tables detected
   - Migration recommendations
   ```

2. **Document Downgrade in Enterprise README**
   - Add explicit downgrade instructions
   - Warn about data loss
   - Provide export scripts for enterprise data

3. **Add Downgrade Safety Script**
   ```bash
   # repos/ninai-enterprise/scripts/safe_downgrade.py
   # - Checks for enterprise data
   # - Offers export options
   # - Confirms before dropping tables
   # - Validates OSS can run without enterprise
   ```

### Medium Priority

4. **Version Compatibility Matrix**
   - Document which OSS versions work with which Enterprise versions
   - Add compatibility check in enterprise `register()`

5. **Enterprise Data Export Tool**
   ```bash
   # Export drift reports to JSON for archival
   python -m ninai_enterprise.tools.export_data --output drift_reports.json
   ```

### Low Priority

6. **Unified Migration Command**
   ```bash
   # Run both OSS and Enterprise migrations in correct order
   python -m scripts.migrate_all
   ```

---

## Database Dependency Analysis

**Enterprise → OSS Dependencies:**
- `drift_reports.organization_id` → `organizations.id` (FK, CASCADE delete)
- Uses OSS Base metadata for table registration
- No other foreign keys

**OSS → Enterprise Dependencies:**
- **None** - OSS code has zero references to enterprise tables
- OSS Alembic doesn't import enterprise models
- Safe to run OSS against DB with enterprise tables (they're ignored)

**Conclusion:**
- ✅ Upgrade is straightforward and safe
- ⚠️ Downgrade works but needs better tooling/documentation
- 🔧 Recommend adding migration health checks and guided downgrade scripts
