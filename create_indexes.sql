-- Week 3 Query Optimization - Index Creation Script
-- ===================================================
-- Creates composite indexes on hot tables for 30-50% performance improvement
-- 
-- Usage: psql -h localhost -U ninai -d ninai -f create_indexes.sql
--
-- Note: CONCURRENTLY flag allows queries during index creation (non-blocking)

-- =====================================================================
-- 1. Memories Table - Filter by org + order by created_at
-- =====================================================================
-- Most common query: SELECT * FROM memories WHERE organization_id = ? ORDER BY created_at DESC
-- This index supports both filtering and sorting

CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_memories_org_id_created
  ON memories(organization_id, created_at DESC)
  WHERE deleted_at IS NULL;

-- Additional: Filter by user_id (for personal queries)
CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_memories_user_id_created
  ON memories(user_id, created_at DESC)
  WHERE deleted_at IS NULL;

-- =====================================================================
-- 2. Audit Events Table - Filter by org, type, timestamp
-- =====================================================================
-- Compliance queries: SELECT * FROM audit_events 
--   WHERE organization_id = ? AND event_type IN (...) AND timestamp > ?
-- Composite index for optimal filtering and sorting

CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_audit_events_org_type_ts
  ON audit_events(organization_id, event_type, timestamp DESC);

-- Alternative: For queries only by org + timestamp (without type filter)
CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_audit_events_org_ts
  ON audit_events(organization_id, timestamp DESC);

-- =====================================================================
-- 3. Capability Grants - Filter by user + org  
-- =====================================================================
-- Permission checks: SELECT * FROM capability_grants 
--   WHERE user_id = ? AND organization_id = ? AND revoked_at IS NULL
-- This is called frequently during access control checks

CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_capability_grants_user_org
  ON capability_grants(user_id, organization_id)
  WHERE revoked_at IS NULL;

-- =====================================================================
-- 4. Memory Access Logs - Filter by memory + user
-- =====================================================================
-- Access tracking: SELECT * FROM memory_access_logs 
--   WHERE memory_id = ? AND user_id = ? ORDER BY accessed_at DESC

CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_memory_access_logs_memory_user
  ON memory_access_logs(memory_id, user_id, accessed_at DESC);

-- =====================================================================
-- 5. Users Table - Lookups by org + email
-- =====================================================================
-- User lookups: SELECT * FROM users WHERE organization_id = ? AND email = ?

CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_users_org_email
  ON users(organization_id, email);

-- =====================================================================
-- Verify Index Creation
-- =====================================================================
-- After running the above, verify indexes are created and active:

-- List all new indexes
SELECT 
  indexname,
  tablename,
  indexdef,
  idx_scan,
  idx_tup_read
FROM pg_stat_user_indexes
WHERE indexname LIKE 'ix_memories_%' 
   OR indexname LIKE 'ix_audit_%'
   OR indexname LIKE 'ix_capability_%'
   OR indexname LIKE 'ix_memory_access_%'
   OR indexname LIKE 'ix_users_%'
ORDER BY tablename, indexname;

-- Check index sizes
SELECT 
  schemaname,
  tablename,
  indexname,
  pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE indexname LIKE 'ix_memories_%' 
   OR indexname LIKE 'ix_audit_%'
   OR indexname LIKE 'ix_capability_%'
   OR indexname LIKE 'ix_memory_access_%'
   OR indexname LIKE 'ix_users_%'
ORDER BY pg_relation_size(indexrelid) DESC;
