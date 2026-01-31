-- =============================================================================
-- Ninai Database Initialization Script
-- =============================================================================
-- This script runs when PostgreSQL container is first created.
-- It sets up required extensions and initial configuration.
-- =============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "ltree";

-- Create application user with limited privileges (for RLS)
-- The main user is created by Docker, but we set up session variables

-- Set timezone
SET timezone = 'UTC';

-- Create custom configuration parameter namespace for RLS
-- These will be set per-transaction by the application
DO $$
BEGIN
    -- These settings will be used by RLS policies
    PERFORM set_config('app.current_user_id', '', false);
    PERFORM set_config('app.current_org_id', '', false);
    PERFORM set_config('app.current_roles', '', false);
    PERFORM set_config('app.current_clearance_level', '0', false);
    PERFORM set_config('app.current_justification', '', false);
EXCEPTION
    WHEN OTHERS THEN
        -- Ignore errors if settings already exist
        NULL;
END
$$;

-- Grant usage on extensions
GRANT USAGE ON SCHEMA public TO ninai;

-- Log initialization
DO $$
BEGIN
    RAISE NOTICE 'Ninai database initialized successfully';
END
$$;
