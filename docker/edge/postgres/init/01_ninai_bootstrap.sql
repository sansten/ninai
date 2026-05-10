-- Ninai edge bootstrap SQL (idempotent on first init of a new PGDATA volume)
-- This script runs automatically via docker-entrypoint-initdb.d.

-- Core extensions used by the backend.
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Harden defaults for the main application role/database.
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'ninai') THEN
    ALTER ROLE ninai SET statement_timeout = '60s';
    ALTER ROLE ninai SET lock_timeout = '5s';
    ALTER ROLE ninai SET idle_in_transaction_session_timeout = '60s';
    ALTER ROLE ninai SET client_min_messages = WARNING;
  END IF;
END $$;

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_database WHERE datname = 'ninai') THEN
    ALTER DATABASE ninai SET timezone = 'UTC';
    ALTER DATABASE ninai SET default_transaction_isolation = 'read committed';
    ALTER DATABASE ninai SET statement_timeout = '60s';
  END IF;
END $$;
