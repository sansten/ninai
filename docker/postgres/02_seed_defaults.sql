-- =============================================================================
-- Ninai Default Data Seeding
-- =============================================================================
-- This script creates default data required for the application to function:
-- - Default organization
-- - System roles (admin, member)
-- - Default admin user
-- - Role assignments
--
-- This script is idempotent - safe to run multiple times
-- =============================================================================

-- Wait for tables to be created (they're created by SQLAlchemy on backend startup)
DO $$
DECLARE
    max_attempts INTEGER := 30;
    attempt INTEGER := 0;
    tables_exist BOOLEAN := FALSE;
BEGIN
    WHILE attempt < max_attempts AND NOT tables_exist LOOP
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('users', 'organizations', 'roles', 'user_roles')
        ) INTO tables_exist;
        
        IF NOT tables_exist THEN
            RAISE NOTICE 'Waiting for tables to be created... (attempt % of %)', attempt + 1, max_attempts;
            PERFORM pg_sleep(2);
            attempt := attempt + 1;
        END IF;
    END LOOP;
    
    IF NOT tables_exist THEN
        RAISE NOTICE 'Tables not created yet - seeding will be skipped. Backend will create tables on first startup.';
    ELSE
        RAISE NOTICE 'Tables found - proceeding with seeding';
    END IF;
END $$;

-- =============================================================================
-- 1. Create Default Organization
-- =============================================================================
INSERT INTO organizations (
    id,
    name,
    slug,
    description,
    settings,
    is_active,
    created_at,
    updated_at
)
SELECT 
    '550e8400-e29b-41d4-a716-446655440000'::uuid,
    'Default Organization',
    'default',
    'Default organization for initial setup',
    '{}'::jsonb,
    true,
    now(),
    now()
WHERE NOT EXISTS (
    SELECT 1 FROM organizations WHERE slug = 'default'
);

-- =============================================================================
-- 2. Create System Roles
-- =============================================================================

-- Admin Role
INSERT INTO roles (
    id,
    name,
    display_name,
    description,
    permissions,
    is_system,
    is_default,
    created_at,
    updated_at
)
SELECT 
    '550e8400-e29b-41d4-a716-446655440010'::uuid,
    'admin',
    'Administrator',
    'Full system administrator with all permissions',
    ARRAY['*']::varchar[],
    true,
    false,
    now(),
    now()
WHERE NOT EXISTS (
    SELECT 1 FROM roles WHERE name = 'admin' AND organization_id IS NULL
);

-- Member Role
INSERT INTO roles (
    id,
    name,
    display_name,
    description,
    permissions,
    is_system,
    is_default,
    created_at,
    updated_at
)
SELECT 
    '550e8400-e29b-41d4-a716-446655440011'::uuid,
    'member',
    'Member',
    'Standard member with read and write permissions',
    ARRAY['read', 'write', 'memory:create', 'memory:read', 'memory:update', 'memory:delete']::varchar[],
    true,
    true,
    now(),
    now()
WHERE NOT EXISTS (
    SELECT 1 FROM roles WHERE name = 'member' AND organization_id IS NULL
);

-- Viewer Role
INSERT INTO roles (
    id,
    name,
    display_name,
    description,
    permissions,
    is_system,
    is_default,
    created_at,
    updated_at
)
SELECT 
    '550e8400-e29b-41d4-a716-446655440012'::uuid,
    'viewer',
    'Viewer',
    'Read-only access',
    ARRAY['read', 'memory:read']::varchar[],
    true,
    false,
    now(),
    now()
WHERE NOT EXISTS (
    SELECT 1 FROM roles WHERE name = 'viewer' AND organization_id IS NULL
);

-- =============================================================================
-- 3. Create Default Admin User
-- =============================================================================
-- Password: admin123
-- Hashed using bcrypt with cost factor 12
INSERT INTO users (
    id,
    email,
    full_name,
    password_hash,
    is_active,
    is_verified,
    auth_provider,
    created_at,
    updated_at
)
SELECT 
    '550e8400-e29b-41d4-a716-446655440001'::uuid,
    'admin@ninai.dev',
    'System Administrator',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYKKAx4L82u',
    true,
    true,
    'password',
    now(),
    now()
WHERE NOT EXISTS (
    SELECT 1 FROM users WHERE email = 'admin@ninai.dev'
);

-- =============================================================================
-- 4. Assign Admin User to Default Organization with Admin Role
-- =============================================================================
INSERT INTO user_roles (
    id,
    user_id,
    organization_id,
    role_id,
    created_at,
    updated_at
)
SELECT 
    '550e8400-e29b-41d4-a716-446655440020'::uuid,
    '550e8400-e29b-41d4-a716-446655440001'::uuid,
    '550e8400-e29b-41d4-a716-446655440000'::uuid,
    '550e8400-e29b-41d4-a716-446655440010'::uuid,
    now(),
    now()
WHERE NOT EXISTS (
    SELECT 1 FROM user_roles 
    WHERE user_id = '550e8400-e29b-41d4-a716-446655440001'::uuid
    AND organization_id = '550e8400-e29b-41d4-a716-446655440000'::uuid
    AND role_id = '550e8400-e29b-41d4-a716-446655440010'::uuid
);

-- =============================================================================
-- Verification & Summary
-- =============================================================================
DO $$
DECLARE
    org_count INTEGER;
    role_count INTEGER;
    user_count INTEGER;
    user_role_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO org_count FROM organizations WHERE slug = 'default';
    SELECT COUNT(*) INTO role_count FROM roles WHERE is_system = true;
    SELECT COUNT(*) INTO user_count FROM users WHERE email = 'admin@ninai.dev';
    SELECT COUNT(*) INTO user_role_count FROM user_roles WHERE user_id = '550e8400-e29b-41d4-a716-446655440001'::uuid;
    
    RAISE NOTICE '';
    RAISE NOTICE '=============================================================================';
    RAISE NOTICE 'Ninai Database Initialization Complete';
    RAISE NOTICE '=============================================================================';
    RAISE NOTICE 'Default Organization: % created', org_count;
    RAISE NOTICE 'System Roles: % created (admin, member, viewer)', role_count;
    RAISE NOTICE 'Admin User: % created', user_count;
    RAISE NOTICE 'Role Assignments: % created', user_role_count;
    RAISE NOTICE '';
    RAISE NOTICE 'Default Login Credentials:';
    RAISE NOTICE '  Email: admin@ninai.dev';
    RAISE NOTICE '  Password: admin123';
    RAISE NOTICE '';
    RAISE NOTICE 'IMPORTANT: Change the default password after first login!';
    RAISE NOTICE '=============================================================================';
    RAISE NOTICE '';
END $$;
