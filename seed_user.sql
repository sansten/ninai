-- Seed default user and organization for development

-- Insert default organization if it doesn't exist
INSERT INTO organizations (id, name, slug, description, settings, is_active, created_at, updated_at)
SELECT 
  '550e8400-e29b-41d4-a716-446655440000'::uuid,
  'Default Organization',
  'default-org',
  'Default organization for initial setup',
  '{"features": {}, "settings": {}}'::jsonb,
  true,
  now(),
  now()
WHERE NOT EXISTS (SELECT 1 FROM organizations WHERE slug = 'default-org');

-- Insert default admin user if it doesn't exist
INSERT INTO users (
  id, email, hashed_password, full_name, is_active, 
  is_superuser, is_admin, role, clearance_level, preferences, created_at, updated_at
)
SELECT
  '550e8400-e29b-41d4-a716-446655440001'::uuid,
  'admin@ninai.dev',
  '$2b$12$QIvTZk/LS0l0fMJVA/yJ4u9JCLL5hjGF9aPv0cPKlLmLGO/bKZ3lO',
  'Admin User',
  true,
  true,
  true,
  'admin',
  100,
  '{}'::jsonb,
  now(),
  now()
WHERE NOT EXISTS (SELECT 1 FROM users WHERE email = 'admin@ninai.dev');
