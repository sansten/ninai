-- Create admin role and assign user to organization

-- Create roles if they don't exist
INSERT INTO roles (id, name, display_name, description, permissions, is_system, is_default, created_at, updated_at)
SELECT 
  '550e8400-e29b-41d4-a716-446655440010'::uuid,
  'admin',
  'Administrator',
  'Administrator role',
  ARRAY['*']::varchar[],
  true,
  false,
  now(),
  now()
WHERE NOT EXISTS (SELECT 1 FROM roles WHERE name = 'admin');

INSERT INTO roles (id, name, display_name, description, permissions, is_system, is_default, created_at, updated_at)
SELECT 
  '550e8400-e29b-41d4-a716-446655440011'::uuid,
  'member',
  'Member',
  'Member role',
  ARRAY['read', 'write']::varchar[],
  true,
  true,
  now(),
  now()
WHERE NOT EXISTS (SELECT 1 FROM roles WHERE name = 'member');

-- Assign admin user to default organization with admin role
INSERT INTO user_roles (id, user_id, organization_id, role_id, created_at)
SELECT 
  '550e8400-e29b-41d4-a716-446655440020'::uuid,
  '550e8400-e29b-41d4-a716-446655440001'::uuid,
  '550e8400-e29b-41d4-a716-446655440000'::uuid,
  (SELECT id FROM roles WHERE name = 'admin'),
  now()
WHERE NOT EXISTS (
  SELECT 1 FROM user_roles 
  WHERE user_id = '550e8400-e29b-41d4-a716-446655440001'::uuid
  AND organization_id = '550e8400-e29b-41d4-a716-446655440000'::uuid
);
