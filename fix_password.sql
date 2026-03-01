-- Fix the password hash for admin user
UPDATE users 
SET hashed_password = '$2b$12$aOYy.zj.TRqAmJ8ZPX6ZbuXokMPir/MZPSt4DmDUel1G05iSTPfSi'
WHERE email = 'admin@ninai.dev';
