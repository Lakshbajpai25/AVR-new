-- Add is_admin column to users table
ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT 0;

-- Add last_login column to users table
ALTER TABLE users ADD COLUMN last_login TIMESTAMP; 