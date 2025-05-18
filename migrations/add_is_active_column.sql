-- Add is_active column to users table
ALTER TABLE users ADD COLUMN is_active BOOLEAN DEFAULT 1;

-- Update existing users to be active
UPDATE users SET is_active = 1; 