-- Add is_admin column to users table
ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT 0;

-- Add last_login column to users table
ALTER TABLE users ADD COLUMN last_login TIMESTAMP;

-- Create system_settings table
CREATE TABLE IF NOT EXISTS system_settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL UNIQUE,
    value TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create system_logs table
CREATE TABLE IF NOT EXISTS system_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert default system settings
INSERT OR IGNORE INTO system_settings (key, value) VALUES
    ('site_name', 'AVR Video Editor'),
    ('admin_email', 'admin@avr.com'),
    ('timezone', 'UTC'),
    ('session_timeout', '30'),
    ('max_login_attempts', '5'),
    ('enable_2fa', '0'),
    ('max_upload_size', '500'),
    ('storage_limit_per_user', '10'),
    ('allowed_file_types', '.mp4,.avi,.mov,.wmv'),
    ('smtp_server', 'smtp.gmail.com'),
    ('smtp_port', '587'),
    ('smtp_username', 'avr.videoeditor@gmail.com'),
    ('smtp_password', ''); 