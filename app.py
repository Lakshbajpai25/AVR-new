from flask import Flask, render_template, request, jsonify, send_file, Response, redirect, url_for, flash, session
import os
import json
import time
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import sqlite3
from functools import wraps
from flask import Flask
from jinja2 import ChoiceLoader, FileSystemLoader
import cv2
import numpy as np
from object_detector import ObjectDetector
from ultralytics import YOLO
from moviepy import VideoFileClip
import torch
from datetime import datetime, timedelta
from PIL import Image
import humanize
import secrets
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name__)
# Set up multiple template folders
template_dirs = ['templates', 'Templates']
app.jinja_loader = ChoiceLoader([FileSystemLoader(template_dir) for template_dir in template_dirs])

# Configure session and security
app.secret_key = os.urandom(24)  # Generate a random secret key
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)  # Session expires after 30 minutes
app.config['SESSION_COOKIE_SECURE'] = True  # Only send cookies over HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True  # Prevent JavaScript access to cookies
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # Prevent CSRF attacks

# Email Configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'avr.videoeditor@gmail.com'  # Replace this with your Gmail address
app.config['MAIL_PASSWORD'] = 'xfwu qxvp rnxm rnxm'  # Replace this with your Gmail App Password
app.config['MAIL_DEFAULT_SENDER'] = 'AVR Video Editor <avr.videoeditor@gmail.com>'

# Configure upload folder
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize LoginManager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Add these configuration settings after the existing app configurations
app.config['MAIL_SERVER'] = 'smtp.gmail.com'  # Replace with your SMTP server
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your-email@gmail.com'  # Replace with your email
app.config['MAIL_PASSWORD'] = 'your-app-password'  # Replace with your app password
app.config['MAIL_DEFAULT_SENDER'] = 'your-email@gmail.com'  # Replace with your email

# Database setup
def get_db():
    db = sqlite3.connect('instance/avr.db')
    db.row_factory = sqlite3.Row
    return db

def init_db():
    db = get_db()
    with app.open_resource('schema.sql') as f:
        db.executescript(f.read().decode('utf8'))
    # Execute password reset tokens migration
    with app.open_resource('migrations/add_password_reset_tokens.sql') as f:
        db.executescript(f.read().decode('utf8'))
    db.commit()
    db.close()

def migrate_db():
    db = get_db()
    try:
        # Check if is_favorite column exists
        cursor = db.execute('SELECT * FROM projects LIMIT 1')
        columns = [description[0] for description in cursor.description]
        
        # Add is_favorite column if it doesn't exist
        if 'is_favorite' not in columns:
            print("Adding is_favorite column to projects table...")
            db.execute('ALTER TABLE projects ADD COLUMN is_favorite BOOLEAN DEFAULT 0')
            db.commit()
            print("Migration completed successfully")

        # Check if is_admin and last_login columns exist in users table
        cursor = db.execute('SELECT * FROM users LIMIT 1')
        user_columns = [description[0] for description in cursor.description]
        
        # Add is_admin column if it doesn't exist
        if 'is_admin' not in user_columns:
            print("Adding is_admin column to users table...")
            db.execute('ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT 0')
            db.commit()
            print("Admin column migration completed successfully")

        # Add last_login column if it doesn't exist
        if 'last_login' not in user_columns:
            print("Adding last_login column to users table...")
            db.execute('ALTER TABLE users ADD COLUMN last_login TIMESTAMP')
            db.commit()
            print("Last login column migration completed successfully")

        # Add is_active column if it doesn't exist
        if 'is_active' not in user_columns:
            print("Adding is_active column to users table...")
            db.execute('ALTER TABLE users ADD COLUMN is_active BOOLEAN DEFAULT 1')
            db.execute('UPDATE users SET is_active = 1')
            db.commit()
            print("Active status column migration completed successfully")

        # Check if password_reset_tokens table exists
        cursor = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='password_reset_tokens'")
        if not cursor.fetchone():
            print("Creating password_reset_tokens table...")
            with app.open_resource('migrations/add_password_reset_tokens.sql') as f:
                db.executescript(f.read().decode('utf8'))
            db.commit()
            print("Password reset tokens table created successfully")
            
    except Exception as e:
        print(f"Error during migration: {str(e)}")
    finally:
        db.close()

def ensure_db_exists():
    try:
        # Check if the instance directory exists
        if not os.path.exists('instance'):
            os.makedirs('instance')
        
        # Check if the database file exists
        db_path = 'instance/avr.db'
        if not os.path.exists(db_path):
            init_db()
            print("Database initialized successfully")
        else:
            print("Database already exists")
            # Run migration for existing database
            migrate_db()
    except Exception as e:
        print(f"Error initializing database: {str(e)}")

# Initialize database when app starts
ensure_db_exists()

# User model
class User(UserMixin):
    def __init__(self, user_id, username, email, first_name=None, last_name=None):
        self.id = user_id
        self.username = username
        self.email = email
        self.first_name = first_name
        self.last_name = last_name
        
    @property
    def full_name(self):
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.username

@login_manager.user_loader
def load_user(user_id):
    db = get_db()
    user = db.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    db.close()
    if user:
        return User(user['id'], user['username'], user['email'], user['first_name'], user['last_name'])
    return None

# Dictionary to store processing progress
processing_progress = {}

detector = ObjectDetector()


# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    # Check if user is already logged in
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    # Clear any existing flash messages
    session.pop('_flashes', None)
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not email or not password:
            flash('Please enter both email and password', 'error')
            return render_template('login.html')
        
        db = get_db()
        user = db.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        db.close()
        
        if not user:
            flash('No account found with this email', 'error')
            return render_template('login.html')
        
        if not check_password_hash(user['password_hash'], password):
            flash('Incorrect password', 'error')
            return render_template('login.html')
        
        user_obj = User(user['id'], user['username'], user['email'], user['first_name'], user['last_name'])
        login_user(user_obj)
        flash('Login successful!', 'success')
        return redirect(url_for('dashboard'))
        
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Check if user is admin - redirect to admin dashboard if true
    db = get_db()
    user = db.execute('SELECT is_admin FROM users WHERE id = ?', (current_user.id,)).fetchone()
    
    if user and user['is_admin']:
        return redirect(url_for('admin_dashboard'))
    
    if not current_user.is_authenticated:
        flash('Please log in to access the dashboard', 'error')
        return redirect(url_for('login'))
    
    db = get_db()
    
    # Get favorite count
    favorite_count = db.execute('''
        SELECT COUNT(*) as count 
        FROM projects 
        WHERE user_id = ? AND is_favorite = 1
    ''', (current_user.id,)).fetchone()['count']
    
    # Count all videos from the projects table for the current user
    processed_videos_count = db.execute('''
        SELECT COUNT(*) as count 
        FROM projects 
        WHERE user_id = ?
    ''', (current_user.id,)).fetchone()['count']
    
    # Calculate success rate
    total_projects = db.execute('''
        SELECT COUNT(*) as count, status
        FROM projects
        WHERE user_id = ?
        GROUP BY status
    ''', (current_user.id,)).fetchall()
    
    total_count = sum(row['count'] for row in total_projects)
    success_count = next((row['count'] for row in total_projects if row['status'] == 'completed'), 0)
    success_rate = (success_count / total_count * 100) if total_count > 0 else 0
    
    # Calculate average processing time
    avg_processing_time = db.execute('''
        SELECT AVG(JULIANDAY(processing_end) - JULIANDAY(processing_start)) * 24 * 60 as avg_minutes
        FROM projects
        WHERE user_id = ? AND status = 'completed'
        AND processing_start IS NOT NULL AND processing_end IS NOT NULL
    ''', (current_user.id,)).fetchone()['avg_minutes']
    
    # Calculate storage savings
    storage_stats = db.execute('''
        SELECT 
            SUM(original_size) as total_original,
            SUM(processed_size) as total_processed
        FROM projects
        WHERE user_id = ? AND status = 'completed'
        AND original_size IS NOT NULL AND processed_size IS NOT NULL
    ''', (current_user.id,)).fetchone()
    
    storage_saved = (storage_stats['total_original'] - storage_stats['total_processed']) if storage_stats['total_original'] else 0
    storage_saved_percent = (storage_saved / storage_stats['total_original'] * 100) if storage_stats['total_original'] else 0
    
    # Calculate total storage used (original + processed files)
    total_storage = db.execute('''
        SELECT 
            SUM(original_size) + SUM(processed_size) as total_storage
        FROM projects
        WHERE user_id = ?
        AND original_size IS NOT NULL
    ''', (current_user.id,)).fetchone()['total_storage'] or 0
    
    # Calculate storage usage breakdown
    storage_breakdown = db.execute('''
        SELECT 
            COUNT(*) as total_files,
            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_files,
            SUM(CASE WHEN status = 'processing' THEN 1 ELSE 0 END) as processing_files,
            SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_files,
            SUM(original_size) as total_original_size,
            SUM(processed_size) as total_processed_size
        FROM projects
        WHERE user_id = ?
    ''', (current_user.id,)).fetchone()
    
    # Format storage breakdown for display
    storage_metrics = {
        'total_files': storage_breakdown['total_files'],
        'completed_files': storage_breakdown['completed_files'],
        'processing_files': storage_breakdown['processing_files'],
        'failed_files': storage_breakdown['failed_files'],
        'original_total': humanize.naturalsize(storage_breakdown['total_original_size']) if storage_breakdown['total_original_size'] else '0 B',
        'processed_total': humanize.naturalsize(storage_breakdown['total_processed_size']) if storage_breakdown['total_processed_size'] else '0 B',
    }
    
    # Get recent activities with sorting
    sort_by = request.args.get('sort', 'date')  # date, name, size
    sort_order = request.args.get('order', 'desc')  # asc, desc
    
    sort_clause = {
        'date': 'created_at',
        'name': 'original_filename',
        'size': 'original_size'
    }.get(sort_by, 'created_at')
    
    order_clause = 'DESC' if sort_order == 'desc' else 'ASC'
    
    recent_activities = db.execute(f'''
        SELECT *
        FROM projects
        WHERE user_id = ?
        ORDER BY {sort_clause} {order_clause}
        LIMIT 5
    ''', (current_user.id,)).fetchall()
    
    # Format the activities for display
    formatted_activities = []
    for activity in recent_activities:
        # Calculate relative time
        relative_time = get_relative_time(activity['created_at'])
        
        # Calculate processing time if available
        processing_time = None
        if activity['processing_start'] and activity['processing_end']:
            try:
                try:
                    start = datetime.strptime(activity['processing_start'], '%Y-%m-%d %H:%M:%S.%f')
                except ValueError:
                    start = datetime.strptime(activity['processing_start'], '%Y-%m-%d %H:%M:%S')
                try:
                    end = datetime.strptime(activity['processing_end'], '%Y-%m-%d %H:%M:%S.%f')
                except ValueError:
                    end = datetime.strptime(activity['processing_end'], '%Y-%m-%d %H:%M:%S')
                processing_time = humanize.naturaldelta(end - start)
            except Exception as e:
                print(f"Error calculating processing time: {str(e)}")
                processing_time = None
        
        # Parse processing parameters
        processing_params = json.loads(activity['processing_parameters']) if activity['processing_parameters'] else {}
        
        formatted_activities.append({
            'id': activity['id'],
            'original_filename': activity['original_filename'],
            'filename': activity['filename'],
            'created_at': activity['created_at'],
            'relative_time': relative_time,
            'status': activity['status'],
            'thumbnail_path': activity['thumbnail_path'],
            'processing_time': processing_time,
            'original_size': humanize.naturalsize(activity['original_size']) if activity['original_size'] else None,
            'processed_size': humanize.naturalsize(activity['processed_size']) if activity['processed_size'] else None,
            'size_reduction': f"{((activity['original_size'] - activity['processed_size']) / activity['original_size'] * 100):.1f}%" if activity['original_size'] and activity['processed_size'] else None,
            'parameters': processing_params
        })
    
    # Group activities by date
    today = datetime.now().date()
    grouped_activities = {
        'Today': [],
        'Yesterday': [],
        'This Week': [],
        'Older': []
    }
    
    for activity in formatted_activities:
        activity_date = datetime.strptime(activity['created_at'], '%Y-%m-%d %H:%M:%S').date()
        if activity_date == today:
            grouped_activities['Today'].append(activity)
        elif activity_date == today - timedelta(days=1):
            grouped_activities['Yesterday'].append(activity)
        elif activity_date >= today - timedelta(days=7):
            grouped_activities['This Week'].append(activity)
        else:
            grouped_activities['Older'].append(activity)
    
    db.close()
    
    return render_template('dashboard.html', 
                         processed_videos_count=processed_videos_count,
                         favorite_count=favorite_count,
                         recent_activities=formatted_activities,
                         grouped_activities=grouped_activities,
                         success_rate=success_rate,
                         avg_processing_time=avg_processing_time,
                         storage_saved=humanize.naturalsize(storage_saved),
                         storage_saved_percent=storage_saved_percent,
                         storage_used=humanize.naturalsize(total_storage),
                         storage_metrics=storage_metrics,
                         sort_by=sort_by,
                         sort_order=sort_order)

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/signup', methods=['POST'])
def signup():
    firstname = request.form.get('firstname')
    lastname = request.form.get('lastname')
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')
    confirm_password = request.form.get('confirm_password')
    terms = request.form.get('terms')
    
    # Validate all required fields
    if not all([firstname, lastname, username, email, password, confirm_password]):
        flash('Please fill in all required fields', 'error')
        return redirect(url_for('index'))
    
    # Validate terms acceptance
    if not terms:
        flash('You must accept the Terms and Conditions', 'error')
        return redirect(url_for('index'))
    
    # Validate password match
    if password != confirm_password:
        flash('Passwords do not match', 'error')
        return redirect(url_for('index'))
    
    # Validate password strength
    if len(password) < 8:
        flash('Password must be at least 8 characters long', 'error')
        return redirect(url_for('index'))
    
    db = get_db()
    try:
        # Check if email already exists
        existing_user = db.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        if existing_user:
            flash('Email already registered', 'error')
            return redirect(url_for('index'))
        
        # Check if username already exists
        existing_username = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        if existing_username:
            flash('Username already taken', 'error')
            return redirect(url_for('index'))
        
        # Create new user
        password_hash = generate_password_hash(password)
        db.execute(
            'INSERT INTO users (username, email, password_hash, first_name, last_name) VALUES (?, ?, ?, ?, ?)',
            (username, email, password_hash, firstname, lastname)
        )
        db.commit()
        flash('Account created successfully! Please login.', 'success')
        return redirect(url_for('login'))
    except sqlite3.Error as e:
        db.rollback()
        flash('An error occurred during signup', 'error')
        return redirect(url_for('index'))
    finally:
        db.close()

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contactus')
def contactus():
    return render_template('contactus.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/process')
def process():
    return render_template('process.html')

@app.route('/my_projects')
@login_required
def my_projects():
    db = get_db()
    projects = db.execute('''
        SELECT id, filename, original_filename, created_at, status, processed_size, is_favorite
        FROM projects
        WHERE user_id = ?
        ORDER BY created_at DESC
    ''', (current_user.id,)).fetchall()
    
    # Convert projects to list of dictionaries
    projects_list = []
    for project in projects:
        projects_list.append({
            'id': project['id'],
            'filename': project['filename'],
            'original_filename': project['original_filename'],
            'created_at': project['created_at'],
            'status': project['status'],
            'processed_size': humanize.naturalsize(project['processed_size']) if project['processed_size'] else None,
            'is_favorite': project['is_favorite']
        })
    
    return render_template('my_projects.html', projects=projects_list)


@app.route('/profile')
@login_required
def profile():
    db = get_db()
    user_data = db.execute('SELECT * FROM users WHERE id = ?', (current_user.id,)).fetchone()
    db.close()
    return render_template('profile.html', user=user_data)

@app.route('/update_name', methods=['POST'])
@login_required
def update_name():
    first_name = request.form.get('first_name')
    last_name = request.form.get('last_name')
    
    if not first_name or not last_name:
        flash('Please provide both first and last name', 'error')
        return redirect(url_for('profile'))
    
    try:
        db = get_db()
        db.execute('''
            UPDATE users 
            SET first_name = ?, last_name = ? 
            WHERE id = ?
        ''', (first_name, last_name, current_user.id))
        db.commit()
        flash('Name updated successfully!', 'success')
    except Exception as e:
        flash('An error occurred while updating your name', 'error')
    finally:
        db.close()
    
    return redirect(url_for('profile'))

@app.route('/project')
@login_required
def project():
    return render_template('project.html')


# Initialize YOLO model
model = None
model_seg = None

def get_detection_model():
    global model
    if model is None:
        print("Loading YOLO detection model...")
        model = YOLO('yolov8n.pt')
        print("YOLO detection model loaded successfully")
    return model

def get_segmentation_model():
    global model_seg
    if model_seg is None:
        print("Loading YOLO segmentation model...")
        model_seg = YOLO('yolov8n-seg.pt')
        print("YOLO segmentation model loaded successfully")
    return model_seg

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({
            'success': False,
            'message': 'No file part'
        })
    file = request.files['video']
    if file.filename == '':
        return jsonify({
            'success': False,
            'message': 'No selected file'
        })
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({
            'success': True,
            'filename': filename,
            'message': 'File uploaded successfully'
        })
    return jsonify({
        'success': False,
        'message': 'Invalid file type'
    })

@app.route('/progress/<filename>')
def progress(filename):
    def generate():
        start_time = time.time()
        last_progress = 0
        last_time = start_time
        
        while True:
            current_progress = processing_progress.get(filename, 0)
            current_time = time.time()
            
            # Calculate time remaining
            if current_progress > last_progress:
                time_elapsed = current_time - start_time
                if current_progress > 0:
                    estimated_total_time = time_elapsed / (current_progress / 100)
                    time_remaining = estimated_total_time - time_elapsed
                else:
                    time_remaining = 0
            else:
                time_remaining = 0
            
            yield f"data: {{\"percentage\": {current_progress}, \"time_remaining\": {time_remaining}}}\n\n"
            
            if current_progress >= 100:
                break
                
            last_progress = current_progress
            last_time = current_time
            time.sleep(0.5)
            
    return Response(generate(), mimetype='text/event-stream')

def generate_thumbnail(video_path, output_path, timestamp=1.0):
    """Generate thumbnail from video at given timestamp."""
    try:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        success, frame = cap.read()
        cap.release()
        
        if success:
            # Convert to RGB and resize
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = frame_rgb.shape[:2]
            max_size = 320
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            
            thumbnail = cv2.resize(frame_rgb, (new_width, new_height))
            
            # Save as JPEG
            img = Image.fromarray(thumbnail)
            img.save(output_path, 'JPEG', quality=85)
            return True
    except Exception as e:
        print(f"Error generating thumbnail: {str(e)}")
    return False

def get_relative_time(timestamp_str):
    """Convert timestamp to relative time (e.g., '2 hours ago')."""
    try:
        # Handle timestamps with or without microseconds
        try:
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
        except ValueError:
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            
        now = datetime.now()
        diff = now - timestamp
        
        if diff.days == 0:
            if diff.seconds < 60:
                return 'just now'
            elif diff.seconds < 3600:
                minutes = diff.seconds // 60
                return f'{minutes} minute{"s" if minutes != 1 else ""} ago'
            else:
                hours = diff.seconds // 3600
                return f'{hours} hour{"s" if hours != 1 else ""} ago'
        elif diff.days == 1:
            return 'yesterday'
        elif diff.days < 7:
            return f'{diff.days} days ago'
        elif diff.days < 30:
            weeks = diff.days // 7
            return f'{weeks} week{"s" if weeks != 1 else ""} ago'
        else:
            months = diff.days // 30
            return f'{months} month{"s" if months != 1 else ""} ago'
    except:
        return timestamp_str

@app.route('/process', methods=['POST'])
@login_required
def process_video():
    if not current_user.is_authenticated:
        return jsonify({'success': False, 'message': 'Please log in to process videos'})
        
    data = request.json
    filename = data.get('filename')
    if not filename:
        return jsonify({'success': False, 'message': 'No filename provided'})
    
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(input_path):
        return jsonify({'success': False, 'message': 'Video file not found'})
    
    try:
        # Initialize progress and project in database with 'processing' status
        processing_progress[filename] = 0
        db = get_db()
        output_filename = f'processed_{filename}'
        processing_start = datetime.now()
        original_size = os.path.getsize(input_path)
        
        # Store processing parameters
        processing_params = {
            'aspect_ratio': data.get('aspect_ratio', '16:9'),
            'subject_focus': data.get('subject_focus', 'no'),
            'focus_x': data.get('focus_x', 50),
            'focus_y': data.get('focus_y', 50),
            'auto_focus_mode': data.get('auto_focus_mode'),
            'selected_subject': data.get('selected_subject')
        }
        
        # Create thumbnail directory if it doesn't exist
        thumbnail_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'thumbnails')
        os.makedirs(thumbnail_dir, exist_ok=True)
        thumbnail_filename = f'thumb_{filename}.jpg'
        thumbnail_path = os.path.join(thumbnail_dir, thumbnail_filename)
        
        # Generate thumbnail from original video
        generate_thumbnail(input_path, thumbnail_path)
        
        # Create project entry with 'processing' status
        db.execute('''
            INSERT INTO projects (
                user_id, filename, original_filename, status, processing_start,
                aspect_ratio, subject_focus_mode, focus_coordinates,
                original_size, thumbnail_path, processing_parameters
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            current_user.id, output_filename, filename, 'processing', processing_start,
            data.get('aspect_ratio'), data.get('subject_focus'),
            json.dumps({'x': data.get('focus_x'), 'y': data.get('focus_y')}),
            original_size, thumbnail_filename, json.dumps(processing_params)
        ))
        db.commit()
        
        # Get input video extension
        input_ext = os.path.splitext(filename)[1].lower()
        output_filename = f'processed_{filename}'
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        
        # Remove existing files if they exist
        temp_output = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_{output_filename}')
        if os.path.exists(temp_output):
            os.remove(temp_output)
        if os.path.exists(output_path):
            os.remove(output_path)
        
        # Open video file
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return jsonify({'success': False, 'message': 'Error opening video file'})
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Validate video dimensions
        if width <= 0 or height <= 0:
            raise ValueError("Invalid video dimensions")
        
        # Calculate target dimensions based on aspect ratio
        target_w, target_h = map(int, data.get('aspect_ratio', '16:9').split(':'))
        target_ratio = target_w / target_h
        
        # Calculate output dimensions while maintaining aspect ratio
        if width/height > target_ratio:
            output_w = int(height * target_ratio)
            output_h = height
        else:
            output_w = width
            output_h = int(width / target_ratio)
            
        # Ensure output dimensions are valid
        if output_w <= 0 or output_h <= 0:
            raise ValueError(f"Invalid output dimensions: {output_w}x{output_h}")
            
        print(f"Input dimensions: {width}x{height}")
        print(f"Output dimensions: {output_w}x{output_h}")
        
        # Initialize YOLO model if auto subject focus is enabled
        model = None
        if data.get('subject_focus') == 'auto':
            model = get_detection_model()
        
        # Set up video writer with appropriate codec
        if input_ext in ['.mp4', '.avi']:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif input_ext == '.mov':
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        # Create video writer
        out = cv2.VideoWriter(temp_output, fourcc, fps, (output_w, output_h))
        if not out.isOpened():
            raise ValueError("Failed to create output video file")
        
        frame_count = 0
        first_subject_box = None
        prev_center_x, prev_center_y = None, None  # For temporal smoothing
        chosen_object = None
        chosen_object_label = None
        chosen_object_frame = None
        all_detections = []
        if data.get('subject_focus') == 'auto' and data.get('auto_focus_mode') == 'multiple' and get_segmentation_model() is not None:
            # First pass: collect all detections by frame
            cap1 = cv2.VideoCapture(input_path)
            frame_idx = 0
            all_detections_by_frame = []
            while cap1.isOpened():
                ret, frame = cap1.read()
                if not ret:
                    break
                frame_detections = []
                results = get_segmentation_model()(frame)
                for result in results:
                    boxes = result.boxes
                    for i, box in enumerate(boxes):
                        label = get_segmentation_model().names[int(box.cls[0])]
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        frame_detections.append({
                            'label': label,
                            'x1': x1,
                            'y1': y1,
                            'x2': x2,
                            'y2': y2,
                            'frame': frame_idx
                        })
                all_detections_by_frame.append(frame_detections)
                frame_idx += 1
            cap1.release()
            # Group unique subjects by label and initial position (first frame)
            unique_subjects = []
            if all_detections_by_frame:
                first_frame_detections = all_detections_by_frame[0]
                for idx, det in enumerate(first_frame_detections):
                    unique_subjects.append({
                        'label': det['label'],
                        'x1': det['x1'],
                        'y1': det['y1'],
                        'x2': det['x2'],
                        'y2': det['y2'],
                        'id': idx
                    })
            output_files = []
            for subj in unique_subjects:
                # Prepare output file for this subject
                subj_label = subj['label']
                subj_id = subj['id']
                subj_box = [subj['x1'], subj['y1'], subj['x2'], subj['y2']]
                subj_output_filename = f'processed_{subj_label}_{subj_id}_{filename}'
                subj_temp_output = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_{subj_output_filename}')
                subj_final_output = os.path.join(app.config['UPLOAD_FOLDER'], subj_output_filename)
                # Remove existing files if they exist
                if os.path.exists(subj_temp_output):
                    os.remove(subj_temp_output)
                if os.path.exists(subj_final_output):
                    os.remove(subj_final_output)
                # Second pass: process video for this subject
                cap2 = cv2.VideoCapture(input_path)
                out2 = cv2.VideoWriter(subj_temp_output, fourcc, fps, (output_w, output_h))
                prev_center_x, prev_center_y = None, None
                frame_count2 = 0
                while cap2.isOpened():
                    ret, frame = cap2.read()
                    if not ret:
                        break
                    frame_count2 += 1
                    try:
                        # Find best match for this subject in this frame
                        results = get_segmentation_model()(frame)
                        best_box = None
                        best_score = -1
                        for result in results:
                            boxes = result.boxes
                            for i, box in enumerate(boxes):
                                label = get_segmentation_model().names[int(box.cls[0])]
                                if label == subj_label:
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    # Score by IoU with original subject box
                                    orig = subj
                                    xx1 = max(x1, orig['x1'])
                                    yy1 = max(y1, orig['y1'])
                                    xx2 = min(x2, orig['x2'])
                                    yy2 = min(y2, orig['y2'])
                                    inter_area = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                                    box_area = (x2 - x1) * (y2 - y1)
                                    orig_area = (orig['x2'] - orig['x1']) * (orig['y2'] - orig['y1'])
                                    union_area = box_area + orig_area - inter_area
                                    iou = inter_area / union_area if union_area > 0 else 0
                                    score = iou
                                    if score > best_score:
                                        best_score = score
                                        best_box = (x1, y1, x2, y2)
                        if best_box:
                            center_x = (best_box[0] + best_box[2]) // 2
                            center_y = (best_box[1] + best_box[3]) // 2
                        else:
                            center_x = width // 2
                            center_y = height // 2
                        # Temporal smoothing
                        if prev_center_x is not None and prev_center_y is not None:
                            alpha = 0.7
                            center_x = int(alpha * prev_center_x + (1 - alpha) * center_x)
                            center_y = int(alpha * prev_center_y + (1 - alpha) * center_y)
                        prev_center_x, prev_center_y = center_x, center_y
                        # Calculate crop dimensions
                        if width/height > target_ratio:
                            new_width = int(height * target_ratio)
                            crop_width = new_width
                            crop_height = height
                        else:
                            new_height = int(width / target_ratio)
                            crop_width = width
                            crop_height = new_height
                        crop_width = min(crop_width, width)
                        crop_height = min(crop_height, height)
                        x1 = max(0, min(center_x - crop_width // 2, width - crop_width))
                        x2 = min(width, x1 + crop_width)
                        y1 = max(0, min(center_y - crop_height // 2, height - crop_height))
                        y2 = min(height, y1 + crop_height)
                        if x2 <= x1 or y2 <= y1:
                            raise ValueError(f"Invalid crop coordinates: ({x1}, {y1}, {x2}, {y2})")
                        cropped = frame[y1:y2, x1:x2]
                        if cropped.size == 0:
                            raise ValueError("Empty crop region")
                        resized = cv2.resize(cropped, (output_w, output_h), interpolation=cv2.INTER_LINEAR)
                        if resized.size == 0:
                            raise ValueError("Empty resized frame")
                        out2.write(resized)
                    except Exception as frame_error:
                        print(f"Error processing frame {frame_count2}: {str(frame_error)}")
                        continue
                cap2.release()
                out2.release()
                # Add original audio to processed video using FFmpeg
                try:
                    import subprocess
                    if os.path.exists(subj_final_output):
                        os.remove(subj_final_output)
                    subprocess.run([
                        'ffmpeg', '-y',
                        '-i', subj_temp_output,
                        '-i', input_path,
                        '-c:v', 'copy',
                        '-c:a', 'copy',
                        '-map', '0:v:0',
                        '-map', '1:a:0?',
                        subj_final_output
                    ], check=True)
                    if os.path.exists(subj_temp_output):
                        os.remove(subj_temp_output)
                except (ImportError, subprocess.SubprocessError, FileNotFoundError) as e:
                    print(f"FFmpeg audio muxing failed: {str(e)}")
                    if os.path.exists(subj_temp_output):
                        os.remove(subj_temp_output)
                output_files.append({'filename': subj_output_filename, 'label': subj_label, 'id': subj_id})
            processing_progress[filename] = 100
            return jsonify({
                'success': True,
                'processed_files': output_files,
                'message': f'Generated {len(output_files)} subject videos.'
            })
        else:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                progress = int((frame_count / total_frames) * 100)
                processing_progress[filename] = progress
                try:
                    if data.get('selected_subject') and get_segmentation_model() is not None:
                        # Run segmentation on this frame
                        results = get_segmentation_model()(frame)
                        best_box = None
                        best_score = -1
                        for result in results:
                            boxes = result.boxes
                            for i, box in enumerate(boxes):
                                label = get_segmentation_model().names[int(box.cls[0])]
                                if label == data['selected_subject'].get('label'):
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    # Score by IoU with original box (first frame)
                                    if first_subject_box:
                                        # Calculate intersection-over-union (IoU)
                                        xx1 = max(x1, first_subject_box[0])
                                        yy1 = max(y1, first_subject_box[1])
                                        xx2 = min(x2, first_subject_box[2])
                                        yy2 = min(y2, first_subject_box[3])
                                        inter_area = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                                        box_area = (x2 - x1) * (y2 - y1)
                                        first_area = (first_subject_box[2] - first_subject_box[0]) * (first_subject_box[3] - first_subject_box[1])
                                        union_area = box_area + first_area - inter_area
                                        iou = inter_area / union_area if union_area > 0 else 0
                                        score = iou
                                    else:
                                        # Fallback: use confidence
                                        score = float(box.conf[0])
                                    if score > best_score:
                                        best_score = score
                                        best_box = (x1, y1, x2, y2)
                        if best_box:
                            center_x = (best_box[0] + best_box[2]) // 2
                            center_y = (best_box[1] + best_box[3]) // 2
                        else:
                            center_x = width // 2
                            center_y = height // 2
                    elif data.get('subject_focus') == 'auto' and model is not None:
                        results = model(frame)
                        if len(results) > 0 and len(results[0].boxes) > 0:
                            box = results[0].boxes[0]
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                        else:
                            center_x = width // 2
                            center_y = height // 2
                    elif data.get('subject_focus') == 'manual':
                        center_x = int(width * data['focus_x'])
                        center_y = int(height * data['focus_y'])
                    else:
                        center_x = width // 2
                        center_y = height // 2
                    # Temporal smoothing
                    if prev_center_x is not None and prev_center_y is not None:
                        alpha = 0.7  # smoothing factor (0 = no smoothing, 1 = full smoothing)
                        center_x = int(alpha * prev_center_x + (1 - alpha) * center_x)
                        center_y = int(alpha * prev_center_y + (1 - alpha) * center_y)
                    prev_center_x, prev_center_y = center_x, center_y
                    
                    # Calculate crop dimensions
                    if width/height > target_ratio:
                        new_width = int(height * target_ratio)
                        crop_width = new_width
                        crop_height = height
                    else:
                        new_height = int(width / target_ratio)
                        crop_width = width
                        crop_height = new_height
                    
                    # Ensure crop dimensions are valid
                    crop_width = min(crop_width, width)
                    crop_height = min(crop_height, height)
                    
                    # Calculate crop coordinates with bounds checking
                    x1 = max(0, min(center_x - crop_width // 2, width - crop_width))
                    x2 = min(width, x1 + crop_width)
                    y1 = max(0, min(center_y - crop_height // 2, height - crop_height))
                    y2 = min(height, y1 + crop_height)
                    
                    # Verify crop dimensions
                    if x2 <= x1 or y2 <= y1:
                        raise ValueError(f"Invalid crop coordinates: ({x1}, {y1}, {x2}, {y2})")
                    
                    # Crop frame
                    cropped = frame[y1:y2, x1:x2]
                    if cropped.size == 0:
                        raise ValueError("Empty crop region")
                    
                    # Resize frame
                    resized = cv2.resize(cropped, (output_w, output_h), interpolation=cv2.INTER_LINEAR)
                    if resized.size == 0:
                        raise ValueError("Empty resized frame")
                    
                    # Write frame
                    out.write(resized)
                    
                except Exception as frame_error:
                    print(f"Error processing frame {frame_count}: {str(frame_error)}")
                    continue
            cap.release()
            out.release()
            
            # Add original audio to processed video using FFmpeg
            try:
                import subprocess
                final_output = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
                if os.path.exists(final_output):
                    os.remove(final_output)
                subprocess.run([
                    'ffmpeg', '-y',
                    '-i', temp_output,
                    '-i', input_path,
                    '-c:v', 'copy',
                    '-c:a', 'copy',
                    '-map', '0:v:0',
                    '-map', '1:a:0?',
                    final_output
                ], check=True)
                if os.path.exists(temp_output):
                    os.remove(temp_output)
                output_filename = os.path.basename(final_output)
            except (ImportError, subprocess.SubprocessError, FileNotFoundError) as e:
                print(f"FFmpeg audio muxing failed: {str(e)}")
                if os.path.exists(output_path):
                    os.remove(output_path)
                os.rename(temp_output, output_path)
            
            # Set progress to 100% when done
            processing_progress[filename] = 100
            
            # After successful processing, update project status to 'completed'
            processing_end = datetime.now()
            processed_size = os.path.getsize(os.path.join(app.config['UPLOAD_FOLDER'], output_filename))
            
            db.execute('''
                UPDATE projects 
                SET status = ?, processing_end = ?, processed_size = ?
                WHERE user_id = ? AND filename = ?
            ''', ('completed', processing_end, processed_size, current_user.id, output_filename))
            db.commit()
            db.close()
            
            return jsonify({
                'success': True,
                'processed_filename': output_filename,
                'message': 'Video processed successfully'
            })
        
    except Exception as e:
        # If there's an error, update project status to 'failed'
        try:
            processing_end = datetime.now()
            db.execute('''
                UPDATE projects 
                SET status = ?, processing_end = ?
                WHERE user_id = ? AND filename = ?
            ''', ('failed', processing_end, current_user.id, output_filename))
            db.commit()
        except:
            pass
        finally:
            db.close()
            
        print(f"Error during processing: {str(e)}")
        # Clean up any temporary files
        for file_path in [temp_output, output_path]:
            if 'file_path' in locals() and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as cleanup_error:
                    print(f"Error during cleanup: {str(cleanup_error)}")
        return jsonify({
            'success': False,
            'message': f'Error processing video: {str(e)}'
        })
    finally:
        # Clean up progress tracking
        if filename in processing_progress:
            del processing_progress[filename]

@app.route('/detect-objects', methods=['POST'])
def detect_objects():
    if not get_segmentation_model():
        return jsonify({'error': 'YOLO segmentation model not loaded'}), 500
        
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame provided'}), 400
    
    frame_file = request.files['frame']
    if frame_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Read the frame
        frame_data = frame_file.read()
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Failed to decode frame'}), 400
        
        # Run YOLO segmentation detection
        results = get_segmentation_model()(frame, conf=0.5)
        
        detected_objects = []
        for result in results:
            boxes = result.boxes
            masks = result.masks  # Segmentation masks
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                label = get_segmentation_model().names[class_id]
                mask_points = None
                if masks is not None:
                    # Get segmentation points for this object
                    mask_points = masks.xy[i].tolist()  # List of [x, y] points
                detected_objects.append({
                    'label': label,
                    'confidence': confidence,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'mask': mask_points
                })
        
        return jsonify({
            'success': True,
            'objects': detected_objects
        })
        
    except Exception as e:
        print(f"Error in object detection: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            return jsonify({
                'success': False,
                'message': 'File not found'
            }), 404
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error downloading file: {str(e)}'
        }), 500

@app.route('/reprocess/<int:project_id>', methods=['POST'])
@login_required
def reprocess_video(project_id):
    db = get_db()
    project = db.execute('SELECT * FROM projects WHERE id = ? AND user_id = ?', 
                        (project_id, current_user.id)).fetchone()
    
    if not project:
        return jsonify({'success': False, 'message': 'Project not found'})
    
    # Get the new processing parameters from the request
    data = request.json
    
    # Update the request with the original filename
    data['filename'] = project['original_filename']
    
    # Process the video with new parameters
    return process_video()

@app.route('/delete_project/<int:project_id>', methods=['POST'])
@login_required
def delete_project(project_id):
    db = get_db()
    try:
        # Get project details
        project = db.execute('SELECT * FROM projects WHERE id = ? AND user_id = ?', 
                           (project_id, current_user.id)).fetchone()
        
        if not project:
            return jsonify({'success': False, 'message': 'Project not found'})
        
        # Delete associated files
        for filename in [project['filename'], project['original_filename']]:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Delete thumbnail if exists
        if project['thumbnail_path']:
            thumb_path = os.path.join(app.config['UPLOAD_FOLDER'], 'thumbnails', project['thumbnail_path'])
            if os.path.exists(thumb_path):
                os.remove(thumb_path)
        
        # Delete database entry
        db.execute('DELETE FROM projects WHERE id = ?', (project_id,))
        db.commit()
        
        return jsonify({'success': True, 'message': 'Project deleted successfully'})
    except Exception as e:
        db.rollback()
        return jsonify({'success': False, 'message': str(e)})
    finally:
        db.close()

@app.route('/preview/<int:project_id>')
@login_required
def preview_video(project_id):
    db = get_db()
    project = db.execute('SELECT * FROM projects WHERE id = ? AND user_id = ?', 
                        (project_id, current_user.id)).fetchone()
    db.close()
    
    if not project:
        return jsonify({'success': False, 'message': 'Project not found'})
    
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], project['filename'])
    return send_file(video_path)

@app.route('/favorites')
@login_required
def favorites():
    db = get_db()
    favorite_videos = db.execute('''
        SELECT * FROM projects 
        WHERE user_id = ? AND is_favorite = 1 
        ORDER BY created_at DESC
    ''', (current_user.id,)).fetchall()
    db.close()
    return render_template('favorites.html', favorite_videos=favorite_videos)

@app.route('/toggle_favorite/<int:project_id>', methods=['POST'])
@login_required
def toggle_favorite(project_id):
    db = get_db()
    # Check if project belongs to user
    project = db.execute('SELECT * FROM projects WHERE id = ? AND user_id = ?', 
                        (project_id, current_user.id)).fetchone()
    
    if not project:
        db.close()
        return jsonify({'success': False, 'error': 'Project not found'}), 404
    
    # Toggle favorite status
    current_status = project['is_favorite']
    new_status = 0 if current_status else 1
    
    db.execute('UPDATE projects SET is_favorite = ? WHERE id = ?', 
               (new_status, project_id))
    db.commit()
    db.close()
    
    return jsonify({'success': True})

def send_password_reset_email(user_email, reset_url):
    msg = MIMEMultipart('alternative')
    msg['Subject'] = 'AVR - Password Reset Request'
    msg['From'] = app.config['MAIL_DEFAULT_SENDER']
    msg['To'] = user_email

    # Plain text version
    text = f'''
Hello,

You have requested to reset your password for your AVR Video Editor account.

To reset your password, please click on the following link:
{reset_url}

This link will expire in 1 hour for security reasons.

If you did not request this password reset, please ignore this email and your password will remain unchanged.

Best regards,
AVR Video Editor Team
'''

    # HTML version
    html = f'''
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .button {{ display: inline-block; padding: 12px 24px; background-color: #667eea; color: white; 
                  text-decoration: none; border-radius: 5px; margin: 20px 0; }}
        .button:hover {{ background-color: #5a6fd1; }}
        .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; font-size: 0.9em; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <h2>Password Reset Request</h2>
        <p>Hello,</p>
        <p>You have requested to reset your password for your AVR Video Editor account.</p>
        <p>To reset your password, please click the button below:</p>
        <a href="{reset_url}" class="button">Reset Password</a>
        <p>This link will expire in 1 hour for security reasons.</p>
        <p>If you did not request this password reset, please ignore this email and your password will remain unchanged.</p>
        <div class="footer">
            <p>Best regards,<br>AVR Video Editor Team</p>
        </div>
    </div>
</body>
</html>
'''

    # Attach both versions
    msg.attach(MIMEText(text, 'plain'))
    msg.attach(MIMEText(html, 'html'))

    try:
        with smtplib.SMTP(app.config['MAIL_SERVER'], app.config['MAIL_PORT']) as server:
            server.starttls()
            server.login(app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])
            server.send_message(msg)
            print(f"Password reset email sent successfully to {user_email}")
            return True
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False

def generate_reset_token():
    return secrets.token_urlsafe(32)

@app.route('/reset_password_request', methods=['GET', 'POST'])
def reset_password_request():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        email = request.form.get('email')
        if not email:
            flash('Please enter your email address.', 'error')
            return render_template('reset_password_request.html')
            
        db = get_db()
        user = db.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        
        if user:
            token = generate_reset_token()
            expires_at = datetime.now() + timedelta(hours=1)
            
            # Delete any existing unused tokens for this user
            db.execute('DELETE FROM password_reset_tokens WHERE user_id = ? AND used = 0', (user['id'],))
            
            # Create new token
            db.execute('''
                INSERT INTO password_reset_tokens (user_id, token, expires_at)
                VALUES (?, ?, ?)
            ''', (user['id'], token, expires_at))
            db.commit()
            
            reset_url = url_for('reset_password', token=token, _external=True)
            if send_password_reset_email(email, reset_url):
                flash('Password reset instructions have been sent to your email.', 'success')
            else:
                flash('Error sending password reset email. Please try again later.', 'error')
        else:
            # Don't reveal if email exists or not for security
            flash('Password reset instructions have been sent to your email if it exists in our system.', 'success')
            
        return redirect(url_for('login'))
        
    return render_template('reset_password_request.html')

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    db = get_db()
    token_data = db.execute('''
        SELECT * FROM password_reset_tokens 
        WHERE token = ? AND used = 0 AND expires_at > CURRENT_TIMESTAMP
    ''', (token,)).fetchone()
    
    if not token_data:
        flash('Invalid or expired reset link.', 'error')
        return redirect(url_for('login'))
        
    if request.method == 'POST':
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not password or not confirm_password:
            flash('Please fill in all fields.', 'error')
            return render_template('reset_password.html', token=token)
            
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('reset_password.html', token=token)
            
        if len(password) < 8:
            flash('Password must be at least 8 characters long.', 'error')
            return render_template('reset_password.html', token=token)
            
        # Update password and mark token as used
        password_hash = generate_password_hash(password)
        db.execute('UPDATE users SET password_hash = ? WHERE id = ?', 
                  (password_hash, token_data['user_id']))
        db.execute('UPDATE password_reset_tokens SET used = 1 WHERE id = ?', 
                  (token_data['id'],))
        db.commit()
        
        flash('Your password has been reset successfully. Please login with your new password.', 'success')
        return redirect(url_for('login'))
        
    return render_template('reset_password.html', token=token)

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if current_user.is_authenticated:
        db = get_db()
        user = db.execute('SELECT is_admin FROM users WHERE id = ?', (current_user.id,)).fetchone()
        db.close()
        
        if user and user['is_admin']:
            return redirect(url_for('admin_dashboard'))
        else:
            logout_user()
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        db = get_db()
        user = db.execute('SELECT * FROM users WHERE username = ? AND is_admin = 1', (username,)).fetchone()
        
        if user and check_password_hash(user['password_hash'], password):
            user_obj = User(user['id'], user['username'], user['email'], user['first_name'], user['last_name'])
            login_user(user_obj)
            
            # Update last login time
            db.execute('UPDATE users SET last_login = ? WHERE id = ?', 
                      (datetime.now(), user['id']))
            db.commit()
            
            flash('Welcome to the admin dashboard!', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid admin credentials', 'error')
        
        db.close()
    
    return render_template('admin_login.html')

# Update the admin_required decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            flash('Please log in as admin to access this page.', 'error')
            return redirect(url_for('admin_login'))
        
        db = get_db()
        user = db.execute('SELECT is_admin FROM users WHERE id = ?', (current_user.id,)).fetchone()
        db.close()
        
        if not user or not user['is_admin']:
            flash('You need admin privileges to access this page.', 'error')
            return redirect(url_for('admin_login'))
            
        return f(*args, **kwargs)
    return decorated_function

@app.route('/admin')
@admin_required
def admin_dashboard():
    db = get_db()
    
    # Get total users count
    total_users = db.execute('SELECT COUNT(*) as count FROM users WHERE is_admin = 0').fetchone()['count']
    
    # Get total projects count
    total_projects = db.execute('SELECT COUNT(*) as count FROM projects').fetchone()['count']
    
    # Get total storage used
    storage_stats = db.execute('''
        SELECT SUM(original_size) + SUM(processed_size) as total_storage
        FROM projects
        WHERE original_size IS NOT NULL
    ''').fetchone()
    total_storage = humanize.naturalsize(storage_stats['total_storage']) if storage_stats['total_storage'] else '0 B'
    
    # Get active users (users who logged in within the last 24 hours)
    active_users = db.execute('''
        SELECT COUNT(*) as count FROM users 
        WHERE last_login >= datetime('now', '-1 day')
        AND is_admin = 0
    ''').fetchone()['count']
    
    # Get recent users with more details
    recent_users = db.execute('''
        SELECT u.id, u.username, u.email, u.is_active, u.last_login,
               COUNT(p.id) as project_count,
               SUM(p.original_size) + SUM(p.processed_size) as storage_used
        FROM users u
        LEFT JOIN projects p ON u.id = p.user_id
        WHERE u.is_admin = 0
        GROUP BY u.id
        ORDER BY u.created_at DESC
        LIMIT 5
    ''').fetchall()
    
    # Get recent activities
    recent_activities = db.execute('''
        SELECT 'project' as type,
               u.username,
               p.original_filename,
               p.created_at as timestamp,
               p.status
        FROM projects p
        JOIN users u ON p.user_id = u.id
        WHERE u.is_admin = 0
        ORDER BY p.created_at DESC
        LIMIT 5
    ''').fetchall()
    
    # Format activities for display
    formatted_activities = []
    for activity in recent_activities:
        formatted_activities.append({
            'icon': 'fa-video',
            'color': 'blue',
            'message': f"{activity['username']} processed {activity['original_filename']}",
            'timestamp': activity['timestamp']
        })
    
    # System alerts
    system_alerts = [
        {
            'level': 'warning',
            'message': 'Storage usage above 75%',
            'timestamp': 'Just now'
        },
        {
            'level': 'error',
            'message': 'Failed login attempts detected',
            'timestamp': '5 minutes ago'
        }
    ]
    
    # System stats
    system_stats = {
        'storage': {
            'total': '1 TB',
            'used': '750 GB',
            'free': '250 GB',
            'percent': 75
        },
        'memory': {
            'total': '16 GB',
            'used': '9.6 GB',
            'free': '6.4 GB',
            'percent': 60
        },
        'cpu': {
            'usage': 45,
            'temperature': 65
        }
    }
    
    return render_template('admin_dashboard.html',
                         total_users=total_users,
                         total_projects=total_projects,
                         total_storage=total_storage,
                         active_users=active_users,
                         recent_users=recent_users,
                         recent_activities=formatted_activities,
                         system_alerts=system_alerts,
                         system_stats=system_stats)

@app.route('/admin/users')
@admin_required
def admin_users():
    db = get_db()
    users = db.execute('''
        SELECT u.*, 
               COUNT(p.id) as project_count,
               SUM(p.original_size) + SUM(p.processed_size) as total_storage
        FROM users u
        LEFT JOIN projects p ON u.id = p.user_id
        GROUP BY u.id
        ORDER BY u.created_at DESC
    ''').fetchall()
    
    return render_template('admin_users.html', users=users)

@app.route('/admin/users/<int:user_id>/toggle', methods=['POST'])
@admin_required
def toggle_user(user_id):
    db = get_db()
    user = db.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    
    if user:
        new_status = not user['is_active']
        db.execute('UPDATE users SET is_active = ? WHERE id = ?', (new_status, user_id))
        db.commit()
        flash(f"User {'activated' if new_status else 'deactivated'} successfully.", 'success')
    else:
        flash('User not found.', 'error')
    
    return redirect(url_for('admin_users'))

@app.route('/admin/projects')
@admin_required
def admin_projects():
    db = get_db()
    projects = db.execute('''
        SELECT p.*, u.username
        FROM projects p
        JOIN users u ON p.user_id = u.id
        ORDER BY p.created_at DESC
    ''').fetchall()
    
    return render_template('admin_projects.html', projects=projects)

@app.route('/admin/system')
@admin_required
def admin_system():
    # In a production environment, you would use proper monitoring tools
    # This is a simplified example
    system_stats = {
        'storage': {
            'total': '1 TB',
            'used': '750 GB',
            'free': '250 GB'
        },
        'memory': {
            'total': '16 GB',
            'used': '9.6 GB',
            'free': '6.4 GB'
        },
        'cpu': {
            'usage': '45%',
            'temperature': '65°C'
        }
    }
    
    return render_template('admin_system.html', stats=system_stats)

@app.route('/admin/logs')
@admin_required
def admin_logs():
    # In a production environment, you would implement proper log viewing
    # This is a simplified example
    logs = [
        {'timestamp': '2024-03-20 10:30:00', 'level': 'INFO', 'message': 'System started'},
        {'timestamp': '2024-03-20 10:35:00', 'level': 'WARNING', 'message': 'High CPU usage detected'},
        {'timestamp': '2024-03-20 10:40:00', 'level': 'ERROR', 'message': 'Failed login attempt'}
    ]
    
    return render_template('admin_logs.html', logs=logs)

@app.route('/admin/settings')
@admin_required
def admin_settings():
    return render_template('admin_settings.html')

@app.route('/admin/activity')
@admin_required
def admin_activity():
    db = get_db()
    activities = db.execute('''
        SELECT 'project' as type,
               u.username,
               p.original_filename,
               p.created_at as timestamp,
               p.status
        FROM projects p
        JOIN users u ON p.user_id = u.id
        ORDER BY p.created_at DESC
        LIMIT 20
    ''').fetchall()
    
    return render_template('admin_activity.html', activities=activities)

@app.route('/admin/alerts')
@admin_required
def admin_alerts():
    # Example alerts - in a production environment, these would come from your monitoring system
    alerts = [
        {
            'level': 'warning',
            'message': 'Storage usage above 75%',
            'timestamp': datetime.now() - timedelta(minutes=5)
        },
        {
            'level': 'error',
            'message': 'Failed login attempts detected',
            'timestamp': datetime.now() - timedelta(minutes=10)
        },
        {
            'level': 'info',
            'message': 'System backup completed',
            'timestamp': datetime.now() - timedelta(hours=1)
        }
    ]
    return render_template('admin_alerts.html', alerts=alerts)

# Add datetime filter
@app.template_filter('datetime')
def format_datetime(value):
    if isinstance(value, str):
        try:
            value = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            return value
    
    now = datetime.now()
    diff = now - value
    
    if diff.days == 0:
        if diff.seconds < 60:
            return 'just now'
        elif diff.seconds < 3600:
            minutes = diff.seconds // 60
            return f'{minutes}m ago'
        else:
            hours = diff.seconds // 3600
            return f'{hours}h ago'
    elif diff.days == 1:
        return 'yesterday'
    elif diff.days < 7:
        return f'{diff.days}d ago'
    else:
        return value.strftime('%Y-%m-%d %H:%M')

if __name__ == '__main__':
    if not os.path.exists('instance'):
        os.makedirs('instance')
    if not os.path.exists('instance/avr.db'):
        init_db()
    app.run(debug=True)