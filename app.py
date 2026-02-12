from flask import Flask, render_template, request, jsonify, Response, send_file, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from models import db, Student, FaceEncoding, Timetable, Attendance, PendingConfirmation, Settings, User, AssignedClass
from face_utils import (
    encode_face_from_base64, recognize_face, load_known_faces, 
    add_face_encoding, image_to_base64, draw_face_box, 
    cleanup_old_encodings, get_confidence_thresholds, detect_all_faces
)
from functools import wraps
from datetime import datetime, date, time, timedelta
from zoneinfo import ZoneInfo
import cv2
import numpy as np
import pandas as pd
import json
import os
import base64

# Timezone configuration - set your local timezone
LOCAL_TIMEZONE = ZoneInfo(os.environ.get('TIMEZONE', 'Asia/Kolkata'))

def local_now():
    """Get current datetime in local timezone"""
    return datetime.now(LOCAL_TIMEZONE)

def local_today():
    """Get current date in local timezone"""
    return datetime.now(LOCAL_TIMEZONE).date()

app = Flask(__name__)

# Database configuration - use environment variable for production
import os
database_url = os.environ.get('DATABASE_URL', 'sqlite:///attendance.db')
# Railway uses postgres:// but SQLAlchemy requires postgresql://
if database_url.startswith('postgres://'):
    database_url = database_url.replace('postgres://', 'postgresql://', 1)
app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')

db.init_app(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'warning'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


def admin_required(f):
    """Decorator to require admin role"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'admin':
            flash('Admin access required.', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function


def get_user_classes():
    """Get classes accessible by current user"""
    if not current_user.is_authenticated:
        return []
    if current_user.role == 'admin':
        # Admin can access all classes
        classes = db.session.query(Student.class_name).distinct().all()
        return [c[0] for c in classes]
    return current_user.get_class_names()


# Create folders
os.makedirs('exports', exist_ok=True)
os.makedirs('known_faces', exist_ok=True)

# Global variables for camera
camera = None
known_faces_cache = {}
last_cache_update = None
CACHE_UPDATE_INTERVAL = 60  # seconds

# Track recently marked attendance to avoid duplicates
recently_marked = {}  # {student_id: timestamp}
MARK_COOLDOWN = 300  # 5 minutes cooldown


def get_camera():
    """Get or initialize camera"""
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera


def release_camera():
    """Release camera resource"""
    global camera
    if camera is not None:
        camera.release()
        camera = None


def get_known_faces():
    """Get known faces with caching"""
    global known_faces_cache, last_cache_update
    
    now = datetime.now()
    if last_cache_update is None or (now - last_cache_update).seconds > CACHE_UPDATE_INTERVAL:
        known_faces_cache = load_known_faces()
        last_cache_update = now
    
    return known_faces_cache


def refresh_known_faces():
    """Force refresh of known faces cache"""
    global known_faces_cache, last_cache_update
    known_faces_cache = load_known_faces()
    last_cache_update = datetime.now()


def get_current_class():
    """Get the current class based on timetable"""
    # Use local timezone for correct time matching
    now = datetime.now(LOCAL_TIMEZONE)
    current_time = now.time()
    current_day = now.weekday()
    
    timetable = Timetable.query.filter_by(day_of_week=current_day).all()
    
    for entry in timetable:
        if entry.start_time <= current_time <= entry.end_time:
            return entry
    
    return None


# Initialize database
with app.app_context():
    db.create_all()
    # Set default settings
    if Settings.get('high_confidence_threshold') is None:
        Settings.set('high_confidence_threshold', '0.6')
    if Settings.get('low_confidence_threshold') is None:
        Settings.set('low_confidence_threshold', '0.5')
    if Settings.get('max_encodings_per_student') is None:
        Settings.set('max_encodings_per_student', '10')
    if Settings.get('adaptive_learning', 'true') is None:
        Settings.set('adaptive_learning', 'true')
    
    # Create default admin user if no users exist
    if User.query.count() == 0:
        admin = User(
            username='admin',
            email='admin@college.edu',
            name='Administrator',
            role='admin'
        )
        admin.set_password('admin123')  # Change this in production!
        db.session.add(admin)
        try:
            db.session.commit()
            print("Default admin created - username: admin, password: admin123")
        except:
            db.session.rollback()


# ==================== AUTH ROUTES ====================

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = request.form.get('remember', False)
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            if not user.is_active:
                flash('Your account has been deactivated.', 'danger')
                return redirect(url_for('login'))
            
            login_user(user, remember=remember)
            flash(f'Welcome back, {user.name}!', 'success')
            
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'danger')
    
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    """Logout user"""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """User profile and password change"""
    if request.method == 'POST':
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        if not current_user.check_password(current_password):
            flash('Current password is incorrect.', 'danger')
        elif new_password != confirm_password:
            flash('New passwords do not match.', 'danger')
        elif len(new_password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
        else:
            current_user.set_password(new_password)
            db.session.commit()
            flash('Password updated successfully!', 'success')
        
        return redirect(url_for('profile'))
    
    return render_template('profile.html')


# ==================== MAIN ROUTES ====================

@app.route('/')
@login_required
def dashboard():
    """Main dashboard"""
    today = local_today()
    user_classes = get_user_classes()
    
    # Stats - filtered by user's classes
    if current_user.role == 'admin':
        total_students = Student.query.count()
        today_attendance = Attendance.query.filter_by(date=today).count()
        pending_count = PendingConfirmation.query.count()
    else:
        total_students = Student.query.filter(Student.class_name.in_(user_classes)).count() if user_classes else 0
        today_attendance = db.session.query(Attendance).join(Student).filter(
            Attendance.date == today,
            Student.class_name.in_(user_classes)
        ).count() if user_classes else 0
        pending_count = db.session.query(PendingConfirmation).join(Student).filter(
            Student.class_name.in_(user_classes)
        ).count() if user_classes else 0
    
    # Current class - check if user has access
    current_class = get_current_class()
    if current_class and not current_user.can_access_class(current_class.class_name):
        current_class = None
    
    # Recent attendance - filtered
    if current_user.role == 'admin':
        recent = Attendance.query.filter_by(date=today).order_by(Attendance.time_marked.desc()).limit(10).all()
    else:
        recent = db.session.query(Attendance).join(Student).filter(
            Attendance.date == today,
            Student.class_name.in_(user_classes)
        ).order_by(Attendance.time_marked.desc()).limit(10).all() if user_classes else []
    
    return render_template('dashboard.html', 
                         total_students=total_students,
                         today_attendance=today_attendance,
                         pending_count=pending_count,
                         current_class=current_class,
                         recent_attendance=recent,
                         user_classes=user_classes)


@app.route('/students')
@login_required
def students():
    """List all students"""
    user_classes = get_user_classes()
    if current_user.role == 'admin':
        all_students = Student.query.all()
    else:
        all_students = Student.query.filter(Student.class_name.in_(user_classes)).all() if user_classes else []
    return render_template('students.html', students=all_students, user_classes=user_classes)


@app.route('/students/register', methods=['GET', 'POST'])
@login_required
def register_student():
    """Register a new student with face capture"""
    if request.method == 'POST':
        data = request.get_json()
        
        # Check if roll number exists
        if Student.query.filter_by(roll_number=data['roll_number']).first():
            return jsonify({'success': False, 'error': 'Roll number already exists'})
        
        # Create student
        student = Student(
            name=data['name'],
            roll_number=data['roll_number'],
            class_name=data['class_name'],
            department=data['department']
        )
        db.session.add(student)
        db.session.commit()
        
        # Process face images
        face_count = 0
        for image_data in data.get('images', []):
            encoding, _ = encode_face_from_base64(image_data)
            if encoding is not None:
                add_face_encoding(student.id, encoding, source='registration')
                face_count += 1
        
        if face_count == 0:
            # No faces detected, delete student
            db.session.delete(student)
            db.session.commit()
            return jsonify({'success': False, 'error': 'No face detected in any image'})
        
        # Refresh cache
        refresh_known_faces()
        
        return jsonify({'success': True, 'message': f'Student registered with {face_count} face images'})
    
    return render_template('register_student.html', user_classes=get_user_classes())


@app.route('/students/<int:id>/delete', methods=['POST'])
@login_required
def delete_student(id):
    """Delete a student"""
    student = Student.query.get_or_404(id)
    
    # Check if user has access to this student's class
    if not current_user.can_access_class(student.class_name):
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    db.session.delete(student)
    db.session.commit()
    refresh_known_faces()
    return jsonify({'success': True})


@app.route('/timetable')
@login_required
def timetable():
    """Manage timetable"""
    user_classes = get_user_classes()
    if current_user.role == 'admin':
        entries = Timetable.query.order_by(Timetable.day_of_week, Timetable.start_time).all()
    else:
        entries = Timetable.query.filter(Timetable.class_name.in_(user_classes)).order_by(
            Timetable.day_of_week, Timetable.start_time
        ).all() if user_classes else []
    
    classes = db.session.query(Student.class_name).distinct().all()
    classes = [c[0] for c in classes]
    return render_template('timetable.html', entries=entries, classes=classes, user_classes=user_classes)


@app.route('/timetable/add', methods=['POST'])
@login_required
def add_timetable():
    """Add timetable entry"""
    data = request.get_json()
    
    # Check if user has access to this class
    if not current_user.can_access_class(data['class_name']):
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    entry = Timetable(
        class_name=data['class_name'],
        day_of_week=int(data['day_of_week']),
        start_time=datetime.strptime(data['start_time'], '%H:%M').time(),
        end_time=datetime.strptime(data['end_time'], '%H:%M').time(),
        subject=data['subject']
    )
    db.session.add(entry)
    db.session.commit()
    
    return jsonify({'success': True})


@app.route('/timetable/<int:id>/delete', methods=['POST'])
@login_required
def delete_timetable(id):
    """Delete timetable entry"""
    entry = Timetable.query.get_or_404(id)
    
    # Check if user has access to this class
    if not current_user.can_access_class(entry.class_name):
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    db.session.delete(entry)
    db.session.commit()
    return jsonify({'success': True})


@app.route('/attendance')
@login_required
def attendance():
    """Live attendance page"""
    current_class = get_current_class()
    user_classes = get_user_classes()
    
    # Check if user has access to current class
    if current_class and not current_user.can_access_class(current_class.class_name):
        current_class = None
    
    # Get pending confirmations for user's classes
    if current_user.role == 'admin':
        pending = PendingConfirmation.query.all()
    else:
        pending = db.session.query(PendingConfirmation).join(Student).filter(
            Student.class_name.in_(user_classes)
        ).all() if user_classes else []
    
    return render_template('attendance.html', 
                         current_class=current_class,
                         pending_confirmations=pending,
                         user_classes=user_classes)


@app.route('/attendance/process', methods=['POST'])
@login_required
def process_attendance():
    """Process a frame for attendance"""
    global recently_marked
    
    data = request.get_json()
    image_data = data.get('image')
    
    if not image_data:
        return jsonify({'success': False, 'error': 'No image provided'})
    
    # Get current class
    current_class = get_current_class()
    if not current_class:
        return jsonify({'success': False, 'error': 'No class scheduled', 'no_class': True})
    
    # Encode face
    encoding, image = encode_face_from_base64(image_data)
    
    if encoding is None:
        return jsonify({'success': False, 'error': 'No face detected'})
    
    # Recognize face
    known_faces = get_known_faces()
    student_id, confidence, match_type = recognize_face(encoding, known_faces)
    
    result = {
        'success': True,
        'match_type': match_type,
        'confidence': confidence
    }
    
    if match_type == 'unknown':
        result['message'] = 'Unknown face detected'
        return jsonify(result)
    
    student = Student.query.get(student_id)
    result['student'] = {
        'id': student.id,
        'name': student.name,
        'roll_number': student.roll_number,
        'class_name': student.class_name
    }
    
    # Check if student is in the current class
    if student.class_name != current_class.class_name:
        result['message'] = f'{student.name} is not in this class'
        result['wrong_class'] = True
        return jsonify(result)
    
    # Check cooldown
    now = datetime.now()
    if student_id in recently_marked:
        time_diff = (now - recently_marked[student_id]).seconds
        if time_diff < MARK_COOLDOWN:
            result['message'] = f'{student.name} - Already marked ({MARK_COOLDOWN - time_diff}s cooldown)'
            result['already_marked'] = True
            return jsonify(result)
    
    # Check if already marked today for this subject
    existing = Attendance.query.filter_by(
        student_id=student_id,
        date=local_today(),
        subject=current_class.subject
    ).first()
    
    if existing:
        result['message'] = f'{student.name} - Already marked for {current_class.subject}'
        result['already_marked'] = True
        return jsonify(result)
    
    if match_type == 'high':
        # High confidence - mark attendance directly
        attendance_record = Attendance(
            student_id=student_id,
            date=local_today(),
            time_marked=local_now().time(),
            subject=current_class.subject,
            confidence=confidence,
            confirmed=True
        )
        db.session.add(attendance_record)
        db.session.commit()
        
        recently_marked[student_id] = now
        
        # Adaptive learning - add this encoding if enabled
        if Settings.get('adaptive_learning', 'true') == 'true':
            add_face_encoding(student_id, encoding, source='adaptive')
            cleanup_old_encodings(student_id)
            refresh_known_faces()
        
        result['message'] = f'✓ {student.name} - Attendance marked!'
        result['marked'] = True
        
    else:
        # Low confidence - create pending confirmation
        pending = PendingConfirmation(
            student_id=student_id,
            subject=current_class.subject,
            confidence=confidence
        )
        pending.set_encoding(encoding)
        pending.face_image = image_data
        db.session.add(pending)
        db.session.commit()
        
        result['message'] = f'? {student.name} - Low confidence, needs confirmation'
        result['pending'] = True
        result['pending_id'] = pending.id
    
    return jsonify(result)


@app.route('/attendance/detect', methods=['POST'])
@login_required
def detect_faces():
    """Detect faces in image and return locations with names for live preview"""
    data = request.get_json()
    image_data = data.get('image')
    
    if not image_data:
        return jsonify({'success': False, 'faces': []})
    
    # Get current class for context
    current_class = get_current_class()
    
    # Detect and recognize all faces
    known_faces = get_known_faces()
    faces, _ = detect_all_faces(image_data, known_faces)
    
    # Format results for frontend
    face_results = []
    for face in faces:
        face_data = {
            'location': face['location'],
            'match_type': face['match_type'],
            'confidence': round(face['confidence'] * 100),
            'name': face['student_name'] or 'Unknown',
            'student_id': face['student_id']
        }
        
        # Add class info if recognized
        if face['student_id']:
            student = Student.query.get(face['student_id'])
            if student:
                face_data['roll_number'] = student.roll_number
                face_data['class_name'] = student.class_name
                # Check if student is in current class
                if current_class and student.class_name != current_class.class_name:
                    face_data['wrong_class'] = True
        
        face_results.append(face_data)
    
    return jsonify({
        'success': True,
        'faces': face_results,
        'current_class': current_class.class_name if current_class else None
    })


@app.route('/attendance/confirm/<int:id>', methods=['POST'])
@login_required
def confirm_attendance(id):
    """Confirm a pending attendance"""
    data = request.get_json()
    pending = PendingConfirmation.query.get_or_404(id)
    
    # Check access
    student = Student.query.get(pending.student_id)
    if not current_user.can_access_class(student.class_name):
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    confirmed = data.get('confirmed', False)
    correct_student_id = data.get('correct_student_id')
    
    if confirmed:
        student_id = correct_student_id or pending.student_id
        
        # Check if already marked
        existing = Attendance.query.filter_by(
            student_id=student_id,
            date=local_today(),
            subject=pending.subject
        ).first()
        
        if not existing:
            # Mark attendance
            attendance_record = Attendance(
                student_id=student_id,
                date=local_today(),
                time_marked=local_now().time(),
                subject=pending.subject,
                confidence=pending.confidence,
                confirmed=True
            )
            db.session.add(attendance_record)
            
            # Adaptive learning - add encoding to improve recognition
            if Settings.get('adaptive_learning', 'true') == 'true':
                encoding = np.array(pending.get_encoding())
                add_face_encoding(student_id, encoding, source='adaptive')
                cleanup_old_encodings(student_id)
                refresh_known_faces()
    
    # Delete pending record
    db.session.delete(pending)
    db.session.commit()
    
    return jsonify({'success': True})


@app.route('/attendance/pending')
@login_required
def get_pending():
    """Get all pending confirmations"""
    user_classes = get_user_classes()
    
    if current_user.role == 'admin':
        pending = PendingConfirmation.query.all()
    else:
        pending = db.session.query(PendingConfirmation).join(Student).filter(
            Student.class_name.in_(user_classes)
        ).all() if user_classes else []
    
    result = []
    
    for p in pending:
        student = Student.query.get(p.student_id)
        result.append({
            'id': p.id,
            'student_name': student.name,
            'student_roll': student.roll_number,
            'confidence': p.confidence,
            'subject': p.subject,
            'image': p.face_image,
            'created_at': p.created_at.strftime('%H:%M:%S')
        })
    
    return jsonify(result)


@app.route('/reports')
@login_required
def reports():
    """View and export reports"""
    user_classes = get_user_classes()
    
    # Get unique dates with attendance - filtered by user's classes
    if current_user.role == 'admin':
        dates = db.session.query(Attendance.date).distinct().order_by(Attendance.date.desc()).all()
    else:
        dates = db.session.query(Attendance.date).join(Student).filter(
            Student.class_name.in_(user_classes)
        ).distinct().order_by(Attendance.date.desc()).all() if user_classes else []
    dates = [d[0] for d in dates]
    
    # Get classes - filtered by user's access
    if current_user.role == 'admin':
        classes = db.session.query(Student.class_name).distinct().all()
        classes = [c[0] for c in classes]
    else:
        classes = user_classes
    
    return render_template('reports.html', dates=dates, classes=classes, user_classes=user_classes)


@app.route('/reports/data')
@login_required
def get_report_data():
    """Get attendance data for reporting"""
    report_date = request.args.get('date', local_today().isoformat())
    class_name = request.args.get('class', '')
    user_classes = get_user_classes()
    
    if isinstance(report_date, str):
        report_date = datetime.strptime(report_date, '%Y-%m-%d').date()
    
    query = db.session.query(Attendance, Student).join(Student)
    query = query.filter(Attendance.date == report_date)
    
    # Filter by user's classes
    if current_user.role != 'admin':
        query = query.filter(Student.class_name.in_(user_classes)) if user_classes else query.filter(False)
    
    if class_name:
        # Verify user has access to this class
        if not current_user.can_access_class(class_name):
            return jsonify([])
        query = query.filter(Student.class_name == class_name)
    
    records = query.all()
    
    result = []
    for attendance, student in records:
        result.append({
            'student_name': student.name,
            'roll_number': student.roll_number,
            'class_name': student.class_name,
            'department': student.department,
            'subject': attendance.subject,
            'time': attendance.time_marked.strftime('%H:%M:%S'),
            'confidence': f'{attendance.confidence:.0%}',
            'status': attendance.status
        })
    
    return jsonify(result)


@app.route('/reports/export')
@login_required
def export_report():
    """Export attendance to Excel"""
    report_date = request.args.get('date', local_today().isoformat())
    class_name = request.args.get('class', '')
    user_classes = get_user_classes()
    
    if isinstance(report_date, str):
        report_date = datetime.strptime(report_date, '%Y-%m-%d').date()
    
    query = db.session.query(Attendance, Student).join(Student)
    query = query.filter(Attendance.date == report_date)
    
    # Filter by user's classes
    if current_user.role != 'admin':
        query = query.filter(Student.class_name.in_(user_classes)) if user_classes else query.filter(False)
    
    if class_name:
        if not current_user.can_access_class(class_name):
            return "Access denied", 403
        query = query.filter(Student.class_name == class_name)
    
    records = query.all()
    
    data = []
    for attendance, student in records:
        data.append({
            'Roll Number': student.roll_number,
            'Student Name': student.name,
            'Class': student.class_name,
            'Department': student.department,
            'Subject': attendance.subject,
            'Time Marked': attendance.time_marked.strftime('%H:%M:%S'),
            'Confidence': f'{attendance.confidence:.0%}',
            'Status': attendance.status
        })
    
    df = pd.DataFrame(data)
    
    filename = f'attendance_{report_date}_{class_name or "all"}.xlsx'
    filepath = os.path.join('exports', filename)
    
    df.to_excel(filepath, index=False)
    
    return send_file(filepath, as_attachment=True, download_name=filename)


@app.route('/settings')
@login_required
@admin_required
def settings():
    """Settings page (Admin only)"""
    current_settings = {
        'high_confidence_threshold': float(Settings.get('high_confidence_threshold', 0.6)),
        'low_confidence_threshold': float(Settings.get('low_confidence_threshold', 0.5)),
        'max_encodings_per_student': int(Settings.get('max_encodings_per_student', 10)),
        'adaptive_learning': Settings.get('adaptive_learning', 'true') == 'true'
    }
    return render_template('settings.html', settings=current_settings)


@app.route('/settings/update', methods=['POST'])
@login_required
@admin_required
def update_settings():
    """Update settings (Admin only)"""
    data = request.get_json()
    
    for key, value in data.items():
        if isinstance(value, bool):
            value = 'true' if value else 'false'
        Settings.set(key, str(value))
    
    return jsonify({'success': True})


@app.route('/api/students')
@login_required
def api_students():
    """Get all students (API) - filtered by user's classes"""
    user_classes = get_user_classes()
    
    if current_user.role == 'admin':
        students = Student.query.all()
    else:
        students = Student.query.filter(Student.class_name.in_(user_classes)).all() if user_classes else []
    
    return jsonify([{
        'id': s.id,
        'name': s.name,
        'roll_number': s.roll_number,
        'class_name': s.class_name
    } for s in students])


# ==================== ADMIN ROUTES ====================

@app.route('/admin/staff')
@login_required
@admin_required
def manage_staff():
    """Staff management page (Admin only)"""
    staff_list = User.query.all()
    all_classes = db.session.query(Student.class_name).distinct().all()
    all_classes = [c[0] for c in all_classes]
    return render_template('staff.html', staff=staff_list, all_classes=all_classes)


@app.route('/admin/staff/add', methods=['POST'])
@login_required
@admin_required
def add_staff():
    """Add new staff member"""
    data = request.get_json()
    
    # Check if username or email already exists
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'success': False, 'error': 'Username already exists'})
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'success': False, 'error': 'Email already exists'})
    
    user = User(
        username=data['username'],
        email=data['email'],
        name=data['name'],
        role=data.get('role', 'staff')
    )
    user.set_password(data['password'])
    db.session.add(user)
    db.session.commit()
    
    # Assign classes
    for class_name in data.get('classes', []):
        assigned = AssignedClass(user_id=user.id, class_name=class_name)
        db.session.add(assigned)
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Staff member added successfully'})


@app.route('/admin/staff/<int:id>/update', methods=['POST'])
@login_required
@admin_required
def update_staff(id):
    """Update staff member"""
    user = User.query.get_or_404(id)
    data = request.get_json()
    
    # Update basic info
    if 'name' in data:
        user.name = data['name']
    if 'email' in data:
        if data['email'] != user.email and User.query.filter_by(email=data['email']).first():
            return jsonify({'success': False, 'error': 'Email already exists'})
        user.email = data['email']
    if 'role' in data:
        user.role = data['role']
    if 'is_active' in data:
        user.is_active = data['is_active']
    if 'password' in data and data['password']:
        user.set_password(data['password'])
    
    # Update classes
    if 'classes' in data:
        # Remove existing assignments
        AssignedClass.query.filter_by(user_id=user.id).delete()
        # Add new assignments
        for class_name in data['classes']:
            assigned = AssignedClass(user_id=user.id, class_name=class_name)
            db.session.add(assigned)
    
    db.session.commit()
    return jsonify({'success': True, 'message': 'Staff member updated successfully'})


@app.route('/admin/staff/<int:id>/delete', methods=['POST'])
@login_required
@admin_required
def delete_staff(id):
    """Delete staff member"""
    user = User.query.get_or_404(id)
    
    # Prevent deleting the last admin
    if user.role == 'admin':
        admin_count = User.query.filter_by(role='admin').count()
        if admin_count <= 1:
            return jsonify({'success': False, 'error': 'Cannot delete the last admin'})
    
    # Prevent deleting yourself
    if user.id == current_user.id:
        return jsonify({'success': False, 'error': 'Cannot delete yourself'})
    
    db.session.delete(user)
    db.session.commit()
    return jsonify({'success': True, 'message': 'Staff member deleted'})


def generate_ssl_cert():
    """Generate a self-signed SSL certificate for development"""
    from OpenSSL import crypto
    
    # Generate key
    key = crypto.PKey()
    key.generate_key(crypto.TYPE_RSA, 2048)
    
    # Generate certificate
    cert = crypto.X509()
    cert.get_subject().CN = "localhost"
    cert.set_serial_number(1000)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(365 * 24 * 60 * 60)  # Valid for 1 year
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(key)
    cert.sign(key, 'sha256')
    
    # Save certificate and key
    with open("cert.pem", "wb") as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
    with open("key.pem", "wb") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, key))
    
    return "cert.pem", "key.pem"


if __name__ == '__main__':
    import sys
    
    # Check if SSL certificates exist, if not generate them
    if not (os.path.exists('cert.pem') and os.path.exists('key.pem')):
        print("Generating SSL certificate for HTTPS...")
        generate_ssl_cert()
        print("SSL certificate generated!")
    
    print("\n" + "="*60)
    print("Smart Attendance System")
    print("="*60)
    print("\nAccess the app at:")
    print("  Local:   https://localhost:5000")
    print("  Network: https://192.168.0.172:5000")
    print("\n⚠️  Your browser will show a security warning.")
    print("   Click 'Advanced' -> 'Proceed anyway' to continue.")
    print("="*60 + "\n")
    
    # Run with HTTPS
    app.run(debug=True, host='0.0.0.0', port=5000, ssl_context=('cert.pem', 'key.pem'))
