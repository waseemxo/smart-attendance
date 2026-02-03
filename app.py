from flask import Flask, render_template, request, jsonify, Response, send_file
from models import db, Student, FaceEncoding, Timetable, Attendance, PendingConfirmation, Settings
from face_utils import (
    encode_face_from_base64, recognize_face, load_known_faces, 
    add_face_encoding, image_to_base64, draw_face_box, 
    cleanup_old_encodings, get_confidence_thresholds
)
from datetime import datetime, date, time, timedelta
import cv2
import numpy as np
import pandas as pd
import json
import os
import base64

app = Flask(__name__)

# Database configuration - use environment variable for production
import os
database_url = os.environ.get('DATABASE_URL', 'sqlite:///attendance.db')
app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')

db.init_app(app)

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
    now = datetime.now()
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


# ==================== ROUTES ====================

@app.route('/')
def dashboard():
    """Main dashboard"""
    today = date.today()
    
    # Stats
    total_students = Student.query.count()
    today_attendance = Attendance.query.filter_by(date=today).count()
    pending_count = PendingConfirmation.query.count()
    
    # Current class
    current_class = get_current_class()
    
    # Recent attendance
    recent = Attendance.query.filter_by(date=today).order_by(Attendance.time_marked.desc()).limit(10).all()
    
    return render_template('dashboard.html', 
                         total_students=total_students,
                         today_attendance=today_attendance,
                         pending_count=pending_count,
                         current_class=current_class,
                         recent_attendance=recent)


@app.route('/students')
def students():
    """List all students"""
    all_students = Student.query.all()
    return render_template('students.html', students=all_students)


@app.route('/students/register', methods=['GET', 'POST'])
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
    
    return render_template('register_student.html')


@app.route('/students/<int:id>/delete', methods=['POST'])
def delete_student(id):
    """Delete a student"""
    student = Student.query.get_or_404(id)
    db.session.delete(student)
    db.session.commit()
    refresh_known_faces()
    return jsonify({'success': True})


@app.route('/timetable')
def timetable():
    """Manage timetable"""
    entries = Timetable.query.order_by(Timetable.day_of_week, Timetable.start_time).all()
    classes = db.session.query(Student.class_name).distinct().all()
    classes = [c[0] for c in classes]
    return render_template('timetable.html', entries=entries, classes=classes)


@app.route('/timetable/add', methods=['POST'])
def add_timetable():
    """Add timetable entry"""
    data = request.get_json()
    
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
def delete_timetable(id):
    """Delete timetable entry"""
    entry = Timetable.query.get_or_404(id)
    db.session.delete(entry)
    db.session.commit()
    return jsonify({'success': True})


@app.route('/attendance')
def attendance():
    """Live attendance page"""
    current_class = get_current_class()
    pending = PendingConfirmation.query.all()
    return render_template('attendance.html', 
                         current_class=current_class,
                         pending_confirmations=pending)


@app.route('/attendance/process', methods=['POST'])
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
        date=date.today(),
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
            date=date.today(),
            time_marked=now.time(),
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


@app.route('/attendance/confirm/<int:id>', methods=['POST'])
def confirm_attendance(id):
    """Confirm a pending attendance"""
    data = request.get_json()
    pending = PendingConfirmation.query.get_or_404(id)
    
    confirmed = data.get('confirmed', False)
    correct_student_id = data.get('correct_student_id')
    
    if confirmed:
        student_id = correct_student_id or pending.student_id
        
        # Check if already marked
        existing = Attendance.query.filter_by(
            student_id=student_id,
            date=date.today(),
            subject=pending.subject
        ).first()
        
        if not existing:
            # Mark attendance
            attendance_record = Attendance(
                student_id=student_id,
                date=date.today(),
                time_marked=datetime.now().time(),
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
def get_pending():
    """Get all pending confirmations"""
    pending = PendingConfirmation.query.all()
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
def reports():
    """View and export reports"""
    # Get unique dates with attendance
    dates = db.session.query(Attendance.date).distinct().order_by(Attendance.date.desc()).all()
    dates = [d[0] for d in dates]
    
    # Get classes
    classes = db.session.query(Student.class_name).distinct().all()
    classes = [c[0] for c in classes]
    
    return render_template('reports.html', dates=dates, classes=classes)


@app.route('/reports/data')
def get_report_data():
    """Get attendance data for reporting"""
    report_date = request.args.get('date', date.today().isoformat())
    class_name = request.args.get('class', '')
    
    if isinstance(report_date, str):
        report_date = datetime.strptime(report_date, '%Y-%m-%d').date()
    
    query = db.session.query(Attendance, Student).join(Student)
    query = query.filter(Attendance.date == report_date)
    
    if class_name:
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
def export_report():
    """Export attendance to Excel"""
    report_date = request.args.get('date', date.today().isoformat())
    class_name = request.args.get('class', '')
    
    if isinstance(report_date, str):
        report_date = datetime.strptime(report_date, '%Y-%m-%d').date()
    
    query = db.session.query(Attendance, Student).join(Student)
    query = query.filter(Attendance.date == report_date)
    
    if class_name:
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
def settings():
    """Settings page"""
    current_settings = {
        'high_confidence_threshold': float(Settings.get('high_confidence_threshold', 0.6)),
        'low_confidence_threshold': float(Settings.get('low_confidence_threshold', 0.5)),
        'max_encodings_per_student': int(Settings.get('max_encodings_per_student', 10)),
        'adaptive_learning': Settings.get('adaptive_learning', 'true') == 'true'
    }
    return render_template('settings.html', settings=current_settings)


@app.route('/settings/update', methods=['POST'])
def update_settings():
    """Update settings"""
    data = request.get_json()
    
    for key, value in data.items():
        if isinstance(value, bool):
            value = 'true' if value else 'false'
        Settings.set(key, str(value))
    
    return jsonify({'success': True})


@app.route('/api/students')
def api_students():
    """Get all students (API)"""
    students = Student.query.all()
    return jsonify([{
        'id': s.id,
        'name': s.name,
        'roll_number': s.roll_number,
        'class_name': s.class_name
    } for s in students])


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
