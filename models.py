from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, date
import json

db = SQLAlchemy()

class Student(db.Model):
    __tablename__ = 'students'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    roll_number = db.Column(db.String(50), unique=True, nullable=False)
    class_name = db.Column(db.String(50), nullable=False)
    department = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    face_encodings = db.relationship('FaceEncoding', backref='student', lazy=True, cascade='all, delete-orphan')
    attendances = db.relationship('Attendance', backref='student', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Student {self.name} - {self.roll_number}>'


class FaceEncoding(db.Model):
    """Store multiple face encodings per student for better recognition"""
    __tablename__ = 'face_encodings'
    
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=False)
    encoding = db.Column(db.Text, nullable=False)  # JSON-encoded numpy array
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    source = db.Column(db.String(50), default='registration')  # 'registration' or 'adaptive'
    
    def set_encoding(self, encoding_array):
        """Convert numpy array to JSON string"""
        self.encoding = json.dumps(encoding_array.tolist())
    
    def get_encoding(self):
        """Convert JSON string back to list"""
        return json.loads(self.encoding)
    
    def __repr__(self):
        return f'<FaceEncoding for Student {self.student_id}>'


class Timetable(db.Model):
    __tablename__ = 'timetables'
    
    id = db.Column(db.Integer, primary_key=True)
    class_name = db.Column(db.String(50), nullable=False)
    day_of_week = db.Column(db.Integer, nullable=False)  # 0=Monday, 6=Sunday
    start_time = db.Column(db.Time, nullable=False)
    end_time = db.Column(db.Time, nullable=False)
    subject = db.Column(db.String(100), nullable=False)
    
    def __repr__(self):
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        return f'<Timetable {self.class_name} - {days[self.day_of_week]} {self.subject}>'


class Attendance(db.Model):
    __tablename__ = 'attendances'
    
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=False)
    date = db.Column(db.Date, nullable=False, default=date.today)
    time_marked = db.Column(db.Time, nullable=False)
    subject = db.Column(db.String(100), nullable=False)
    status = db.Column(db.String(20), default='present')  # present, absent, late
    confidence = db.Column(db.Float, default=1.0)  # Recognition confidence
    confirmed = db.Column(db.Boolean, default=True)  # Manual confirmation status
    
    # Unique constraint: one attendance per student per subject per day
    __table_args__ = (
        db.UniqueConstraint('student_id', 'date', 'subject', name='unique_attendance'),
    )
    
    def __repr__(self):
        return f'<Attendance {self.student_id} - {self.date} - {self.subject}>'


class PendingConfirmation(db.Model):
    """Store low-confidence recognitions pending manual confirmation"""
    __tablename__ = 'pending_confirmations'
    
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=False)
    face_encoding = db.Column(db.Text, nullable=False)  # Captured face encoding
    face_image = db.Column(db.Text, nullable=False)  # Base64 encoded image
    confidence = db.Column(db.Float, nullable=False)
    subject = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    student = db.relationship('Student', backref='pending_confirmations')
    
    def set_encoding(self, encoding_array):
        self.face_encoding = json.dumps(encoding_array.tolist())
    
    def get_encoding(self):
        return json.loads(self.face_encoding)
    
    def __repr__(self):
        return f'<PendingConfirmation Student {self.student_id} - {self.confidence:.2f}>'


class Settings(db.Model):
    """Store system settings"""
    __tablename__ = 'settings'
    
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(50), unique=True, nullable=False)
    value = db.Column(db.String(200), nullable=False)
    
    @staticmethod
    def get(key, default=None):
        setting = Settings.query.filter_by(key=key).first()
        return setting.value if setting else default
    
    @staticmethod
    def set(key, value):
        try:
            setting = Settings.query.filter_by(key=key).first()
            if setting:
                setting.value = str(value)
            else:
                setting = Settings(key=key, value=str(value))
                db.session.add(setting)
            db.session.commit()
        except Exception:
            # Handle race condition - another worker may have inserted
            db.session.rollback()
            setting = Settings.query.filter_by(key=key).first()
            if setting:
                setting.value = str(value)
                db.session.commit()
