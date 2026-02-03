import face_recognition
import numpy as np
import cv2
import base64
from models import db, Student, FaceEncoding, Settings

# Default confidence thresholds
DEFAULT_HIGH_CONFIDENCE = 0.6  # Distance below this = high confidence match
DEFAULT_LOW_CONFIDENCE = 0.5   # Distance below this = low confidence (needs confirmation)
# Distance above LOW_CONFIDENCE = unknown face


def get_confidence_thresholds():
    """Get confidence thresholds from settings"""
    high = float(Settings.get('high_confidence_threshold', DEFAULT_HIGH_CONFIDENCE))
    low = float(Settings.get('low_confidence_threshold', DEFAULT_LOW_CONFIDENCE))
    return high, low


def encode_face(image):
    """
    Extract face encoding from an image.
    
    Args:
        image: numpy array (BGR format from OpenCV)
    
    Returns:
        tuple: (face_encoding, face_location) or (None, None) if no face found
    """
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Find face locations
    face_locations = face_recognition.face_locations(rgb_image, model='hog')
    
    if not face_locations:
        return None, None
    
    # Get face encoding for the first face found
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    if not face_encodings:
        return None, None
    
    return face_encodings[0], face_locations[0]


def encode_face_from_base64(base64_string):
    """
    Extract face encoding from a base64 encoded image.
    
    Args:
        base64_string: Base64 encoded image string
    
    Returns:
        tuple: (face_encoding, image) or (None, None) if no face found
    """
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Decode base64 to image
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return None, None
    
    encoding, location = encode_face(image)
    return encoding, image


def recognize_face(face_encoding, known_encodings_dict):
    """
    Compare a face encoding against known encodings.
    
    Args:
        face_encoding: numpy array of the face to recognize
        known_encodings_dict: dict mapping student_id to list of encodings
    
    Returns:
        tuple: (student_id, confidence, match_type)
               match_type: 'high', 'low', or 'unknown'
    """
    if not known_encodings_dict:
        return None, 0, 'unknown'
    
    high_threshold, low_threshold = get_confidence_thresholds()
    
    best_match_id = None
    best_distance = float('inf')
    
    # Compare against all known faces
    for student_id, encodings in known_encodings_dict.items():
        for known_encoding in encodings:
            # Calculate face distance (lower = more similar)
            distance = face_recognition.face_distance([known_encoding], face_encoding)[0]
            
            if distance < best_distance:
                best_distance = distance
                best_match_id = student_id
    
    # Convert distance to confidence (0-1 scale, higher = better)
    confidence = 1 - best_distance
    
    # Determine match type based on thresholds
    if best_distance <= low_threshold:
        match_type = 'high'
    elif best_distance <= high_threshold:
        match_type = 'low'
    else:
        match_type = 'unknown'
        best_match_id = None
    
    return best_match_id, confidence, match_type


def load_known_faces():
    """
    Load all known face encodings from database.
    
    Returns:
        dict: mapping student_id to list of face encodings
    """
    known_encodings = {}
    
    face_records = FaceEncoding.query.all()
    
    for record in face_records:
        student_id = record.student_id
        encoding = np.array(record.get_encoding())
        
        if student_id not in known_encodings:
            known_encodings[student_id] = []
        
        known_encodings[student_id].append(encoding)
    
    return known_encodings


def add_face_encoding(student_id, encoding, source='registration'):
    """
    Add a new face encoding for a student.
    
    Args:
        student_id: ID of the student
        encoding: numpy array of face encoding
        source: 'registration' or 'adaptive'
    
    Returns:
        FaceEncoding object
    """
    face_record = FaceEncoding(student_id=student_id, source=source)
    face_record.set_encoding(encoding)
    db.session.add(face_record)
    db.session.commit()
    return face_record


def image_to_base64(image):
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')


def base64_to_image(base64_string):
    """Convert base64 string to OpenCV image"""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def draw_face_box(image, location, name, confidence, match_type):
    """
    Draw a box around the face with name and confidence.
    
    Args:
        image: OpenCV image
        location: (top, right, bottom, left) tuple
        name: Name to display
        confidence: Recognition confidence
        match_type: 'high', 'low', or 'unknown'
    
    Returns:
        Image with annotations
    """
    top, right, bottom, left = location
    
    # Color based on match type
    colors = {
        'high': (0, 255, 0),      # Green
        'low': (0, 255, 255),     # Yellow
        'unknown': (0, 0, 255)   # Red
    }
    color = colors.get(match_type, (255, 255, 255))
    
    # Draw rectangle
    cv2.rectangle(image, (left, top), (right, bottom), color, 2)
    
    # Draw label background
    label = f"{name} ({confidence:.0%})" if name else "Unknown"
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(image, (left, bottom), (left + label_size[0], bottom + 25), color, -1)
    
    # Draw label text
    cv2.putText(image, label, (left + 2, bottom + 18), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return image


def get_max_encodings_per_student():
    """Get maximum number of adaptive encodings to store per student"""
    return int(Settings.get('max_encodings_per_student', 10))


def cleanup_old_encodings(student_id):
    """Remove oldest adaptive encodings if exceeding limit"""
    max_encodings = get_max_encodings_per_student()
    
    # Count current encodings
    encodings = FaceEncoding.query.filter_by(student_id=student_id).order_by(FaceEncoding.created_at.desc()).all()
    
    if len(encodings) > max_encodings:
        # Keep registration encodings, remove oldest adaptive ones
        adaptive_encodings = [e for e in encodings if e.source == 'adaptive']
        
        # Remove excess adaptive encodings (oldest first)
        excess = len(encodings) - max_encodings
        for encoding in adaptive_encodings[-excess:]:
            db.session.delete(encoding)
        
        db.session.commit()
