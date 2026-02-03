# Smart Attendance Tracker

A face recognition-based attendance system for colleges built with Python Flask.

## Features

- **Face Recognition**: Automatic student identification using webcam
- **Confidence Levels**: 
  - 🟢 **High confidence**: Auto-marks attendance
  - 🟡 **Low confidence**: Requires manual confirmation
  - 🔴 **Unknown**: Face not recognized
- **Adaptive Learning**: System improves recognition over time by learning from confirmed faces
- **Timetable Management**: Manual class schedule configuration
- **Reports**: View and export attendance to Excel
- **Settings**: Configurable confidence thresholds

## Tech Stack

- **Backend**: Python + Flask
- **Face Recognition**: face_recognition (dlib-based)
- **Database**: SQLite
- **Frontend**: Bootstrap 5 + JavaScript
- **Webcam**: OpenCV

## Prerequisites

Before installation, you need:

1. **Python 3.8+**
2. **CMake** - Required for building dlib
   ```
   Download from: https://cmake.org/download/
   Or: pip install cmake
   ```
3. **Visual Studio Build Tools** (Windows only)
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Install "Desktop development with C++"

## Installation

1. **Navigate to project folder**:
   ```bash
   cd c:\Users\Waseem\Documents\minip
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   > ⚠️ **Note**: `face_recognition` installation may take several minutes as it compiles dlib.

4. **Run the application**:
   ```bash
   python app.py
   ```

5. **Open in browser**:
   ```
   http://localhost:5000
   ```

## Quick Start Guide

### 1. Add Class Timetable
- Go to **Timetable** → **Add Entry**
- Set class name (e.g., "CSE-A"), day, time, and subject
- The system uses this to determine which class to mark attendance for

### 2. Register Students
- Go to **Students** → **Register New Student**
- Fill in student details
- Click **Start Camera** and capture 3-5 face images from different angles
- Click **Register Student**

### 3. Take Attendance
- Go to **Attendance**
- Click **Start Attendance**
- Students walk in front of the camera
- System automatically marks attendance or flags for confirmation

### 4. View Reports
- Go to **Reports**
- Select date and class
- Click **Export Excel** to download

## Configuration

Access **Settings** to configure:

| Setting | Description | Default |
|---------|-------------|---------|
| Low Confidence Threshold | Below this = needs confirmation | 50% |
| High Confidence Threshold | Above this = unknown face | 60% |
| Adaptive Learning | Learn from confirmed faces | On |
| Max Encodings | Face images per student | 10 |

## How Face Recognition Works

1. **Registration**: Student's face is encoded into a 128-dimensional vector
2. **Recognition**: Incoming face is compared against all stored encodings
3. **Matching**: Distance-based similarity determines confidence level
4. **Adaptive**: Confirmed faces are saved to improve future recognition

## Project Structure

```
minip/
├── app.py                 # Main Flask application
├── models.py              # Database models
├── face_utils.py          # Face recognition utilities
├── requirements.txt       # Python dependencies
├── attendance.db          # SQLite database (auto-created)
├── static/
│   └── css/
│       └── style.css      # Custom styles
├── templates/
│   ├── base.html          # Base template
│   ├── dashboard.html     # Home page
│   ├── students.html      # Student list
│   ├── register_student.html
│   ├── timetable.html     # Schedule management
│   ├── attendance.html    # Live attendance
│   ├── reports.html       # View/export reports
│   └── settings.html      # Configuration
├── exports/               # Excel exports folder
└── known_faces/           # Face images folder
```

## Troubleshooting

### "No face detected"
- Ensure good lighting
- Face the camera directly
- Keep face within frame

### "dlib installation failed"
- Install CMake: `pip install cmake`
- Install Visual Studio Build Tools
- Try: `pip install dlib --verbose`

### Camera not working
- Check browser camera permissions
- Try a different browser (Chrome recommended)
- Ensure no other app is using the camera

## License

MIT License - Free for educational use
