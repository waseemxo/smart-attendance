FROM python:3.11-slim

# Install system dependencies for dlib/face_recognition
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for caching
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p exports known_faces instance

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Railway uses PORT env variable
CMD gunicorn wsgi:app --bind 0.0.0.0:${PORT:-5000} --timeout 120 --workers 2
