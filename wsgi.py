"""
WSGI entry point for production deployment
Used by Gunicorn, uWSGI, or other WSGI servers
"""
import os
from app import app, db

# Initialize database tables on startup
with app.app_context():
    db.create_all()

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
