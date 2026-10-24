"""
WSGI entry point for Crop Advisory Agent.
Used by Gunicorn in production (e.g. Render, Railway, Heroku).
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.app import app

if __name__ == "__main__":
    app.run()
