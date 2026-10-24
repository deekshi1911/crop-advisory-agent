"""
Crop Advisory Agent web application package.
"""
import os
import sys

# Ensure project root is on sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app.app import app

__all__ = ["app"]
