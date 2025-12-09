"""
Pytest configuration and shared fixtures for view selection tests.
"""

import sys
import os

# Add parent directory to path so we can import view_selection
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
