import sys
import os

# Add the backend directory to sys.path so tests can import backend modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
