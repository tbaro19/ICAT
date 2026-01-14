"""
Setup script to ensure local pyribs is used
Add this to the beginning of scripts that use pyribs
"""
import sys
import os

# Add local pyribs to Python path
PYRIBS_PATH = '/root/ICAT/pyribs'
if os.path.exists(PYRIBS_PATH) and PYRIBS_PATH not in sys.path:
    sys.path.insert(0, PYRIBS_PATH)
    print(f"Using local pyribs from: {PYRIBS_PATH}")

# Verify pyribs location
try:
    import ribs
    print(f"pyribs loaded from: {ribs.__file__}")
except ImportError as e:
    print(f"Warning: Could not import pyribs: {e}")
