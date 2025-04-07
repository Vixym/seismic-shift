#!/usr/bin/env python3
"""
Simple test script for the seismic_cpp Python bindings.
"""

import sys
import os

# Add the build directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build/pylib')))

try:
    # Try to import the module
    from seismic_cpp import get_seismic_string, SeismicIndex, SeismicIndexRaw
    print(f"Successfully imported seismic_cpp module")
    print(f"Seismic string: {get_seismic_string()}")
    
    # Test creating a simple index
    print("\nCreating a simple index...")
    
    # Print module information
    print("\nModule information:")
    print(f"Module location: {os.path.dirname(os.path.abspath(SeismicIndex.__module__))}")
    
    print("\nTest completed successfully!")
    
except ImportError as e:
    print(f"Error importing seismic_cpp: {e}")
    print(f"Python path: {sys.path}")
    
    # Check if the .so file exists
    build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../build/pylib'))
    so_files = [f for f in os.listdir(build_dir) if f.endswith('.so')]
    print(f"Available .so files in {build_dir}: {so_files}")
    
    sys.exit(1)
