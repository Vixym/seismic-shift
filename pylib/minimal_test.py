#!/usr/bin/env python3
"""
Minimal test script for the seismic_cpp Python bindings.
"""

import sys
import os
import importlib
import glob

# Print Python version
print(f"Python version: {sys.version}")

# Add the build directory to the Python path
build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../build/pylib'))
sys.path.append(build_dir)

# List all .so files in the build directory
so_files = glob.glob(os.path.join(build_dir, '*.so'))
print(f"Found .so files: {so_files}")

# Try to import the module directly
try:
    # First try direct import
    import seismic_cpp
    print("Successfully imported seismic_cpp module")
    
    # Print available attributes
    print("\nAvailable attributes in the module:")
    for attr in dir(seismic_cpp):
        if not attr.startswith('__'):
            print(f"  {attr}")
    
    # Try to call a simple function
    if hasattr(seismic_cpp, 'get_seismic_string'):
        print(f"\nSeismic string: {seismic_cpp.get_seismic_string()}")
    
    print("\nTest completed successfully!")
    
except ImportError as e:
    print(f"Error importing seismic_cpp: {e}")
    
    # Try to manually load the .so file
    if so_files:
        print("\nTrying to manually load the module...")
        try:
            # Get the first .so file
            so_file = so_files[0]
            print(f"Loading: {so_file}")
            
            # Try to load it using importlib
            spec = importlib.util.find_spec('seismic_cpp')
            if spec:
                print(f"Found spec: {spec}")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                print("Successfully loaded module using importlib")
            else:
                print("Could not find spec for seismic_cpp")
        except Exception as e2:
            print(f"Error manually loading module: {e2}")
    
    sys.exit(1)
