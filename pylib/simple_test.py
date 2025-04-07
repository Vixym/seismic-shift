#!/usr/bin/env python3
"""
Simple test script for the seismic_cpp Python bindings.
This script uses a more direct approach to load the extension module.
"""

import sys
import os
import importlib.util
import importlib.machinery

# Find the compiled module
build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../build/pylib'))
module_files = [f for f in os.listdir(build_dir) if f.endswith('.so')]

if not module_files:
    print(f"No .so files found in {build_dir}")
    sys.exit(1)

module_path = os.path.join(build_dir, module_files[0])
print(f"Found module: {module_path}")

try:
    # Try to load the module directly
    loader = importlib.machinery.ExtensionFileLoader('seismic_cpp', module_path)
    spec = importlib.util.spec_from_file_location('seismic_cpp', module_path, loader=loader)
    
    if spec is None:
        print(f"Failed to create spec for {module_path}")
        sys.exit(1)
        
    seismic_cpp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(seismic_cpp)
    
    # Test the module
    print(f"Successfully loaded seismic_cpp module")
    
    if hasattr(seismic_cpp, 'get_seismic_string'):
        print(f"Seismic string: {seismic_cpp.get_seismic_string()}")
    else:
        print("get_seismic_string function not found in module")
    
    # Print available attributes
    print("\nAvailable attributes in the module:")
    for attr in dir(seismic_cpp):
        if not attr.startswith('__'):
            print(f"  {attr}")
    
    print("\nTest completed successfully!")
    
except Exception as e:
    print(f"Error loading seismic_cpp: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
