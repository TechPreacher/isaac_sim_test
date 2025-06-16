"""
Helper script to fix the NumPy compatibility issue with Isaac Sim.
Run this script to install a compatible version of NumPy.
"""
import sys
import subprocess
import os

def fix_numpy_compatibility():
    print("Checking current NumPy installation...")
    try:
        import numpy
        current_version = numpy.__version__
        print(f"Current NumPy version: {current_version}")
    except ImportError:
        current_version = "Not installed"
        print("NumPy is not currently installed")
    
    # The error message mentioned "Expected 96 from C header, got 88 from PyObject"
    # This suggests we need an older version of NumPy that matches the C API expectations
    # Let's try a commonly compatible version
    compatible_version = "1.24.4"  # This is often a stable version for many libraries
    
    print(f"\nAttempting to install NumPy {compatible_version} for compatibility with Isaac Sim...")
    
    # Determine the pip command (might be pip3 on some systems)
    pip_cmd = [sys.executable, "-m", "pip"]
    
    # Install the specific version
    cmd = pip_cmd + ["install", f"numpy=={compatible_version}", "--force-reinstall"]
    print(f"Running: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\nSuccessfully installed NumPy {compatible_version}")
        print("\nPlease restart your script or Python environment to use the new NumPy version.")
    except subprocess.CalledProcessError as e:
        print(f"\nFailed to install NumPy {compatible_version}.")
        print(f"Error: {e}")
        print("\nAlternative options:")
        print("1. Try a different NumPy version (1.23.x or 1.22.x might be compatible)")
        print("2. Create a separate virtual environment for Isaac Sim")
        print("3. Contact NVIDIA support for specific version requirements")

if __name__ == "__main__":
    fix_numpy_compatibility()
