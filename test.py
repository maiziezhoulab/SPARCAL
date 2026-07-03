#!/usr/bin/env python3
"""
Test script to check Python version and package versions
"""

import sys
import importlib

def get_package_version(package_name):
    """Get the version of a package"""
    try:
        package = importlib.import_module(package_name)
        if hasattr(package, '__version__'):
            return package.__version__
        elif hasattr(package, 'version'):
            return package.version
        else:
            return "Version not found"
    except ImportError:
        return "Not installed"

def main():
    print("=" * 50)
    print("PYTHON AND PACKAGE VERSION CHECK")
    print("=" * 50)
    
    # Print Python version
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print("-" * 50)
    
    # Check and print package versions
    packages = ['numpy', 'sklearn']
    
    for package in packages:
        version = get_package_version(package)
        print(f"{package:10} : {version}")
    
    print("-" * 50)
    
    # Additional check for sklearn submodules
    try:
        import sklearn
        print("Sklearn available modules:")
        print(f"  - Linear Regression: {'Available' if hasattr(sklearn, 'linear_model') else 'Not available'}")
        print(f"  - Preprocessing: {'Available' if hasattr(sklearn, 'preprocessing') else 'Not available'}")
    except ImportError:
        pass
    
    print("=" * 50)

if __name__ == "__main__":
    main()