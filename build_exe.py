#!/usr/bin/env python3
"""
Build script for creating standalone MIB Viewer executable

Handles both PyQt5 and PyQtGraph versions with fallback strategies for uv issues on Windows.
"""

import os
import subprocess
import sys
import platform

def build_with_uv(entry_point):
    """Attempt to build with uv run pyinstaller"""
    print("Attempting build with uv run pyinstaller...")
    
    # Build command for different platforms
    if platform.system() == "Windows":
        cmd = [
            "uv", "run", "pyinstaller",
            "--onefile",
            "--windowed", 
            "--name=MibViewer",
            "--icon=app.ico",  # Add icon if available
            "--add-data", "src;src",  # Include source for imports
            entry_point
        ]
    else:
        cmd = [
            "uv", "run", "pyinstaller",
            "--onefile",
            "--windowed",
            "--name=MibViewer",
            "--noupx",  # Disable UPX compression
            "--copy-metadata", "pyqtgraph",
            "--copy-metadata", "matplotlib", 
            "--add-data", "src:src",
            entry_point
        ]
    
    # Remove icon option if file doesn't exist
    if "--icon=app.ico" in cmd and not os.path.exists("app.ico"):
        cmd.remove("--icon=app.ico")
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"ERROR: uv build failed: {e}")  # Remove Unicode for Windows compatibility
        return False

def build_with_spec_file():
    """Build using PyInstaller spec file for better control"""
    print("Using PyInstaller spec file for precise control...")
    
    spec_file = "MibViewer.spec"
    if not os.path.exists(spec_file):
        print(f"ERROR: {spec_file} not found")
        return False
    
    cmd = ["pyinstaller", spec_file]
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"ERROR: Spec file build failed: {e}")
        return False

def build_with_direct_pyinstaller(entry_point):
    """Fallback: build with direct pyinstaller"""
    print("Fallback: Using direct pyinstaller...")
    
    cmd = [
        "pyinstaller",
        "--onefile",
        "--windowed",
        "--name=MibViewer",
        "--add-data", "src;src" if platform.system() == "Windows" else "src:src",
        "--hidden-import=mib_viewer.gui.mib_viewer_pyqtgraph",
        "--hidden-import=mib_viewer.gui.mib_viewer_qt",
        "--hidden-import=PyQt5",
        "--hidden-import=pyqtgraph",
        "--hidden-import=numpy",
        "--hidden-import=matplotlib",
        "--hidden-import=h5py"
    ]
    
    # Add platform-specific options
    if platform.system() == "Windows":
        if os.path.exists("app.ico"):
            cmd.extend(["--icon=app.ico"])
    else:
        cmd.extend(["--noupx"])
    
    cmd.append(entry_point)
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"ERROR: Direct pyinstaller failed: {e}")  # Remove Unicode for Windows compatibility
        return False

def main():
    """Build the executable using PyInstaller with fallbacks"""
    
    print("Building MIB Viewer standalone executable...")
    print(f"Platform: {platform.system()}")
    
    # Use standalone entry point for PyInstaller compatibility
    entry_point = "standalone_main.py"
    
    if not os.path.exists(entry_point):
        print(f"ERROR: {entry_point} not found!")
        print("This file is required for standalone executable creation.")
        sys.exit(1)
        
    print(f"Using PyInstaller-compatible entry point: {entry_point}")
    
    # Clean previous builds
    if os.path.exists("build"):
        print("Cleaning previous build...")
        if platform.system() == "Windows":
            os.system("rmdir /s /q build")
            os.system("rmdir /s /q dist")
        else:
            os.system("rm -rf build dist")
    
    # Try multiple build strategies
    build_success = False
    
    # Strategy 1: Try PyInstaller spec file (most reliable)
    if not build_success:
        build_success = build_with_spec_file()
    
    # Strategy 2: Try uv run pyinstaller (if spec fails)
    if not build_success:
        print("\nTrying uv run pyinstaller...")
        build_success = build_with_uv(entry_point)
    
    # Strategy 3: Fallback to direct pyinstaller 
    if not build_success:
        print("\nTrying fallback approach...")
        build_success = build_with_direct_pyinstaller(entry_point)
    
    # Strategy 4: Try with pip-installed pyinstaller
    if not build_success:
        print("\nInstalling pyinstaller with pip and retrying...")
        try:
            subprocess.run(["pip", "install", "pyinstaller"], check=True)
            build_success = build_with_direct_pyinstaller(entry_point)
        except subprocess.CalledProcessError:
            print("ERROR: Failed to install pyinstaller with pip")
    
    if not build_success:
        print("\nERROR: All build strategies failed!")
        print("\nTroubleshooting tips:")
        print("1. Make sure PyQt5 is installed: pip install PyQt5")
        print("2. Try manual: pip install pyinstaller && pyinstaller --onefile --windowed standalone_main.py")
        print("3. Check Windows PyQt5 compatibility issues")
        sys.exit(1)
    
    print("\nSUCCESS: Build completed!")
    
    # Show executable info
    if platform.system() == "Windows":
        exe_path = "dist/MibViewer.exe"
    else:
        exe_path = "dist/MibViewer"
        
    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print(f"Executable: {exe_path}")
        print(f"Size: {size_mb:.1f} MB")
        print(f"Version: PyQtGraph-based MIB Viewer with 4D STEM support")
        
        if platform.system() != "Windows":
            # Make executable on Unix systems
            os.chmod(exe_path, 0o755)
            print("Made executable")
    else:
        print("ERROR: Executable not found!")

if __name__ == "__main__":
    main()