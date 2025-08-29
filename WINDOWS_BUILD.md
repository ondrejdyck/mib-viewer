# Building MIB Viewer Standalone Executable on Windows

**Goal**: Create `MibViewer.exe` - a standalone executable that runs on any Windows machine without Python or dependencies.

**Version**: PyQtGraph-based viewer with 4D STEM virtual detector support  
**Target**: Offline deployment to acquisition computers and other Windows machines

---

## 🎯 Build Process Overview

1. **Development machine** (Windows with internet): Build the .exe 
2. **Target machine** (Windows, potentially offline): Just run the .exe

---

## 📋 Prerequisites (Development Machine Only)

### Required Software
1. **Python 3.10+** from https://python.org/downloads/
   - ✅ **CRITICAL**: Check "Add Python to PATH" during installation
   - ✅ Choose "Install for all users" if you have admin rights

2. **Internet connection** (for downloading dependencies during build)

### Verify Installation
```cmd
python --version
pip --version
```
Should show Python 3.10+ and pip version.

---

## 🚀 Building the Standalone Executable

### Method 1: One-Click Build (Recommended)

**Just run the batch file:**
```cmd
build_windows_exe.bat
```

**What it does:**
- Installs all dependencies with pip (bypasses uv PyQt5 issues)
- Runs PyInstaller with multiple fallback strategies
- Creates `dist\MibViewer.exe` (80-120 MB)
- Shows success/error messages

### Method 2: Manual Build Steps

If the batch file fails, build manually:

```cmd
REM 1. Install dependencies (bypasses uv PyQt5 wheel issue)
pip install PyQt5 pyqtgraph numpy matplotlib h5py emdfile tqdm py4dstem pyinstaller

REM 2. Build executable
python build_exe.py
```

### Method 3: Advanced Build (Using Spec File)

For maximum control:
```cmd
pip install PyQt5 pyqtgraph numpy matplotlib h5py emdfile tqdm pyinstaller
pyinstaller MibViewer.spec
```

---

## 📦 Build Output

### Successful Build
```
✅ SUCCESS: MibViewer.exe created!
📍 Location: dist\MibViewer.exe
📊 Size: ~80-120 MB
🎯 Version: PyQtGraph-based MIB Viewer with 4D STEM support
```

### What Gets Created
```
dist/
└── MibViewer.exe    # ← This is your standalone executable
build/               # ← Build artifacts (can delete)
MibViewer.spec       # ← PyInstaller configuration
```

---

## 🔧 Troubleshooting Build Issues

### "Python not found"
```
ERROR: Python not found in PATH
```
**Solution**: Reinstall Python with "Add to PATH" checked

### "PyQt5 wheel not found" 
```
error: pyqt5-qt5==5.15.17 doesn't have a wheel for win_amd64
```
**Solution**: The batch file handles this automatically by using pip instead of uv

### "Build failed" - General PyInstaller Issues
```cmd
REM Try direct PyInstaller command
pyinstaller --onefile --windowed --add-data "src;src" standalone_main.py
```

### "Import errors in .exe"
- Make sure `standalone_main.py` exists
- Verify `src/mib_viewer/` directory structure is intact
- Check that all Python files are in the project directory

---

## ✅ Testing Your Built Executable

### Quick Test on Build Machine
```cmd
dist\MibViewer.exe
```
Should launch the PyQtGraph interface with EELS and 4D STEM tabs.

### Deploy to Target Machine
1. **Copy** `dist\MibViewer.exe` to target Windows machine
2. **Double-click** to run - no installation needed
3. **No Python required** on target machine
4. **No internet required** on target machine

---

## 🎯 What You Get - Standalone Features

### EELS Analysis
- Hardware-accelerated PyQtGraph graphics
- Advanced ROI widgets (resizable, rotatable)
- Real-time spectrum updates
- Energy range selection with LinearRegion widgets

### 4D STEM Analysis  
- Scan position navigation with click selection
- Virtual detector overlays (BF disk, DF annular)
- Real-time virtual imaging calculations
- Optimized updates (300ms delay for smooth resizing)
- Detector visibility toggle

### Technical Benefits
- Complete independence from Python environment
- Works on Windows machines without admin rights
- Suitable for offline/restricted acquisition computers
- Single file deployment (no installation wizards)

---

## 📝 Build Environment Notes

### Recommended Build Setup
- **Windows 10/11** (64-bit)
- **4GB+ RAM** (for PyInstaller process)
- **1GB free disk space** (for build process)
- **Admin rights** (helpful but not required)

### File Structure (Development Machine)
```
mib-viewer/
├── src/mib_viewer/              # Source code
├── standalone_main.py           # PyInstaller entry point  
├── build_exe.py                 # Build script with fallbacks
├── MibViewer.spec               # PyInstaller spec file
├── build_windows_exe.bat        # One-click build
└── dist/MibViewer.exe          # ← Final product
```

---

## 🚀 Ready for Deployment

Once you have `dist\MibViewer.exe`:

1. **Copy to USB drive** or network location
2. **Transfer to acquisition computer**  
3. **Run directly** - no setup required
4. **Works offline** - perfect for restricted environments

The executable contains everything needed: Python runtime, PyQt5, PyQtGraph, NumPy, matplotlib, and all MIB analysis code in a single 80-120MB file.

---

*Build once, deploy everywhere - no Python installation headaches!*