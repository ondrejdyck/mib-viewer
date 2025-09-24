# MIB EELS Viewer with 4D STEM Support

**Hardware-accelerated PyQtGraph GUI for analyzing MIB EELS and 4D STEM datasets with virtual detector capabilities.**

![Status](https://img.shields.io/badge/Status-Release-green)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-blue)
![Build](https://img.shields.io/badge/Build-Automated-brightgreen)

## 🚀 Quick Start - Download & Run

### **Recommended: Get the Latest Release**

1. **Download**: Go to [**Releases**](https://github.com/ondrejdyck/mib-viewer/releases) and download for your platform:
   - **Windows**: `MibViewer.exe` - Double-click to run
   - **Linux**: `MibViewer` - Run with `chmod +x MibViewer && ./MibViewer`
2. **Load Data**: File → Load MIB File → select your dataset
3. **Analyze**: Switch between EELS and 4D STEM tabs

### **Alternative: Development Builds**

1. **Go to**: [Actions](https://github.com/ondrejdyck/mib-viewer/actions) 
2. **Download**: Build artifacts from latest successful run:
   - "MibViewer-Windows-exe" for Windows
   - "MibViewer-Linux" for Ubuntu/Debian
3. **Extract**: and run the appropriate executable

---

## ✨ Features

### **EELS Analysis**
- Hardware-accelerated PyQtGraph interface
- Interactive ROI selection (resizable, rotatable)
- Real-time spectrum visualization
- Energy range analysis

### **4D STEM Analysis**
- Virtual detector overlays (BF/DF imaging)
- Interactive scan position navigation
- Real-time virtual imaging updates
- Optimized performance (300ms update delay)

### **4D FFT Analysis** ✨ *NEW*
- **Chunked 4D FFT computation** for memory-efficient processing
- **Auto-persistent results** - compute once, use forever
- **Three-phase optimization** - instant display with background saving
- **Adaptive bright field detection** and cropping
- **Interactive spatial frequency analysis** with ROI/point selectors
- **Amplitude and phase visualization** with specialized colormaps

### **Technical Capabilities**
- **Standalone**: No Python installation required (~80-120 MB)
- **Performance**: Hardware-accelerated OpenGL rendering
- **Memory Efficient**: Memory-mapped file access
- **Format Support**: MIB files, ndata contextual images

---

## 🖥️ System Requirements

### **Windows**
- **Windows 10/11** (64-bit)
- **4GB+ RAM** (8GB+ recommended for large datasets)  
- **100MB free disk space**
- **No admin rights required**

### **Linux (Ubuntu/Debian)**
- **Ubuntu 18.04+** or **Debian 10+** (64-bit)
- **4GB+ RAM** (8GB+ recommended for large datasets)
- **100MB free disk space**
- **No root access required**

---

## 📁 File Format Support

- **Primary**: `.mib` files (4D EELS/STEM datasets)
- **Contextual**: `.ndata1` files (2D reference images)
- **Future**: EMD compressed format support

---

## 🛠️ For Developers

### **Repository Structure**
```
src/mib_viewer/          # Main application code
├── gui/                 # PyQtGraph interface
├── io/                  # File format handlers (including mib_loader.py)  
├── analysis/            # Data processing tools
├── __main__.py          # Standard Python module entry point
└── main.py              # Main function for PyInstaller

MibViewer.spec          # Build configuration
.github/workflows/      # Automated CI/CD
```

### **Build from Source**
```bash
git clone https://github.com/ondrejdyck/mib-viewer.git
cd mib-viewer
pip install PyQt5 pyqtgraph numpy matplotlib h5py emdfile tqdm pyinstaller
pyinstaller MibViewer.spec
```
