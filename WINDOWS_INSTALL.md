# Using MIB Viewer on Windows

**For Users**: Running the standalone MIB Viewer executable on Windows machines

**For Developers**: See `WINDOWS_BUILD.md` for building the executable

---

## 🎯 Quick Start (For Most Users)

### If you have `MibViewer.exe`:
1. **Double-click** `MibViewer.exe`
2. **Done!** - The application launches immediately

No Python, no installation, no internet required.

---

## 📥 Getting MibViewer.exe

### From Developer/IT
Ask your developer or IT administrator for the latest `MibViewer.exe` file (~80-120 MB).

### Build It Yourself
See `WINDOWS_BUILD.md` for complete build instructions.

---

## 🚀 Using the Application

### EELS Analysis Workflow
1. **File** → **Load MIB File** → Select your `.mib` EELS dataset
2. **Optional**: **File** → **Load ndata File** → Select corresponding `.ndata` image
3. **Switch to EELS tab** (usually default)
4. **Interact with data**:
   - **Drag ROI rectangles** to select analysis regions
   - **Resize/rotate ROIs** using corner handles
   - **Adjust energy range** with red selection bar on spectrum
   - **Spectra update in real-time** as you move ROIs

### 4D STEM Analysis Workflow  
1. **File** → **Load MIB File** → Select your `.mib` 4D STEM dataset
2. **Switch to 4D STEM tab**
3. **Navigate scan positions**: Click on scan overview image
4. **Adjust virtual detectors**:
   - **BF (green circle)**: Bright field disk detector
   - **DF (blue rings)**: Dark field annular detector  
   - **Resize detectors**: Drag edges to adjust size
   - **Toggle visibility**: Uncheck "Show Detector Overlays" for clear diffraction view
5. **Virtual images update automatically** after 300ms delay

---

## 🖥️ System Requirements

### Minimum Requirements
- **Windows 10** (64-bit) or later
- **4GB RAM** (8GB+ recommended for large datasets)
- **100MB disk space** (for the executable)
- **No admin rights required**

### Recommended for Large Datasets
- **16GB+ RAM** (for datasets >4GB)
- **SSD storage** (faster file loading)
- **Dedicated graphics** (better PyQtGraph performance)

---

## 📁 File Formats Supported

### MIB Files (Primary Data)
- **4D EELS datasets**: Rectangular detector (energy × detector)
- **4D STEM datasets**: Square detector (diffraction patterns)
- **Automatic scan size detection** - no manual input needed
- **Memory-mapped loading** for efficient access

### ndata Files (Complementary Images)  
- **2D images** with metadata (ZIP format)
- **Optional**: Provides context images for EELS analysis

### Future Support
- **EMD files**: Compressed format for large datasets (in development)

---

## ⚠️ Troubleshooting

### Application Won't Start
**"Application failed to start"**
- **Check Windows version**: Requires Windows 10+ (64-bit)
- **Antivirus blocking**: Add MibViewer.exe to antivirus exceptions
- **Corrupted download**: Re-download the .exe file

### "DLL Load Failed" 
**Missing Visual C++ Redistributables**
```
Download and install: Microsoft Visual C++ Redistributable (x64)
https://aka.ms/vs/17/release/vc_redist.x64.exe
```

### Application Crashes on File Load
**Large dataset memory issues**
- **Check available RAM**: Task Manager → Performance → Memory
- **Close other applications** to free memory
- **For 8GB+ files**: Use machine with 32GB+ RAM

### Slow Performance
**Graphics/rendering issues**
- **Update graphics drivers** (especially Intel/AMD integrated graphics)
- **Close other GPU-intensive applications**
- **Reduce dataset size** if possible

---

## 🎛️ Application Features

### Hardware Acceleration
- **PyQtGraph with OpenGL**: Smooth real-time graphics
- **Memory-mapped file access**: Efficient large dataset handling
- **Optimized calculations**: NumPy boolean masking for virtual detectors

### User Interface
- **Tabbed interface**: Clean separation of EELS and 4D STEM workflows
- **Resizable panels**: Adjust layout to your preference  
- **Real-time updates**: Immediate feedback during analysis
- **Non-blocking operations**: UI stays responsive during calculations

### Analysis Capabilities
- **Interactive ROI selection**: Point-and-click analysis regions
- **Virtual detector analysis**: BF/DF imaging from 4D datasets
- **Energy range integration**: Custom spectrum analysis
- **Automatic calibration**: No manual scan size input required

---

## 📞 Getting Help

### For Users
1. **Check this guide** for common solutions
2. **Contact your IT administrator** for technical issues
3. **Verify file formats** are supported (.mib, .ndata)

### For Developers
1. **See `WINDOWS_BUILD.md`** for build instructions
2. **Check source code** in `src/mib_viewer/` directory
3. **Review build logs** in `build/` directory for errors

---

## 🔄 Updates

### Getting New Versions
- **New .exe file** from developer/IT (replaces old version)
- **No uninstallation needed** - just replace the file
- **Settings are preserved** (stored in Windows user profile)

### Version Information
Current build shows version info on startup and in window title:
- **PyQtGraph-based**: Hardware-accelerated graphics
- **4D STEM support**: Virtual detector capabilities
- **Alpha status**: Ready for testing and feedback

---

*Simple deployment, powerful analysis - designed for the electron microscopy workflow!*