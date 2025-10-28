# Super-Resolution Tab Redesign - Implementation Summary

**Date**: 2025-01-XX
**Status**: Partially Complete - Needs Manual Integration
**Files Modified**:
- `src/mib_viewer/gui/mib_viewer_pyqtgraph.py`
- `src/mib_viewer/processing/superres_processor.py`

**Files Created**:
- `superres_steps_to_add.py` (temporary helper file with Step 2-4 implementations)

---

## Overview

The Super-Resolution tab was redesigned from a single-step workflow into a **4-step vertical workflow** that allows users to:
1. Verify intermediate results at each stage
2. Adjust parameters before proceeding
3. Iterate on reconstruction parameters without recomputing expensive steps
4. Understand what the algorithm is doing through targeted visualizations

This matches the exploratory workflow from the Jupyter notebook "Upscaler For Ondrej.ipynb".

---

## Design Decisions

### Layout: Vertical Workflow (User Preference)

```
┌─────────────────────────┬──────────────────────────────────┐
│  WORKFLOW STEPS         │   VISUALIZATION AREA             │
│  (Scrollable)           │   (Changes per step)             │
│                         │                                  │
│  ┌─ Step 1: BF ───┐    │                                  │
│  │ [params]        │    │   [Dynamic visualization         │
│  │ [Detect & Prev] │    │    based on active step]         │
│  │ ✓ Complete      │    │                                  │
│  └─────────────────┘    │                                  │
│                         │                                  │
│  ┌─ Step 2: XCorr ┐    │                                  │
│  │ [params]        │    │                                  │
│  │ [Compute]       │    │                                  │
│  │ Ready...        │    │                                  │
│  └─────────────────┘    │                                  │
│                         │                                  │
│  ┌─ Step 3: Shifts┐    │                                  │
│  │ [Compute]       │    │                                  │
│  │ (disabled)      │    │                                  │
│  └─────────────────┘    │                                  │
│                         │                                  │
│  ┌─ Step 4: Recon ┐    │                                  │
│  │ [params]        │    │                                  │
│  │ [Reconstruct]   │    │                                  │
│  │ (disabled)      │    │                                  │
│  └─────────────────┘    │                                  │
└─────────────────────────┴──────────────────────────────────┘
```

**Rationale**: All steps visible at once; user can scroll back to see what they've done.

---

## The 4-Step Workflow

### Step 1: BF Detection & Cropping
**Purpose**: Verify bright field disk is correctly located and appropriately sized

**Parameters**:
- BF center: Auto-detect (default) / Manual
- Crop radius: 8-64 pixels (default: 32)

**Visualizations**:
- **Left**: Sample diffraction pattern (0,0) with BF center marked and crop circle overlaid
- **Right**: Radial intensity profile with crop radius indicated by red dashed line

**User Validation**:
- Is the BF center correct?
- Is the crop radius capturing the full BF disk without too much background?

**Cached Data**: `self.superres_data_cropped`, `self.superres_bf_center`

**Invalidation**: Changing parameters clears Steps 2, 3, 4

---

### Step 2: Cross-Correlation Analysis
**Purpose**: Verify cross-correlations produce clear peaks

**Parameters**:
- Reference smoothing σ: 0-5 (default: 0.5)

**Visualizations**:
- **Left**: Reference image (smoothed central detector pixel)
- **Right**: Example correlation map (detector pixel at +10, +10 from center)

**User Validation**:
- Does correlation map show a clear peak?
- Is reference image smooth enough?

**Computation Time**: ~20-25 seconds for 256×256 scan

**Cached Data**: `self.superres_correlations`, `self.superres_reference`

**Invalidation**: Changing parameters clears Steps 3, 4

---

### Step 3: Shift Map Computation ⭐ **MAIN DIAGNOSTIC STEP**
**Purpose**: Verify shift maps show smooth radial patterns (critical validation!)

**Parameters**: None (uses results from Step 2)

**Visualizations**:
- **MAIN (large, top)**: **Quiver plot** showing shift vectors (user's priority)
  - Arrows indicating direction and magnitude of shifts
  - Should show smooth radial/circular pattern
- **Insets (small, bottom row)**:
  - Shift Map Y (heatmap, centered)
  - Shift Map X (heatmap, centered)
  - Quality Map (correlation strength at peak)

**Statistics Display**:
```
Shift Y range: [-A, +B]
Shift X range: [-C, +D]
Quality: mean=X, median=Y
Quality range: [min, max]
```

**User Validation**:
- **CRITICAL**: Do shift maps show smooth gradients? If noisy → something is wrong!
- Are quality values high (>0.8)?
- Does quiver plot show expected radial pattern?

**What Good Looks Like**:
- Shift ranges: ±10 to ±20 pixels (not ±128!)
- Quality mean: >0.9
- Smooth, continuous shift patterns

**Cached Data**: `self.superres_shift_maps` (tuple of xm_sub, ym_sub, im_sub)

**Invalidation**: N/A (no parameters to change)

---

### Step 4: Super-Resolution Reconstruction
**Purpose**: Generate final super-res image with adjustable reconstruction parameters

**Parameters** (can iterate quickly without recomputing Steps 1-3):
- Upscaling factor: 1-8× (default: 4)
- Detector radius: ±4-32 pixels (default: 16)
- Quality threshold: 0-1 (default: 0.7)

**Visualizations** (2×2 grid):
- **Top-left**: Standard BF (central detector pixel)
- **Top-right**: Super-Resolution BF
- **Bottom-left**: Standard BF FFT (log scale)
- **Bottom-right**: Super-Res FFT (log scale) - **should show higher frequencies**

**User Validation**:
- Does super-res image show finer detail?
- Does FFT show extended frequency response?

**Computation Time**: ~2 seconds (fast iteration)

**Cached Data**: `self.superres_results`

**Invalidation**: Changing parameters only invalidates Step 4 (can re-run immediately)

---

## State Management & Caching

### Cached Intermediate Results
```python
self.superres_data_cropped = None    # Step 1 result (256,256,64,64)
self.superres_correlations = None    # Step 2 result (~4GB for 256×256)
self.superres_shift_maps = None      # Step 3 result (3 arrays: xm, ym, quality)
self.superres_results = None         # Step 4 result (super-res image + metadata)
```

### Invalidation Logic

**When Step 1 parameters change** (`invalidate_step1_onwards`):
- Clear ALL cached data (steps 2, 3, 4)
- Disable buttons for Steps 2, 3, 4
- Set status to "Re-run Step 1 first"

**When Step 2 parameters change** (`invalidate_step2_onwards`):
- Clear cached data for steps 3, 4 (keep Step 1)
- Disable buttons for Steps 3, 4
- Set status to "Re-run Step 2 first"

**When Step 4 parameters change** (`invalidate_step4`):
- Clear only Step 4 results
- Button stays enabled (can re-run immediately)

### Button State Flow

```
Load 4D Data
    ↓
Step 1 button ENABLED
    ↓
[User clicks Step 1]
    ↓
Step 2 button ENABLED
    ↓
[User clicks Step 2]
    ↓
Step 3 button ENABLED
    ↓
[User clicks Step 3]
    ↓
Step 4 button ENABLED
    ↓
[User can iterate Step 4 quickly]
```

---

## Implementation Status

### ✅ Completed

1. **Tab Layout** (`create_superres_tab` method)
   - Left panel: Scrollable workflow with 4 step groups
   - Right panel: Stacked widget with 5 pages (instructions + 4 step visualizations)
   - Proper sizing and proportions

2. **Parameter Controls**
   - All spinboxes, radio buttons, labels created
   - Connected to invalidation methods via `valueChanged` signals

3. **Visualization Widgets**
   - All plot widgets created and configured
   - Stacked widget for switching between visualizations
   - Instructions page (index 0)
   - Step 1 viz page (index 1): diffraction + radial profile
   - Step 2 viz page (index 2): reference + correlation
   - Step 3 viz page (index 3): quiver plot + 3 insets
   - Step 4 viz page (index 4): 2×2 BF/FFT comparison

4. **State Management**
   - Cache variables initialized
   - Invalidation methods fully implemented
   - `update_superres_button_state` connected to data loading

5. **Step 1 Implementation** (`superres_step1_detect_bf`)
   - BF center detection
   - Data cropping
   - Diffraction pattern visualization with center marker
   - Crop region circle overlay
   - Radial profile computation and display
   - Status updates and error handling

6. **Missing Import Fixed**
   - Added `QDoubleSpinBox` to imports
   - Added `QScrollArea`, `QStackedWidget` imports (inline)

### ⚠️ Needs Manual Addition

**Steps 2, 3, 4 implementations** are in `superres_steps_to_add.py`:

1. `superres_step2_correlations` (~50 lines)
2. `superres_step3_shifts` (~90 lines)
3. `superres_step4_reconstruct` (~80 lines)

**Where to add**: Insert at **line ~4365** in `mib_viewer_pyqtgraph.py`
**Location**: Right before `def auto_detect_and_load_fft(self):`

---

## How to Complete Integration

### Option 1: Manual Copy-Paste

1. Open `mib_viewer_pyqtgraph.py`
2. Find `def auto_detect_and_load_fft(self):` (around line 4365)
3. Open `superres_steps_to_add.py`
4. Copy the 3 methods (without the `def` at the very top explanation)
5. Paste RIGHT BEFORE `auto_detect_and_load_fft`
6. Delete `superres_steps_to_add.py` (cleanup)

### Option 2: Automated Insertion

```python
# Read existing file
with open('src/mib_viewer/gui/mib_viewer_pyqtgraph.py', 'r') as f:
    lines = f.readlines()

# Find insertion point
for i, line in enumerate(lines):
    if 'def auto_detect_and_load_fft(self):' in line:
        insertion_point = i
        break

# Read methods to insert
with open('superres_steps_to_add.py', 'r') as f:
    new_methods = f.readlines()[2:]  # Skip first 2 comment lines

# Insert
lines[insertion_point:insertion_point] = new_methods

# Write back
with open('src/mib_viewer/gui/mib_viewer_pyqtgraph.py', 'w') as f:
    f.writelines(lines)
```

---

## Testing Checklist

After integration, test this workflow:

1. **Load 4D STEM data**
   - [ ] Step 1 button becomes enabled
   - [ ] Status shows "Ready - click to detect BF region"

2. **Step 1: BF Detection**
   - [ ] Click "Detect & Preview BF Region"
   - [ ] Diffraction pattern appears with red center marker
   - [ ] Red circle shows crop region
   - [ ] Radial profile shows BF disk with red dashed line at crop radius
   - [ ] Status shows "✓ BF center: (x, y)"
   - [ ] Step 2 button becomes enabled

3. **Parameter Change Test**
   - [ ] Change BF radius
   - [ ] Step 2 button becomes disabled
   - [ ] Status shows "Re-run Step 1 first"

4. **Step 2: Cross-Correlations**
   - [ ] Click "Compute Cross-Correlations"
   - [ ] Progress shows "Computing... (~20s)"
   - [ ] Reference image appears (left)
   - [ ] Example correlation map appears (right) with clear peak
   - [ ] Status shows "✓ Cross-correlations computed"
   - [ ] Step 3 button becomes enabled

5. **Step 3: Shift Maps** ⭐
   - [ ] Click "Compute Shift Maps"
   - [ ] Large quiver plot appears with arrows
   - [ ] Arrows show radial pattern (not random!)
   - [ ] 3 inset images show shift Y, shift X, quality
   - [ ] Statistics display shows reasonable shift ranges (±10-20, not ±128!)
   - [ ] Quality mean >0.9
   - [ ] Status shows "✓ Shift maps computed"
   - [ ] Step 4 button becomes enabled

6. **Step 4: Reconstruction**
   - [ ] Click "Reconstruct Super-Res Image"
   - [ ] 2×2 grid appears with BF comparison and FFTs
   - [ ] Super-res image is larger than standard BF
   - [ ] Super-res FFT shows higher frequencies than standard FFT
   - [ ] Status shows "✓ Reconstruction complete! (4× upscaling)"

7. **Iteration Test**
   - [ ] Change quality threshold or detector radius
   - [ ] Click "Reconstruct" again
   - [ ] Result updates quickly (~2s, not 20s)

---

## Known Issues

1. **Quiver plot rendering**: Uses `pg.ArrowItem` in a loop which may be slow for dense grids. If performance is poor, consider:
   - Using PyQtGraph's built-in quiver functionality (if available)
   - Drawing with QPainter directly
   - Increasing subsampling step size

2. **Memory usage**: Correlations cache is ~4GB for 256×256 scans. Consider:
   - Adding a "Clear Cache" button
   - Showing memory usage estimate
   - Auto-clearing on tab switch

3. **EllipseROI**: Fixed syntax error (was `pg.EllipseItemROI`, now `pg.EllipseROI`)

---

## Future Enhancements

1. **Export Options**
   - Save shift maps as HDF5
   - Export super-res image as TIFF
   - Export quiver plot as PNG/SVG

2. **Advanced Diagnostics**
   - Show multiple example correlation maps (grid of 4-6)
   - Aberration fitting from shift maps
   - Line profiles comparing standard vs super-res

3. **Parameter Presets**
   - "Fast Preview" (smaller detector radius, lower threshold)
   - "High Quality" (large detector radius, high threshold)
   - Save/load custom presets

4. **Progress Bars**
   - Replace label-based progress with QProgressBar
   - Show percentage complete for Step 2 FFT

---

## Algorithm Reference

**Notebook**: `Upscaler For Ondrej.ipynb`
**Documentation**: `4D-STEM-Super-Resolution-Algorithm.md`
**Processor**: `src/mib_viewer/processing/superres_processor.py`

**Test Dataset**: `/media/o2d/data/ORNL Dropbox/Ondrej Dyck/Code/mib-viewer-project/SS_a7_2ms_256x256 4D_16nmFoV -20nmFocus.emd`

**Expected Results** (for test dataset):
- BF center: ~(127, 126)
- Shift range: ±10-20 pixels
- Quality mean: 0.92-0.93
- Quiver plot: Smooth radial/circular pattern

---

## Contact

For questions about this implementation:
- Algorithm: See `4D-STEM-Super-Resolution-Algorithm.md`
- Code: Check `superres_processor.py` docstrings
- Workflow: This document

**Next Step**: Manually insert methods from `superres_steps_to_add.py` and test!
