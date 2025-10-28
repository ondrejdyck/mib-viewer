# Task 2: Add New Step 2 - Compute & Explore 4D FFT

**Date:** 2025-01-07
**Status:** Planning
**Dependencies:** Task 1 (Cache Manager) ✅ Complete

---

## Overview

Insert a new Step 2 between current Step 1 (BF Detection) and current Step 2 (Cross-Correlations) that:
1. Computes the 4D FFT (bigFT) and writes to cache file
2. Provides interactive FFT exploration (same as 4D FFT Explorer tab)
3. Requires user to click "Proceed" button before advancing to Step 3

This allows users to inspect reciprocal space structure before cross-correlation.

---

## Architecture Analysis

### Current 4D FFT Explorer Tab Structure

**Layout:**
```
┌─────────────────────────────────────────────┬──────────┐
│  Plots (2x2 Grid)                           │ Controls │
│  ┌──────────────┬──────────────┐           │          │
│  │ FFT Overview │ Sub-select   │           │ [Compute]│
│  │  (center)    │ (freq slice) │           │ [Modes]  │
│  ├──────────────┼──────────────┤           │ [Info]   │
│  │ Amplitude    │ Phase        │           │          │
│  │ (at ky,kx)   │ (at ky,kx)   │           │          │
│  └──────────────┴──────────────┘           │          │
└─────────────────────────────────────────────┴──────────┘
```

**Key Components:**

1. **Plots** (lines 544-589 in mib_viewer_pyqtgraph.py):
   - `fft_overview_plot`: Log(|FFT[sy,sx]|) at detector center
   - `fft_subselect_plot`: Spatial frequency map for selected detector pixel/ROI
   - `fft_amp_plot`: |FFT[ky,kx]| amplitude at selected spatial frequency
   - `fft_phase_plot`: angle(FFT[ky,kx]) phase at selected spatial frequency

2. **Selectors**:
   - `fft_scan_selector_overview`: Crosshair on overview (selects ky,kx spatial frequency)
   - `fft_scan_selector_subselect`: Linked crosshair on sub-selection
   - `fft_freq_selector_amp`: Crosshair on amplitude (selects qy,qx detector position)
   - `fft_freq_selector_phase`: Linked crosshair on phase

3. **Update Logic** (lines 3957-4044):
   ```python
   def update_fft_displays():
       # Overview: FFT at detector center
       spatial_freq_map = fft4d_data[:, :, qy//2, qx//2]
       overview_item.setImage(log(abs(spatial_freq_map)))

       # Get selected spatial frequency (ky, kx) from selector
       ky, kx = get_fft_scan_position()

       # Amplitude/Phase: FFT slice at (ky, kx)
       fft_slice = fft4d_data[ky, kx, :, :]
       amp_item.setImage(abs(fft_slice))
       phase_item.setImage(angle(fft_slice))

       # Sub-select: Selected detector pixel across all (ky, kx)
       freq_y, freq_x = get_fft_freq_position()
       subselect_map = abs(fft4d_data[:, :, freq_y, freq_x])
       subselect_item.setImage(log(subselect_map))
   ```

4. **Data Access**:
   - Reads from `self.fft4d_data` (complex128 array in memory)
   - For super-res, we'll read from cache file in chunks

---

## Integration Strategy

### Option 1: Embed FFT Explorer in Step 2 Group (RECOMMENDED)

**Layout for new Step 2:**
```
┌──────────────────────────────────────────────────────┐
│ Step 2: Compute & Explore 4D FFT                     │
├──────────────────────────────────────────────────────┤
│ [Compute 4D FFT] button                              │
│ Status: "Computing... 45% (chunk 12/27)"             │
├──────────────────────────────────────────────────────┤
│ ┌──── FFT Explorer (appears after computation) ────┐│
│ │ ┌──────────┬──────────┐  ┌────────────────────┐ ││
│ │ │ Overview │Sub-select│  │ Parameters:        │ ││
│ │ ├──────────┼──────────┤  │ - Ref smoothing    │ ││
│ │ │ Amp      │ Phase    │  │ - (future params)  │ ││
│ │ └──────────┴──────────┘  │ [Proceed to Step3] │ ││
│ │                           └────────────────────┘ ││
│ └──────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────┘
```

**Advantages:**
- All in one place (vertical workflow preserved)
- User can't miss the FFT explorer
- Proceed button clearly gates Step 3

**Disadvantages:**
- Step 2 group becomes large (~500 pixels tall)
- May need scrolling on smaller screens

### Option 2: FFT Explorer in Visualization Panel

Use existing `self.sr_viz_stack` (stacked widget for visualizations):
- Index 0: Instructions
- Index 1: Step 1 results (BF detection preview)
- Index 2: **Step 2 results (FFT explorer)** ← NEW
- Index 3: Step 3+ results

**Advantages:**
- Reuses existing visualization space
- Keeps controls panel compact

**Disadvantages:**
- FFT plots might be too small (competing with detection preview)
- Less clear that user should explore before proceeding

**Recommendation: Use Option 1** - Embed in Step 2 group, make it collapsible.

---

## Implementation Plan

### Part 1: Modify Step 2 GUI Structure

**Current Step 2 (lines 734-758):**
```python
step2_group = QGroupBox("Step 2: Cross-Correlations")
step2_layout = QVBoxLayout(step2_group)

step2_layout.addWidget(QLabel("<b>Parameters:</b>"))
self.sr_ref_smoothing = create_spinbox(...)
step2_layout.addWidget(self.sr_ref_smoothing)

self.sr_step2_btn = QPushButton("Compute Cross-Correlations")
self.sr_step2_btn.clicked.connect(self.superres_step2_correlations)
self.sr_step2_btn.setEnabled(False)
step2_layout.addWidget(self.sr_step2_btn)

self.sr_step2_status = QLabel("Complete Step 1 first")
step2_layout.addWidget(self.sr_step2_status)

workflow_layout.addWidget(step2_group)
```

**New Step 2 - Compute & Explore 4D FFT:**
```python
step2_group = QGroupBox("Step 2: Compute & Explore 4D FFT")
step2_layout = QVBoxLayout(step2_group)

# Compute button
self.sr_step2_compute_btn = QPushButton("Compute 4D FFT")
self.sr_step2_compute_btn.clicked.connect(self.superres_step2_compute_fft)
self.sr_step2_compute_btn.setEnabled(False)
step2_layout.addWidget(self.sr_step2_compute_btn)

# Progress/status
self.sr_step2_status = QLabel("Complete Step 1 first")
self.sr_step2_status.setStyleSheet("color: #666; font-size: 10px;")
step2_layout.addWidget(self.sr_step2_status)

# FFT Explorer (hidden initially)
self.sr_fft_explorer_widget = self.create_sr_fft_explorer()
self.sr_fft_explorer_widget.setVisible(False)
step2_layout.addWidget(self.sr_fft_explorer_widget)

workflow_layout.addWidget(step2_group)
```

**New Step 3 - Cross-Correlations (renumbered from old Step 2):**
```python
step3_group = QGroupBox("Step 3: Cross-Correlations")
step3_layout = QVBoxLayout(step3_group)

step3_layout.addWidget(QLabel("<b>Parameters:</b>"))
self.sr_ref_smoothing = create_spinbox(...)
step3_layout.addWidget(self.sr_ref_smoothing)

self.sr_step3_btn = QPushButton("Compute Cross-Correlations")  # Renamed from sr_step2_btn
self.sr_step3_btn.clicked.connect(self.superres_step3_correlations)  # Renamed from step2
self.sr_step3_btn.setEnabled(False)
step3_layout.addWidget(self.sr_step3_btn)

self.sr_step3_status = QLabel("Complete Step 2 first")
step3_layout.addWidget(self.sr_step3_status)

workflow_layout.addWidget(step3_group)
```

---

### Part 2: Create FFT Explorer Widget

**New method:**
```python
def create_sr_fft_explorer(self):
    """
    Create embedded FFT explorer for Step 2.
    Similar to 4D FFT tab but:
    - Reads from cache file instead of memory
    - Simplified controls (no mode selection, compute button)
    - Has "Proceed to Step 3" button
    """
    explorer_widget = QWidget()
    explorer_layout = QHBoxLayout(explorer_widget)

    # Left: Plots (2x2 grid) - 70% width
    plots_widget = QWidget()
    plots_layout = QGridLayout(plots_widget)

    # Overview plot
    self.sr_fft_overview_plot = pg.PlotWidget(title="FFT Overview (center)")
    self.sr_fft_overview_plot.setAspectLocked(True)
    self.sr_fft_overview_item = pg.ImageItem()
    self.sr_fft_overview_plot.addItem(self.sr_fft_overview_item)
    plots_layout.addWidget(self.sr_fft_overview_plot, 0, 0)

    # Sub-selection plot
    self.sr_fft_subselect_plot = pg.PlotWidget(title="Sub-selection")
    self.sr_fft_subselect_plot.setAspectLocked(True)
    self.sr_fft_subselect_item = pg.ImageItem()
    self.sr_fft_subselect_plot.addItem(self.sr_fft_subselect_item)
    plots_layout.addWidget(self.sr_fft_subselect_plot, 0, 1)

    # Amplitude plot
    self.sr_fft_amp_plot = pg.PlotWidget(title="Amplitude")
    self.sr_fft_amp_plot.setAspectLocked(True)
    self.sr_fft_amp_item = pg.ImageItem()
    self.sr_fft_amp_plot.addItem(self.sr_fft_amp_item)
    plots_layout.addWidget(self.sr_fft_amp_plot, 1, 0)

    # Phase plot
    self.sr_fft_phase_plot = pg.PlotWidget(title="Phase")
    self.sr_fft_phase_plot.setAspectLocked(True)
    self.sr_fft_phase_item = pg.ImageItem()
    self.sr_fft_phase_plot.addItem(self.sr_fft_phase_item)
    # Use seismic colormap for phase
    colormap = pg.colormap.get('seismic', source='matplotlib')
    self.sr_fft_phase_item.setColorMap(colormap)
    plots_layout.addWidget(self.sr_fft_phase_plot, 1, 1)

    explorer_layout.addWidget(plots_widget, stretch=7)

    # Right: Controls - 30% width
    controls_widget = QWidget()
    controls_layout = QVBoxLayout(controls_widget)

    # Instructions
    info_label = QLabel(
        "Click on plots to explore FFT:\n"
        "• Overview/Sub-select: Select spatial frequency\n"
        "• Amplitude/Phase: Select detector position"
    )
    info_label.setStyleSheet("color: #666; font-size: 10px;")
    info_label.setWordWrap(True)
    controls_layout.addWidget(info_label)

    controls_layout.addSpacing(20)

    # Parameters (for future use)
    params_group = QGroupBox("Parameters")
    params_layout = QVBoxLayout(params_group)

    # Reference smoothing (preview for Step 3)
    params_layout.addWidget(QLabel("Reference Smoothing (σ):"))
    # Note: self.sr_ref_smoothing already exists, just display value
    self.sr_ref_smoothing_display = QLabel(f"{self.sr_ref_smoothing.value()}")
    params_layout.addWidget(self.sr_ref_smoothing_display)
    # Link to update when main parameter changes
    self.sr_ref_smoothing.valueChanged.connect(
        lambda v: self.sr_ref_smoothing_display.setText(f"{v}")
    )

    controls_layout.addWidget(params_group)

    controls_layout.addStretch()

    # Proceed button
    self.sr_step2_proceed_btn = QPushButton("✓ Proceed to Cross-Correlation (Step 3)")
    self.sr_step2_proceed_btn.setStyleSheet("font-weight: bold; padding: 10px;")
    self.sr_step2_proceed_btn.clicked.connect(self.superres_step2_proceed)
    controls_layout.addWidget(self.sr_step2_proceed_btn)

    explorer_layout.addWidget(controls_widget, stretch=3)

    return explorer_widget
```

---

### Part 3: Add Selectors and Interaction

**Setup selectors (called after FFT computation):**
```python
def setup_sr_fft_selectors(self):
    """Setup crosshair selectors for FFT explorer"""

    # Spatial frequency selectors (on overview and subselect)
    self.sr_fft_scan_selector_overview = pg.CrosshairROI(
        pos=(self.sr_fft_shape[0]//2, self.sr_fft_shape[1]//2),
        size=(1, 1),
        pen='r'
    )
    self.sr_fft_overview_plot.addItem(self.sr_fft_scan_selector_overview)

    self.sr_fft_scan_selector_subselect = pg.CrosshairROI(
        pos=(self.sr_fft_shape[0]//2, self.sr_fft_shape[1]//2),
        size=(1, 1),
        pen='r'
    )
    self.sr_fft_subselect_plot.addItem(self.sr_fft_scan_selector_subselect)

    # Detector position selectors (on amp and phase)
    self.sr_fft_freq_selector_amp = pg.CrosshairROI(
        pos=(self.sr_fft_shape[2]//2, self.sr_fft_shape[3]//2),
        size=(1, 1),
        pen='g'
    )
    self.sr_fft_amp_plot.addItem(self.sr_fft_freq_selector_amp)

    self.sr_fft_freq_selector_phase = pg.CrosshairROI(
        pos=(self.sr_fft_shape[2]//2, self.sr_fft_shape[3]//2),
        size=(1, 1),
        pen='g'
    )
    self.sr_fft_phase_plot.addItem(self.sr_fft_freq_selector_phase)

    # Connect signals for linked movement
    self.sr_fft_scan_selector_overview.sigRegionChanged.connect(
        lambda: self.on_sr_fft_scan_changed('overview')
    )
    self.sr_fft_scan_selector_subselect.sigRegionChanged.connect(
        lambda: self.on_sr_fft_scan_changed('subselect')
    )
    self.sr_fft_freq_selector_amp.sigRegionChanged.connect(
        lambda: self.on_sr_fft_freq_changed('amp')
    )
    self.sr_fft_freq_selector_phase.sigRegionChanged.connect(
        lambda: self.on_sr_fft_freq_changed('phase')
    )

def on_sr_fft_scan_changed(self, source):
    """Handle spatial frequency selector changes with linking"""
    if source == 'overview':
        pos = self.sr_fft_scan_selector_overview.pos()
        self.sr_fft_scan_selector_subselect.blockSignals(True)
        self.sr_fft_scan_selector_subselect.setPos(pos, update=False)
        self.sr_fft_scan_selector_subselect.blockSignals(False)
    else:  # subselect
        pos = self.sr_fft_scan_selector_subselect.pos()
        self.sr_fft_scan_selector_overview.blockSignals(True)
        self.sr_fft_scan_selector_overview.setPos(pos, update=False)
        self.sr_fft_scan_selector_overview.blockSignals(False)

    # Update displays
    self.update_sr_fft_displays()

def on_sr_fft_freq_changed(self, source):
    """Handle detector position selector changes with linking"""
    if source == 'amp':
        pos = self.sr_fft_freq_selector_amp.pos()
        self.sr_fft_freq_selector_phase.blockSignals(True)
        self.sr_fft_freq_selector_phase.setPos(pos, update=False)
        self.sr_fft_freq_selector_phase.blockSignals(False)
    else:  # phase
        pos = self.sr_fft_freq_selector_phase.pos()
        self.sr_fft_freq_selector_amp.blockSignals(True)
        self.sr_fft_freq_selector_amp.setPos(pos, update=False)
        self.sr_fft_freq_selector_amp.blockSignals(False)

    # Update displays
    self.update_sr_fft_subselection()
```

---

### Part 4: Update Display Logic with Cache Reading

**Key difference from 4D FFT tab:** Read from cache file instead of `self.fft4d_data`

```python
def update_sr_fft_displays(self):
    """Update FFT explorer displays, reading from cache file"""

    if self.superres_cache_manager is None:
        return

    if not self.superres_cache_manager.has_bigft():
        return

    # Get current spatial frequency position
    pos = self.sr_fft_scan_selector_overview.pos()
    ky = int(np.clip(pos[1], 0, self.sr_fft_shape[0] - 1))
    kx = int(np.clip(pos[0], 0, self.sr_fft_shape[1] - 1))

    # Read overview: FFT at detector center (read single slice)
    sy, sx, dy, dx = self.sr_fft_shape
    center_y, center_x = dy // 2, dx // 2

    # Read: [all ky, all kx, center_y, center_x]
    overview_slice = slice(None), slice(None), center_y, center_x
    spatial_freq_map = self.superres_cache_manager.read_bigft(overview_slice)
    overview_data = np.log(np.abs(spatial_freq_map) + 1e-10)
    self.sr_fft_overview_item.setImage(overview_data.T)

    # Read amplitude/phase: FFT at selected (ky, kx)
    # Read: [ky, kx, all qy, all qx]
    amp_phase_slice = (ky, kx, slice(None), slice(None))
    fft_slice = self.superres_cache_manager.read_bigft(amp_phase_slice)

    # Update amplitude
    amp_data = np.abs(fft_slice)
    self.sr_fft_amp_item.setImage(amp_data.T)

    # Update phase
    phase_data = np.angle(fft_slice)
    self.sr_fft_phase_item.setImage(phase_data.T)

    # Update sub-selection
    self.update_sr_fft_subselection()

def update_sr_fft_subselection(self):
    """Update sub-selection display"""

    if self.superres_cache_manager is None:
        return

    # Get selected detector position
    pos = self.sr_fft_freq_selector_amp.pos()
    freq_y = int(np.clip(pos[1], 0, self.sr_fft_shape[2] - 1))
    freq_x = int(np.clip(pos[0], 0, self.sr_fft_shape[3] - 1))

    # Read: [all ky, all kx, freq_y, freq_x]
    subselect_slice = (slice(None), slice(None), freq_y, freq_x)
    spatial_freq_map = self.superres_cache_manager.read_bigft(subselect_slice)

    # Apply log scaling
    log_spatial_freq_map = np.log(np.abs(spatial_freq_map) + 1e-10)
    self.sr_fft_subselect_item.setImage(log_spatial_freq_map.T)
```

---

### Part 5: Step Methods

**Step 2 Compute:**
```python
def superres_step2_compute_fft(self):
    """Step 2: Compute 4D FFT and write to cache"""

    if self.superres_data_cropped is None:
        QMessageBox.warning(self, "Error", "Complete Step 1 first")
        return

    try:
        from ..processing.superres_processor import SuperResProcessor
        from ..processing.superres_cache import SuperResCacheManager

        self.sr_step2_compute_btn.setEnabled(False)
        self.sr_step2_status.setText("Creating cache file...")
        QApplication.processEvents()

        # Create cache manager
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        cache_manager = SuperResCacheManager(
            self.stem4d_filename,
            timestamp=timestamp,
            create_new=True
        )

        # Create cache file structure
        cache_manager.create_cache_file(
            crop_info=self.superres_crop_info,
            original_shape=self.stem4d_data.shape,
            cropped_shape=self.superres_data_cropped.shape,
            reference_smoothing=self.sr_ref_smoothing.value()
        )

        self.sr_step2_status.setText("Computing 4D FFT...")
        QApplication.processEvents()

        # Compute bigFT to cache
        processor = SuperResProcessor(log_callback=self.log_message)
        reference_image = processor.compute_bigft_to_cache(
            self.superres_data_cropped,
            cache_manager,
            reference_smoothing=self.sr_ref_smoothing.value()
        )

        # Store cache manager and metadata
        self.superres_cache_manager = cache_manager
        self.superres_reference_image = reference_image
        self.sr_fft_shape = self.superres_data_cropped.shape

        # Show FFT explorer
        self.sr_fft_explorer_widget.setVisible(True)
        self.setup_sr_fft_selectors()
        self.update_sr_fft_displays()

        # Update status
        self.sr_step2_status.setText(
            f"✓ FFT computed ({cache_manager.get_cache_size_gb():.2f} GB cache)"
        )

        self.log_message(f"Step 2 complete: FFT explorer ready")

    except Exception as e:
        QMessageBox.critical(self, "Error", f"Step 2 failed:\n{str(e)}")
        self.sr_step2_status.setText(f"❌ Error: {str(e)}")
        traceback.print_exc()

    finally:
        self.sr_step2_compute_btn.setEnabled(True)

def superres_step2_proceed(self):
    """Proceed from Step 2 to Step 3"""

    # Enable Step 3
    self.sr_step3_btn.setEnabled(True)
    self.sr_step3_status.setText("Ready - click to compute cross-correlations")

    # Scroll to Step 3 (optional - helps user see next step)
    # self.sr_workflow_scroll.ensureWidgetVisible(self.sr_step3_group)

    self.log_message("Ready for Step 3: Cross-correlations")
```

---

## Variable Naming Convention

To avoid conflicts with existing 4D FFT tab, prefix all super-res FFT explorer widgets with `sr_fft_`:

| 4D FFT Tab | Super-Res Step 2 |
|------------|------------------|
| `fft_overview_plot` | `sr_fft_overview_plot` |
| `fft_overview_item` | `sr_fft_overview_item` |
| `fft_subselect_plot` | `sr_fft_subselect_plot` |
| `fft_amp_plot` | `sr_fft_amp_plot` |
| `fft_phase_plot` | `sr_fft_phase_plot` |
| `fft_scan_selector_overview` | `sr_fft_scan_selector_overview` |
| `fft_freq_selector_amp` | `sr_fft_freq_selector_amp` |
| `fft4d_data` | (read from cache) |
| `update_fft_displays()` | `update_sr_fft_displays()` |

---

## Files Modified

1. **`mib_viewer_pyqtgraph.py`**:
   - `create_superres_tab()`: Add new Step 2 group
   - `create_sr_fft_explorer()`: NEW method
   - `setup_sr_fft_selectors()`: NEW method
   - `update_sr_fft_displays()`: NEW method
   - `update_sr_fft_subselection()`: NEW method
   - `on_sr_fft_scan_changed()`: NEW method
   - `on_sr_fft_freq_changed()`: NEW method
   - `superres_step2_compute_fft()`: NEW method (replaces old superres_step2_correlations)
   - `superres_step2_proceed()`: NEW method
   - Rename all `sr_step2_*` → `sr_step3_*` (old cross-correlation step)
   - Rename all `sr_step3_*` → `sr_step4_*` (old shift maps step)
   - Rename all `sr_step4_*` → `sr_step5_*` (old reconstruction step)

2. **`superres_processor.py`**:
   - ✅ Already added `compute_bigft_to_cache()` method

---

## Testing Plan

1. **Load 4D data and run Step 1**
   - Verify BF detection works as before

2. **Click "Compute 4D FFT"**
   - Verify cache file created
   - Verify progress updates
   - Verify FFT explorer appears

3. **Interact with FFT explorer**
   - Click on overview → verify amp/phase update
   - Click on amp → verify sub-selection updates
   - Verify crosshairs linked correctly
   - Verify log scaling applied

4. **Click "Proceed to Step 3"**
   - Verify Step 3 button enabled
   - Verify status updated

5. **Continue with Step 3**
   - Verify reads bigFT from cache
   - Verify cross-correlations computed correctly

6. **Close and reopen viewer**
   - Verify cache detection and resume capability (Task 6)

---

## Performance Considerations

**Cache File Reads:**
- Overview: `read_bigft([:, :, center_y, center_x])` → 256×256 complex = 0.5 MB
- Amp/Phase: `read_bigft([ky, kx, :, :])` → 64×64 complex = 32 KB
- Sub-select: `read_bigft([:, :, freq_y, freq_x])` → 256×256 complex = 0.5 MB

**Total per interaction:** ~1 MB read from HDF5 (should be fast with caching)

**HDF5 Read Optimization:**
- Uses chunked storage (32×32×32×32) for efficient slicing
- Gzip compression level 1 (minimal overhead)
- OS filesystem cache helps with repeated reads

---

## Estimated Effort

- **Part 1**: Modify Step 2 GUI structure → 30 min
- **Part 2**: Create FFT explorer widget → 1 hour
- **Part 3**: Add selectors and interaction → 1 hour
- **Part 4**: Update display logic with cache reading → 1 hour
- **Part 5**: Step methods → 30 min
- **Testing**: 1 hour

**Total: ~5 hours**

---

## Success Criteria

✅ New Step 2 group added with compute button
✅ FFT explorer appears after computation
✅ Can click on plots to explore FFT
✅ Crosshairs linked correctly
✅ Reads from cache file efficiently
✅ "Proceed" button enables Step 3
✅ bigFT written to cache correctly
✅ No interference with existing 4D FFT tab

---

**Status:** Ready for implementation
**Next:** Begin Part 1 implementation
