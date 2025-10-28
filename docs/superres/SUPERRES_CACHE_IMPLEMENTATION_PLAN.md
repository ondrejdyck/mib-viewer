# Super-Resolution Cache File Implementation Plan

**Date:** 2025-01-07
**Purpose:** Add 4D FFT exploration step and implement cache file strategy for super-resolution reconstruction

## Executive Summary

Restructure super-resolution workflow from 4 steps to 5 steps by adding an interactive 4D FFT exploration phase. Implement cache file system to store large intermediate arrays (bigFT and correlations) separately from main EMD file. This allows:
1. User to explore reciprocal space structure before cross-correlation
2. Resume interrupted computations
3. Keep main EMD file small (~8 MB per reconstruction)
4. User control over 34-68 GB cache file retention

---

## Current Workflow (Before Changes)

```
Step 1: Detect & Preview BF Disk
  ↓
Step 2: Compute Cross-Correlations
  ↓
Step 3: Find Shift Maps
  ↓
Step 4: Reconstruct Super-Res
```

**Problems:**
- No 4D FFT exploration (exists in separate tab)
- bigFT computed and immediately discarded
- All data stored in main EMD (would add 34-68 GB)
- No resume capability

---

## New Workflow (After Changes)

```
Step 1: Detect & Preview BF Disk
  ↓
Step 2: Compute & Explore 4D FFT ← NEW
  ↓
Step 3: Compute Cross-Correlations (modified to read from cache)
  ↓
Step 4: Find Shift Maps (renumbered from 3)
  ↓
Step 5: Reconstruct Super-Res (renumbered from 4)
```

**Benefits:**
- User can explore 4D FFT before proceeding
- bigFT stored in cache for Step 3
- Correlations stored in cache for Step 4
- Resume capability at any step
- Main EMD stays small

---

## Cache File Structure

### File Naming Convention
```
{base_emd_name}_superres_cache_{timestamp}.h5
```

Example: `mydata_superres_cache_20250107_153045.h5`

### HDF5 Internal Structure
```
/bigFT/
    /data [sy, sx, crop_dy, crop_dx] complex128 (~34 GB)
    /metadata/
        crop_bounds: [y1, y2, x1, x2]
        bf_center: [cy, cx]
        bf_radius: scalar
        computation_date: string
        reference_smoothing: scalar

/correlations/
    /data [sy, sx, crop_dy, crop_dx] float64 (~34 GB)
    /metadata/
        reference_pixel: [w_y, w_x]
        normalization_applied: bool

/metadata/
    linked_emd_path: string
    step_completed: int (1-5)
    original_data_shape: [sy, sx, dy, dx]
    cropped_data_shape: [sy, sx, crop_dy, crop_dx]
    version: "1.0"
```

### Size Estimates (256×256 scan, 64×64 crop)
- bigFT: 256×256×64×64 × 16 bytes (complex128) = 34.4 GB
- correlations: 256×256×64×64 × 8 bytes (float64) = 17.2 GB
- Total: ~52 GB
- After final step: Can delete bigFT, keep only correlations (17 GB)

---

## Main EMD File Structure (Unchanged Size)

```
mydata.emd
├── /version_1/data/datacubes/... [original data]
└── /superres_results/
    └── /run_20250107_153045/
        ├── parameters (attrs)
        │   ├── timestamp
        │   ├── bf_center: [cy, cx]
        │   ├── bf_radius: scalar
        │   ├── crop_bounds: [y1, y2, x1, x2]
        │   ├── upscale_factor: 4
        │   ├── reference_smoothing: 0.5
        │   ├── detector_radius: 16
        │   ├── quality_threshold: 0.8
        │   └── cache_file: "mydata_superres_cache_20250107_153045.h5"
        ├── shift_y [crop_dy, crop_dx] float64 (~100 KB)
        ├── shift_x [crop_dy, crop_dx] float64 (~100 KB)
        ├── quality [crop_dy, crop_dx] float64 (~100 KB)
        ├── superres_image [sy*fac, sx*fac] float64 (~8 MB for 4x)
        ├── standard_bf [sy, sx] float64 (~0.5 MB)
        └── reference_image [sy, sx] float64 (~0.5 MB)
```

**Total per run: ~10 MB in EMD**

---

## Implementation Tasks

### Task 1: Cache File Infrastructure

**Files to modify:**
- `mib-viewer/src/mib_viewer/processing/superres_processor.py`

**New components:**
1. **CacheFileManager class**
   ```python
   class SuperResCacheManager:
       def __init__(self, emd_path: str, timestamp: str = None)
       def create_cache_file(self, crop_info: dict, data_shape: tuple)
       def write_bigft(self, bigft: np.ndarray, chunk_info: ChunkInfo)
       def read_bigft(self, chunk_info: ChunkInfo = None) -> np.ndarray
       def write_correlations(self, correlations: np.ndarray, chunk_info: ChunkInfo)
       def read_correlations(self, chunk_info: ChunkInfo = None) -> np.ndarray
       def get_step_completed(self) -> int
       def set_step_completed(self, step: int)
       def delete_bigft(self)  # Optional space savings
       def close(self)
   ```

2. **Cache detection on data load**
   - Search for `{basename}_superres_cache_*.h5`
   - Parse metadata to determine resume point
   - Display status to user

3. **Cache cleanup utilities**
   - List all cache files for current EMD
   - Delete selected cache files
   - Auto-delete on app close (if enabled)

**Estimated effort:** 1-2 hours

---

### Task 2: New Step 2 - Compute & Explore 4D FFT

**Files to modify:**
- `mib-viewer/src/mib_viewer/gui/mib_viewer_pyqtgraph.py` (GUI)
- `mib-viewer/src/mib_viewer/processing/superres_processor.py` (computation)

**2.1 Computation Logic**

Modify `SuperResProcessor.compute_cross_correlations()`:
- Split into two methods: `compute_bigft()` and `compute_correlations_from_bigft()`

```python
def compute_bigft(self,
                  data_4d_cropped: np.ndarray,
                  cache_manager: SuperResCacheManager,
                  reference_smoothing: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 4D FFT over scan dimensions using adaptive chunking.
    Writes results to cache file incrementally.

    Uses AdaptiveChunkCalculator with:
    - chunk_detector_dims=True
    - Single-threaded for I/O optimization (Scenario B)

    Returns:
        reference_image: For diagnostics
        bigft_shape: Shape of computed bigFT
    """
```

**Key implementation details:**
- Use `AdaptiveChunkCalculator` from `adaptive_chunking.py`
- Analyze memory with `analyze_memory_strategy()`
- If Scenario B (memory constrained):
  - Single-threaded, large detector chunks
  - Write directly to cache file
- If Scenario A (fits in memory):
  - Multi-threaded computation to memory
  - Background write to cache file

**2.2 GUI Components**

Add to super-res tab layout:
```python
# Step 2: Compute & Explore 4D FFT
step2_group = QGroupBox("Step 2: Compute & Explore 4D FFT")
step2_layout = QVBoxLayout()

# Compute button
self.sr_step2_compute_btn = QPushButton("Compute 4D FFT")
self.sr_step2_compute_btn.clicked.connect(self.sr_compute_bigft)

# Progress/status
self.sr_step2_status = QLabel("Waiting for Step 1...")

# 4D FFT Explorer (same as current 4D FFT tab)
fft_explorer_layout = QHBoxLayout()

# Left: Plots (2x2 grid)
plots_widget = QWidget()
plots_layout = QGridLayout()
self.sr_fft_overview_plot = pg.PlotWidget(title="FFT Overview")
self.sr_fft_subselect_plot = pg.PlotWidget(title="Sub-selection")
self.sr_fft_amp_plot = pg.PlotWidget(title="Amplitude")
self.sr_fft_phase_plot = pg.PlotWidget(title="Phase")
plots_layout.addWidget(self.sr_fft_overview_plot, 0, 0)
plots_layout.addWidget(self.sr_fft_subselect_plot, 0, 1)
plots_layout.addWidget(self.sr_fft_amp_plot, 1, 0)
plots_layout.addWidget(self.sr_fft_phase_plot, 1, 1)

# Right: Controls
controls_widget = QWidget()
controls_layout = QVBoxLayout()
# ... (copy from current 4D FFT tab)

# Ready button (unlocks Step 3)
self.sr_step2_ready_btn = QPushButton("Proceed to Cross-Correlation")
self.sr_step2_ready_btn.clicked.connect(self.sr_unlock_step3)
self.sr_step2_ready_btn.setEnabled(False)
```

**Interaction flow:**
1. User clicks "Compute 4D FFT"
2. Computation runs with progress bar
3. bigFT written to cache file
4. FFT explorer becomes interactive
5. User browses FFT
6. User clicks "Proceed to Cross-Correlation"
7. Step 3 unlocks

**Estimated effort:** 3-4 hours

---

### Task 3: Modify Step 3 - Cross-Correlations from Cache

**Current code:**
```python
def compute_cross_correlations(self, data_4d_cropped, ...):
    # Step 1: FFT over scan dimensions
    bigFT = np.fft.fft2(data_4d_cropped, axes=(0, 1))  # ← Computed here

    # Step 2: Create reference
    central_image = gaussian_filter(...)
    central_slice = np.fft.fft2(central_image)

    # Step 3: Cross-correlation
    cross_corr_ft = bigFT * np.conj(central_slice)[...]

    # Step 4: IFFT back
    correlations = np.abs((np.fft.ifft2(...))**2)

    return correlations, central_image
```

**New code:**
```python
def compute_correlations_from_bigft(self,
                                   cache_manager: SuperResCacheManager,
                                   data_4d_cropped: np.ndarray,
                                   reference_smoothing: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cross-correlations from cached bigFT.
    Writes correlations to cache file incrementally.
    """

    # Step 1: Create reference (same as before)
    w_y, w_x = data_4d_cropped.shape[2] // 2, data_4d_cropped.shape[3] // 2
    central_image = gaussian_filter(data_4d_cropped[:, :, w_y, w_x], reference_smoothing)
    central_slice = np.fft.fft2(central_image)

    # Step 2: Process in chunks (same chunking strategy as bigFT computation)
    chunking_result = ...  # Use same chunking as Step 2

    for chunk in chunks:
        # Read bigFT chunk from cache
        bigft_chunk = cache_manager.read_bigft(chunk)

        # Cross-correlation in Fourier space
        cross_corr_ft = bigft_chunk * np.conj(central_slice)[...]

        # IFFT back
        corr_chunk = np.abs((np.fft.ifft2(cross_corr_ft, axes=(0,1)))**2)

        # Normalize and write to cache
        corr_chunk = fftshift(corr_chunk, axes=(0,1))
        # ... normalization ...
        cache_manager.write_correlations(corr_chunk, chunk)

    return correlations, central_image  # Return reference for diagnostics
```

**GUI changes:**
- Renumber button: "Step 2" → "Step 3: Compute Cross-Correlations"
- Update status label
- Button enabled only after Step 2 completes

**Estimated effort:** 1-2 hours

---

### Task 4: Renumber Steps 3 → 4, 4 → 5

**Simple renumbering:**

| Old | New | Method Name |
|-----|-----|-------------|
| Step 3: Find Shift Maps | Step 4: Find Shift Maps | `sr_compute_shifts()` |
| Step 4: Reconstruct | Step 5: Reconstruct | `sr_compute_reconstruction()` |

**GUI updates:**
- Update all labels: "Step 3" → "Step 4", "Step 4" → "Step 5"
- Update button enables/disables logic
- Update status tracking

**Implementation note:**
- Step 4 now reads correlations from cache instead of memory
- Add cache read in `find_shift_maps()` method
- All other logic unchanged

**Estimated effort:** 30 minutes

---

### Task 5: Settings & Cleanup UI

**5.1 Settings Panel**

Add to main window settings or super-res tab:
```python
# Cache file management
cache_group = QGroupBox("Cache File Management")
cache_layout = QVBoxLayout()

# Auto-delete checkbox
self.auto_delete_cache_checkbox = QCheckBox("Auto-delete cache files on close")
self.auto_delete_cache_checkbox.setChecked(False)  # Default: KEEP
self.auto_delete_cache_checkbox.setToolTip(
    "If checked, cache files are automatically deleted when closing the viewer.\n"
    "If unchecked, you'll be prompted to keep or delete cache files."
)

# Cache info label
self.cache_info_label = QLabel("No cache files")

# Manual cleanup button
self.cleanup_cache_btn = QPushButton("Manage Cache Files...")
self.cleanup_cache_btn.clicked.connect(self.show_cache_manager_dialog)
```

**5.2 Cache Manager Dialog**

```python
def show_cache_manager_dialog(self):
    """Show dialog to manage cache files"""
    dialog = QDialog(self)
    dialog.setWindowTitle("Manage Cache Files")

    layout = QVBoxLayout()

    # List of cache files
    cache_list = QListWidget()
    # ... populate with cache files and sizes ...

    # Buttons
    delete_selected_btn = QPushButton("Delete Selected")
    delete_all_btn = QPushButton("Delete All")
    close_btn = QPushButton("Close")

    # ... connect signals ...
```

**5.3 App Close Handler**

In `MIBViewerGUI.closeEvent()`:
```python
def closeEvent(self, event):
    # Check for active cache files
    cache_files = self.find_superres_cache_files()

    if cache_files:
        if self.auto_delete_cache_checkbox.isChecked():
            # Auto-delete all
            for cache_file in cache_files:
                os.remove(cache_file)
            self.log_message(f"Auto-deleted {len(cache_files)} cache file(s)")
        else:
            # Prompt user
            total_size_gb = sum([os.path.getsize(f) for f in cache_files]) / (1024**3)

            reply = QMessageBox.question(
                self,
                "Super-Resolution Cache Files",
                f"You have {len(cache_files)} cache file(s) ({total_size_gb:.1f} GB).\n\n"
                "These can be used to resume computations later.\n\n"
                "Delete now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No  # Default: keep
            )

            if reply == QMessageBox.Yes:
                for cache_file in cache_files:
                    os.remove(cache_file)

    # Continue with normal close
    event.accept()
```

**Estimated effort:** 1-2 hours

---

### Task 6: Resume Capability

**On data load**, check for existing cache:
```python
def load_stem4d_data_completed(self):
    # ... existing load logic ...

    # Check for cache files
    cache_files = self.find_superres_cache_files(self.stem4d_filename)

    if cache_files:
        # Found cache, show resume option
        cache_file = cache_files[0]  # Use most recent
        cache_manager = SuperResCacheManager.from_existing(cache_file)

        step_completed = cache_manager.get_step_completed()

        msg = QMessageBox.question(
            self,
            "Resume Super-Resolution?",
            f"Found cache from previous session (Step {step_completed} completed).\n\n"
            f"Resume from Step {step_completed + 1}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        if msg == QMessageBox.Yes:
            self.resume_superres_from_cache(cache_manager)
        else:
            # Start fresh option (delete old cache?)
            pass
```

**Resume logic:**
```python
def resume_superres_from_cache(self, cache_manager):
    """Restore state from cache"""
    metadata = cache_manager.get_metadata()

    # Restore crop info
    self.superres_crop_info = {
        'center': metadata['bf_center'],
        'radius': metadata['bf_radius'],
        'bounds': metadata['crop_bounds']
    }

    # Update UI state based on step_completed
    step = cache_manager.get_step_completed()

    if step >= 1:
        # Step 1 complete: show BF detection results
        self.sr_step1_complete(...)

    if step >= 2:
        # Step 2 complete: bigFT ready, load FFT visualization
        self.sr_load_bigft_from_cache(cache_manager)
        self.sr_step2_ready_btn.setEnabled(True)

    if step >= 3:
        # Step 3 complete: correlations ready
        self.sr_step3_status.setText("✓ Correlations computed")
        self.sr_step4_btn.setEnabled(True)

    # ... etc ...
```

**Estimated effort:** 1-2 hours

---

## Testing Plan

### Test 1: Fresh Computation (No Cache)
1. Load 4D STEM data
2. Step 1: Detect BF disk
3. Step 2: Compute 4D FFT
   - Verify cache file created
   - Verify bigFT written correctly
   - Verify FFT visualization works
4. Step 3: Compute cross-correlations
   - Verify reads bigFT from cache
   - Verify writes correlations to cache
5. Step 4: Find shift maps
   - Verify reads correlations from cache
6. Step 5: Reconstruct
   - Verify final result saved to EMD
7. Close app
   - Verify cleanup prompt appears
   - Test both keep and delete options

### Test 2: Resume from Cache
1. Complete Steps 1-2
2. Close app (keep cache)
3. Reopen app, load same data
4. Verify resume prompt appears
5. Accept resume
6. Verify Step 2 results loaded correctly
7. Continue with Step 3
8. Verify computation resumes correctly

### Test 3: Memory Scenarios
1. Test with small dataset (Scenario A - fits in memory)
   - Verify multi-threaded processing
2. Test with large dataset (Scenario B - memory constrained)
   - Verify single-threaded with large chunks
   - Verify no memory exhaustion

### Test 4: Cache Management
1. Create multiple cache files (different timestamps)
2. Open cache manager dialog
3. Verify all caches listed with sizes
4. Delete selected cache
5. Verify deletion successful

### Test 5: Error Handling
1. Interrupt computation mid-step
2. Verify cache metadata reflects partial completion
3. Delete cache file manually while app running
4. Verify app handles missing cache gracefully
5. Corrupt cache file
6. Verify error message and fallback to fresh computation

---

## File Organization Summary

```
mib-viewer/src/mib_viewer/
├── processing/
│   ├── superres_processor.py           [MODIFIED - split methods, add cache I/O]
│   └── superres_cache.py               [NEW - cache file manager]
├── gui/
│   └── mib_viewer_pyqtgraph.py         [MODIFIED - new Step 2, renumber, cleanup UI]
└── io/
    └── adaptive_chunking.py            [EXISTING - reuse for chunking strategy]
```

---

## Implementation Timeline

| Task | Estimated Time | Priority |
|------|---------------|----------|
| 1. Cache File Infrastructure | 1-2 hours | HIGH |
| 2. New Step 2 GUI | 3-4 hours | HIGH |
| 3. New Step 2 Computation | 2-3 hours | HIGH |
| 4. Modify Step 3 | 1-2 hours | HIGH |
| 5. Renumber Steps 3→4, 4→5 | 30 min | MEDIUM |
| 6. Settings & Cleanup UI | 1-2 hours | MEDIUM |
| 7. Resume Capability | 1-2 hours | LOW |
| 8. Testing | 2-3 hours | HIGH |

**Total: 12-18 hours**

---

## Future Enhancements (Out of Scope)

1. **Automatic bigFT deletion after Step 3**
   - Save 34 GB by deleting bigFT after no longer needed
   - Requires disabling "go back" functionality

2. **Cloud storage integration**
   - Upload cache files to cloud storage
   - Resume on different machines

3. **Compressed cache storage**
   - Use compression (gzip level 1) to reduce cache size
   - Trade-off: slower I/O

4. **Multiple reconstruction runs**
   - Support multiple parameter sets in one session
   - Each gets separate cache file

5. **Cache analytics**
   - Show disk space usage
   - Estimate remaining space needed
   - Warn before running out of space

---

## Open Questions

1. **Should we allow "go back" to previous steps?**
   - If yes: must keep all cache data
   - If no: can delete intermediate data for space savings
   - **Decision: For v1, allow go back (keep all data)**

2. **What if user changes parameters mid-workflow?**
   - Invalidate cache and start fresh?
   - Allow parameter adjustment before Step 2?
   - **Decision: For v1, disable parameter changes after Step 1**

3. **Should Step 2 FFT exploration be optional?**
   - Add "Skip to Step 3" button?
   - Auto-proceed after timeout?
   - **Decision: For v1, require user to click "Proceed" button**

4. **Cache file location**
   - Same directory as EMD?
   - User-configurable temp directory?
   - **Decision: For v1, same directory as EMD**

---

## Success Criteria

✅ User can compute super-resolution in 5 steps
✅ bigFT and correlations stored in separate cache file
✅ Main EMD stays small (~10 MB per run)
✅ User can explore 4D FFT before cross-correlation
✅ Cache persists across sessions
✅ Resume capability works
✅ User can delete cache files easily
✅ No memory exhaustion on 64GB system
✅ Existing 4D FFT tab untouched

---

## Notes

- This plan assumes adaptive chunking logic from `adaptive_chunking.py` works correctly
- Memory analysis from `chunked_fft_processor.py` should be reused
- All existing Step 3-4 logic should work unchanged (just renumbered)
- 4D FFT tab remains functional for other analyses

---

**Plan Status:** Draft for review
**Next Step:** Review with user, then begin Task 1 implementation
