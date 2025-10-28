# Super-Resolution Feature - Release Notes (v1.1.0)

## Overview

MIB Viewer v1.1.0 introduces a powerful **super-resolution reconstruction feature** for 4D STEM data, enabling sub-pixel resolution enhancement through Fourier-space analysis and cross-correlation.

## Key Features

### 1. Interactive 4-Step Workflow

The super-resolution feature is integrated into the GUI as a dedicated tab with a clear, guided workflow:

**Step 1: Select BF Region**
- Interactive selection of bright-field (BF) detector region
- Visual feedback with overlay on diffraction pattern
- Adjustable center position and radius

**Step 2: Compute 4D FFT**
- Memory-efficient computation with HDF5 caching
- Interactive FFT explorer for visualization
- Real-time updates with probe position
- **NEW: Intelligent cache detection** - prevents duplicate 50+ GB cache files

**Step 3: Compute Cross-Correlations**
- Automated cross-correlation calculation
- Reference image and example correlation display
- Progress tracking

**Step 4: Compute Shift Maps**
- Sub-pixel shift map generation
- Visualization of x/y shift components

**Step 5: Reconstruct Image**
- Final super-resolved image reconstruction
- Side-by-side comparison with original

### 2. Memory-Efficient Caching System

- **HDF5-based caching** for large 4D datasets
- **Incremental computation** - resume from any step
- **Metadata tracking** - stores all parameters for reproducibility
- **Cache validation** - ensures parameter consistency

### 3. Cache Detection Dialog (NEW in v1.1.0)

When clicking "Compute 4D FFT" with existing cache files:
- **Automatic detection** of existing cache files
- **Interactive dialog** with three options:
  - **Load from Cache** - Instant resume (saves hours!)
  - **Recompute** - Create fresh cache
  - **Cancel** - Return without action
- **Cache metadata display** - size, date, completion status
- **Parameter validation** - warns if settings don't match

### 4. FFT Explorer

Interactive visualization tool for exploring Fourier space:
- **Real-time updates** with probe position
- **Multiple views**: Amplitude, phase, power spectrum
- **Colormap controls** with context menus
- **Zoom and pan** capabilities

## Technical Details

### Algorithm

The super-resolution reconstruction uses:
1. **Fourier-space analysis** - 4D FFT of cropped diffraction patterns
2. **Cross-correlation** - Between reference and individual patterns
3. **Sub-pixel registration** - Shift map calculation
4. **Image reconstruction** - Shift-corrected averaging

### Performance

- **Memory-efficient**: Processes data in chunks, never loads full 4D array
- **Cached computation**: HDF5 files enable instant resume
- **Progress tracking**: Real-time feedback for long operations
- **Typical cache size**: 50-100 GB for 256×256 scan

### File Format

Cache files are stored as HDF5 with structure:
```
cache_file.h5
├── metadata/           # Parameters, timestamps, completion status
├── cropped_data/       # Cropped diffraction patterns
├── bigft/             # 4D FFT results
├── correlations/      # Cross-correlation data
├── reference_image/   # Reference BF image
└── shift_maps/        # Sub-pixel shift maps (x, y components)
```

## Usage Instructions

### Basic Workflow

1. **Load 4D STEM data** (EMD format)
2. **Switch to Super-Resolution tab**
3. **Step 1**: Click "Select BF Region" and adjust center/radius
4. **Step 2**: Click "Compute 4D FFT"
   - If cache exists, choose Load/Recompute/Cancel
   - Explore FFT with interactive viewer
5. **Step 3**: Click "Compute Cross-Correlations"
6. **Step 4**: Click "Compute Shift Maps"
7. **Step 5**: Click "Reconstruct Image"

### Tips

- **Start with small datasets** (64×64 or 128×128) to test parameters
- **Cache files are large** - ensure sufficient disk space
- **Use cache detection** - always choose "Load from Cache" to save time
- **Adjust BF radius** - larger radius = more signal, but slower
- **Reference smoothing** - adjust if needed for noisy data

### Cache Management

Cache files are automatically created in the same directory as the EMD file with naming:
```
{basename}_superres_cache_{timestamp}.h5
```

To manage caches:
- **Delete old caches** manually if disk space is limited
- **Keep most recent** for quick resume
- **Archive important results** before deleting caches

## Known Limitations

### ADF/DF Super-Resolution (Experimental)

- **Status**: Code implemented but NOT integrated into GUI
- **Reason**: Alignment issues with noise in annular detector data
- **Impact**: No user-facing issues - feature is disabled
- **Future**: May be enabled if alignment algorithm improves

### Memory Requirements

- **Minimum**: 8 GB RAM for small datasets (64×64)
- **Recommended**: 16+ GB RAM for typical datasets (256×256)
- **Cache storage**: 50-100 GB per dataset

### Dataset Compatibility

- **Format**: EMD files with 4D STEM data
- **Detector**: Works best with centered BF disk
- **Scan size**: Tested up to 256×256 (larger may work but slower)

## Troubleshooting

### "Cache file not found" error
- Ensure EMD file is in the same location as when cache was created
- Check cache file exists in same directory as EMD

### "Cache parameters don't match" warning
- Step 1 settings changed since cache creation
- Choose "Recompute" to create new cache with current settings

### Out of memory errors
- Reduce scan size or crop region
- Close other applications
- Increase system RAM

### Slow performance
- Use cache detection to resume from previous computation
- Reduce BF radius in Step 1
- Use smaller scan sizes for testing

## Migration from Previous Versions

If upgrading from v1.0.x:
- **No breaking changes** - existing workflows continue to work
- **New cache detection** - will automatically detect old cache files
- **Old caches compatible** - can load caches from v1.0.12+

## Future Enhancements

Planned improvements:
- [ ] ADF super-resolution (if alignment issues resolved)
- [ ] Batch processing for multiple datasets
- [ ] Export shift maps as separate files
- [ ] GPU acceleration for FFT computation
- [ ] Automatic parameter optimization
- [ ] Cache compression to reduce file sizes

## Credits

Super-resolution algorithm based on Fourier-space cross-correlation techniques commonly used in electron microscopy.

Implementation by Ondrej Dyck, 2025.

## References

For more technical details, see:
- `SUPERRES_CACHE_IMPLEMENTATION_PLAN.md` - Cache system design
- `TASK2_STEP2_FFT_EXPLORER_PLAN.md` - FFT explorer implementation
- `ADF_SUPERRES_RESULTS.md` - ADF experiments (not GUI-integrated)

---

**Version**: 1.1.0  
**Release Date**: October 28, 2025  
**Branch**: feature/super-resolution → main
