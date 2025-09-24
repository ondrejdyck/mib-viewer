# EMD Export Implementation for EELS + HAADF Data

## Overview

This document describes the implementation of unified EMD export functionality that enables users to save paired EELS spectroscopic data (.mib files) and HAADF structural data (.ndata1 files) into a single EMD file format. This transforms the workflow from "two separate files" to "single comprehensive EMD file" containing both structural and spectroscopic information with proper spatial registration.

## Problem Statement

Users had:
- **EELS data**: 4D spectroscopic datasets in .mib format
- **HAADF data**: 2D structural images in .ndata1 format (Nion's zip-based format)
- **Manual pairing**: Users had to manually correlate these complementary datasets
- **Separate storage**: No unified format to store both datasets together
- **Export duplication**: Multiple confusing export options in the GUI

## Solution Architecture

### Three Export Scenarios Supported

1. **HAADF only**: .ndata1 → EMD with `/data/images/image_000/` structure
2. **EELS only**: .mib → EMD with `/data/datacubes/datacube_000/` structure
3. **Combined**: Both datasets in single EMD with both image and datacube structures

### EMD File Structure Design

```
version_1/
├── data/
│   ├── datacubes/           # 4D EELS data (if present)
│   │   └── datacube_000/
│   │       ├── data         # 4D array (sy, sx, qy, qx)
│   │       ├── dim1-4       # Dimension datasets
│   │       └── attributes   # Spatial/spectral calibrations
│   └── images/              # 2D HAADF data (if present)
│       └── image_000/
│           ├── data         # 2D array (sy, sx)
│           ├── dim1-2       # Dimension datasets
│           └── attributes   # Spatial calibrations
├── metadata/
│   ├── eels_metadata/       # Original MIB metadata
│   ├── haadf_metadata/      # Processed Nion metadata
│   └── additional_metadata/ # GUI state (ROI, energy range, etc.)
└── log/                     # Conversion metadata
```

## Implementation Details

### Key Design Decisions

1. **Reuse existing GUI loader**: Enhanced existing .ndata1 loader instead of duplicating functionality
2. **Extend MibToEmdConverter**: Added new methods while preserving all existing functionality
3. **EMD 1.0 compliance**: Follows standard EMD structure for maximum compatibility
4. **Non-breaking changes**: All existing MIB→EMD and EMD→EMD functionality preserved

### Code Architecture

#### Enhanced GUI Loader (`mib_viewer_pyqtgraph.py`)

**Key Addition**: Store metadata from .ndata1 files
```python
# Added metadata storage
self.ndata_metadata = None  # Store ndata metadata for export

# Enhanced load_ndata_file() method
with zipfile.ZipFile(filename, 'r') as zip_file:
    # Load numpy data
    with zip_file.open('data.npy') as npy_file:
        self.ndata_data = np.load(npy_file)

    # Load metadata and store it (NEW)
    with zip_file.open('metadata.json') as json_file:
        self.ndata_metadata = json.load(json_file)
```

#### Extended Converter (`mib_to_emd_converter.py`)

**New Methods Added**:
- `process_ndata_metadata()`: Convert Nion metadata for EMD export
- `_write_image_dataset()`: Create 2D image datasets in EMD structure
- `_add_image_metadata()`: Store image metadata with spatial calibrations
- `export_display_data()`: Main export method for GUI-loaded data

**Key Method**: `export_display_data()`
```python
def export_display_data(self, output_path: str, display_data: Dict, metadata_extra: Optional[Dict] = None):
    """
    Export display data from GUI to EMD format
    Supports: HAADF only, EELS only, or combined HAADF+EELS
    """
    # Automatically detects what data is available
    # Creates appropriate EMD structure
    # Preserves all metadata from both sources
```

#### GUI Integration (`mib_viewer_pyqtgraph.py`)

**New Menu Option**: "Save EELS Display as EMD..."
```python
def save_eels_display_as_emd(self):
    """Save EELS display data (EELS and/or HAADF) to EMD format"""
    # Check available data
    has_eels = self.eels_data is not None
    has_ndata = self.ndata_data is not None

    # Prepare display_data dictionary
    display_data = {}
    if has_eels:
        display_data['eels_data'] = self.eels_data
        display_data['eels_metadata'] = {...}
    if has_ndata:
        display_data['haadf_data'] = self.ndata_data
        display_data['haadf_metadata'] = processed_metadata

    # Export using converter
    converter = MibToEmdConverter(log_callback=self.log_message)
    stats = converter.export_display_data(filename, display_data, metadata_extra)
```

## Code Cleanup Performed

### Removed Duplicate Functionality
- **Removed**: `analyze_ndata1_file()` and `load_ndata1()` methods from converter (duplicated GUI functionality)
- **Removed**: `_convert_2d_image()` method (consolidated into `export_display_data()`)
- **Removed**: Old `save_eels_as_emd()` method that used separate `EelsEmdSaver` class

### Menu Consolidation
- **Before**: "Save EELS as EMD..." and "Save Display as EMD..." (confusing)
- **After**: Single "Save EELS Display as EMD..." option (clear and specific)

### Workflow Simplification
- **Before**: Multiple code paths, file re-loading, duplicate logic
- **After**: Single export path using already-loaded GUI data

## Critical Code Files to Review Post-Autocompact

### 🔥 HIGHEST PRIORITY

1. **`/src/mib_viewer/gui/mib_viewer_pyqtgraph.py`**
   - **Lines ~210**: Metadata storage initialization (`self.ndata_metadata = None`)
   - **Lines ~1526-1537**: Enhanced `load_ndata_file()` with metadata storage
   - **Lines ~1549-1654**: Complete `save_eels_display_as_emd()` method
   - **Lines ~1168-1171**: Menu option creation

2. **`/src/mib_viewer/io/mib_to_emd_converter.py`**
   - **Lines ~297-358**: `process_ndata_metadata()` method
   - **Lines ~604-676**: `_write_image_dataset()` method
   - **Lines ~678-722**: `_add_image_metadata()` method
   - **Lines ~891-1164**: Complete `export_display_data()` method
   - **Lines ~739-741**: Updated file type routing to prevent ndata1 direct conversion

### 🔸 MEDIUM PRIORITY

3. **`/src/mib_viewer/io/mib_to_emd_converter.py`**
   - **Lines ~158-168**: Enhanced `detect_file_type()` with .ndata1 support
   - **Lines ~17**: Added imports (`json`, `zipfile`)

## Testing Scenarios

### Test Case 1: HAADF Only Export
1. Load .ndata1 file in GUI
2. Click "Save EELS Display as EMD..."
3. Verify EMD contains `/data/images/image_000/` with proper spatial calibrations

### Test Case 2: EELS Only Export
1. Load .mib file in GUI
2. Click "Save EELS Display as EMD..."
3. Verify EMD contains `/data/datacubes/datacube_000/` structure

### Test Case 3: Combined Export
1. Load both .mib and .ndata1 files in GUI
2. Click "Save EELS Display as EMD..."
3. Verify EMD contains both `/data/images/` and `/data/datacubes/` structures
4. Verify metadata preservation from both sources

## Usage Workflow

```
User loads .ndata1 (HAADF) ✓ [existing enhanced]
    ↓
User optionally loads .mib (EELS) ✓ [existing]
    ↓
User clicks "Save EELS Display as EMD..." ✓ [new]
    ↓
Single EMD file created with appropriate structure ✓ [new]
    ↓
Compatible with py4DSTEM, STEMTooL, etc. ✓
```

## Benefits Achieved

1. **Unified Storage**: Single EMD file contains both structural and spectroscopic data
2. **Metadata Preservation**: All original metadata preserved from both sources
3. **Standard Compliance**: EMD 1.0 format ensures broad compatibility
4. **No Duplication**: Clean, consolidated export code path
5. **User Clarity**: Single, clearly-named menu option
6. **Efficiency**: Uses already-loaded data instead of re-reading files

## Future Extensions

- **4D STEM Support**: Can easily add "Save 4D STEM Display as EMD..." for 4D STEM datasets
- **Progress Dialogs**: Can add progress callbacks to export method
- **Batch Export**: Could extend to export multiple datasets
- **Custom Metadata**: Could add user-defined metadata fields

## Dependencies

- **PyQt5**: GUI framework
- **h5py**: HDF5/EMD file operations
- **numpy**: Data array operations
- **json**: Metadata parsing
- **zipfile**: .ndata1 file access

---

**Implementation Status**: ✅ Complete and tested
**Breaking Changes**: ❌ None - all existing functionality preserved
**User Impact**: ➕ Positive - simplified workflow, new capabilities