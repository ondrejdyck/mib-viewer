# Large File Lazy Loading Implementation Plan

**Date**: September 16, 2025
**Goal**: Handle datasets larger than available memory (e.g., 130GB dataset on 64GB machine)

## 🎯 Core Strategy

### **Dual-Mode Architecture**
- **MIB Files**: Read-only lazy access with immediate basic functionality
- **EMD Files**: Enhanced with persistent caching for advanced computations
- **Philosophy**: Treat MIB as immutable source data, EMD as enhanced working format

### **Key Principle**: Progressive Enhancement
Files become "smarter" over time through cached computations, while preserving original data safety.

## 🏗️ Technical Architecture

### **Mode 1: MIB Read-Only Access**
```
MIB File (130GB) → Memory-mapped access
├── Individual Frames: Direct access data[y,x,:,:] (fast)
├── Scan Overview: On-the-fly chunked sum (moderate)
├── ROI Analysis: Computed as needed (moderate)
└── 4D FFT Explorer: "Convert to EMD first" (blocked)
```

**Benefits:**
- ✅ Immediate access to basic functionality
- ✅ Zero risk to original data
- ✅ Memory usage stays bounded
- ✅ Direct frame access is very fast with proper chunking

**Limitations:**
- ❌ No 4D FFT without conversion
- ❌ Recomputes overviews on each session
- ❌ Limited to views that can be computed on-the-fly

### **Mode 2: EMD Enhanced Access**
```
EMD File → Enhanced with cached computations
├── cached/scan_overview (persistent 2D overview)
├── cached/detector_integrated (persistent detector sum)
├── cached/fft_4d_complex (parallel computed, chunk-stored)
├── cached/fft_amplitude (derived from complex)
├── cached/fft_phase (derived from complex)
└── metadata/computation_log (versioning, validation)
```

**Benefits:**
- ✅ First computation cached permanently
- ✅ Subsequent loads nearly instant
- ✅ Full 4D FFT functionality available
- ✅ Files get smarter over time

## 🚀 Chunked 4D FFT Strategy

### **Core Concept**
Reuse the parallel processing architecture from the converter restoration:
- **Chunk the computation** across scan dimensions
- **Parallel workers** process chunks independently
- **Write results directly to file** (never hold full FFT in memory)
- **Lazy load results** on demand for visualization

### **Implementation Architecture**
```python
class ChunkedFFTProcessor:
    """Parallel 4D FFT computation with direct file storage"""

    def compute_and_store_fft(self):
        # 1. Create HDF5 dataset for FFT results
        # 2. Chunk input data optimally for memory budget
        # 3. Submit chunks to ThreadPoolExecutor
        # 4. Write results as they complete
        # 5. Add metadata for lazy loading
```

### **Memory Management**
- **Input chunking**: Process scan regions that fit in memory budget
- **No double buffering**: Read chunk → compute FFT → write result → discard
- **Optimal chunk sizing**: Reuse factor-based algorithm from converter
- **Memory-safe processing**: Never exceed calculated memory limits

### **Performance Characteristics**
- **First FFT computation**: Moderate time (parallel processing)
- **Subsequent access**: Fast (lazy loading from cached results)
- **Memory usage**: Bounded by chunk size, not dataset size
- **Scalability**: Handles arbitrary file sizes

## 📁 EMD File Structure Enhancement

```
dataset.emd
├── version_1/data/datacubes/datacube_000/
│   ├── data (original 4D dataset)
│   ├── dim1, dim2, dim3, dim4 (coordinate arrays)
│   └── [existing EMD structure]
├── cached/                     # ← NEW: Computed results cache
│   ├── scan_overview           # 2D: sum over detector dimensions
│   │   ├── data (2D array)
│   │   └── attrs: computation_date, source_shape
│   ├── detector_integrated/    # Various detector integrations
│   │   ├── sum_over_x (3D)
│   │   ├── sum_over_y (3D)
│   │   └── sum_over_both (2D)
│   ├── fft_4d/                 # 4D FFT results
│   │   ├── complex (4D complex64, chunked storage)
│   │   ├── amplitude (4D float32, computed from complex)
│   │   ├── phase (4D float32, computed from complex)
│   │   └── metadata/
│   │       ├── computation_params
│   │       ├── chunk_info
│   │       └── fft_axes_info
│   └── metadata/
│       ├── computation_log     # What's been computed when
│       ├── source_file_hash    # Validate cache validity
│       └── viewer_version      # Cache format version
└── [standard EMD structure continues]
```

### **Cache Validation Strategy**
- **Source hash checking**: Detect if original data changed
- **Version compatibility**: Handle viewer updates gracefully
- **Incremental computation**: Add new cached results over time
- **Selective invalidation**: Remove outdated cached results only

## 🔄 User Experience Flow

### **Opening Large MIB File**
```
User Action: Open "dataset.mib" (130GB)
├── System: "Analyzing file..." (immediate)
├── System: "Basic views ready" (1-2 seconds)
├── User: Browse scan overview, individual frames ✅
├── User: Clicks "4D FFT Explorer"
└── System: "4D FFT requires conversion to EMD. Convert now?"
    ├── User: "Yes" → Conversion workflow
    │   ├── Parallel conversion (reuse existing converter)
    │   ├── Optional: Compute common overviews during conversion
    │   └── Result: Enhanced EMD file
    └── User: "No" → Continue with basic functionality
```

### **4D FFT Computation Flow**
```
User: Clicks "Compute 4D FFT" (in EMD file)
├── System: Check for cached FFT
│   ├── Found: "Loading cached FFT..." (instant)
│   └── Not found: "Computing 4D FFT..."
│       ├── Chunked parallel computation (progress bar)
│       ├── Direct file storage (never fills memory)
│       └── "FFT computation complete, results cached"
└── User: Full FFT Explorer functionality available
```

### **Subsequent Opens**
```
User: Open "dataset.emd" (second time)
├── System: "Loading... found cached computations"
├── User: All views instantly available ✅
├── User: FFT Explorer → immediate access ✅
└── Performance: Near-instant startup
```

## 💾 Implementation Classes

### **Core Data Access Layer**
```python
class LazyDataManager:
    """Factory for appropriate data access strategy"""

    @staticmethod
    def create_viewer(file_path):
        if file_path.endswith('.mib'):
            return MIBLazyViewer(file_path)
        elif file_path.endswith(('.emd', '.h5')):
            return EMDEnhancedViewer(file_path)
        else:
            raise ValueError("Unsupported file format")

class MIBLazyViewer:
    """Read-only lazy access to MIB files"""
    - Memory-mapped data access
    - On-the-fly computation cache (RAM only)
    - Direct frame access
    - Chunked overview computation

class EMDEnhancedViewer:
    """Enhanced viewer with persistent caching"""
    - Cached computation discovery
    - Persistent overview storage
    - Lazy FFT result access
    - Cache invalidation management
```

### **Computation Layer**
```python
class ChunkedFFTProcessor:
    """Parallel 4D FFT with direct file storage"""
    - Reuse ThreadPoolExecutor pattern from converter
    - Memory-aware chunk sizing
    - Direct HDF5 dataset writing
    - Progress reporting integration

class OverviewComputer:
    """Compute and cache common overview images"""
    - Scan integrated views
    - Detector summed views
    - Statistical summaries
    - Histogram data for contrast
```

### **Cache Management Layer**
```python
class EMDCacheManager:
    """Manage cached computations in EMD files"""
    - Cache discovery and validation
    - Incremental computation scheduling
    - Version compatibility checking
    - Storage space management

class LazyFFTAccessor:
    """Lazy loading interface for FFT results"""
    - On-demand slice loading
    - Memory-efficient data access
    - Amplitude/phase computation from complex
    - ROI extraction from cached results
```

## 🎯 Implementation Phases

### **Phase 1: MIB Read-Only Foundation** ⭐ (High Impact)
**Goal**: Immediate functionality for large MIB files

**Tasks**:
1. Implement `MIBLazyViewer` with memory-mapped access
2. Add chunked scan overview computation
3. Integrate with existing GUI (detect large files, switch modes)
4. Add "Convert for advanced features" messaging
5. Test with 130GB dataset

**Deliverable**: Can immediately view large MIB files with basic functionality

### **Phase 2: EMD Cache Infrastructure** ⭐⭐ (Foundation)
**Goal**: Persistent caching system for EMD files

**Tasks**:
1. Design EMD file structure enhancement
2. Implement `EMDCacheManager` for cache discovery
3. Add overview computation and storage
4. Integrate cache validation (hashing, versioning)
5. Update converter to optionally generate overviews

**Deliverable**: EMD files become "smarter" over time

### **Phase 3: Chunked 4D FFT** ⭐⭐⭐ (Advanced Feature)
**Goal**: Full 4D FFT functionality for large files

**Tasks**:
1. Implement `ChunkedFFTProcessor` using ThreadPoolExecutor pattern
2. Design optimal chunking strategy for FFT computation
3. Add direct HDF5 writing with proper chunking
4. Implement `LazyFFTAccessor` for result visualization
5. Integrate with existing FFT Explorer interface

**Deliverable**: Complete 4D FFT functionality regardless of file size

### **Phase 4: Optimization & Polish** ⭐ (Performance)
**Goal**: Production-ready performance and user experience

**Tasks**:
1. Performance optimization (chunk sizes, caching strategies)
2. Memory usage profiling and optimization
3. Progress reporting and user feedback improvements
4. Error handling and recovery for interrupted computations
5. Documentation and user guide

**Deliverable**: Production-ready large file handling

## 🧪 Testing Strategy

### **Test Cases**
1. **130GB MIB file**: Verify immediate basic functionality
2. **EMD conversion**: Verify cached computation persistence
3. **Memory constraints**: Test on 32GB, 64GB, 128GB systems
4. **4D FFT accuracy**: Verify chunked FFT matches full computation
5. **Cache invalidation**: Verify proper cache management
6. **Concurrent access**: Multiple viewers, file locking

### **Performance Benchmarks**
- **MIB basic views**: < 5 seconds startup time
- **EMD cached access**: < 2 seconds for any view
- **4D FFT first computation**: Reasonable progress, complete successfully
- **4D FFT cached access**: < 5 seconds startup time
- **Memory usage**: Never exceed 80% of available RAM

## 🔧 Integration Points

### **Existing Code Reuse**
- **Parallel processing**: Reuse ThreadPoolExecutor pattern from converter
- **Memory management**: Reuse chunking algorithms from converter
- **Progress reporting**: Reuse callback system from conversion
- **GUI integration**: Extend existing plot widgets with lazy loading

### **New Dependencies**
- **Memory mapping**: May need platform-specific optimizations
- **HDF5 chunking**: Advanced chunking strategies for FFT storage
- **Cache management**: Robust file locking and concurrent access

## 🎯 Success Metrics

### **Functional Goals**
- ✅ Can open and explore 130GB+ files on 64GB machine
- ✅ All existing functionality preserved
- ✅ 4D FFT works for arbitrarily large files
- ✅ Subsequent opens are dramatically faster

### **Performance Goals**
- ✅ Memory usage bounded regardless of file size
- ✅ Startup time acceptable (< 10 seconds for basic views)
- ✅ FFT computation reasonable (progress indication, completion)
- ✅ Cache hits provide near-instant access

### **User Experience Goals**
- ✅ Clear workflow (MIB → basic, EMD → full)
- ✅ Transparent caching (files get smarter)
- ✅ Graceful degradation (large files still work)
- ✅ Progress feedback (users understand what's happening)

---

**Next Step**: Begin Phase 1 - MIB Read-Only Foundation to provide immediate value for the 130GB dataset challenge.

*This plan builds on the successful parallel processing restoration and provides a path to handle datasets of any size while preserving data safety and user experience.*