---
title: "4D FFT Zero-Data Bug Fix"
date: 2025-10-28
base_commit: f34f6db
scope_paths:
  - src/mib_viewer/io/chunked_fft_processor.py
stale_policy.max_commits_since_base: 30
---

# 4D FFT Zero-Data Bug Fix

## Problem
When trying to recompute 4D FFT on a file with existing but corrupted FFT data (all zeros), the application crashes with:
```
Failed to compute 4D FFT:
'Unable to synchronously open object (component not found)'
```

## Root Cause
1. **Detection passes**: `detect_existing_fft()` only checks if dataset path exists, not if data is valid
2. **All-zero data**: FFT computation may have been interrupted, leaving zero-filled datasets
3. **No validation**: `load_fft_results()` doesn't validate data before loading
4. **Confusing error**: h5py error message doesn't clearly indicate the issue

## Solution
Added three layers of validation to `chunked_fft_processor.py`:

### 1. Enhanced Detection (lines 647-672)
**Before**: Only checked if dataset path exists
```python
if cropped_path in f:
    result['cropped'] = True
```

**After**: Samples data to verify it's not all zeros
```python
if cropped_path in f:
    dataset = f[cropped_path]
    if dataset.size > 0:
        # Sample first 10x10 pixels to check validity
        sample = dataset[0, 0, :min(10, dataset.shape[2]), :min(10, dataset.shape[3])]
        if not np.all(sample == 0):
            result['cropped'] = True
        else:
            self.log(f"Cropped FFT exists but appears to be all zeros (corrupted)", LogLevel.WARNING)
```

### 2. Dataset Existence Validation (lines 1293-1300)
**Added**: Check if dataset paths exist before attempting to load
```python
if dataset_path not in f:
    raise KeyError(f"FFT dataset not found at path: {dataset_path}\n"
                 f"The file may have corrupted FFT data. Try recomputing the FFT.")

if metadata_path not in f:
    raise KeyError(f"FFT metadata not found at path: {metadata_path}\n"
                 f"The file may have corrupted FFT data. Try recomputing the FFT.")
```

### 3. Zero-Data Validation (lines 1302-1306)
**Added**: Detect all-zero datasets after loading
```python
# Validate data is not all zeros (common corruption indicator)
if fft_data.size > 0 and np.all(fft_data == 0):
    raise ValueError(f"FFT data exists but contains only zeros.\n"
                   f"This indicates corrupted or incomplete FFT computation.\n"
                   f"Please recompute the FFT.")
```

## Benefits

1. **Early detection**: Corrupted FFT data detected during file opening
2. **Clear errors**: Specific error messages explain the problem
3. **Auto-recovery**: Detection now skips corrupted data, allowing recomputation
4. **User guidance**: Error messages tell user to recompute FFT

## User Experience

### Before Fix
```
User: *clicks Recompute FFT*
App: Failed to compute 4D FFT: 'Unable to synchronously open object (component not found)'
User: ??? What does that mean?
```

### After Fix
```
User: *opens EMD file with corrupted FFT*
App: [Log] Cropped FFT exists but appears to be all zeros (corrupted)
App: [Tab shows] "4D FFT" (not "4D FFT ✅")
User: *clicks Compute FFT*
App: *deletes corrupted data and recomputes successfully*
```

OR if user tries to load corrupted data:
```
App: FFT data exists but contains only zeros.
     This indicates corrupted or incomplete FFT computation.
     Please recompute the FFT.
```

## Testing Recommendations

1. **Test with corrupted file**: Open your EMD file with all-zero FFT data
   - Should see warning in log
   - Should NOT auto-load FFT
   - Should allow recomputation

2. **Test recomputation**: Click "Compute FFT" on file with corrupted data
   - Should delete old data
   - Should compute fresh FFT
   - Should display correctly

3. **Test with valid FFT**: Open EMD file with good FFT data
   - Should auto-load normally
   - Should show "4D FFT ✅" tab

## Files Modified

```
src/mib_viewer/io/chunked_fft_processor.py | 35 lines changed
```

## Related to File Inspector Port

The File Inspector we just added will be very useful for diagnosing these issues:
- Open "File Inspector" tab
- Navigate to `cached/fft_4d_cropped/complex` or `cached/fft_4d_full/complex`
- View dataset metadata (shape, dtype, size)
- Check statistics (min, max, mean) - will show all zeros if corrupted
