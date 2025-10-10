# Weekend Status Note - Super-Resolution Work

**Date:** Friday, October 10, 2025  
**Status:** Feature branch created, exploring annular detector super-resolution

---

## What We Accomplished This Week

### 1. Super-Resolution Feature Implementation ✅
- **Completed** full super-resolution tab in GUI with 4-step workflow
- **Completed** memory-efficient caching system using HDF5
- **Completed** all core algorithms (FFT, correlation, shift maps, reconstruction)
- **Fixed** coordinate system issues and UI bugs
- **Tested** successfully on appropriate dataset (SS_a7_2ms_256x256)

### 2. Git Status ✅
- Created feature branch: `feature/super-resolution`
- Committed all changes (5,347 lines added!)
- Main branch remains clean
- Commit hash: `12882cc`

### 3. Key Discovery 🎯
**The algorithm works, but requires the RIGHT dataset!**
- Needs sufficient probe overlap
- Needs good SNR in BF disk
- Dataset `SS_a7_2ms_256x256 4D_16nmFoV -20nmFocus.emd` works perfectly
- Dataset `1_256x256_2msec_graphene8x8.emd` doesn't work (wrong experimental conditions)

---

## Current Experiment: Annular Detector Super-Resolution

### The Big Idea 💡
**Can we do super-resolution on ANNULAR DARK FIELD (ADF) detector regions?**

Instead of using the bright field disk (center pixels), use the annular region (high-angle scattered electrons) to get:
- Super-resolution Z-contrast imaging (HAADF)
- Complementary information to BF super-res
- Potentially better for certain samples

### The Breakthrough Insight 🌟
**Use a VIRTUAL ADF reference image!**

Just like BF uses the central pixel as reference, we should:
1. Sum all pixels in the annular region → creates "virtual ADF detector"
2. Use this virtual ADF image as the reference for cross-correlation
3. Correlate each individual annular pixel against this reference

This is brilliant because:
- Same contrast mechanism (Z-contrast)
- Better SNR (averaging over many pixels)
- Rotationally symmetric (no arbitrary choice)
- Exactly analogous to what BF center pixel does

### Current Status: Memory Issue 🔴

**Problem:** Script `test_superres_annular.py` runs out of memory

**Why:** 
- Loading full 256x256x256x256 dataset (~8GB)
- Computing FFT on full dataset in memory
- Need to either:
  - Use smaller subset (128x128 scan)
  - Use cache-based approach (like GUI does)
  - Run on machine with more RAM

**What to try Monday:**

#### Option 1: Quick Test (Recommended First)
```python
# Modify test_superres_annular.py line ~115
# Change from:
data_4d = f['version_1/data/datacubes/datacube_000/data'][:]

# To:
data_4d = f['version_1/data/datacubes/datacube_000/data'][:128, :128, :, :]
```
This loads only 128x128 scan (1/4 the data) - should fit in memory

#### Option 2: Use Cache-Based Approach
Modify the script to use `SuperResCacheManager` like the GUI does:
- Create cache file
- Process in chunks
- Won't run out of memory

#### Option 3: Run on Different Machine
If you have access to a machine with more RAM (32GB+), run there

---

## Files to Remember

### New Files Created:
- `test_superres_annular.py` - **THE EXPERIMENT** (needs memory fix)
- `test_superres_real.py` - Working BF super-res test
- `src/mib_viewer/processing/superres_processor.py` - Core algorithm
- `src/mib_viewer/processing/superres_cache.py` - Cache management

### Modified Files:
- `src/mib_viewer/gui/mib_viewer_pyqtgraph.py` - Super-res tab

### Documentation:
- `SUPERRES-TAB-REDESIGN.md`
- `SUPERRES_CACHE_IMPLEMENTATION_PLAN.md`

---

## Quick Start for Monday

### To Resume Annular Detector Experiment:

1. **Activate environment:**
   ```bash
   cd "/media/o2d/data/ORNL Dropbox/Ondrej Dyck/Code/mib-viewer-project/mib-viewer"
   . .venv/bin/activate
   ```

2. **Check git status:**
   ```bash
   git branch  # Should show: * feature/super-resolution
   git log --oneline -1  # Should show: 12882cc Add super-resolution...
   ```

3. **Fix memory issue in test_superres_annular.py:**
   - Edit line ~115 to load smaller subset (see Option 1 above)
   - Or implement cache-based approach

4. **Run test:**
   ```bash
   python test_superres_annular.py
   ```

5. **Check results:**
   - Look at `superres_annular_test.png`
   - Check if shift maps show coherent patterns
   - Compare quality values to BF (might be lower, that's OK)
   - See if super-res ADF image looks reasonable

---

## Expected Outcomes

### If It Works 🎉
- Shift maps should show smooth patterns (like BF does)
- Quality values might be lower than BF (0.3-0.5 instead of 0.7-0.9)
- Super-res ADF image should show enhanced resolution
- **This would be REALLY COOL** - super-res Z-contrast imaging!

### If It Doesn't Work 😕
- Shift maps might be noisy/random
- Quality values very low (<0.1)
- Reconstruction might be blank or garbage

**Possible reasons:**
- ADF correlations too weak (fewer electrons)
- Different aberration patterns at high angles
- Need different processing parameters

**What to try:**
- Lower quality threshold (try 0.1 instead of 0.7)
- More smoothing on reference
- Different annular region (try different radii)

---

## Questions to Explore

1. **Do ADF shift maps match BF shift maps?**
   - Run both BF and ADF super-res
   - Compare the shift patterns
   - If similar → aberrations affect all angles similarly
   - If different → interesting physics!

2. **What's the optimal annular region?**
   - Try different inner/outer radii
   - Find sweet spot for correlation quality

3. **Can we combine BF and ADF super-res?**
   - Get two super-res images with different contrast
   - Overlay them for multi-modal imaging

---

## Important Notes

- **Don't merge to main yet** - still experimental
- **Dataset matters** - use SS_a7 dataset for testing
- **Memory is tight** - 256x256 scan is pushing limits
- **Virtual ADF reference is the key insight** - this is the right approach

---

## Contact Info for Future You

If you're reading this Monday and confused:
1. Read the "Quick Start for Monday" section above
2. The key file is `test_superres_annular.py`
3. The key insight is "virtual ADF reference"
4. The key problem is "memory - use smaller subset"

**Good luck! This could be really cool if it works!** 🚀

---

## Bonus: If You Want to Show Someone

The working BF super-resolution is already impressive:
```bash
python test_superres_real.py
# Look at superres_test_real.png
```

This shows the algorithm works perfectly on BF data.
