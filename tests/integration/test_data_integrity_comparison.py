#!/usr/bin/env python3
"""
Data Integrity Comparison Test

This test validates that both the original and V2 converters preserve data integrity
by comparing 1000 random positions between the source MIB file and converted EMD outputs.
"""

import sys
import os
from pathlib import Path
import numpy as np
import h5py
import random

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mib_viewer.io.mib_loader import load_mib

def test_data_integrity():
    """
    Test data integrity by comparing 1000 random positions between:
    1. Original MIB file (ground truth)
    2. Fixed original converter output
    3. V2 converter output
    """

    print("="*70)
    print("DATA INTEGRITY COMPARISON TEST")
    print("="*70)

    # File paths
    mib_path = "/media/o2d/data/ORNL Dropbox/Ondrej Dyck/Code/mib-viewer-project/Example 4D/64x64 Test.mib"
    original_emd = "/media/o2d/data/ORNL Dropbox/Ondrej Dyck/Code/mib-viewer-project/Example 4D/test_outputs/original_converter_output.emd"
    fixed_original_emd = "/media/o2d/data/ORNL Dropbox/Ondrej Dyck/Code/mib-viewer-project/Example 4D/test_outputs/fixed_original_converter_output.emd"
    v2_emd = "/media/o2d/data/ORNL Dropbox/Ondrej Dyck/Code/mib-viewer-project/Example 4D/test_outputs/v2_converter_output.emd"

    # Verify all files exist
    for file_path, name in [(mib_path, "MIB"), (original_emd, "Original EMD"), (fixed_original_emd, "Fixed Original EMD"), (v2_emd, "V2 EMD")]:
        if not os.path.exists(file_path):
            print(f"❌ {name} file not found: {file_path}")
            return False

    try:
        # Step 1: Load original MIB data (ground truth)
        print("\n📖 Loading original MIB file...")
        mib_data = load_mib(mib_path)
        mib_shape = mib_data.shape
        print(f"   MIB shape: {mib_shape}")

        # Step 2: Load all EMD files
        print("\n📖 Loading converted EMD files...")
        with h5py.File(original_emd, 'r') as f:
            original_data = f['version_1/data/datacubes/datacube_000/data'][:]
            original_shape = original_data.shape

        with h5py.File(fixed_original_emd, 'r') as f:
            fixed_original_data = f['version_1/data/datacubes/datacube_000/data'][:]
            fixed_original_shape = fixed_original_data.shape

        with h5py.File(v2_emd, 'r') as f:
            v2_data = f['version_1/data/datacubes/datacube_000/data'][:]
            v2_shape = v2_data.shape

        print(f"   Original converter shape: {original_shape}")
        print(f"   Fixed original converter shape: {fixed_original_shape}")
        print(f"   V2 converter shape: {v2_shape}")

        # Step 3: Check which shapes match MIB
        shape_matches = {
            'Original': mib_shape == original_shape,
            'Fixed Original': mib_shape == fixed_original_shape,
            'V2': mib_shape == v2_shape
        }

        print(f"\n📊 Shape compatibility check:")
        print(f"   MIB: {mib_shape}")
        print(f"   Original: {original_shape} {'✅' if shape_matches['Original'] else '❌'}")
        print(f"   Fixed Original: {fixed_original_shape} {'✅' if shape_matches['Fixed Original'] else '❌'}")
        print(f"   V2: {v2_shape} {'✅' if shape_matches['V2'] else '❌'}")

        # Only test converters with matching shapes
        if not shape_matches['Fixed Original']:
            print("❌ Fixed Original converter still has shape issues!")
            return False

        sy, sx, qy, qx = mib_shape

        # Step 4: Generate 1000 random test positions
        print(f"\n🎯 Generating 1000 random test positions...")
        random.seed(42)  # Reproducible random positions
        test_positions = []

        for i in range(1000):
            scan_y = random.randint(0, sy - 1)
            scan_x = random.randint(0, sx - 1)
            det_y = random.randint(0, qy - 1)
            det_x = random.randint(0, qx - 1)
            test_positions.append((scan_y, scan_x, det_y, det_x))

        print(f"   Sample positions: {test_positions[:5]}...")

        # Step 5: Extract values from all three sources
        print(f"\n🔍 Extracting values from all sources...")

        mib_values = []
        original_values = []
        fixed_original_values = []
        v2_values = []

        for scan_y, scan_x, det_y, det_x in test_positions:
            mib_val = mib_data[scan_y, scan_x, det_y, det_x]
            mib_values.append(mib_val)

            # Only sample from converters with correct shapes
            if shape_matches['Original'] and scan_y < original_shape[0] and scan_x < original_shape[1] and det_y < original_shape[2] and det_x < original_shape[3]:
                orig_val = original_data[scan_y, scan_x, det_y, det_x]
                original_values.append(orig_val)
            else:
                original_values.append(-1)  # Invalid marker

            if shape_matches['Fixed Original']:
                fixed_orig_val = fixed_original_data[scan_y, scan_x, det_y, det_x]
                fixed_original_values.append(fixed_orig_val)
            else:
                fixed_original_values.append(-1)

            if shape_matches['V2'] and scan_y < v2_shape[0] and scan_x < v2_shape[1] and det_y < v2_shape[2] and det_x < v2_shape[3]:
                v2_val = v2_data[scan_y, scan_x, det_y, det_x]
                v2_values.append(v2_val)
            else:
                v2_values.append(-1)

        # Step 6: Compare values
        print(f"\n📊 Comparing values...")

        # Compare MIB vs Original converter
        original_matches = 0
        for i, (mib_val, orig_val) in enumerate(zip(mib_values, original_values)):
            if orig_val != -1 and mib_val == orig_val:
                original_matches += 1
            elif orig_val != -1 and i < 5:  # Show first 5 mismatches for debugging
                pos = test_positions[i]
                print(f"   Original mismatch at {pos}: MIB={mib_val}, Original={orig_val}")

        # Compare MIB vs Fixed Original converter
        fixed_original_matches = 0
        for i, (mib_val, fixed_orig_val) in enumerate(zip(mib_values, fixed_original_values)):
            if fixed_orig_val != -1 and mib_val == fixed_orig_val:
                fixed_original_matches += 1
            elif fixed_orig_val != -1 and i < 5:  # Show first 5 mismatches for debugging
                pos = test_positions[i]
                print(f"   Fixed Original mismatch at {pos}: MIB={mib_val}, Fixed={fixed_orig_val}")

        # Compare MIB vs V2 converter
        v2_matches = 0
        for i, (mib_val, v2_val) in enumerate(zip(mib_values, v2_values)):
            if v2_val != -1 and mib_val == v2_val:
                v2_matches += 1
            elif v2_val != -1 and i < 5:  # Show first 5 mismatches for debugging
                pos = test_positions[i]
                print(f"   V2 mismatch at {pos}: MIB={mib_val}, V2={v2_val}")

        # Step 7: Report results
        print(f"\n" + "="*70)
        print("INTEGRITY COMPARISON RESULTS")
        print("="*70)

        original_accuracy = (original_matches / 1000) * 100
        fixed_original_accuracy = (fixed_original_matches / 1000) * 100
        v2_accuracy = (v2_matches / 1000) * 100

        print(f"Original Converter vs MIB:      {original_matches}/1000 matches ({(original_matches/1000)*100:.2f}%)")
        print(f"Fixed Original Converter vs MIB: {fixed_original_matches}/1000 matches ({(fixed_original_matches/1000)*100:.2f}%)")
        print(f"V2 Converter vs MIB:            {v2_matches}/1000 matches ({(v2_matches/1000)*100:.2f}%)")

        # Step 8: Analysis and conclusions
        print(f"\n" + "="*70)
        print("ANALYSIS")
        print("="*70)

        if original_accuracy >= 99.0:
            print("✅ Original Converter: EXCELLENT data integrity")
        elif original_accuracy >= 95.0:
            print("⚠️ Original Converter: Good data integrity (minor issues)")
        else:
            print("❌ Original Converter: Poor data integrity (significant scrambling)")

        if fixed_original_accuracy >= 99.0:
            print("✅ Fixed Original Converter: EXCELLENT data integrity")
        elif fixed_original_accuracy >= 95.0:
            print("⚠️ Fixed Original Converter: Good data integrity (minor issues)")
        else:
            print("❌ Fixed Original Converter: Poor data integrity (significant scrambling)")

        if v2_accuracy >= 99.0:
            print("✅ V2 Converter: EXCELLENT data integrity")
        elif v2_accuracy >= 95.0:
            print("⚠️ V2 Converter: Good data integrity (minor issues)")
        else:
            print("❌ V2 Converter: Poor data integrity (significant scrambling)")

        # Success criteria - fixed original should be perfect
        success = (fixed_original_accuracy >= 99.0)
        return success

        # Statistical summary
        print(f"\n" + "="*70)
        print("STATISTICAL SUMMARY")
        print("="*70)

        mib_stats = f"min={np.min(mib_values)}, max={np.max(mib_values)}, mean={np.mean(mib_values):.2f}"

        valid_original = [v for v in original_values if v != -1]
        orig_stats = f"min={np.min(valid_original) if valid_original else 'N/A'}, max={np.max(valid_original) if valid_original else 'N/A'}, mean={np.mean(valid_original):.2f if valid_original else 'N/A'}"

        valid_fixed_original = [v for v in fixed_original_values if v != -1]
        fixed_orig_stats = f"min={np.min(valid_fixed_original) if valid_fixed_original else 'N/A'}, max={np.max(valid_fixed_original) if valid_fixed_original else 'N/A'}, mean={np.mean(valid_fixed_original):.2f if valid_fixed_original else 'N/A'}"

        valid_v2 = [v for v in v2_values if v != -1]
        v2_stats = f"min={np.min(valid_v2) if valid_v2 else 'N/A'}, max={np.max(valid_v2) if valid_v2 else 'N/A'}, mean={np.mean(valid_v2):.2f if valid_v2 else 'N/A'}"

        print(f"MIB values:           {mib_stats}")
        print(f"Original values:      {orig_stats}")
        print(f"Fixed Original values: {fixed_orig_stats}")
        print(f"V2 values:            {v2_stats}")


    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_integrity()

    print(f"\n" + "="*70)
    if success:
        print("🎉 DATA INTEGRITY TEST PASSED!")
        print("Fixed original converter preserves data correctly.")
    else:
        print("💥 DATA INTEGRITY TEST FAILED!")
        print("Fixed original converter still has data corruption issues.")
    print("="*70)

    sys.exit(0 if success else 1)