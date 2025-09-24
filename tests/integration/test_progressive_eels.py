#!/usr/bin/env python3
"""
Test script for Progressive EELS Loader

Tests the progressive loading system with a smaller file first to verify functionality.
"""

import sys
import time
import os

# Add the source directory to Python path
sys.path.insert(0, "src")

from mib_viewer.io.progressive_eels_loader import ProgressiveEELSLoader, should_use_progressive_loading


def progress_callback(status):
    """Progress callback function"""
    print(f"Progress: {status['progress_percent']:.1f}% "
          f"({status['chunks_completed']}/{status['total_chunks']} chunks, "
          f"{status['elapsed_time']:.1f}s elapsed)")


def test_progressive_loading():
    """Test progressive loading with the test file"""

    # Test file path
    test_file = "/media/o2d/data/ORNL Dropbox/Ondrej Dyck/Code/mib-viewer-project/Example 4D/64x64 Test.mib"

    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return

    print("=== Progressive EELS Loader Test ===")
    print(f"Test file: {os.path.basename(test_file)}")

    # Check if progressive loading is recommended
    should_use_progressive = should_use_progressive_loading(test_file, memory_threshold_gb=0.1)  # Low threshold for testing
    print(f"Progressive loading recommended: {should_use_progressive}")

    try:
        # Initialize loader
        print("\n1. Initializing progressive loader...")
        loader = ProgressiveEELSLoader(test_file, progress_callback=progress_callback)

        # Start loading
        print("\n2. Starting progressive loading...")
        result = loader.start_loading()

        print(f"Data shape: {result.data_shape}")
        print(f"Estimated time: {result.estimated_time_s:.1f}s")
        print(f"Memory usage: {result.memory_usage_mb:.1f} MB")

        # Monitor progress for a bit
        print("\n3. Monitoring progress...")
        start_time = time.time()

        while loader.is_loading and (time.time() - start_time) < 30:  # Max 30 seconds
            time.sleep(1)

            # Show some data as it becomes available
            data = loader.get_processed_data()
            completion_mask = loader.get_completion_mask()

            completed_pct = (completion_mask.sum() / completion_mask.size) * 100
            data_max = data.max()
            data_nonzero = (data > 0).sum()

            print(f"  Completed regions: {completed_pct:.1f}%, "
                  f"Data max: {data_max:.0f}, "
                  f"Non-zero pixels: {data_nonzero}")

        # Final results
        print("\n4. Final results:")
        final_data = loader.get_processed_data()
        print(f"Final data shape: {final_data.shape}")
        print(f"Final data range: {final_data.min():.0f} to {final_data.max():.0f}")
        print(f"Non-zero pixels: {(final_data > 0).sum()} / {final_data.size}")

        # Test region checking
        print("\n5. Testing region readiness:")
        test_regions = [
            (slice(0, 10), slice(0, 10)),      # Top-left
            (slice(100, 110), slice(100, 110)), # Center
            (slice(-10, None), slice(-10, None)) # Bottom-right
        ]

        for i, (y_slice, x_slice) in enumerate(test_regions):
            is_ready = loader.is_region_ready(y_slice, x_slice)
            print(f"  Region {i+1}: {'Ready' if is_ready else 'Not ready'}")

        # Stop loading
        loader.stop_loading()
        print("\n✅ Test completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_progressive_loading()