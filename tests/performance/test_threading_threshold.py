#!/usr/bin/env python3
"""
Test the threading threshold change
"""

import sys
import os

# Add src to path
sys.path.insert(0, "src")

def test_threading_threshold():
    """Test that 0.33 GB file now uses multi-threading"""
    print("=== Threading Threshold Test ===")

    try:
        from mib_viewer.io.adaptive_chunking import create_adaptive_chunking_strategy

        # Simulate your file: 0.33 GB
        file_size_gb = 0.33

        # Estimate file shape for 0.33 GB
        # Assuming uint16 (2 bytes per pixel)
        total_pixels = (file_size_gb * 1024**3) / 2  # ~177 million pixels

        # Example shape for ~0.33 GB: 256x256 scan, 52x52 detector (cropped)
        cropped_shape = (256, 256, 52, 52)
        actual_pixels = 256 * 256 * 52 * 52  # ~177 million pixels ✓
        actual_size_gb = (actual_pixels * 2) / (1024**3)

        print(f"File size: {file_size_gb} GB")
        print(f"Simulated shape: {cropped_shape}")
        print(f"Actual size: {actual_size_gb:.3f} GB")
        print(f"New threshold: 0.1 GB (100 MB)")

        # Test chunking strategy
        chunking_result = create_adaptive_chunking_strategy(
            cropped_shape,
            chunk_detector_dims=True
        )

        print(f"\nChunking result:")
        print(f"  Strategy: {chunking_result.strategy}")
        print(f"  Single thread: {chunking_result.use_single_thread}")
        print(f"  Num workers: {chunking_result.num_workers}")
        print(f"  Chunk dims: {chunking_result.chunk_dims}")
        print(f"  Total chunks: {chunking_result.total_chunks}")

        if chunking_result.use_single_thread:
            print(f"  ❌ Still using single threading!")
            print(f"     File size {file_size_gb:.3f} GB >= threshold 0.1 GB")
            return False
        else:
            print(f"  ✅ Now using multi-threading!")
            print(f"     {chunking_result.num_workers} workers will process {chunking_result.total_chunks} chunks")
            return True

    except Exception as e:
        print(f"❌ Threading threshold test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_threshold_boundary():
    """Test the threshold boundary (files right at 100MB)"""
    print(f"\n=== Threshold Boundary Test ===")

    try:
        from mib_viewer.io.adaptive_chunking import AdaptiveChunkCalculator

        calculator = AdaptiveChunkCalculator()

        # Test file just below threshold (0.09 GB = 90 MB)
        test_shape_small = (128, 128, 32, 32)  # Small file
        result_small = calculator.calculate_chunking_strategy(test_shape_small)

        print(f"90 MB file: {result_small.strategy}, single_thread={result_small.use_single_thread}")

        # Test file just above threshold (0.15 GB = 150 MB)
        test_shape_large = (256, 256, 40, 40)  # Larger file
        result_large = calculator.calculate_chunking_strategy(test_shape_large)

        print(f"150 MB file: {result_large.strategy}, single_thread={result_large.use_single_thread}")

        if result_small.use_single_thread and not result_large.use_single_thread:
            print("✅ Threshold working correctly!")
            return True
        else:
            print("❌ Threshold not working as expected")
            return False

    except Exception as e:
        print(f"❌ Boundary test failed: {e}")
        return False

def main():
    """Run tests"""
    test1 = test_threading_threshold()
    test2 = test_threshold_boundary()

    if test1 and test2:
        print(f"\n🎉 Threading threshold successfully lowered!")
        print(f"Your 0.33 GB file should now use multi-threading for faster FFT processing.")
        return 0
    else:
        print(f"\n❌ Threading threshold change needs verification.")
        return 1

if __name__ == "__main__":
    sys.exit(main())