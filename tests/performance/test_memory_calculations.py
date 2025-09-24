#!/usr/bin/env python3
"""
Test memory calculations after the fix
"""

import sys
import os

# Add src to path
sys.path.insert(0, "src")

def test_detector_chunking_memory():
    """Test detector chunking memory calculations"""
    print("=== Detector Chunking Memory Test ===")

    try:
        from mib_viewer.io.adaptive_chunking import create_adaptive_chunking_strategy

        # Your actual file dimensions
        cropped_shape = (256, 256, 52, 52)

        chunking_result = create_adaptive_chunking_strategy(
            cropped_shape,
            chunk_detector_dims=True
        )

        print(f"File shape: {cropped_shape}")
        print(f"Chunk dims: {chunking_result.chunk_dims}")
        print(f"Chunk size reported: {chunking_result.chunk_size_mb:.1f} MB")
        print(f"Estimated memory: {chunking_result.estimated_memory_usage_gb:.2f} GB")

        # Manual verification
        chunk_sy, chunk_sx, chunk_qy, chunk_qx = chunking_result.chunk_dims
        expected_pixels = chunk_sy * chunk_sx * chunk_qy * chunk_qx
        expected_size_mb = (expected_pixels * 2) / (1024**2)  # 2 bytes per pixel
        expected_memory_gb = (expected_size_mb * chunking_result.num_workers) / 1024

        print(f"\nManual verification:")
        print(f"  Expected pixels per chunk: {expected_pixels:,}")
        print(f"  Expected size per chunk: {expected_size_mb:.1f} MB")
        print(f"  Expected total memory: {expected_memory_gb:.2f} GB")

        # Check if calculations match
        size_match = abs(chunking_result.chunk_size_mb - expected_size_mb) < 1
        memory_match = abs(chunking_result.estimated_memory_usage_gb - expected_memory_gb) < 0.01

        if size_match and memory_match:
            print("  ✅ Memory calculations are correct!")

            # Check if memory is reasonable (should be under 1GB total)
            if chunking_result.estimated_memory_usage_gb < 1.0:
                print("  ✅ Memory usage is reasonable!")
                return True
            else:
                print(f"  ⚠️  Memory usage seems high: {chunking_result.estimated_memory_usage_gb:.2f} GB")
                return False
        else:
            print("  ❌ Memory calculations don't match!")
            return False

    except Exception as e:
        print(f"❌ Detector chunking memory test failed: {e}")
        return False

def test_scan_chunking_memory():
    """Test scan chunking memory calculations"""
    print(f"\n=== Scan Chunking Memory Test ===")

    try:
        from mib_viewer.io.adaptive_chunking import create_adaptive_chunking_strategy

        # Test scan chunking mode (conversion mode)
        test_shape = (256, 256, 64, 64)

        chunking_result = create_adaptive_chunking_strategy(
            test_shape,
            chunk_detector_dims=False  # Scan chunking mode
        )

        print(f"File shape: {test_shape}")
        print(f"Chunk dims: {chunking_result.chunk_dims}")
        print(f"Chunk size reported: {chunking_result.chunk_size_mb:.1f} MB")
        print(f"Estimated memory: {chunking_result.estimated_memory_usage_gb:.2f} GB")

        # Manual verification
        chunk_sy, chunk_sx, chunk_qy, chunk_qx = chunking_result.chunk_dims
        expected_pixels = chunk_sy * chunk_sx * chunk_qy * chunk_qx
        expected_size_mb = (expected_pixels * 2) / (1024**2)  # 2 bytes per pixel
        expected_memory_gb = (expected_size_mb * chunking_result.num_workers) / 1024

        print(f"\nManual verification:")
        print(f"  Expected pixels per chunk: {expected_pixels:,}")
        print(f"  Expected size per chunk: {expected_size_mb:.1f} MB")
        print(f"  Expected total memory: {expected_memory_gb:.2f} GB")

        # Check if calculations match
        size_match = abs(chunking_result.chunk_size_mb - expected_size_mb) < 1
        memory_match = abs(chunking_result.estimated_memory_usage_gb - expected_memory_gb) < 0.01

        if size_match and memory_match and chunking_result.estimated_memory_usage_gb < 5.0:
            print("  ✅ Scan chunking memory calculations are correct!")
            return True
        else:
            print("  ❌ Scan chunking memory calculations are incorrect!")
            return False

    except Exception as e:
        print(f"❌ Scan chunking memory test failed: {e}")
        return False

def main():
    """Run comprehensive memory calculation tests"""
    print("=== Memory Calculation Verification ===")

    test1 = test_detector_chunking_memory()
    test2 = test_scan_chunking_memory()

    if test1 and test2:
        print(f"\n🎉 All memory calculations are now correct!")
        print(f"Your FFT processing should work without hanging.")
        return 0
    else:
        print(f"\n❌ Memory calculation issues detected.")
        return 1

if __name__ == "__main__":
    sys.exit(main())