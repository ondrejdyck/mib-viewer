#!/usr/bin/env python3
"""
Test script for chunked FFT processor
"""

import sys
import os

# Add src to path
sys.path.insert(0, "src")

def test_imports():
    """Test that all imports work correctly"""
    print("Testing imports...")

    try:
        from mib_viewer.io.chunked_fft_processor import (
            Chunked4DFFTProcessor, FFTMode, FFTResult,
            create_chunked_fft_processor
        )
        print("✅ chunked_fft_processor imports successful")
    except Exception as e:
        print(f"❌ chunked_fft_processor import failed: {e}")
        return False

    try:
        from mib_viewer.io.adaptive_chunking import create_adaptive_chunking_strategy
        print("✅ adaptive_chunking imports successful")
    except Exception as e:
        print(f"❌ adaptive_chunking import failed: {e}")
        return False

    return True

def test_initialization():
    """Test processor initialization"""
    print("\nTesting initialization...")

    try:
        from mib_viewer.io.chunked_fft_processor import create_chunked_fft_processor

        def test_progress_callback(message):
            print(f"Progress: {message}")

        processor = create_chunked_fft_processor(
            conservative=True,
            progress_callback=test_progress_callback
        )

        print("✅ Processor initialization successful")
        return True
    except Exception as e:
        print(f"❌ Processor initialization failed: {e}")
        return False

def test_detector_chunking():
    """Test detector dimension chunking"""
    print("\nTesting detector dimension chunking...")

    try:
        from mib_viewer.io.adaptive_chunking import create_adaptive_chunking_strategy

        # Test data shape: 256x256 scan, 128x128 detector
        file_shape = (256, 256, 128, 128)

        # Test detector chunking
        chunking_result = create_adaptive_chunking_strategy(
            file_shape,
            chunk_detector_dims=True
        )

        print(f"File shape: {file_shape}")
        print(f"Chunk dims: {chunking_result.chunk_dims}")
        print(f"Strategy: {chunking_result.strategy}")
        print(f"Total chunks: {chunking_result.total_chunks}")
        print(f"Chunk detector dims: {chunking_result.chunk_detector_dims}")

        # Verify that scan dimensions are preserved
        sy, sx, chunk_qy, chunk_qx = chunking_result.chunk_dims
        original_sy, original_sx, original_qy, original_qx = file_shape

        if sy == original_sy and sx == original_sx:
            print("✅ Scan dimensions preserved correctly")
        else:
            print(f"❌ Scan dimensions not preserved: got ({sy}, {sx}), expected ({original_sy}, {original_sx})")
            return False

        if chunk_qy <= original_qy and chunk_qx <= original_qx:
            print("✅ Detector dimensions chunked correctly")
        else:
            print(f"❌ Detector dimensions not chunked: got ({chunk_qy}, {chunk_qx}), original ({original_qy}, {original_qx})")
            return False

        return True

    except Exception as e:
        print(f"❌ Detector chunking test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Chunked FFT Processor Test ===")

    tests = [
        test_imports,
        test_initialization,
        test_detector_chunking
    ]

    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()

    print(f"=== Results: {passed}/{len(tests)} tests passed ===")

    if passed == len(tests):
        print("🎉 All tests passed! Chunked FFT processor is ready.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())