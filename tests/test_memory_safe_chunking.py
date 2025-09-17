#!/usr/bin/env python3
"""
Test script to verify memory-safe chunking algorithm
Ensures that chunks are always ≤ target size for memory safety
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mib_viewer.io.mib_to_emd_converter import MibToEmdConverter

def test_memory_safe_chunking():
    """Test that chunking algorithm respects memory safety"""
    print("=== Testing Memory-Safe Chunking Algorithm ===")

    # Create converter
    converter = MibToEmdConverter(log_callback=lambda msg, level: print(f"CONVERTER: {msg}"))

    # Test cases: various file shapes
    test_cases = [
        # (shape, description, expected_behavior)
        ((256, 256, 256, 256), "Your original case", "should choose 4,096 not 8,192"),
        ((128, 128, 512, 512), "Medium file", "should stay under target"),
        ((64, 64, 1024, 256), "Test file shape", "should stay under target"),
        ((512, 512, 256, 256), "Large scan", "should stay under target"),
    ]

    all_passed = True

    for shape_4d, description, expected_behavior in test_cases:
        print(f"\n--- Testing {description}: {shape_4d} ---")
        print(f"Expected: {expected_behavior}")

        try:
            # Calculate available memory (simulate limited memory for testing)
            import psutil
            available_memory = psutil.virtual_memory().available

            # Calculate chunking
            result = converter.calculate_optimal_chunk_size(shape_4d, available_memory)

            sy, sx, qy, qx = shape_4d
            chunk_sy, chunk_sx, chunk_qy, chunk_qx = result

            total_frames = sy * sx
            available_workers = max(1, os.cpu_count() - 2)
            target_frames_per_worker = total_frames // available_workers
            actual_frames_per_chunk = chunk_sy * chunk_sx

            print(f"Target frames per worker: {target_frames_per_worker:,}")
            print(f"Actual frames per chunk: {actual_frames_per_chunk:,}")
            print(f"Chunk dimensions: ({chunk_sy}, {chunk_sx})")

            # CRITICAL TEST: Verify memory safety
            if actual_frames_per_chunk <= target_frames_per_worker:
                print(f"✓ MEMORY SAFE: {actual_frames_per_chunk:,} ≤ {target_frames_per_worker:,}")
            else:
                print(f"✗ MEMORY UNSAFE: {actual_frames_per_chunk:,} > {target_frames_per_worker:,}")
                all_passed = False

            # Additional checks
            total_chunks = (sy // chunk_sy) * (sx // chunk_sx)
            print(f"Total chunks: {total_chunks}")
            print(f"Workers: {available_workers}")

            if total_chunks >= available_workers:
                print(f"✓ Parallelization possible: {total_chunks} chunks ≥ {available_workers} workers")
            else:
                print(f"⚠ Limited parallelization: {total_chunks} chunks < {available_workers} workers")

        except Exception as e:
            print(f"✗ Test failed with error: {str(e)}")
            all_passed = False

    return all_passed

def test_original_case_specifically():
    """Test the specific case that was problematic"""
    print("\n=== Testing Your Original Case Specifically ===")

    converter = MibToEmdConverter(log_callback=lambda msg, level: print(f"CONVERTER: {msg}"))

    # Your exact case: 256×256 scan
    shape_4d = (256, 256, 256, 256)
    import psutil
    available_memory = psutil.virtual_memory().available

    print(f"File shape: {shape_4d}")
    print("Testing chunking algorithm...")

    result = converter.calculate_optimal_chunk_size(shape_4d, available_memory)
    chunk_sy, chunk_sx, chunk_qy, chunk_qx = result

    sy, sx, qy, qx = shape_4d
    total_frames = sy * sx
    available_workers = max(1, os.cpu_count() - 2)
    target_frames_per_worker = total_frames // available_workers
    actual_frames_per_chunk = chunk_sy * chunk_sx

    print(f"\nResults:")
    print(f"  Total frames: {total_frames:,}")
    print(f"  Available workers: {available_workers}")
    print(f"  Target per worker: {target_frames_per_worker:,}")
    print(f"  Chosen chunk size: ({chunk_sy}, {chunk_sx}) = {actual_frames_per_chunk:,} frames")

    # This should now choose (16, 256) = 4,096 frames instead of (32, 256) = 8,192 frames
    if actual_frames_per_chunk == 4096:
        print(f"✅ CORRECT: Chose 4,096 frames (memory safe)")
        return True
    elif actual_frames_per_chunk == 8192:
        print(f"❌ STILL BROKEN: Chose 8,192 frames (exceeds target of {target_frames_per_worker:,})")
        return False
    else:
        print(f"🤔 UNEXPECTED: Chose {actual_frames_per_chunk:,} frames")
        return False

if __name__ == "__main__":
    print("Testing Memory-Safe Chunking Algorithm")
    print("=" * 50)

    success = True

    # Test 1: General memory safety
    if not test_memory_safe_chunking():
        success = False

    # Test 2: Specific original case
    if not test_original_case_specifically():
        success = False

    print("\n" + "=" * 50)
    if success:
        print("🎉 MEMORY SAFETY TESTS PASSED!")
        print("✓ Chunks always ≤ target size")
        print("✓ Memory safety prioritized over optimization")
        print("✓ Algorithm fixed successfully")
    else:
        print("❌ MEMORY SAFETY TESTS FAILED!")
        print("Algorithm still needs fixes")

    print("=" * 50)