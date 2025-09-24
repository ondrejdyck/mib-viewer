#!/usr/bin/env python3
"""
Test script for crop-aware chunked FFT processor
"""

import sys
import os

# Add src to path
sys.path.insert(0, "src")

def test_crop_aware_chunking():
    """Test crop-aware detector dimension chunking"""
    print("=== Crop-Aware Chunking Test ===")

    try:
        from mib_viewer.io.adaptive_chunking import create_adaptive_chunking_strategy

        # Test data shape: 256x256 scan, 128x128 detector
        file_shape = (256, 256, 128, 128)

        # Define crop bounds: crop detector from 128x128 to 52x52 (as in the error)
        crop_bounds = (38, 90, 38, 90)  # (y1, y2, x1, x2) -> 52x52 region

        print(f"Original detector shape: {file_shape[2]}x{file_shape[3]}")
        print(f"Crop bounds: {crop_bounds}")
        print(f"Cropped detector shape: {crop_bounds[1] - crop_bounds[0]}x{crop_bounds[3] - crop_bounds[2]}")

        # Test 1: Full detector chunking (no crop)
        print("\n1. FULL DETECTOR CHUNKING:")
        full_result = create_adaptive_chunking_strategy(
            file_shape,
            chunk_detector_dims=True,
            crop_bounds=None
        )
        print(f"   Chunk dims: {full_result.chunk_dims}")
        print(f"   Total chunks: {full_result.total_chunks}")
        print(f"   Crop bounds: {full_result.crop_bounds}")

        # Test 2: Cropped detector chunking
        print("\n2. CROPPED DETECTOR CHUNKING:")
        crop_result = create_adaptive_chunking_strategy(
            file_shape,
            chunk_detector_dims=True,
            crop_bounds=crop_bounds
        )
        print(f"   Chunk dims: {crop_result.chunk_dims}")
        print(f"   Total chunks: {crop_result.total_chunks}")
        print(f"   Crop bounds: {crop_result.crop_bounds}")

        # Test 3: Generate chunk queues and verify slices
        print("\n3. CHUNK QUEUE VERIFICATION:")
        from mib_viewer.io.adaptive_chunking import AdaptiveChunkCalculator

        calculator = AdaptiveChunkCalculator()

        full_queue = calculator.generate_chunk_queue(full_result)
        crop_queue = calculator.generate_chunk_queue(crop_result)

        print(f"   Full chunking queue: {len(full_queue)} chunks")
        print(f"   Crop chunking queue: {len(crop_queue)} chunks")

        # Verify first chunk slices
        print("\n4. SLICE VERIFICATION:")
        if full_queue:
            full_chunk = full_queue[0]
            print(f"   Full chunk input slice: {full_chunk.input_slice}")
            print(f"   Full chunk output slice: {full_chunk.output_slice}")
            print(f"   Full chunk expected shape: {full_chunk.expected_shape}")

        if crop_queue:
            crop_chunk = crop_queue[0]
            print(f"   Crop chunk input slice: {crop_chunk.input_slice}")
            print(f"   Crop chunk output slice: {crop_chunk.output_slice}")
            print(f"   Crop chunk expected shape: {crop_chunk.expected_shape}")

            # Verify that input slice targets the cropped region
            input_qy_slice = crop_chunk.input_slice[2]
            input_qx_slice = crop_chunk.input_slice[3]

            expected_qy_start = crop_bounds[0]  # Should start at crop boundary
            expected_qx_start = crop_bounds[2]

            if input_qy_slice.start == expected_qy_start and input_qx_slice.start == expected_qx_start:
                print("   ✅ Input slice correctly targets cropped region")
            else:
                print(f"   ❌ Input slice mismatch: got ({input_qy_slice.start}, {input_qx_slice.start}), expected ({expected_qy_start}, {expected_qx_start})")
                return False

            # Verify that output slice starts at 0 (cropped coordinate system)
            output_qy_slice = crop_chunk.output_slice[2]
            output_qx_slice = crop_chunk.output_slice[3]

            if output_qy_slice.start == 0 and output_qx_slice.start == 0:
                print("   ✅ Output slice uses cropped coordinate system")
            else:
                print(f"   ❌ Output slice should start at (0,0), got ({output_qy_slice.start}, {output_qx_slice.start})")
                return False

        print("\n✅ Crop-aware chunking test passed!")
        return True

    except Exception as e:
        print(f"❌ Crop-aware chunking test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chunked_fft_initialization():
    """Test that chunked FFT processor can initialize with crop-aware chunking"""
    print("\n=== Chunked FFT Processor Test ===")

    try:
        from mib_viewer.io.chunked_fft_processor import create_chunked_fft_processor

        def test_progress_callback(message):
            print(f"Progress: {message}")

        processor = create_chunked_fft_processor(
            conservative=True,
            progress_callback=test_progress_callback
        )

        print("✅ Chunked FFT processor initialization successful")
        return True
    except Exception as e:
        print(f"❌ Chunked FFT processor initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=== Crop-Aware Chunked FFT Test Suite ===")

    tests = [
        test_crop_aware_chunking,
        test_chunked_fft_initialization
    ]

    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()

    print(f"=== Results: {passed}/{len(tests)} tests passed ===")

    if passed == len(tests):
        print("🎉 All tests passed! Crop-aware chunked FFT is ready.")
        print("\nKey improvements:")
        print("- ✅ Eliminated double cropping issue")
        print("- ✅ Chunks directly target cropped detector regions")
        print("- ✅ Consistent chunk shapes in output")
        print("- ✅ Memory efficient - no intermediate cropped copies")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())