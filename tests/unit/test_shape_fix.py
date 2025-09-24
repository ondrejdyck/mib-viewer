#!/usr/bin/env python3
"""
Test script for FFT shape-based chunking fix
"""

import sys
import os

# Add src to path
sys.path.insert(0, "src")

def test_fft_shape_chunking():
    """Test that chunking strategy now uses FFT shape instead of original shape"""
    print("=== FFT Shape-Based Chunking Test ===")

    try:
        from mib_viewer.io.adaptive_chunking import create_adaptive_chunking_strategy

        # Original file shape: 256x256 scan, 128x128 detector
        original_shape = (256, 256, 128, 128)

        # Cropped shape: 256x256 scan, 52x52 detector (like the error)
        cropped_shape = (256, 256, 52, 52)

        # Crop bounds that produce the 52x52 region
        crop_bounds = (38, 90, 38, 90)  # (y1, y2, x1, x2) -> 52x52 region

        print(f"Original shape: {original_shape}")
        print(f"Cropped shape: {cropped_shape}")
        print(f"Crop bounds: {crop_bounds}")

        # Test 1: Chunking based on original shape (old approach - should work for full mode)
        print("\n1. CHUNKING BASED ON ORIGINAL SHAPE:")
        original_result = create_adaptive_chunking_strategy(
            original_shape,
            chunk_detector_dims=True,
            crop_bounds=None  # No crop bounds
        )
        print(f"   File shape used: {original_shape}")
        print(f"   Chunk dims: {original_result.chunk_dims}")
        print(f"   Max chunk detector size: {original_result.chunk_dims[2]}x{original_result.chunk_dims[3]}")

        # Test 2: Chunking based on cropped shape (new approach - should work for cropped mode)
        print("\n2. CHUNKING BASED ON CROPPED SHAPE:")
        cropped_result = create_adaptive_chunking_strategy(
            cropped_shape,
            chunk_detector_dims=True,
            crop_bounds=crop_bounds  # Pass crop bounds for slice generation
        )
        print(f"   File shape used: {cropped_shape}")
        print(f"   Chunk dims: {cropped_result.chunk_dims}")
        print(f"   Max chunk detector size: {cropped_result.chunk_dims[2]}x{cropped_result.chunk_dims[3]}")

        # Test 3: Verify that cropped chunking produces chunks that fit in output
        max_chunk_qy = cropped_result.chunk_dims[2]
        max_chunk_qx = cropped_result.chunk_dims[3]
        max_output_qy = cropped_shape[2]  # 52
        max_output_qx = cropped_shape[3]  # 52

        print(f"\n3. COMPATIBILITY CHECK:")
        print(f"   Max chunk size: {max_chunk_qy}x{max_chunk_qx}")
        print(f"   Output file size: {max_output_qy}x{max_output_qx}")

        if max_chunk_qy <= max_output_qy and max_chunk_qx <= max_output_qx:
            print("   ✅ Chunk sizes compatible with output file")
        else:
            print("   ❌ Chunk sizes too large for output file - would cause broadcasting error")
            return False

        # Test 4: Generate chunk queue and verify slice bounds
        print(f"\n4. CHUNK QUEUE VERIFICATION:")
        from mib_viewer.io.adaptive_chunking import AdaptiveChunkCalculator

        calculator = AdaptiveChunkCalculator()
        chunk_queue = calculator.generate_chunk_queue(cropped_result)

        print(f"   Generated {len(chunk_queue)} chunks")

        max_input_qy_end = 0
        max_input_qx_end = 0
        max_output_qy_end = 0
        max_output_qx_end = 0

        for chunk in chunk_queue:
            input_qy_slice = chunk.input_slice[2]
            input_qx_slice = chunk.input_slice[3]
            output_qy_slice = chunk.output_slice[2]
            output_qx_slice = chunk.output_slice[3]

            max_input_qy_end = max(max_input_qy_end, input_qy_slice.stop)
            max_input_qx_end = max(max_input_qx_end, input_qx_slice.stop)
            max_output_qy_end = max(max_output_qy_end, output_qy_slice.stop)
            max_output_qx_end = max(max_output_qx_end, output_qx_slice.stop)

        print(f"   Max input slice end: ({max_input_qy_end}, {max_input_qx_end})")
        print(f"   Max output slice end: ({max_output_qy_end}, {max_output_qx_end})")
        print(f"   Expected crop region end: ({crop_bounds[1]}, {crop_bounds[3]}) = (90, 90)")
        print(f"   Expected output end: ({cropped_shape[2]}, {cropped_shape[3]}) = (52, 52)")

        # Verify input slices don't exceed crop bounds
        if max_input_qy_end <= crop_bounds[1] and max_input_qx_end <= crop_bounds[3]:
            print("   ✅ Input slices stay within crop bounds")
        else:
            print("   ❌ Input slices exceed crop bounds")
            return False

        # Verify output slices don't exceed output dimensions
        if max_output_qy_end <= cropped_shape[2] and max_output_qx_end <= cropped_shape[3]:
            print("   ✅ Output slices stay within output dimensions")
        else:
            print("   ❌ Output slices exceed output dimensions")
            return False

        print("\n✅ FFT shape-based chunking test passed!")
        print("\nKey insight: Chunking strategy now correctly optimizes for the FFT output size,")
        print("not the original file size, preventing broadcasting errors.")
        return True

    except Exception as e:
        print(f"❌ FFT shape-based chunking test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the test"""
    if test_fft_shape_chunking():
        print("\n🎉 Shape-based chunking fix is working correctly!")
        return 0
    else:
        print("\n❌ Shape-based chunking fix needs more work.")
        return 1

if __name__ == "__main__":
    sys.exit(main())