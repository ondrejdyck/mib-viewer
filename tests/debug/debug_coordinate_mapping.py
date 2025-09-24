#!/usr/bin/env python3
"""
Debug coordinate mapping in chunked FFT processing
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, "src")

def test_coordinate_mapping():
    """Test coordinate mapping logic step by step"""
    print("=== Coordinate Mapping Debug ===")

    # Simulate real parameters
    original_shape = (256, 256, 128, 128)  # Original file shape
    sy, sx, original_qy, original_qx = original_shape

    # Simulate BF detection results (example from real data)
    bf_center_y, bf_center_x = 64, 70  # BF disk center in original coordinates
    crop_bounds = (38, 90, 44, 96)  # (y1, y2, x1, x2) - crop boundaries
    y1, y2, x1, x2 = crop_bounds

    cropped_shape = (sy, sx, y2-y1, x2-x1)  # (256, 256, 52, 52)

    print(f"Original detector shape: {original_qy}×{original_qx}")
    print(f"BF disk center (original coords): ({bf_center_y}, {bf_center_x})")
    print(f"Crop bounds: y={y1}:{y2}, x={x1}:{x2}")
    print(f"Cropped detector shape: {y2-y1}×{x2-x1}")

    # Calculate where BF center should be in cropped coordinates
    bf_center_cropped_y = bf_center_y - y1  # 64 - 38 = 26
    bf_center_cropped_x = bf_center_x - x1  # 70 - 44 = 26

    print(f"BF disk center (cropped coords): ({bf_center_cropped_y}, {bf_center_cropped_x})")

    # Expected center of cropped region
    expected_center_y = (y2 - y1) / 2  # 52 / 2 = 26
    expected_center_x = (x2 - x1) / 2  # 52 / 2 = 26

    print(f"Expected center of crop: ({expected_center_y}, {expected_center_x})")

    offset_y = bf_center_cropped_y - expected_center_y
    offset_x = bf_center_cropped_x - expected_center_x

    print(f"BF disk offset from crop center: ({offset_y}, {offset_x})")

    if abs(offset_y) > 2 or abs(offset_x) > 2:
        print("⚠️  BF disk is significantly off-center in crop region!")
        print("   This suggests the crop bounds calculation has an issue.")
    else:
        print("✅ BF disk is properly centered in crop region.")

    print(f"\n=== Chunked FFT Coordinate Mapping ===")

    # Test our chunking coordinate mapping
    try:
        from mib_viewer.io.adaptive_chunking import create_adaptive_chunking_strategy, AdaptiveChunkCalculator

        # Create chunking strategy using cropped shape (new approach)
        chunking_result = create_adaptive_chunking_strategy(
            cropped_shape,  # Use cropped shape for optimization
            chunk_detector_dims=True
        )

        print(f"Chunking result - chunk dims: {chunking_result.chunk_dims}")
        print(f"Chunking result - total chunks: {chunking_result.total_chunks}")

        # Generate work chunk queue
        calculator = AdaptiveChunkCalculator()
        work_chunks = calculator.generate_chunk_queue(chunking_result)

        print(f"Generated {len(work_chunks)} work chunks")

        # Test coordinate offset logic manually
        print(f"\n4. COORDINATE OFFSET SIMULATION:")
        y_offset = y1  # 38
        x_offset = x1  # 44

        if work_chunks:
            work_chunk = work_chunks[0]
            work_input_slice = work_chunk.input_slice
            work_output_slice = work_chunk.output_slice

            print(f"  Work chunk input slice: [{work_input_slice[2].start}:{work_input_slice[2].stop}, {work_input_slice[3].start}:{work_input_slice[3].stop}]")
            print(f"  Work chunk output slice: [{work_output_slice[2].start}:{work_output_slice[2].stop}, {work_output_slice[3].start}:{work_output_slice[3].stop}]")

            # Apply coordinate offset (simulating _apply_crop_offset_to_chunks)
            dataset_qy_slice = slice(
                work_input_slice[2].start + y_offset,
                work_input_slice[2].stop + y_offset
            )
            dataset_qx_slice = slice(
                work_input_slice[3].start + x_offset,
                work_input_slice[3].stop + x_offset
            )

            print(f"  Dataset slice (after offset): [{dataset_qy_slice.start}:{dataset_qy_slice.stop}, {dataset_qx_slice.start}:{dataset_qx_slice.stop}]")

            # Check if dataset slice targets the crop region correctly
            expected_start_y = y1  # Should start at crop boundary
            expected_start_x = x1

            if (dataset_qy_slice.start >= expected_start_y and dataset_qy_slice.stop <= y2 and
                dataset_qx_slice.start >= expected_start_x and dataset_qx_slice.stop <= x2):
                print("  ✅ Dataset slice correctly targets crop region")
            else:
                print("  ❌ Dataset slice does not target crop region correctly!")
                return False

            # Verify BF center mapping
            print(f"\n  BF center coordinate verification:")
            print(f"    Original BF center: ({bf_center_y}, {bf_center_x})")
            print(f"    Dataset slice reads: [{dataset_qy_slice.start}:{dataset_qy_slice.stop}, {dataset_qx_slice.start}:{dataset_qx_slice.stop}]")
            print(f"    Work slice writes: [{work_output_slice[2].start}:{work_output_slice[2].stop}, {work_output_slice[3].start}:{work_output_slice[3].stop}]")

            # BF center in dataset coordinates should map to correct position in work coordinates
            if (bf_center_y >= dataset_qy_slice.start and bf_center_y < dataset_qy_slice.stop and
                bf_center_x >= dataset_qx_slice.start and bf_center_x < dataset_qx_slice.stop):

                # Calculate where BF center appears in the loaded chunk
                bf_in_chunk_y = bf_center_y - dataset_qy_slice.start
                bf_in_chunk_x = bf_center_x - dataset_qx_slice.start

                # This should map to the correct position in work coordinates
                bf_in_work_y = bf_in_chunk_y + work_output_slice[2].start
                bf_in_work_x = bf_in_chunk_x + work_output_slice[3].start

                print(f"    BF center in loaded chunk: ({bf_in_chunk_y}, {bf_in_chunk_x})")
                print(f"    BF center in work output: ({bf_in_work_y}, {bf_in_work_x})")
                print(f"    Expected BF in work output: ({bf_center_cropped_y}, {bf_center_cropped_x})")

                if (abs(bf_in_work_y - bf_center_cropped_y) < 2 and
                    abs(bf_in_work_x - bf_center_cropped_x) < 2):
                    print("  ✅ BF center mapping is correct")
                    return True
                else:
                    print("  ❌ BF center mapping is incorrect!")
                    return False
            else:
                print("  ❌ BF center not contained in this chunk!")
                return False

    except Exception as e:
        print(f"❌ Coordinate mapping test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_coordinate_mapping()
    print(f"\n{'✅' if success else '❌'} Coordinate mapping {'correct' if success else 'needs fixing'}")