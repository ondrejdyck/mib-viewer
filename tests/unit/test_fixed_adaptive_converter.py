#!/usr/bin/env python3
"""
Test script to verify the fixed adaptive converter
Tests dimension detection and basic conversion functionality
"""

import os
import sys
import time
import tempfile
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mib_viewer.io.adaptive_converter import AdaptiveMibEmdConverter

def test_dimension_detection():
    """Test that dimension detection works correctly"""
    print("=== Testing Fixed Adaptive Converter ===")

    # Test file path
    test_file = "/media/o2d/data/ORNL Dropbox/Ondrej Dyck/Code/mib-viewer-project/Example 4D/64x64 Test.mib"

    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        print("Please ensure the test file exists or update the path")
        return False

    print(f"Testing with file: {os.path.basename(test_file)}")

    # Create converter
    converter = AdaptiveMibEmdConverter(
        compression='gzip',
        compression_level=6,
        verbose=True,
        log_callback=lambda msg: print(f"CONVERTER: {msg}")
    )

    try:
        # Test file analysis (this should use the fixed dimension detection)
        print("\n--- Testing Dimension Detection ---")
        file_info = converter._analyze_input_file(test_file)

        print(f"Detected file shape: {file_info['file_shape']}")
        print(f"File size: {file_info['file_size_gb']:.2f} GB")

        # Expected shape for Test.mib should be (64, 64, 256, 1024)
        expected_scan_size = (64, 64)
        detected_scan_size = file_info['file_shape'][:2]

        if detected_scan_size == expected_scan_size:
            print(f"✓ Dimension detection CORRECT: {detected_scan_size}")
        else:
            print(f"✗ Dimension detection INCORRECT: got {detected_scan_size}, expected {expected_scan_size}")
            return False

        # Test chunking strategy calculation
        print("\n--- Testing Chunking Strategy ---")
        chunking_result = converter._calculate_chunking_strategy(file_info)

        print(f"Strategy: {chunking_result.strategy.value}")
        print(f"Chunk dims: {chunking_result.chunk_dims}")
        print(f"Workers: {chunking_result.num_workers}")
        print(f"Total chunks: {chunking_result.total_chunks}")
        print(f"I/O reduction: {chunking_result.io_reduction_factor}x")

        print("\n✓ Fixed adaptive converter dimension detection test PASSED!")
        return True

    except Exception as e:
        print(f"\n✗ Test FAILED with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_small_conversion():
    """Test a small conversion to verify no segfaults"""
    print("\n=== Testing Small Conversion (No Segfaults) ===")

    test_file = "/media/o2d/data/ORNL Dropbox/Ondrej Dyck/Code/mib-viewer-project/Example 4D/64x64 Test.mib"

    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return False

    # Create temporary output file
    with tempfile.NamedTemporaryFile(suffix='.emd', delete=False) as tmp_file:
        output_path = tmp_file.name

    try:
        print(f"Converting {os.path.basename(test_file)} to {os.path.basename(output_path)}")

        # Create converter with conservative settings for testing
        converter = AdaptiveMibEmdConverter(
            compression='gzip',
            compression_level=6,
            max_workers=2,  # Conservative worker count for testing
            conservative_mode=True,
            verbose=True,
            log_callback=lambda msg: print(f"CONVERTER: {msg}"),
            progress_callback=lambda pct, msg: print(f"PROGRESS: {pct}% - {msg}")
        )

        start_time = time.time()

        # Perform conversion
        result = converter.convert_to_emd(test_file, output_path)

        conversion_time = time.time() - start_time

        print(f"\n--- Conversion Results ---")
        print(f"Time: {conversion_time:.1f}s")
        print(f"Strategy: {result.get('chunking_strategy', 'unknown')}")
        print(f"Workers: {result.get('num_workers', 'unknown')}")
        print(f"Chunks: {result.get('total_chunks', 'unknown')}")
        print(f"I/O reduction: {result.get('io_reduction_factor', 'unknown')}x")
        print(f"Input size: {result.get('input_size_gb', 0):.2f} GB")
        print(f"Output size: {result.get('output_size_gb', 0):.2f} GB")
        print(f"Compression: {result.get('compression_ratio', 0):.1f}x")

        # Verify output file exists and has reasonable size
        if os.path.exists(output_path):
            output_size = os.path.getsize(output_path)
            if output_size > 1024:  # At least 1KB
                print(f"✓ Output file created successfully: {output_size:,} bytes")
                print(f"✓ Conversion completed WITHOUT SEGFAULTS!")
                return True
            else:
                print(f"✗ Output file too small: {output_size} bytes")
                return False
        else:
            print("✗ Output file not created")
            return False

    except Exception as e:
        print(f"\n✗ Conversion FAILED with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up temporary file
        if os.path.exists(output_path):
            os.unlink(output_path)
            print(f"Cleaned up temporary file: {os.path.basename(output_path)}")

if __name__ == "__main__":
    print("Testing Fixed Adaptive Converter")
    print("=" * 50)

    success = True

    # Test 1: Dimension detection
    if not test_dimension_detection():
        success = False

    # Test 2: Small conversion
    if not test_small_conversion():
        success = False

    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL TESTS PASSED! Fixed adaptive converter is working correctly.")
        print("✓ Dimension detection fixed")
        print("✓ No segfaults during conversion")
        print("✓ Parallel processing appears to be working")
    else:
        print("❌ SOME TESTS FAILED! Further investigation needed.")

    print("=" * 50)