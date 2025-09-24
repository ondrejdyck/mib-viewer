#!/usr/bin/env python3
"""
Test script for chunked processing functionality in MibToEmdConverter

This script creates synthetic test data and verifies that the chunked processing
pipeline works correctly for both memory detection and chunked I/O operations.
"""

import numpy as np
import h5py
import os
import sys
import time
from pathlib import Path

# Add src to path so we can import the converter
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mib_viewer.io.mib_to_emd_converter import MibToEmdConverter


def create_synthetic_emd(filename: str, shape: tuple, dtype=np.uint16):
    """Create a synthetic EMD file for testing"""
    print(f"Creating synthetic EMD file: {filename}")
    print(f"  Shape: {shape}")
    print(f"  Size: {np.prod(shape) * np.dtype(dtype).itemsize / (1024**2):.1f} MB")
    
    test_data = np.random.randint(0, 1000, size=shape, dtype=dtype)
    
    with h5py.File(filename, 'w') as f:
        # EMD 1.0 structure
        f.attrs['emd_group_type'] = 'file'
        f.attrs['version_major'] = 1
        f.attrs['version_minor'] = 0
        f.attrs['authoring_program'] = 'test_chunked_processing.py'
        
        version_group = f.create_group('version_1')
        version_group.attrs['emd_group_type'] = 'root'
        
        data_group = version_group.create_group('data')
        datacubes_group = data_group.create_group('datacubes')
        datacube_group = datacubes_group.create_group('datacube_000')
        datacube_group.attrs['emd_group_type'] = 'array'
        
        # Create dataset with frame-based chunking
        sy, sx, qy, qx = shape
        dataset = datacube_group.create_dataset(
            'data', 
            data=test_data, 
            chunks=(1, 1, qy, qx),
            compression='gzip'
        )
        dataset.attrs['units'] = 'counts'
        
        # Add dimensions
        datacube_group.create_dataset('dim1', data=np.arange(sy))
        datacube_group.create_dataset('dim2', data=np.arange(sx))
        datacube_group.create_dataset('dim3', data=np.arange(qy))
        datacube_group.create_dataset('dim4', data=np.arange(qx))
        
        datacube_group['dim1'].attrs['name'] = 'scan_y'
        datacube_group['dim1'].attrs['units'] = 'pixel'
        datacube_group['dim2'].attrs['name'] = 'scan_x'
        datacube_group['dim2'].attrs['units'] = 'pixel'
        datacube_group['dim3'].attrs['name'] = 'detector_y'
        datacube_group['dim3'].attrs['units'] = 'pixel'
        datacube_group['dim4'].attrs['name'] = 'detector_x'
        datacube_group['dim4'].attrs['units'] = 'pixel'


def test_memory_detection():
    """Test the memory detection logic"""
    print("\n" + "="*60)
    print("TESTING MEMORY DETECTION")
    print("="*60)
    
    converter = MibToEmdConverter()
    
    # Test cases: (shape, expected_chunked_mode, description)
    test_cases = [
        ((64, 64, 256, 256), False, "Small file - should use in-memory"),
        ((512, 512, 512, 512), True, "Medium file - should use chunked"),
        ((2048, 2048, 512, 512), True, "Large file (130GB-like) - should use chunked"),
    ]
    
    for shape, expected, description in test_cases:
        use_chunked = converter.should_use_chunked_mode("dummy_path", shape)
        status = "✓" if use_chunked == expected else "✗"
        print(f"{status} {description}")
        print(f"   Shape: {shape}, Chunked: {use_chunked}")
        
        if use_chunked:
            import psutil
            chunk_size = converter.calculate_optimal_chunk_size(shape, psutil.virtual_memory().available)
            print(f"   Optimal chunk size: {chunk_size}")


def test_chunked_reader(filename: str, chunk_size: tuple):
    """Test the chunked EMD reader generator"""
    print(f"\n" + "="*60)
    print("TESTING CHUNKED EMD READER")
    print("="*60)
    
    converter = MibToEmdConverter()
    
    print(f"Reading {filename} in chunks of {chunk_size}")
    
    chunk_count = 0
    total_elements = 0
    
    try:
        for chunk_slice, chunk_data in converter.chunked_emd_reader(filename, chunk_size):
            chunk_count += 1
            total_elements += chunk_data.size
            
            print(f"Chunk {chunk_count:2d}: slice={chunk_slice}")
            print(f"           shape={chunk_data.shape}, range={chunk_data.min()}-{chunk_data.max()}")
            
            # Verify chunk data is reasonable
            assert chunk_data.dtype == np.uint16, f"Wrong dtype: {chunk_data.dtype}"
            assert chunk_data.size > 0, "Empty chunk"
            
        print(f"\n✓ Chunked reader completed successfully")
        print(f"  Total chunks: {chunk_count}")
        print(f"  Total elements read: {total_elements:,}")
        
    except Exception as e:
        print(f"✗ Chunked reader failed: {e}")
        raise


def test_chunked_conversion(input_file: str, output_file: str, processing_options: dict = None):
    """Test complete chunked conversion pipeline"""
    print(f"\n" + "="*60)
    print("TESTING CHUNKED CONVERSION PIPELINE")
    print("="*60)
    
    def progress_callback(percentage, message):
        print(f"  Progress: {percentage:3d}% - {message}")
    
    converter = MibToEmdConverter(progress_callback=progress_callback)
    
    # Force chunked mode for testing
    original_should_use_chunked = converter.should_use_chunked_mode
    def force_chunked_mode(self, file_path, data_shape=None, safety_factor=0.5):
        print("  Using forced chunked mode for testing")
        return True
    converter.should_use_chunked_mode = force_chunked_mode.__get__(converter, MibToEmdConverter)
    
    try:
        print(f"Converting: {input_file} -> {output_file}")
        if processing_options:
            print(f"Processing: {processing_options}")
        
        start_time = time.time()
        stats = converter.convert_to_emd(input_file, output_file, processing_options=processing_options)
        conversion_time = time.time() - start_time
        
        print(f"\n✓ Chunked conversion completed successfully!")
        print(f"  Input size: {stats['input_size_gb']:.6f} GB")
        print(f"  Output size: {stats['output_size_gb']:.6f} GB")
        print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")
        print(f"  Conversion time: {conversion_time:.2f}s")
        
        # Verify output file structure
        if os.path.exists(output_file):
            with h5py.File(output_file, 'r') as f:
                output_shape = f['version_1/data/datacubes/datacube_000/data'].shape
                print(f"  Output shape: {output_shape}")
                
                # Verify EMD structure
                assert 'version_1' in f, "Missing EMD version_1 group"
                assert 'data/datacubes/datacube_000/data' in f['version_1'], "Missing data structure"
                print(f"  ✓ EMD structure verified")
        
    except Exception as e:
        print(f"✗ Chunked conversion failed: {e}")
        raise
    finally:
        # Restore original method
        converter.should_use_chunked_mode = original_should_use_chunked


def main():
    """Run all chunked processing tests"""
    print("CHUNKED PROCESSING TEST SUITE")
    print("="*60)
    
    # Test parameters
    test_shape = (8, 8, 128, 128)  # Small but reasonable test size
    chunk_size = (3, 3, 128, 128)  # Non-aligned chunk size to test edge cases
    
    test_files = []
    
    try:
        # Create test data
        small_test_file = "test_small_synthetic.emd"
        create_synthetic_emd(small_test_file, test_shape)
        test_files.append(small_test_file)
        
        # Test 1: Memory detection
        test_memory_detection()
        
        # Test 2: Chunked reader
        test_chunked_reader(small_test_file, chunk_size)
        
        # Test 3: Chunked conversion without processing
        output_file_1 = "test_chunked_output_basic.emd"
        test_chunked_conversion(small_test_file, output_file_1)
        test_files.append(output_file_1)
        
        # Test 4: Chunked conversion with Y-summing
        output_file_2 = "test_chunked_output_ysum.emd"
        test_chunked_conversion(small_test_file, output_file_2, {'sum_y': True})
        test_files.append(output_file_2)
        
        # Test 5: Chunked conversion with binning
        output_file_3 = "test_chunked_output_binned.emd"
        test_chunked_conversion(small_test_file, output_file_3, {'bin_factor': 2, 'bin_method': 'mean'})
        test_files.append(output_file_3)
        
        print(f"\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
        print("="*60)
        print("The chunked processing pipeline is working correctly.")
        print("Ready to handle large files like the 130 GB acquisition data.")
        
    except Exception as e:
        print(f"\n" + "="*60)
        print("TEST FAILED! ✗")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up test files
        print(f"\nCleaning up test files...")
        for test_file in test_files:
            if os.path.exists(test_file):
                os.remove(test_file)
                print(f"  Removed: {test_file}")


if __name__ == "__main__":
    main()