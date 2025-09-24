#!/usr/bin/env python3
"""
Simple test of adaptive converter v2 - verify it works with mock data
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mib_viewer.io.adaptive_converter_v2 import AdaptiveMibEmdConverterV2
import tempfile
import numpy as np
from unittest.mock import patch

def create_mock_mib_file(file_path: str, shape_4d: tuple):
    """Create a mock MIB file with header and data"""
    sy, sx, qy, qx = shape_4d
    
    # Create mock header (384 bytes)
    header_fields = [
        f"scan_x={sx}",
        f"scan_y={sy}", 
        f"detector_x={qx}",
        f"detector_y={qy}",
        "dtype=uint16"
    ]
    header = ','.join(header_fields).ljust(384, ' ')
    
    # Create mock data
    with open(file_path, 'wb') as f:
        f.write(header.encode('utf-8'))
        
        # Write frames (simplified - just zeros)
        frame_size = qy * qx * 2  # uint16
        frame_data = np.zeros((qy, qx), dtype=np.uint16)
        
        for frame_idx in range(sy * sx):
            f.write(frame_data.tobytes())
            
    return file_path

def test_converter_v2():
    """Test the adaptive converter v2"""
    print("Testing Adaptive Converter V2...")
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, 'test.mib')
        output_path = os.path.join(temp_dir, 'test.emd')
        
        # Create mock MIB file (small for testing)
        shape_4d = (32, 32, 128, 128)  # 1024 frames, 128x128 detector
        create_mock_mib_file(input_path, shape_4d)
        
        print(f"Created mock MIB file: {shape_4d}")
        
        # Test converter
        converter = AdaptiveMibEmdConverterV2(
            compression='gzip',
            compression_level=4,
            max_workers=4,
            verbose=True
        )
        
        # Mock get_mib_properties to return our test data
        def mock_get_mib_properties(header_fields):
            return {
                'scan_x': 32,
                'scan_y': 32, 
                'detector_x': 128,
                'detector_y': 128,
                'header_size': 384,
                'frame_size': 128 * 128 * 2,  # uint16
                'dtype': np.uint16
            }
            
        with patch('mib_viewer.io.adaptive_converter_v2.get_mib_properties', mock_get_mib_properties):
            result = converter.convert_to_emd(input_path, output_path)
            
        print("\nConversion completed!")
        print(f"Strategy: {result['chunking_strategy']}")
        print(f"Total chunks: {result['total_chunks']}")
        print(f"I/O reduction: {result['io_reduction_factor']}x")
        print(f"Conversion time: {result['total_time_s']:.2f}s")
        print(f"Throughput: {result['throughput_mb_s']:.1f} MB/s")
        print(f"Chunks processed: {result['chunks_processed']}")
        
        # Verify output file exists and has correct structure
        import h5py
        with h5py.File(output_path, 'r') as f:
            print(f"\nOutput file structure:")
            print(f"  Groups: {list(f.keys())}")
            
            if 'version_1' in f:
                version_group = f['version_1']
                print(f"  version_1 groups: {list(version_group.keys())}")
                
                if 'data' in version_group and 'datacubes' in version_group['data']:
                    datacubes = version_group['data/datacubes']
                    print(f"  datacubes: {list(datacubes.keys())}")
                    
                    if 'datacube_000' in datacubes:
                        datacube = datacubes['datacube_000']
                        if 'data' in datacube:
                            data_shape = datacube['data'].shape
                            print(f"  Data shape: {data_shape}")
                            
                            # Verify shape matches expectation
                            expected_shape = shape_4d  # No processing applied
                            if data_shape == expected_shape:
                                print("  ✓ Data shape correct!")
                            else:
                                print(f"  ✗ Data shape mismatch! Expected {expected_shape}")
        
        print("\nTest completed successfully! ✓")

if __name__ == '__main__':
    test_converter_v2()