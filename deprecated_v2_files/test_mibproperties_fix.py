#!/usr/bin/env python3
"""
Simple test to verify the MibProperties fix is working
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import tempfile
import numpy as np
from mib_viewer.io.adaptive_converter_v2 import AdaptiveMibEmdConverterV2

def create_mock_mib_file(file_path: str, shape_4d: tuple):
    """Create a mock MIB file with header and data"""
    sy, sx, qy, qx = shape_4d
    
    # Create mock header (384 bytes) with proper MIB format
    # Based on get_mib_properties expectations:
    header_fields = [
        "HDR",          # head[0]
        "VERSION",      # head[1] 
        "00384",        # head[2] - header size
        "TIMESTAMP",    # head[3]
        str(qx),        # head[4] - detector width
        str(qy),        # head[5] - detector height
        "U16",          # head[6] - data type
        "1x1",          # head[7] - detector geometry
    ]
    header = ','.join(header_fields).ljust(384, ' ')
    
    # Create mock data
    with open(file_path, 'wb') as f:
        f.write(header.encode('utf-8'))
        
        # Write frames (simplified - just zeros)
        frame_data = np.zeros((qy, qx), dtype=np.uint16)
        
        for frame_idx in range(sy * sx):
            f.write(frame_data.tobytes())
            
    return file_path

def test_mibproperties_fix():
    """Test that MibProperties access is working"""
    print("Testing MibProperties fix...")
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, 'test.mib')
        output_path = os.path.join(temp_dir, 'test.emd')
        
        # Create small mock MIB file for testing
        shape_4d = (4, 4, 32, 32)  # Very small dataset - 16 frames
        create_mock_mib_file(input_path, shape_4d)
        
        print(f"Created mock MIB file: {shape_4d}")
        
        # Create converter
        converter = AdaptiveMibEmdConverterV2(
            compression='gzip',
            compression_level=4,
            max_workers=2,
            verbose=True  # Enable detailed logging
        )
        
        try:
            # Test conversion
            print("Starting conversion test...")
            stats = converter.convert_to_emd(
                input_path,
                output_path,
                processing_options=None  # No processing for this test
            )
            
            print("✅ Conversion completed successfully!")
            print(f"Strategy: {stats.get('chunking_strategy', 'unknown')}")
            print(f"Total chunks: {stats.get('total_chunks', 'unknown')}")
            print(f"Conversion time: {stats.get('total_time_s', 0):.2f}s")
            print(f"Chunks processed: {stats.get('chunks_processed', 'unknown')}")
            
            # Verify output file exists
            if not os.path.exists(output_path):
                print("❌ Output file not created")
                return False
                
            import h5py
            try:
                with h5py.File(output_path, 'r') as f:
                    print(f"✅ Output file structure: {list(f.keys())}")
                    
                    if 'version_1' in f and 'data' in f['version_1']:
                        datacubes = f['version_1/data/datacubes']
                        if 'datacube_000' in datacubes:
                            datacube = datacubes['datacube_000']
                            if 'data' in datacube:
                                data_shape = datacube['data'].shape
                                print(f"✅ Data shape: {data_shape}")
                                
                                if data_shape == shape_4d:
                                    print("✅ Data shape matches expected!")
                                    return True
                                else:
                                    print(f"❌ Data shape mismatch! Expected {shape_4d}")
                                    
            except Exception as e:
                print(f"❌ Error reading output file: {e}")
                return False
                
        except Exception as e:
            print(f"❌ Conversion failed: {e}")
            return False
        
        return True

if __name__ == '__main__':
    success = test_mibproperties_fix()
    if success:
        print("\n🎉 MibProperties fix test passed!")
    else:
        print("\n❌ MibProperties fix test failed!")
        sys.exit(1)