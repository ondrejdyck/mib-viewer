#!/usr/bin/env python3
"""
Test GUI integration with adaptive converter v2
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mib_viewer.gui.enhanced_conversion_worker import EnhancedConversionWorker
from PyQt5.QtCore import QObject, QCoreApplication
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
        frame_data = np.zeros((qy, qx), dtype=np.uint16)
        
        for frame_idx in range(sy * sx):
            f.write(frame_data.tobytes())
            
    return file_path

def test_gui_integration():
    """Test GUI integration with converter v2"""
    print("Testing GUI Integration with Adaptive Converter V2...")
    
    # Initialize Qt application
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, 'test.mib')
        output_path = os.path.join(temp_dir, 'test.emd')
        
        # Create small mock MIB file for testing
        shape_4d = (16, 16, 64, 64)  # Small dataset - 256 frames
        create_mock_mib_file(input_path, shape_4d)
        
        print(f"Created mock MIB file: {shape_4d}")
        
        # Create worker
        worker = EnhancedConversionWorker(
            input_path=input_path,
            output_path=output_path,
            compression='gzip',
            compression_level=4,
            max_workers=4,
            processing_options=None  # No processing for this test
        )
        
        # Track conversion completion
        conversion_completed = False
        conversion_result = None
        conversion_error = None
        
        def on_finished(result):
            nonlocal conversion_completed, conversion_result
            conversion_completed = True
            conversion_result = result
            app.quit()
            
        def on_error(error):
            nonlocal conversion_completed, conversion_error
            conversion_completed = True
            conversion_error = error
            app.quit()
            
        # Connect signals
        worker.conversion_finished.connect(on_finished)
        worker.conversion_failed.connect(on_error)
        
        # Mock get_mib_properties to return our test data
        def mock_get_mib_properties(header_fields):
            return {
                'scan_x': 16,
                'scan_y': 16, 
                'detector_x': 64,
                'detector_y': 64,
                'header_size': 384,
                'frame_size': 64 * 64 * 2,  # uint16
                'dtype': np.uint16
            }
            
        # Mock get_data_file_info for GUI worker
        def mock_get_data_file_info(file_path):
            return {
                'file_size_bytes': os.path.getsize(file_path),
                'file_size_gb': os.path.getsize(file_path) / (1024**3),
                'shape_4d': shape_4d
            }
            
        with patch('mib_viewer.io.adaptive_converter_v2.get_mib_properties', mock_get_mib_properties), \
             patch('mib_viewer.gui.enhanced_conversion_worker.get_data_file_info', mock_get_data_file_info):
            
            # Start conversion
            print("Starting GUI conversion...")
            worker.run_conversion()
            
            # Run event loop until conversion completes
            app.exec_()
        
        # Check results
        if conversion_error:
            print(f"❌ Conversion failed: {conversion_error}")
            return False
            
        if not conversion_completed:
            print("❌ Conversion did not complete")
            return False
            
        if not conversion_result:
            print("❌ No conversion result returned")
            return False
            
        print("\n✅ GUI Conversion completed successfully!")
        print(f"Strategy: {conversion_result.get('chunking_strategy', 'unknown')}")
        print(f"Total chunks: {conversion_result.get('total_chunks', 'unknown')}")
        print(f"I/O reduction: {conversion_result.get('io_reduction_factor', 'unknown')}x")
        print(f"Conversion time: {conversion_result.get('total_time_s', 0):.2f}s")
        print(f"Chunks processed: {conversion_result.get('chunks_processed', 'unknown')}")
        
        # Verify output file exists and has correct structure
        if not os.path.exists(output_path):
            print("❌ Output file not created")
            return False
            
        import h5py
        try:
            with h5py.File(output_path, 'r') as f:
                print(f"✅ Output file structure:")
                print(f"  Groups: {list(f.keys())}")
                
                if 'version_1' in f:
                    version_group = f['version_1']
                    if 'data' in version_group and 'datacubes' in version_group['data']:
                        datacubes = version_group['data/datacubes']
                        if 'datacube_000' in datacubes:
                            datacube = datacubes['datacube_000']
                            if 'data' in datacube:
                                data_shape = datacube['data'].shape
                                print(f"  Data shape: {data_shape}")
                                
                                if data_shape == shape_4d:
                                    print("  ✅ Data shape correct!")
                                    return True
                                else:
                                    print(f"  ❌ Data shape mismatch! Expected {shape_4d}")
                                    
        except Exception as e:
            print(f"❌ Error reading output file: {e}")
            return False
        
        return True

if __name__ == '__main__':
    success = test_gui_integration()
    if success:
        print("\n🎉 All GUI integration tests passed!")
    else:
        print("\n❌ GUI integration tests failed!")
        sys.exit(1)