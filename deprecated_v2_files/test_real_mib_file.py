#!/usr/bin/env python3
"""
Test with real MIB file to verify the MibProperties fix and data integrity
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import tempfile
from mib_viewer.io.adaptive_converter_v2 import AdaptiveMibEmdConverterV2

def test_real_mib_file():
    """Test conversion with real MIB file"""
    
    input_path = "/media/o2d/data/ORNL Dropbox/Ondrej Dyck/Code/mib-viewer-project/Example 4D/1_256x256_2msec_graphene.mib"
    log_path = "/media/o2d/data/ORNL Dropbox/Ondrej Dyck/Code/mib-viewer-project/mib-viewer/conversion_test.log"
    
    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        return False
    
    print(f"Testing with real MIB file: {os.path.basename(input_path)}")
    print(f"Log file: {log_path}")
    file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    print(f"File size: {file_size_mb:.1f} MB")
    
    # Create temporary output file
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, 'test_output.emd')
        
        # Custom log callback that writes to file
        def log_to_file(message, level="INFO"):
            timestamp = __import__('datetime').datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {level}: {message}\n"
            print(message)  # Also print to console
            with open(log_path, 'a') as f:
                f.write(log_entry)
        
        # Clear log file
        with open(log_path, 'w') as f:
            f.write("=== ADAPTIVE CONVERTER V2 TEST LOG ===\n")
        
        # Create converter with more workers for 16-thread machine
        converter = AdaptiveMibEmdConverterV2(
            compression='gzip',
            compression_level=4,
            max_workers=12,  # Use 12 workers like we used to have
            log_callback=log_to_file,
            verbose=True  # Enable detailed logging
        )
        
        try:
            print("\nStarting conversion with real MIB file...")
            stats = converter.convert_to_emd(
                input_path,
                output_path,
                processing_options=None  # No processing for this test
            )
            
            print("\n" + "="*60)
            print("CONVERSION RESULTS:")
            print("="*60)
            print(f"Strategy: {stats.get('chunking_strategy', 'unknown')}")
            print(f"Total chunks: {stats.get('total_chunks', 'unknown')}")
            print(f"I/O reduction: {stats.get('io_reduction_factor', 'unknown')}x")
            print(f"Conversion time: {stats.get('total_time_s', 0):.2f}s")
            print(f"Chunks processed: {stats.get('chunks_processed', 'unknown')}")
            
            # Verify output file
            if not os.path.exists(output_path):
                print("❌ Output file not created")
                return False
                
            import h5py
            try:
                with h5py.File(output_path, 'r') as f:
                    print(f"\nOutput file structure: {list(f.keys())}")
                    
                    if 'version_1' in f and 'data' in f['version_1']:
                        datacubes = f['version_1/data/datacubes']
                        if 'datacube_000' in datacubes:
                            datacube = datacubes['datacube_000']
                            if 'data' in datacube:
                                data_shape = datacube['data'].shape
                                print(f"Output data shape: {data_shape}")
                                
                                # Check if it's a reasonable 4D shape
                                if len(data_shape) == 4:
                                    sy, sx, qy, qx = data_shape
                                    total_frames = sy * sx
                                    print(f"Scan size: {sy} x {sx} = {total_frames} frames")
                                    print(f"Detector size: {qy} x {qx}")
                                    
                                    # Based on filename, expect 256x256 scan
                                    if sy == 256 and sx == 256:
                                        print("✅ Scan size matches expected 256x256!")
                                        return True
                                    else:
                                        print(f"❌ Expected 256x256 scan, got {sy}x{sx}")
                                        return False
                                else:
                                    print(f"❌ Expected 4D shape, got {len(data_shape)}D")
                                    return False
                                    
            except Exception as e:
                print(f"❌ Error reading output file: {e}")
                return False
                
        except Exception as e:
            print(f"❌ Conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return False

if __name__ == '__main__':
    success = test_real_mib_file()
    if success:
        print("\n🎉 Real MIB file test PASSED!")
    else:
        print("\n❌ Real MIB file test FAILED!")
        sys.exit(1)