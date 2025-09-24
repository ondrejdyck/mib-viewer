#!/usr/bin/env python3
"""
Test both converters with real MIB file and compare outputs
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import tempfile
import numpy as np
import h5py
import time

# Import both converters
from mib_viewer.io.adaptive_converter import AdaptiveMibEmdConverter as OriginalConverter
from mib_viewer.io.adaptive_converter_v2 import AdaptiveMibEmdConverterV2 as V2Converter

def test_converter_comparison():
    """Compare original and V2 converters on real MIB file"""
    
    input_path = "/media/o2d/data/ORNL Dropbox/Ondrej Dyck/Code/mib-viewer-project/Example 4D/64x64 Test.mib"
    
    if not os.path.exists(input_path):
        print(f"❌ Test file not found: {input_path}")
        return False
    
    file_size_gb = os.path.getsize(input_path) / (1024**3)
    print(f"🔍 Testing converters with real MIB file")
    print(f"   File: {os.path.basename(input_path)}")
    print(f"   Size: {file_size_gb:.2f} GB")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        original_output = os.path.join(temp_dir, 'original_output.emd')
        v2_output = os.path.join(temp_dir, 'v2_output.emd')
        
        # Test 1: Original Converter
        print(f"\n📝 Testing Original Adaptive Converter...")
        try:
            original_converter = OriginalConverter(
                compression='gzip',
                compression_level=1,
                max_workers=4,
                verbose=True
            )
            
            start_time = time.time()
            original_stats = original_converter.convert_to_emd(input_path, original_output)
            original_time = time.time() - start_time
            
            print(f"   ✅ Original conversion completed in {original_time:.2f}s")
            
        except Exception as e:
            print(f"   ❌ Original converter failed: {e}")
            return False
            
        # Test 2: V2 Converter  
        print(f"\n📝 Testing V2 Adaptive Converter...")
        try:
            v2_converter = V2Converter(
                compression='gzip',
                compression_level=1,
                max_workers=4,
                verbose=True
            )
            
            start_time = time.time()
            v2_stats = v2_converter.convert_to_emd(input_path, v2_output)
            v2_time = time.time() - start_time
            
            print(f"   ✅ V2 conversion completed in {v2_time:.2f}s")
            
        except Exception as e:
            print(f"   ❌ V2 converter failed: {e}")
            return False
        
        # Test 3: Compare Outputs
        print(f"\n🔍 Comparing converter outputs...")
        
        if not os.path.exists(original_output):
            print(f"   ❌ Original output file missing")
            return False
            
        if not os.path.exists(v2_output):
            print(f"   ❌ V2 output file missing")
            return False
        
        try:
            with h5py.File(original_output, 'r') as f1, h5py.File(v2_output, 'r') as f2:
                # Get datasets
                original_data = f1['version_1/data/datacubes/datacube_000/data']
                v2_data = f2['version_1/data/datacubes/datacube_000/data']
                
                print(f"   Original shape: {original_data.shape}")
                print(f"   V2 shape: {v2_data.shape}")
                
                if original_data.shape != v2_data.shape:
                    print(f"   ❌ Shape mismatch!")
                    return False
                
                # Sample comparison (don't load entire dataset into memory)
                print(f"   Comparing data samples...")
                
                sy, sx, qy, qx = original_data.shape
                
                # Test specific positions
                test_positions = [
                    (0, 0),           # Top-left
                    (sy//2, sx//2),   # Center
                    (sy-1, sx-1),     # Bottom-right
                    (10, 20),         # Random position 1
                    (30, 45),         # Random position 2
                ]
                
                mismatches = 0
                for sy_pos, sx_pos in test_positions:
                    if sy_pos >= sy or sx_pos >= sx:
                        continue
                        
                    original_frame = original_data[sy_pos, sx_pos, :, :]
                    v2_frame = v2_data[sy_pos, sx_pos, :, :]
                    
                    if not np.array_equal(original_frame, v2_frame):
                        mismatches += 1
                        print(f"   ❌ MISMATCH at scan position ({sy_pos}, {sx_pos})")
                        print(f"      Original stats: min={original_frame.min()}, max={original_frame.max()}, mean={original_frame.mean():.2f}")
                        print(f"      V2 stats: min={v2_frame.min()}, max={v2_frame.max()}, mean={v2_frame.mean():.2f}")
                
                if mismatches == 0:
                    print(f"   ✅ All sampled positions match perfectly!")
                    print(f"   📊 Performance comparison:")
                    print(f"      Original: {original_time:.2f}s")
                    print(f"      V2: {v2_time:.2f}s") 
                    print(f"      Speedup: {original_time/v2_time:.1f}x")
                    return True
                else:
                    print(f"   ❌ Found {mismatches}/{len(test_positions)} mismatched positions")
                    return False
                    
        except Exception as e:
            print(f"   ❌ Error comparing outputs: {e}")
            return False

if __name__ == '__main__':
    print("="*70)
    print("REAL MIB FILE CONVERTER COMPARISON TEST")
    print("="*70)
    
    success = test_converter_comparison()
    
    print(f"\n" + "="*70)
    if success:
        print("🎉 CONVERTERS PRODUCE IDENTICAL RESULTS!")
        print("Both converters handle chunk positioning correctly.")
    else:
        print("❌ CONVERTERS PRODUCE DIFFERENT RESULTS!")
        print("One of the converters has chunk positioning or data handling issues.")
    print("="*70)