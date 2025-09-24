#!/usr/bin/env python3
"""
Debug script to compare chunked vs in-memory processing outputs

This script loads both EMD files and compares their data values to identify
where the chunked processing is corrupting the data.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_emd_data(file_path: str, label: str):
    """Analyze an EMD file and return statistics"""
    print(f"\n=== ANALYZING {label.upper()} ===")
    
    with h5py.File(file_path, 'r') as f:
        data = f['version_1/data/datacubes/datacube_000/data'][:]
        
    print(f"Shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Total elements: {data.size:,}")
    print(f"Memory size: {data.nbytes / (1024**2):.1f} MB")
    
    # Statistical analysis
    print(f"Min value: {data.min()}")
    print(f"Max value: {data.max()}")
    print(f"Mean value: {data.mean():.3f}")
    print(f"Std dev: {data.std():.3f}")
    
    # Count zeros
    zero_count = np.sum(data == 0)
    zero_percentage = (zero_count / data.size) * 100
    print(f"Zero values: {zero_count:,} ({zero_percentage:.1f}%)")
    
    # Count non-zero values
    nonzero_count = np.sum(data > 0)
    nonzero_percentage = (nonzero_count / data.size) * 100
    print(f"Non-zero values: {nonzero_count:,} ({nonzero_percentage:.1f}%)")
    
    # Sample a few frames for detailed analysis
    sample_frame = data[128, 128]  # Middle frame
    print(f"Sample frame (128,128) stats:")
    print(f"  Shape: {sample_frame.shape}")
    print(f"  Min: {sample_frame.min()}, Max: {sample_frame.max()}")
    print(f"  Mean: {sample_frame.mean():.3f}")
    print(f"  Zeros: {np.sum(sample_frame == 0)} / {sample_frame.size}")
    
    return data

def compare_files(chunked_file: str, inmemory_file: str):
    """Compare the two EMD files"""
    print("="*80)
    print("COMPARING CHUNKED VS IN-MEMORY PROCESSING")
    print("="*80)
    
    # Load both datasets
    chunked_data = analyze_emd_data(chunked_file, "CHUNKED PROCESSING")
    inmemory_data = analyze_emd_data(inmemory_file, "IN-MEMORY PROCESSING")
    
    # Direct comparison
    print(f"\n=== DIRECT COMPARISON ===")
    print(f"Shape match: {chunked_data.shape == inmemory_data.shape}")
    
    if chunked_data.shape == inmemory_data.shape:
        # Element-wise comparison
        difference = inmemory_data - chunked_data
        print(f"Difference stats:")
        print(f"  Min diff: {difference.min()}")
        print(f"  Max diff: {difference.max()}")
        print(f"  Mean diff: {difference.mean():.3f}")
        print(f"  Std diff: {difference.std():.3f}")
        
        # Count different pixels
        different_pixels = np.sum(difference != 0)
        different_percentage = (different_pixels / difference.size) * 100
        print(f"Different pixels: {different_pixels:,} ({different_percentage:.1f}%)")
        
        # Check if chunked has zeros where inmemory has values
        chunked_zeros = (chunked_data == 0)
        inmemory_nonzeros = (inmemory_data > 0)
        corrupted_pixels = np.sum(chunked_zeros & inmemory_nonzeros)
        print(f"Pixels zeroed by chunked processing: {corrupted_pixels:,}")
        
        # Sample comparison
        print(f"\nSample frame comparison (frame 128,128):")
        chunked_sample = chunked_data[128, 128]
        inmemory_sample = inmemory_data[128, 128]
        sample_diff = inmemory_sample - chunked_sample
        
        print(f"  In-memory non-zeros: {np.sum(inmemory_sample > 0)}")
        print(f"  Chunked non-zeros: {np.sum(chunked_sample > 0)}")
        print(f"  Sample max diff: {sample_diff.max()}")
        
        # Look for patterns in the corruption
        print(f"\nCorruption pattern analysis:")
        
        # Check if corruption follows detector geometry
        sy, sx, qy, qx = chunked_data.shape
        
        # Sum over scan dimensions to see detector pattern
        inmemory_detector_sum = np.sum(inmemory_data, axis=(0,1))
        chunked_detector_sum = np.sum(chunked_data, axis=(0,1))
        detector_diff = inmemory_detector_sum - chunked_detector_sum
        
        print(f"  Detector sum difference max: {detector_diff.max()}")
        print(f"  Detector pixels with corruption: {np.sum(detector_diff > 0)}")
        
        # Check if it's a circular bright field pattern
        center_y, center_x = qy//2, qx//2
        y, x = np.ogrid[:qy, :qx]
        radius_map = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        
        # Check corruption vs radius
        for radius in [qy//8, qy//4, qy//2]:
            mask = radius_map > radius
            corruption_outside = np.sum(detector_diff[mask] > 0)
            total_outside = np.sum(mask)
            print(f"  Corruption outside radius {radius}: {corruption_outside}/{total_outside}")
    
    return chunked_data, inmemory_data

def main():
    chunked_file = "/media/o2d/data/ORNL Dropbox/Ondrej Dyck/Code/mib-viewer-project/Example 4D/1_256x256_2msec_graphene_chunked_test.emd"
    inmemory_file = "/media/o2d/data/ORNL Dropbox/Ondrej Dyck/Code/mib-viewer-project/Example 4D/1_256x256_2msec_graphene_inmemory_test.emd"
    
    if not Path(chunked_file).exists():
        print(f"Chunked file not found: {chunked_file}")
        return
    
    if not Path(inmemory_file).exists():
        print(f"In-memory file not found: {inmemory_file}")
        return
    
    chunked_data, inmemory_data = compare_files(chunked_file, inmemory_file)

if __name__ == "__main__":
    main()