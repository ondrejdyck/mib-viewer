#!/usr/bin/env python3
"""
Compression Benchmark for 4D STEM Data

Tests various compression strategies on MIB files to determine optimal
storage and access patterns for large 4D datasets.
"""

import numpy as np
import h5py
import time
import os
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import sparse
import gzip
import pickle

# Import our MIB loading functions
try:
    from ..io.mib_loader import load_mib, get_mib_properties, MibProperties
except ImportError:
    from mib_viewer.io.mib_loader import load_mib, get_mib_properties, MibProperties

def analyze_data_sparsity(data_4d):
    """Analyze sparsity and distribution of 4D data"""
    print("=== Data Sparsity Analysis ===")
    
    # Flatten for analysis
    flat_data = data_4d.ravel()
    
    # Basic statistics
    print(f"Shape: {data_4d.shape}")
    print(f"Total elements: {flat_data.size:,}")
    print(f"Data type: {data_4d.dtype}")
    print(f"Memory size: {data_4d.nbytes / (1024**3):.3f} GB")
    
    # Value distribution
    unique_vals, counts = np.unique(flat_data, return_counts=True)
    zero_count = counts[0] if unique_vals[0] == 0 else 0
    
    print(f"\nValue Distribution:")
    print(f"  Min: {flat_data.min()}")
    print(f"  Max: {flat_data.max()}")
    print(f"  Mean: {flat_data.mean():.2f}")
    print(f"  Std: {flat_data.std():.2f}")
    print(f"  Zeros: {zero_count:,} ({100*zero_count/flat_data.size:.1f}%)")
    print(f"  Non-zeros: {flat_data.size - zero_count:,}")
    print(f"  Unique values: {len(unique_vals):,}")
    
    # Show value histogram for small values
    small_vals = flat_data[flat_data <= 50]
    if len(small_vals) > 0:
        print(f"  Values ≤ 50: {len(small_vals):,} ({100*len(small_vals)/flat_data.size:.1f}%)")
    
    return {
        'sparsity': zero_count / flat_data.size,
        'unique_values': len(unique_vals),
        'max_value': flat_data.max(),
        'mean_value': flat_data.mean()
    }

def benchmark_hdf5_compression(data_4d, output_dir):
    """Test HDF5 compression algorithms"""
    print("\n=== HDF5 Compression Benchmark ===")
    
    results = {}
    original_size = data_4d.nbytes
    
    # Test different compression algorithms
    compression_methods = {
        'none': None,
        'gzip_1': ('gzip', 1),
        'gzip_6': ('gzip', 6), 
        'gzip_9': ('gzip', 9),
        'lzf': ('lzf', None),
        'szip': ('szip', None)
    }
    
    # Generate chunk sizes based on actual data shape
    sy, sx, qy, qx = data_4d.shape
    chunk_sizes = [
        (min(32, sy), min(32, sx), min(256, qy), min(256, qx)),  # Real-space optimized
        (min(16, sy), min(16, sx), min(128, qy), min(128, qx)),  # Balanced
        (1, 1, qy, qx),                                          # Single frame chunks
    ]
    
    for chunk_name, chunk_size in zip(['real_space', 'balanced', 'single_frame'], chunk_sizes):
        print(f"\nChunk size: {chunk_size}")
        
        for method_name, method_config in compression_methods.items():
            try:
                filename = output_dir / f"test_{chunk_name}_{method_name}.h5"
                
                start_time = time.time()
                
                with h5py.File(filename, 'w') as f:
                    if method_config is None:
                        # No compression
                        dataset = f.create_dataset('data', data=data_4d, chunks=chunk_size)
                    else:
                        compression, opts = method_config
                        if opts is not None:
                            dataset = f.create_dataset('data', data=data_4d, 
                                                     compression=compression,
                                                     compression_opts=opts,
                                                     chunks=chunk_size)
                        else:
                            dataset = f.create_dataset('data', data=data_4d, 
                                                     compression=compression,
                                                     chunks=chunk_size)
                
                write_time = time.time() - start_time
                file_size = filename.stat().st_size
                compression_ratio = original_size / file_size
                
                # Test read speed
                start_time = time.time()
                with h5py.File(filename, 'r') as f:
                    # Read a few sample chunks
                    _ = f['data'][0:32, 0:32, :, :]  # One chunk
                    _ = f['data'][0, 0, :, :]        # Single frame
                    _ = f['data'][0:64, 0:64, 100, 100]  # K-space slice
                read_time = time.time() - start_time
                
                results[f"{chunk_name}_{method_name}"] = {
                    'file_size_mb': file_size / (1024**2),
                    'compression_ratio': compression_ratio,
                    'write_time': write_time,
                    'read_time': read_time,
                    'chunk_size': chunk_size
                }
                
                print(f"  {method_name:8s}: {file_size/(1024**2):6.1f} MB "
                      f"({compression_ratio:4.1f}x) "
                      f"W:{write_time:5.1f}s R:{read_time:5.2f}s")
                
                # Clean up
                filename.unlink()
                
            except Exception as e:
                print(f"  {method_name:8s}: FAILED ({e})")
    
    return results

def benchmark_sparse_storage(data_4d, output_dir):
    """Test sparse matrix storage"""
    print("\n=== Sparse Storage Benchmark ===")
    
    original_size = data_4d.nbytes
    results = {}
    
    # Test storing as 2D sparse matrices (frame by frame)
    sparse_frames = []
    start_time = time.time()
    
    for sy in range(data_4d.shape[0]):
        for sx in range(data_4d.shape[1]):
            frame = data_4d[sy, sx, :, :]
            # Convert to little-endian uint16 for sparse matrix compatibility
            frame_le = frame.astype(np.uint16)
            sparse_frame = sparse.csr_matrix(frame_le)
            sparse_frames.append(sparse_frame)
    
    creation_time = time.time() - start_time
    
    # Save to file
    filename = output_dir / "sparse_frames.pkl"
    start_time = time.time()
    with open(filename, 'wb') as f:
        pickle.dump(sparse_frames, f)
    save_time = time.time() - start_time
    
    file_size = filename.stat().st_size
    compression_ratio = original_size / file_size
    
    # Test loading speed
    start_time = time.time()
    with open(filename, 'rb') as f:
        loaded_frames = pickle.load(f)
    # Convert a few frames back to dense
    _ = loaded_frames[0].toarray()
    _ = loaded_frames[100].toarray()
    load_time = time.time() - start_time
    
    results['sparse_csr'] = {
        'file_size_mb': file_size / (1024**2),
        'compression_ratio': compression_ratio,
        'creation_time': creation_time,
        'save_time': save_time,
        'load_time': load_time
    }
    
    print(f"Sparse CSR: {file_size/(1024**2):6.1f} MB ({compression_ratio:4.1f}x)")
    print(f"  Create: {creation_time:.1f}s, Save: {save_time:.1f}s, Load: {load_time:.1f}s")
    
    # Clean up
    filename.unlink()
    
    return results

def benchmark_custom_compression(data_4d, output_dir):
    """Test custom compression strategies"""
    print("\n=== Custom Compression Benchmark ===")
    
    original_size = data_4d.nbytes
    results = {}
    
    # Strategy 1: uint8 + overflow map (for values mostly < 255)
    start_time = time.time()
    
    # Convert to uint8, track overflow
    data_uint8 = np.clip(data_4d, 0, 254).astype(np.uint8)
    overflow_mask = data_4d >= 255
    overflow_values = data_4d[overflow_mask]
    overflow_coords = np.where(overflow_mask)
    
    # Mark overflow pixels as 255 in uint8 array
    data_uint8[overflow_mask] = 255
    
    conversion_time = time.time() - start_time
    
    # Save compressed version
    filename_main = output_dir / "custom_uint8.npy"
    filename_overflow = output_dir / "custom_overflow.npz"
    
    start_time = time.time()
    np.save(filename_main, data_uint8)
    np.savez_compressed(filename_overflow, 
                       coords=np.column_stack(overflow_coords),
                       values=overflow_values)
    save_time = time.time() - start_time
    
    file_size = filename_main.stat().st_size + filename_overflow.stat().st_size
    compression_ratio = original_size / file_size
    
    print(f"uint8 + overflow: {file_size/(1024**2):6.1f} MB ({compression_ratio:4.1f}x)")
    print(f"  Overflow pixels: {len(overflow_values):,} ({100*len(overflow_values)/data_4d.size:.2f}%)")
    print(f"  Convert: {conversion_time:.1f}s, Save: {save_time:.1f}s")
    
    results['uint8_overflow'] = {
        'file_size_mb': file_size / (1024**2),
        'compression_ratio': compression_ratio,
        'conversion_time': conversion_time,
        'save_time': save_time,
        'overflow_fraction': len(overflow_values) / data_4d.size
    }
    
    # Clean up
    filename_main.unlink()
    filename_overflow.unlink()
    
    # Strategy 2: Simple gzip compression
    start_time = time.time()
    filename_gz = output_dir / "simple_gzip.npy.gz"
    with gzip.open(filename_gz, 'wb') as f:
        np.save(f, data_4d)
    gzip_time = time.time() - start_time
    
    gzip_size = filename_gz.stat().st_size
    gzip_ratio = original_size / gzip_size
    
    print(f"Simple gzip: {gzip_size/(1024**2):6.1f} MB ({gzip_ratio:4.1f}x)")
    print(f"  Time: {gzip_time:.1f}s")
    
    results['simple_gzip'] = {
        'file_size_mb': gzip_size / (1024**2),
        'compression_ratio': gzip_ratio,
        'compression_time': gzip_time
    }
    
    # Clean up
    filename_gz.unlink()
    
    return results

def create_summary_plot(all_results, output_dir):
    """Create summary plots of compression results"""
    print("\n=== Creating Summary Plots ===")
    
    # Extract data for plotting
    methods = []
    ratios = []
    sizes_mb = []
    
    for method, data in all_results.items():
        methods.append(method)
        ratios.append(data['compression_ratio'])
        sizes_mb.append(data['file_size_mb'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Compression ratio plot
    bars1 = ax1.bar(range(len(methods)), ratios)
    ax1.set_xlabel('Compression Method')
    ax1.set_ylabel('Compression Ratio')
    ax1.set_title('Compression Ratios (Higher = Better)')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, ratio in zip(bars1, ratios):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{ratio:.1f}x', ha='center', va='bottom')
    
    # File size plot
    bars2 = ax2.bar(range(len(methods)), sizes_mb)
    ax2.set_xlabel('Compression Method')
    ax2.set_ylabel('File Size (MB)')
    ax2.set_title('Compressed File Sizes (Lower = Better)')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, size in zip(bars2, sizes_mb):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{size:.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'compression_benchmark.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main benchmarking function"""
    print("4D STEM Data Compression Benchmark")
    print("=" * 50)
    
    # Setup
    data_dir = Path('/media/o2d/data/ORNL Dropbox/Ondrej Dyck/TEM data/2025/Andy 4D')
    output_dir = Path('compression_benchmark_results')
    output_dir.mkdir(exist_ok=True)
    
    # Load the smaller dataset for testing
    mib_file = data_dir / '64x64 Test.mib'
    
    if not mib_file.exists():
        print(f"ERROR: Could not find {mib_file}")
        return
    
    print(f"Loading: {mib_file}")
    print("This may take a moment...")
    
    # Load data
    start_time = time.time()
    data_4d = load_mib(str(mib_file))
    load_time = time.time() - start_time
    
    print(f"Loaded in {load_time:.1f}s")
    
    # Analyze sparsity
    sparsity_info = analyze_data_sparsity(data_4d)
    
    # Run benchmarks
    all_results = {}
    
    # HDF5 compression
    hdf5_results = benchmark_hdf5_compression(data_4d, output_dir)
    all_results.update(hdf5_results)
    
    # Sparse storage
    sparse_results = benchmark_sparse_storage(data_4d, output_dir)
    all_results.update(sparse_results)
    
    # Custom compression
    custom_results = benchmark_custom_compression(data_4d, output_dir)
    all_results.update(custom_results)
    
    # Create summary
    print("\n" + "=" * 70)
    print("COMPRESSION BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Method':<25} {'Size (MB)':<12} {'Ratio':<8} {'Notes'}")
    print("-" * 70)
    
    original_size_mb = data_4d.nbytes / (1024**2)
    print(f"{'Original':<25} {original_size_mb:<12.1f} {'1.0x':<8} {'Uncompressed'}")
    
    # Sort by compression ratio
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['compression_ratio'], reverse=True)
    
    for method, data in sorted_results:
        notes = ""
        if 'gzip' in method:
            notes = "Standard compression"
        elif 'sparse' in method:
            notes = f"Sparse matrix"
        elif 'uint8' in method:
            notes = f"8-bit + {data.get('overflow_fraction', 0)*100:.1f}% overflow"
        
        print(f"{method:<25} {data['file_size_mb']:<12.1f} {data['compression_ratio']:<8.1f} {notes}")
    
    # Create plots
    create_summary_plot(all_results, output_dir)
    
    # Save detailed results
    results_file = output_dir / 'detailed_results.txt'
    with open(results_file, 'w') as f:
        f.write("Detailed Compression Benchmark Results\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Original data: {data_4d.shape}, {data_4d.dtype}\n")
        f.write(f"Original size: {original_size_mb:.1f} MB\n")
        f.write(f"Sparsity: {sparsity_info['sparsity']*100:.1f}% zeros\n\n")
        
        for method, data in sorted_results:
            f.write(f"{method}:\n")
            for key, value in data.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
    
    print(f"\nDetailed results saved to: {results_file}")
    print(f"Plots saved to: {output_dir / 'compression_benchmark.png'}")

if __name__ == "__main__":
    main()