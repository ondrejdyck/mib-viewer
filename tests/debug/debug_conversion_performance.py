#!/usr/bin/env python3
"""
Debug Conversion Performance Issues

This script investigates why multithreaded conversion might be slower
than single-threaded conversion and provides recommendations.
"""

import sys
import time
import psutil
import os
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mib_viewer.io.mib_to_emd_converter import MibToEmdConverter
from mib_viewer.io.enhanced_converter import EnhancedMibEmdConverter


class PerformanceMonitor:
    """Monitor system performance during conversion"""
    
    def __init__(self):
        self.monitoring = False
        self.stats = []
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.monitoring = True
        self.stats = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        return self.stats
    
    def _monitor_loop(self):
        """Monitor system stats in background"""
        while self.monitoring:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk_io = psutil.disk_io_counters()
            
            self.stats.append({
                'timestamp': time.time(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_read_mb': disk_io.read_bytes / (1024**2) if disk_io else 0,
                'disk_write_mb': disk_io.write_bytes / (1024**2) if disk_io else 0
            })
    
    def get_summary(self):
        """Get performance summary"""
        if not self.stats:
            return "No monitoring data"
        
        avg_cpu = sum(s['cpu_percent'] for s in self.stats) / len(self.stats)
        max_cpu = max(s['cpu_percent'] for s in self.stats)
        
        avg_memory = sum(s['memory_percent'] for s in self.stats) / len(self.stats)
        max_memory = max(s['memory_percent'] for s in self.stats)
        
        return f"""Performance Summary:
  CPU Usage: avg={avg_cpu:.1f}%, max={max_cpu:.1f}%
  Memory Usage: avg={avg_memory:.1f}%, max={max_memory:.1f}%
  Duration: {len(self.stats)} seconds"""


def test_single_threaded(file_path: str) -> dict:
    """Test original single-threaded conversion"""
    print("\n" + "="*60)
    print("TESTING SINGLE-THREADED CONVERSION")
    print("="*60)
    
    output_file = "debug_single_thread_output.emd"
    
    # Performance monitoring
    monitor = PerformanceMonitor()
    
    def progress_callback(percentage, message):
        if percentage % 20 == 0:  # Print every 20%
            print(f"  Single-threaded: {percentage}% - {message}")
    
    try:
        # Original converter
        converter = MibToEmdConverter(progress_callback=progress_callback)
        
        print("Starting single-threaded conversion...")
        monitor.start_monitoring()
        start_time = time.time()
        
        stats = converter.convert_to_emd(
            file_path, output_file,
            processing_options={'bin_factor': 4, 'bin_method': 'mean'}
        )
        
        end_time = time.time()
        perf_stats = monitor.stop_monitoring()
        
        total_time = end_time - start_time
        
        print(f"✓ Single-threaded completed in {total_time:.1f}s")
        print(f"  Throughput: {(stats['input_size_gb']*1024)/total_time:.1f} MB/s")
        print(f"  Compression: {stats['compression_ratio']:.1f}x")
        print(monitor.get_summary())
        
        # Cleanup
        if os.path.exists(output_file):
            os.remove(output_file)
        
        return {
            'time': total_time,
            'throughput_mb_s': (stats['input_size_gb']*1024)/total_time,
            'compression_ratio': stats['compression_ratio'],
            'performance_stats': perf_stats
        }
        
    except Exception as e:
        monitor.stop_monitoring()
        print(f"✗ Single-threaded conversion failed: {e}")
        return None


def test_multithreaded_variations(file_path: str) -> dict:
    """Test multithreaded conversion with different configurations"""
    print("\n" + "="*60)
    print("TESTING MULTITHREADED VARIATIONS")
    print("="*60)
    
    # Test different worker counts
    worker_configs = [1, 2, 4, 8]  # Don't test 12 initially
    results = {}
    
    for max_workers in worker_configs:
        print(f"\n--- Testing with {max_workers} workers ---")
        
        output_file = f"debug_multithread_{max_workers}_output.emd"
        monitor = PerformanceMonitor()
        
        def progress_callback(percentage, message):
            if percentage % 20 == 0:  # Print every 20%
                print(f"  {max_workers}-threaded: {percentage}% - {message}")
        
        try:
            # Enhanced converter with specific worker count
            converter = EnhancedMibEmdConverter(
                max_workers=max_workers,
                progress_callback=progress_callback
            )
            
            monitor.start_monitoring()
            start_time = time.time()
            
            stats = converter.convert(
                file_path, output_file,
                processing_options={'bin_factor': 4, 'bin_method': 'mean'},
                force_pipeline=True  # Force pipeline even for smaller files
            )
            
            end_time = time.time()
            perf_stats = monitor.stop_monitoring()
            
            total_time = end_time - start_time
            
            print(f"✓ {max_workers}-threaded completed in {total_time:.1f}s")
            
            if 'pipeline_throughput_mb_s' in stats:
                print(f"  Pipeline throughput: {stats['pipeline_throughput_mb_s']:.1f} MB/s")
                print(f"  Parallelization efficiency: {stats['parallelization_efficiency']:.1f}%")
                print(f"  Stage breakdown:")
                print(f"    Load: {stats['load_time_s']:.1f}s")
                print(f"    Process: {stats['process_time_s']:.1f}s") 
                print(f"    Write: {stats['write_time_s']:.1f}s")
            
            print(monitor.get_summary())
            
            # Cleanup
            if os.path.exists(output_file):
                os.remove(output_file)
            
            results[max_workers] = {
                'time': total_time,
                'throughput_mb_s': stats.get('pipeline_throughput_mb_s', 0),
                'efficiency': stats.get('parallelization_efficiency', 0),
                'load_time': stats.get('load_time_s', 0),
                'process_time': stats.get('process_time_s', 0),
                'write_time': stats.get('write_time_s', 0),
                'performance_stats': perf_stats
            }
            
        except Exception as e:
            monitor.stop_monitoring()
            print(f"✗ {max_workers}-threaded conversion failed: {e}")
            if os.path.exists(output_file):
                os.remove(output_file)
    
    return results


def analyze_bottlenecks(single_result: dict, multi_results: dict):
    """Analyze where the bottlenecks are"""
    print("\n" + "="*60)
    print("BOTTLENECK ANALYSIS")
    print("="*60)
    
    if not single_result:
        print("❌ Cannot analyze - single-threaded test failed")
        return
    
    baseline_time = single_result['time']
    baseline_throughput = single_result['throughput_mb_s']
    
    print(f"Baseline (single-threaded): {baseline_time:.1f}s, {baseline_throughput:.1f} MB/s")
    
    print("\nMulti-threaded performance:")
    print("Workers | Time (s) | Speedup | Throughput | Efficiency | Bottleneck")
    print("-" * 70)
    
    for workers, result in multi_results.items():
        speedup = baseline_time / result['time']
        throughput = result['throughput_mb_s']
        efficiency = result['efficiency']
        
        # Identify bottleneck
        load_time = result['load_time']
        process_time = result['process_time'] 
        write_time = result['write_time']
        
        bottleneck = "Unknown"
        if max(load_time, process_time, write_time) == load_time:
            bottleneck = "I/O Loading"
        elif max(load_time, process_time, write_time) == process_time:
            bottleneck = "CPU Processing"
        elif max(load_time, process_time, write_time) == write_time:
            bottleneck = "I/O Writing"
        
        speedup_str = f"{speedup:.2f}x" if speedup > 1 else f"{speedup:.2f}x ⚠"
        print(f"{workers:7} | {result['time']:8.1f} | {speedup_str:7} | {throughput:10.1f} | {efficiency:9.1f}% | {bottleneck}")
    
    # Recommendations
    print(f"\n📊 ANALYSIS:")
    
    best_workers = max(multi_results.keys(), key=lambda w: baseline_time / multi_results[w]['time'])
    best_speedup = baseline_time / multi_results[best_workers]['time']
    
    if best_speedup < 1.1:
        print("⚠️  ISSUE: Multithreading provides little benefit (<10% speedup)")
        print("   Likely cause: I/O bottlenecked workload")
        print("   Recommendation: Use single-threaded for this file size")
    elif best_workers == 1:
        print("⚠️  ISSUE: Single worker performs best in multithreaded pipeline")
        print("   Likely cause: Threading overhead > benefits")
        print("   Recommendation: Optimize pipeline or use original converter")
    elif best_speedup > 1.5:
        print(f"✅ SUCCESS: Best performance with {best_workers} workers ({best_speedup:.1f}x speedup)")
        print(f"   Recommendation: Use {best_workers} workers for this file type/size")
    else:
        print(f"📈 MODERATE: {best_workers} workers provide {best_speedup:.1f}x speedup")
        print("   May be worth using multithreaded pipeline")


def check_system_resources():
    """Check system resources and threading environment"""
    print("\n" + "="*60)
    print("SYSTEM RESOURCE ANALYSIS")
    print("="*60)
    
    # CPU info
    cpu_count = psutil.cpu_count(logical=True)
    cpu_physical = psutil.cpu_count(logical=False)
    print(f"CPU cores: {cpu_physical} physical, {cpu_count} logical")
    
    # Memory info
    memory = psutil.virtual_memory()
    print(f"Memory: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available")
    
    # Disk info
    disk_usage = psutil.disk_usage('/')
    print(f"Disk: {disk_usage.total / (1024**3):.0f} GB total, {disk_usage.free / (1024**3):.0f} GB free")
    
    # Threading recommendations
    print(f"\n💡 THREADING RECOMMENDATIONS:")
    print(f"   Optimal I/O threads: 2-4 (disk limited)")
    print(f"   Optimal CPU threads: {min(cpu_physical, 8)} (for processing)")
    print(f"   Current system can support: {cpu_count} total threads")
    
    if cpu_count > 8:
        print("   ⚠️  High thread count - diminishing returns likely above 8 threads")


def main():
    """Main performance debugging function"""
    if len(sys.argv) < 2:
        print("Usage: python debug_conversion_performance.py <file_path>")
        print("\nExample:")
        print("  python debug_conversion_performance.py '/path/to/large_file.mib'")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        sys.exit(1)
    
    file_size_gb = os.path.getsize(file_path) / (1024**3)
    
    print("CONVERSION PERFORMANCE DEBUGGING")
    print("="*80)
    print(f"Testing file: {os.path.basename(file_path)}")
    print(f"File size: {file_size_gb:.2f} GB")
    
    # System analysis
    check_system_resources()
    
    # Test single-threaded (baseline)
    single_result = test_single_threaded(file_path)
    
    # Test multithreaded variations  
    multi_results = test_multithreaded_variations(file_path)
    
    # Analyze bottlenecks
    analyze_bottlenecks(single_result, multi_results)
    
    print(f"\n{'='*80}")
    print("🔧 DEBUGGING COMPLETE")
    print("="*80)
    print("Use this analysis to optimize your conversion settings!")


if __name__ == "__main__":
    main()