#!/usr/bin/env python3
"""
Test chunked processing with an actual 8 GB file

Usage:
    python experiments/test_8gb_file.py /path/to/your/8gb/file.mib
    python experiments/test_8gb_file.py /path/to/your/8gb/file.emd
"""

import sys
import os
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mib_viewer.io.mib_to_emd_converter import MibToEmdConverter


def test_real_file_chunked_processing(input_file: str):
    """Test chunked processing with a real large file"""
    
    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        return
    
    # Get file info
    file_size_gb = os.path.getsize(input_file) / (1024**3)
    print(f"🔍 Testing file: {os.path.basename(input_file)}")
    print(f"📊 File size: {file_size_gb:.2f} GB")
    
    # Determine output filename
    input_path = Path(input_file)
    output_file = input_path.with_name(f"{input_path.stem}_chunked_test.emd")
    
    def progress_callback(percentage, message):
        print(f"  📈 Progress: {percentage:3d}% - {message}")
    
    # Create converter with progress tracking
    converter = MibToEmdConverter(progress_callback=progress_callback)
    
    # Check if chunked mode will be used
    file_type = converter.detect_file_type(input_file)
    print(f"📁 Detected file type: {file_type.upper()}")
    
    if file_type == 'mib':
        metadata = converter.analyze_mib_file(input_file)
    else:
        metadata = converter.analyze_emd_file(input_file)
    
    will_use_chunked = converter.should_use_chunked_mode(input_file, metadata['shape_4d'])
    print(f"🧮 Data shape: {metadata['shape_4d']}")
    print(f"⚙️  Will use chunked processing: {will_use_chunked}")
    
    if will_use_chunked:
        import psutil
        chunk_size = converter.calculate_optimal_chunk_size(metadata['shape_4d'], psutil.virtual_memory().available)
        total_chunks = ((metadata['shape_4d'][0] + chunk_size[0] - 1) // chunk_size[0]) * \
                      ((metadata['shape_4d'][1] + chunk_size[1] - 1) // chunk_size[1])
        print(f"📦 Chunk size: {chunk_size}")
        print(f"🔢 Total chunks: {total_chunks}")
    
    # Ready to convert
    print(f"\n🎯 Ready to convert:")
    print(f"   Input:  {input_file}")
    print(f"   Output: {output_file}")
    print(f"   Mode:   {'Chunked' if will_use_chunked else 'In-memory'}")
    print(f"   Processing: 4x4 binning")
    print(f"\n🚀 Proceeding with conversion...")
    
    # Run the conversion
    try:
        print(f"\n🚀 Starting conversion...")
        start_time = time.time()
        
        # Test with 4x4 binning
        processing_options = {'bin_factor': 4, 'bin_method': 'mean'}
        
        stats = converter.convert_to_emd(
            input_file, 
            str(output_file), 
            processing_options=processing_options
        )
        
        end_time = time.time()
        
        print(f"\n✅ Conversion completed successfully!")
        print(f"📊 Results:")
        print(f"   Input size:  {stats['input_size_gb']:.3f} GB")
        print(f"   Output size: {stats['output_size_gb']:.3f} GB")
        print(f"   Compression: {stats['compression_ratio']:.1f}x")
        print(f"   Total time:  {end_time - start_time:.1f} seconds")
        print(f"   Rate:        {file_size_gb / (end_time - start_time):.2f} GB/sec")
        
        print(f"\n📁 Output file created: {output_file}")
        print("💾 Test output file kept for verification")
            
    except KeyboardInterrupt:
        print(f"\n⏸️  Conversion interrupted by user")
        if output_file.exists():
            os.remove(output_file)
            print("🗑️  Partial output file cleaned up")
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point"""
    if len(sys.argv) != 2:
        print("Usage: python experiments/test_8gb_file.py <input_file.mib|input_file.emd>")
        print("\nThis script tests chunked processing with your actual large files.")
        print("It will analyze the file, show what processing mode will be used,")
        print("and optionally run a test conversion with Y-summing.")
        sys.exit(1)
    
    input_file = sys.argv[1]
    test_real_file_chunked_processing(input_file)


if __name__ == "__main__":
    main()