#!/usr/bin/env python3
"""
Simple test of adaptive chunking v2 - just test core logic
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mib_viewer.io.adaptive_chunking_v2 import AdaptiveChunkCalculator, ChunkingStrategy
import tempfile
from unittest.mock import patch

def test_basic_chunking():
    """Test basic chunking functionality"""
    print("Testing adaptive chunking v2...")
    
    # Create a small temp file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(b'test data')
        temp_path = temp_file.name
    
    try:
        # Test small file (frame-based)
        shape_4d = (64, 64, 256, 256)
        
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('os.path.getsize') as mock_getsize:
            
            mock_memory.return_value.available = 32 * 1024**3  # 32 GB
            mock_getsize.return_value = 1 * 1024**3  # 1 GB file
            
            result = AdaptiveChunkCalculator.calculate_adaptive_chunks(temp_path, shape_4d)
            
            print(f"Small file test:")
            print(f"  Strategy: {result.strategy}")
            print(f"  Chunk dims: {result.chunk_dims}")
            print(f"  Total chunks: {result.total_chunks}")
            print(f"  I/O reduction: {result.io_reduction_factor}x")
            
            assert result.strategy == ChunkingStrategy.FRAME_BASED
            assert result.chunk_dims == (1, 1, 256, 256)
            
        # Test medium file (scan-line)
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('os.path.getsize') as mock_getsize:
            
            mock_memory.return_value.available = 32 * 1024**3  # 32 GB
            mock_getsize.return_value = 5 * 1024**3  # 5 GB file
            
            result = AdaptiveChunkCalculator.calculate_adaptive_chunks(temp_path, shape_4d)
            
            print(f"\nMedium file test:")
            print(f"  Strategy: {result.strategy}")
            print(f"  Chunk dims: {result.chunk_dims}")
            print(f"  Total chunks: {result.total_chunks}")
            print(f"  I/O reduction: {result.io_reduction_factor}x")
            
            assert result.strategy == ChunkingStrategy.SCAN_LINE
            assert result.chunk_dims[1] == 1  # scan_x should be 1
            assert result.chunk_dims[0] > 1   # scan_y should be > 1
            assert result.total_chunks < 64 * 64  # Fewer than frame-based
            
        # Test large file (block)  
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('os.path.getsize') as mock_getsize:
            
            mock_memory.return_value.available = 32 * 1024**3  # 32 GB
            mock_getsize.return_value = 50 * 1024**3  # 50 GB file
            
            result = AdaptiveChunkCalculator.calculate_adaptive_chunks(temp_path, shape_4d)
            
            print(f"\nLarge file test:")
            print(f"  Strategy: {result.strategy}")
            print(f"  Chunk dims: {result.chunk_dims}")
            print(f"  Total chunks: {result.total_chunks}")
            print(f"  I/O reduction: {result.io_reduction_factor}x")
            
            assert result.strategy == ChunkingStrategy.BLOCK
            assert result.chunk_dims[0] > 1   # scan_y should be > 1
            assert result.chunk_dims[1] > 1   # scan_x should be > 1
            assert result.total_chunks < 64 * 64  # Much fewer than frame-based
            
        print("\nAll tests passed! ✓")
        
    finally:
        os.unlink(temp_path)

if __name__ == '__main__':
    test_basic_chunking()