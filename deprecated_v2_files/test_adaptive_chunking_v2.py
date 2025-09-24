"""
Unit tests for adaptive chunking v2

Tests the adaptive chunking strategy to ensure:
1. Correct strategy selection based on file size
2. Proper I/O reduction calculations  
3. Memory constraints respected
4. Chunk boundary calculations correct
5. Performance improvements over frame-based chunking
"""

import unittest
import tempfile
import os
from unittest.mock import patch
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mib_viewer.io.adaptive_chunking_v2 import (
    AdaptiveChunkCalculator,
    ChunkingStrategy,
    analyze_file_for_chunking,
    print_chunking_analysis
)


class TestAdaptiveChunking(unittest.TestCase):
    """Test the adaptive chunking strategy selection and calculations"""
    
    def setUp(self):
        """Create temporary files for testing"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create small test files but mock their apparent size for testing
        self.small_file = self._create_temp_file("small.mib", 1024)       # 1 KB actual
        self.medium_file = self._create_temp_file("medium.mib", 2048)     # 2 KB actual  
        self.large_file = self._create_temp_file("large.mib", 4096)       # 4 KB actual
        
    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def _create_temp_file(self, name: str, size_bytes: int) -> str:
        """Create a temporary file of specified size"""
        file_path = os.path.join(self.temp_dir, name)
        with open(file_path, 'wb') as f:
            # Write in chunks to avoid memory issues
            chunk_size = 1024 * 1024  # 1MB chunks
            written = 0
            while written < size_bytes:
                remaining = min(chunk_size, size_bytes - written)
                f.write(b'0' * remaining)
                written += remaining
        return file_path
    
    def test_strategy_selection_small_files(self):
        """Test that small files use frame-based chunking"""
        shape_4d = (64, 64, 256, 256)  # Small dataset
        
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('os.path.getsize') as mock_getsize:
            mock_memory.return_value.available = 32 * 1024**3  # 32 GB available
            mock_getsize.return_value = 1 * 1024**3  # Mock 1 GB file
            
            result = AdaptiveChunkCalculator.calculate_adaptive_chunks(
                self.small_file, shape_4d
            )
            
        self.assertEqual(result.strategy, ChunkingStrategy.FRAME_BASED)
        self.assertEqual(result.chunk_dims, (1, 1, 256, 256))
        self.assertEqual(result.total_chunks, 64 * 64)  # One chunk per frame
        
    def test_strategy_selection_medium_files(self):
        """Test that medium files use scan-line chunking"""
        shape_4d = (128, 128, 256, 256)  # Medium dataset
        
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('os.path.getsize') as mock_getsize:
            mock_memory.return_value.available = 32 * 1024**3  # 32 GB available
            mock_getsize.return_value = 5 * 1024**3  # Mock 5 GB file
            
            result = AdaptiveChunkCalculator.calculate_adaptive_chunks(
                self.medium_file, shape_4d
            )
            
        self.assertEqual(result.strategy, ChunkingStrategy.SCAN_LINE)
        self.assertEqual(result.chunk_dims[1], 1)  # scan_x should be 1 for scan-line
        self.assertGreater(result.chunk_dims[0], 1)  # scan_y should be > 1
        self.assertLess(result.total_chunks, 128 * 128)  # Fewer chunks than frames
        
    def test_strategy_selection_large_files(self):
        """Test that large files use block chunking"""
        shape_4d = (256, 256, 512, 512)  # Large dataset
        
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('os.path.getsize') as mock_getsize:
            mock_memory.return_value.available = 32 * 1024**3  # 32 GB available
            mock_getsize.return_value = 50 * 1024**3  # Mock 50 GB file
            
            result = AdaptiveChunkCalculator.calculate_adaptive_chunks(
                self.large_file, shape_4d
            )
            
        self.assertEqual(result.strategy, ChunkingStrategy.BLOCK)
        self.assertGreater(result.chunk_dims[0], 1)  # scan_y should be > 1
        self.assertGreater(result.chunk_dims[1], 1)  # scan_x should be > 1
        self.assertLess(result.total_chunks, 256 * 256)  # Much fewer chunks than frames
        
    def test_io_reduction_calculation(self):
        """Test that I/O reduction is calculated correctly"""
        shape_4d = (100, 100, 256, 256)  # 10,000 frames total
        
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('os.path.getsize') as mock_getsize:
            mock_memory.return_value.available = 32 * 1024**3  # 32 GB available
            mock_getsize.return_value = 5 * 1024**3  # Mock 5 GB file
            
            result = AdaptiveChunkCalculator.calculate_adaptive_chunks(
                self.medium_file, shape_4d
            )
            
        original_operations = 100 * 100  # Frame-based would be 10,000 I/O ops
        expected_reduction = original_operations // result.total_chunks
        
        self.assertEqual(result.io_reduction_factor, expected_reduction)
        self.assertGreaterEqual(result.io_reduction_factor, 4)  # At least 4x reduction
        
    def test_memory_constraint_respected(self):
        """Test that chunks stay within memory limits"""
        shape_4d = (256, 256, 1024, 1024)  # Very large detector
        
        # Simulate low memory system
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('os.path.getsize') as mock_getsize:
            mock_memory.return_value.available = 4 * 1024**3  # Only 4 GB available
            mock_getsize.return_value = 50 * 1024**3  # Mock 50 GB file
            
            result = AdaptiveChunkCalculator.calculate_adaptive_chunks(
                self.large_file, shape_4d
            )
            
        max_allowed_memory = 4 * AdaptiveChunkCalculator.MAX_MEMORY_FRACTION  # 20% of 4GB
        self.assertLessEqual(result.estimated_memory_per_chunk_gb, max_allowed_memory)
        
    def test_chunk_boundary_calculations(self):
        """Test that chunk slices cover entire dataset without gaps or overlaps"""
        shape_4d = (63, 127, 256, 256)  # Non-power-of-2 dimensions
        
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.available = 32 * 1024**3
            
            result = AdaptiveChunkCalculator.calculate_adaptive_chunks(
                self.medium_file, shape_4d
            )
            
        # Check that all scan positions are covered exactly once
        sy, sx = shape_4d[:2]
        covered_positions = set()
        
        for chunk in result.chunks:
            y_slice, x_slice = chunk.input_slice[:2]
            for y in range(y_slice.start, y_slice.stop):
                for x in range(x_slice.start, x_slice.stop):
                    self.assertNotIn((y, x), covered_positions, 
                                   f"Position ({y},{x}) covered by multiple chunks")
                    covered_positions.add((y, x))
        
        # Check that all positions are covered
        expected_positions = {(y, x) for y in range(sy) for x in range(sx)}
        self.assertEqual(covered_positions, expected_positions, 
                        "Some scan positions not covered by any chunk")
        
    def test_chunk_count_optimization(self):
        """Test that chunk count is optimized for threading"""
        shape_4d = (128, 128, 256, 256)
        
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.available = 32 * 1024**3
            
            result = AdaptiveChunkCalculator.calculate_adaptive_chunks(
                self.medium_file, shape_4d
            )
            
        # Should be in optimal range for threading
        min_chunks = AdaptiveChunkCalculator.MIN_CHUNKS
        max_chunks = AdaptiveChunkCalculator.MAX_CHUNKS
        
        self.assertGreaterEqual(result.total_chunks, min_chunks)
        self.assertLessEqual(result.total_chunks, max_chunks)
        
    def test_target_chunk_count_override(self):
        """Test that target_chunk_count parameter works"""
        shape_4d = (128, 128, 256, 256)
        target_chunks = 12
        
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.available = 32 * 1024**3
            
            result = AdaptiveChunkCalculator.calculate_adaptive_chunks(
                self.medium_file, shape_4d, target_chunk_count=target_chunks
            )
            
        # Should be close to target (may not be exact due to boundary constraints)
        self.assertLessEqual(abs(result.total_chunks - target_chunks), 4)
        
    def test_analyze_file_convenience_function(self):
        """Test the convenience function works correctly"""
        shape_4d = (64, 64, 256, 256)
        
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.available = 32 * 1024**3
            
            result = analyze_file_for_chunking(self.small_file, shape_4d)
            
        self.assertEqual(result.strategy, ChunkingStrategy.FRAME_BASED)
        self.assertGreater(result.file_size_gb, 0)
        
    def test_performance_comparison(self):
        """Test that adaptive chunking provides significant performance improvement"""
        
        test_cases = [
            (self.medium_file, (128, 128, 256, 256), 4),    # At least 4x improvement
            (self.large_file, (256, 256, 512, 512), 50),    # At least 50x improvement
        ]
        
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.available = 32 * 1024**3
            
            for file_path, shape_4d, min_improvement in test_cases:
                with self.subTest(file=os.path.basename(file_path)):
                    result = AdaptiveChunkCalculator.calculate_adaptive_chunks(
                        file_path, shape_4d
                    )
                    
                    self.assertGreaterEqual(result.io_reduction_factor, min_improvement,
                                          f"I/O reduction {result.io_reduction_factor}x less than expected {min_improvement}x")
                    
    def test_edge_case_single_frame(self):
        """Test edge case with single frame dataset"""
        shape_4d = (1, 1, 256, 256)
        
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.available = 32 * 1024**3
            
            result = AdaptiveChunkCalculator.calculate_adaptive_chunks(
                self.small_file, shape_4d
            )
            
        self.assertEqual(result.total_chunks, 1)
        self.assertEqual(result.chunks[0].expected_shape, shape_4d)
        
    def test_edge_case_very_constrained_memory(self):
        """Test behavior with very limited memory"""
        shape_4d = (128, 128, 1024, 1024)  # Large detector
        
        # Simulate very low memory (1GB)
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.available = 1024**3
            
            result = AdaptiveChunkCalculator.calculate_adaptive_chunks(
                self.medium_file, shape_4d
            )
            
        # Should still produce valid chunks, just smaller ones
        self.assertGreater(result.total_chunks, 0)
        max_allowed_memory = 1.0 * AdaptiveChunkCalculator.MAX_MEMORY_FRACTION
        self.assertLessEqual(result.estimated_memory_per_chunk_gb, max_allowed_memory)
        

class TestChunkingAnalysisPrinting(unittest.TestCase):
    """Test the analysis printing functionality"""
    
    def test_print_chunking_analysis(self):
        """Test that analysis printing doesn't crash and produces reasonable output"""
        from io import StringIO
        import sys
        
        # Create a test result
        with tempfile.NamedTemporaryFile(suffix='.mib') as temp_file:
            # Write 1GB of data
            temp_file.write(b'0' * 1024**3)
            temp_file.flush()
            
            shape_4d = (64, 64, 256, 256)
            
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value.available = 32 * 1024**3
                
                result = AdaptiveChunkCalculator.calculate_adaptive_chunks(
                    temp_file.name, shape_4d
                )
                
            # Capture printed output
            captured_output = StringIO()
            sys.stdout = captured_output
            
            try:
                print_chunking_analysis(result, temp_file.name)
                output = captured_output.getvalue()
                
                # Check that key information is present
                self.assertIn("ADAPTIVE CHUNKING ANALYSIS", output)
                self.assertIn("File size:", output)
                self.assertIn("Strategy:", output)
                self.assertIn("Total chunks:", output)
                self.assertIn("I/O reduction:", output)
                
            finally:
                sys.stdout = sys.__stdout__


if __name__ == '__main__':
    unittest.main()