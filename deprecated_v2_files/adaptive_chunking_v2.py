"""
Adaptive Chunking V2 - True adaptive chunking strategy

This module implements the adaptive chunking strategy from ADAPTIVE_CHUNKING_MIGRATION_PLAN.md
that reduces I/O operations from 65,536 (frame-based) to 4-16 (adaptive) chunks.

Key improvements:
- Small files (<2GB): Frame-based chunking (1,1,qy,qx) - existing performance is fine
- Medium files (2-20GB): Scan-line chunking (chunk_sy,1,qy,qx) - 4-16x I/O reduction  
- Large files (>20GB): Block chunking (chunk_sy,chunk_sx,qy,qx) - 50-200x I/O reduction
- Memory safety: Never exceed 20% of available RAM per chunk
- Threading efficiency: Optimal chunk count for parallelization (4-32 chunks)
"""

import os
import psutil
import math
from typing import Tuple, List, NamedTuple, Optional
from enum import Enum
import numpy as np


class ChunkingStrategy(Enum):
    """Chunking strategies based on file size and system resources"""
    FRAME_BASED = "frame_based"      # Small files: (1, 1, qy, qx)
    SCAN_LINE = "scan_line"          # Medium files: (chunk_sy, 1, qy, qx) 
    BLOCK = "block"                  # Large files: (chunk_sy, chunk_sx, qy, qx)


class ChunkInfo(NamedTuple):
    """Information about a single chunk"""
    chunk_id: int
    input_slice: Tuple[slice, slice, slice, slice]    # Where to read from input
    output_slice: Tuple[slice, slice, slice, slice]   # Where to write to output
    expected_shape: Tuple[int, int, int, int]         # Expected chunk dimensions
    size_mb: float                                    # Chunk size in MB
    frames_count: int                                 # Number of frames in chunk


class ChunkingResult(NamedTuple):
    """Result of adaptive chunking calculation"""
    strategy: ChunkingStrategy
    chunk_dims: Tuple[int, int, int, int]            # Chunk dimensions (sy, sx, qy, qx)
    chunks: List[ChunkInfo]                          # List of all chunks
    total_chunks: int                                # Total number of chunks
    io_reduction_factor: int                         # I/O operations reduced by this factor
    estimated_memory_per_chunk_gb: float             # Memory usage per chunk
    file_size_gb: float                              # Original file size
    target_chunk_count: int                          # Target number of chunks for threading


class AdaptiveChunkCalculator:
    """Calculate optimal chunk sizes based on system resources and file characteristics"""
    
    # File size thresholds for different strategies (in GB)
    SMALL_FILE_THRESHOLD = 2.0      # < 2GB use frame-based
    LARGE_FILE_THRESHOLD = 20.0     # > 20GB use block chunking
    
    # Memory safety margins
    MAX_MEMORY_FRACTION = 0.2       # Never use more than 20% of available RAM per chunk
    PROCESSING_OVERHEAD = 3.0       # Account for 3x memory (input + processing + output)
    
    # Threading optimization
    MIN_CHUNKS = 4                  # Minimum chunks for threading benefit
    MAX_CHUNKS = 32                 # Maximum chunks before coordination overhead
    OPTIMAL_CHUNK_RANGE = (8, 16)   # Sweet spot for most systems
    
    @classmethod
    def calculate_adaptive_chunks(
        cls,
        file_path: str,
        shape_4d: Tuple[int, int, int, int],
        available_memory_gb: Optional[float] = None,
        target_chunk_count: Optional[int] = None
    ) -> ChunkingResult:
        """
        Calculate optimal chunk strategy for given file and system constraints
        
        Args:
            file_path: Path to input file for size calculation
            shape_4d: File shape as (scan_y, scan_x, detector_y, detector_x)
            available_memory_gb: Available system memory (auto-detected if None)
            target_chunk_count: Desired number of chunks (auto-calculated if None)
            
        Returns:
            ChunkingResult with strategy, chunks, and performance estimates
        """
        # Get system information
        file_size_gb = os.path.getsize(file_path) / (1024**3)
        if available_memory_gb is None:
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            
        sy, sx, qy, qx = shape_4d
        
        # Determine chunking strategy based on file size
        if file_size_gb < cls.SMALL_FILE_THRESHOLD:
            strategy = ChunkingStrategy.FRAME_BASED
        elif file_size_gb < cls.LARGE_FILE_THRESHOLD:
            strategy = ChunkingStrategy.SCAN_LINE
        else:
            strategy = ChunkingStrategy.BLOCK
            
        # Calculate optimal chunk dimensions for chosen strategy
        chunk_dims, estimated_chunks = cls._calculate_chunk_dimensions(
            shape_4d, file_size_gb, available_memory_gb, strategy, target_chunk_count
        )
        
        # Generate chunk list
        chunks = cls._generate_chunk_list(shape_4d, chunk_dims)
        
        # Calculate performance metrics
        original_frame_count = sy * sx  # Original would be 1 chunk per frame
        io_reduction_factor = max(1, original_frame_count // len(chunks))
        
        # Calculate memory usage per chunk
        chunk_sy, chunk_sx, chunk_qy, chunk_qx = chunk_dims
        bytes_per_chunk = chunk_sy * chunk_sx * chunk_qy * chunk_qx * 2  # uint16
        memory_per_chunk_gb = (bytes_per_chunk * cls.PROCESSING_OVERHEAD) / (1024**3)
        
        return ChunkingResult(
            strategy=strategy,
            chunk_dims=chunk_dims,
            chunks=chunks,
            total_chunks=len(chunks),
            io_reduction_factor=io_reduction_factor,
            estimated_memory_per_chunk_gb=memory_per_chunk_gb,
            file_size_gb=file_size_gb,
            target_chunk_count=estimated_chunks
        )
    
    @classmethod
    def _calculate_chunk_dimensions(
        cls,
        shape_4d: Tuple[int, int, int, int],
        file_size_gb: float,
        available_memory_gb: float,
        strategy: ChunkingStrategy,
        target_chunk_count: Optional[int]
    ) -> Tuple[Tuple[int, int, int, int], int]:
        """Calculate chunk dimensions for specific strategy"""
        
        sy, sx, qy, qx = shape_4d
        
        if strategy == ChunkingStrategy.FRAME_BASED:
            # Small files: use existing frame-based approach
            return (1, 1, qy, qx), sy * sx
            
        elif strategy == ChunkingStrategy.SCAN_LINE:
            # Medium files: multi-scan-line chunks
            return cls._calculate_scan_line_chunks(
                shape_4d, available_memory_gb, target_chunk_count
            )
            
        else:  # ChunkingStrategy.BLOCK
            # Large files: 2D spatial block chunks
            return cls._calculate_block_chunks(
                shape_4d, available_memory_gb, target_chunk_count
            )
    
    @classmethod
    def _calculate_scan_line_chunks(
        cls,
        shape_4d: Tuple[int, int, int, int],
        available_memory_gb: float,
        target_chunk_count: Optional[int]
    ) -> Tuple[Tuple[int, int, int, int], int]:
        """Calculate scan-line chunking dimensions (chunk_sy, 1, qy, qx)"""
        
        sy, sx, qy, qx = shape_4d
        
        # Calculate memory constraint: max scan lines per chunk
        bytes_per_scan_line = sx * qy * qx * 2  # uint16, full scan line
        max_memory_per_chunk = available_memory_gb * cls.MAX_MEMORY_FRACTION * (1024**3)
        max_scan_lines_memory = int(max_memory_per_chunk / (bytes_per_scan_line * cls.PROCESSING_OVERHEAD))
        
        # Calculate threading constraint: optimal chunk count
        if target_chunk_count is None:
            target_chunk_count = min(cls.OPTIMAL_CHUNK_RANGE[1], max(cls.MIN_CHUNKS, sy // 4))
            
        optimal_scan_lines_threading = max(1, sy // target_chunk_count)
        
        # Use the more restrictive constraint
        chunk_sy = min(max_scan_lines_memory, optimal_scan_lines_threading)
        chunk_sy = max(1, min(chunk_sy, sy))  # Ensure valid range
        
        actual_chunk_count = math.ceil(sy / chunk_sy)
        
        return (chunk_sy, 1, qy, qx), actual_chunk_count
    
    @classmethod
    def _calculate_block_chunks(
        cls,
        shape_4d: Tuple[int, int, int, int],
        available_memory_gb: float,
        target_chunk_count: Optional[int]
    ) -> Tuple[Tuple[int, int, int, int], int]:
        """Calculate block chunking dimensions (chunk_sy, chunk_sx, qy, qx)"""
        
        sy, sx, qy, qx = shape_4d
        
        # Calculate memory constraint: max frames per chunk
        bytes_per_frame = qy * qx * 2  # uint16
        max_memory_per_chunk = available_memory_gb * cls.MAX_MEMORY_FRACTION * (1024**3)
        max_frames_memory = int(max_memory_per_chunk / (bytes_per_frame * cls.PROCESSING_OVERHEAD))
        
        # Calculate threading constraint: optimal chunk count  
        if target_chunk_count is None:
            total_frames = sy * sx
            target_chunk_count = min(cls.OPTIMAL_CHUNK_RANGE[1], max(cls.MIN_CHUNKS, total_frames // 1000))
            
        optimal_frames_threading = max(1, (sy * sx) // target_chunk_count)
        
        # Use the more restrictive constraint
        max_frames_per_chunk = min(max_frames_memory, optimal_frames_threading)
        max_frames_per_chunk = max(1, min(max_frames_per_chunk, sy * sx))
        
        # Calculate 2D spatial chunk dimensions that best fit the frame limit
        # Try to keep chunks roughly square in scan space
        chunk_sy = min(sy, max(1, int(math.sqrt(max_frames_per_chunk))))
        chunk_sx = min(sx, max(1, max_frames_per_chunk // chunk_sy))
        
        # Adjust to ensure we don't exceed frame limit
        while chunk_sy * chunk_sx > max_frames_per_chunk and chunk_sy > 1:
            chunk_sy -= 1
            chunk_sx = min(sx, max_frames_per_chunk // chunk_sy)
            
        actual_chunk_count = math.ceil(sy / chunk_sy) * math.ceil(sx / chunk_sx)
        
        return (chunk_sy, chunk_sx, qy, qx), actual_chunk_count
    
    @classmethod
    def _generate_chunk_list(
        cls,
        shape_4d: Tuple[int, int, int, int],
        chunk_dims: Tuple[int, int, int, int]
    ) -> List[ChunkInfo]:
        """Generate list of all chunks with their slice information"""
        
        sy, sx, qy, qx = shape_4d
        chunk_sy, chunk_sx, chunk_qy, chunk_qx = chunk_dims
        
        chunks = []
        chunk_id = 0
        
        # Generate chunks covering the entire scan area
        for y_start in range(0, sy, chunk_sy):
            y_end = min(y_start + chunk_sy, sy)
            actual_chunk_sy = y_end - y_start
            
            for x_start in range(0, sx, chunk_sx):
                x_end = min(x_start + chunk_sx, sx)
                actual_chunk_sx = x_end - x_start
                
                # Create slice objects for input and output
                input_slice = (
                    slice(y_start, y_end),    # scan Y
                    slice(x_start, x_end),    # scan X  
                    slice(None),              # detector Y (full)
                    slice(None)               # detector X (full)
                )
                
                # Output slice same as input for now (processing may change detector dims)
                output_slice = input_slice
                
                # Calculate chunk properties
                expected_shape = (actual_chunk_sy, actual_chunk_sx, chunk_qy, chunk_qx)
                frames_count = actual_chunk_sy * actual_chunk_sx
                size_mb = (frames_count * chunk_qy * chunk_qx * 2) / (1024**2)  # uint16
                
                chunk_info = ChunkInfo(
                    chunk_id=chunk_id,
                    input_slice=input_slice,
                    output_slice=output_slice,
                    expected_shape=expected_shape,
                    size_mb=size_mb,
                    frames_count=frames_count
                )
                
                chunks.append(chunk_info)
                chunk_id += 1
                
        return chunks


def analyze_file_for_chunking(file_path: str, shape_4d: Tuple[int, int, int, int]) -> ChunkingResult:
    """
    Convenience function to analyze a file and get optimal chunking strategy
    
    Args:
        file_path: Path to the file to analyze
        shape_4d: File dimensions (scan_y, scan_x, detector_y, detector_x)
        
    Returns:
        ChunkingResult with optimal strategy and performance estimates
    """
    return AdaptiveChunkCalculator.calculate_adaptive_chunks(file_path, shape_4d)


def print_chunking_analysis(result: ChunkingResult, file_path: str = None) -> None:
    """Print human-readable analysis of chunking strategy"""
    
    print("=" * 60)
    print("ADAPTIVE CHUNKING ANALYSIS")
    print("=" * 60)
    
    if file_path:
        print(f"File: {os.path.basename(file_path)}")
    print(f"File size: {result.file_size_gb:.2f} GB")
    print(f"Strategy: {result.strategy.value}")
    print(f"Chunk dimensions: {result.chunk_dims}")
    print(f"Total chunks: {result.total_chunks}")
    print(f"I/O reduction: {result.io_reduction_factor}x")
    print(f"Memory per chunk: {result.estimated_memory_per_chunk_gb:.2f} GB")
    print(f"Target chunk count: {result.target_chunk_count}")
    
    print(f"\nChunk details:")
    for i, chunk in enumerate(result.chunks[:5]):  # Show first 5 chunks
        print(f"  Chunk {chunk.chunk_id}: {chunk.expected_shape} ({chunk.size_mb:.1f} MB, {chunk.frames_count} frames)")
    
    if len(result.chunks) > 5:
        print(f"  ... and {len(result.chunks) - 5} more chunks")
        
    print("=" * 60)