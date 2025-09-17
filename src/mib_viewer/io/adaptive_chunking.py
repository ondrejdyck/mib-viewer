#!/usr/bin/env python3
"""
Adaptive Chunking System for Large File Processing

This module provides intelligent chunking strategies that adapt to:
- File size (small/medium/large)  
- Available system memory
- Number of worker threads
- I/O vs CPU optimization

Key Features:
- Memory-safe chunk sizes that fit within worker memory limits
- Even division of file dimensions (no awkward remainder chunks)
- Automatic worker count determination based on system resources
- Significant I/O operation reduction (10x to 1000x+ improvement)
"""

import os
import math
import psutil
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


class ChunkingStrategy(Enum):
    """Different chunking approaches based on file characteristics"""
    SINGLE_THREAD = "single_thread"    # Small files < 1GB
    SCAN_LINE = "scan_line"            # Medium files, chunks like (8,1,qy,qx)
    BLOCK = "block"                    # Large files, chunks like (16,16,qy,qx)


@dataclass
class ChunkingResult:
    """Result of chunk size calculation with all relevant information"""
    
    # Chunking strategy
    strategy: ChunkingStrategy
    chunk_dims: Tuple[int, int, int, int]  # (chunk_sy, chunk_sx, qy, qx)
    
    # Resource allocation
    num_workers: int
    available_memory_gb: float
    memory_per_worker_gb: float
    
    # Chunk characteristics
    chunk_size_mb: float
    frames_per_chunk: int
    total_chunks: int
    
    # Performance metrics
    io_reduction_factor: int
    estimated_memory_usage_gb: float
    
    # File info
    file_shape: Tuple[int, int, int, int]
    file_size_gb: float
    
    @property
    def use_single_thread(self) -> bool:
        return self.strategy == ChunkingStrategy.SINGLE_THREAD


@dataclass
class ChunkInfo:
    """Lightweight metadata for work queue"""
    id: int
    input_slice: Tuple[slice, slice, slice, slice]   # Where to read from input
    output_slice: Tuple[slice, slice, slice, slice]  # Where to write in output  
    expected_shape: Tuple[int, int, int, int]        # Expected chunk data shape


class AdaptiveChunkCalculator:
    """
    Calculate optimal chunking strategy based on file size, memory, and system resources
    
    This class implements the core adaptive chunking algorithm that:
    1. Determines optimal worker count based on CPU and memory constraints
    2. Calculates chunk sizes that fit within worker memory limits
    3. Ensures chunks divide evenly into file dimensions
    4. Maximizes chunk size to minimize I/O operations
    """
    
    def __init__(self, 
                 single_thread_threshold_gb: float = 1.0,
                 memory_safety_factor: float = 0.8,
                 worker_memory_factor: float = 0.7,
                 conservative_mode: bool = True,
                 max_workers: Optional[int] = None):
        """
        Initialize adaptive chunk calculator
        
        Parameters:
        -----------
        single_thread_threshold_gb : float
            Files smaller than this use single threading
        memory_safety_factor : float
            Fraction of available memory to use (0.8 = 80%)
        worker_memory_factor : float
            Fraction of worker memory allocation for chunk size (0.7 = 70%)
        conservative_mode : bool
            If True, leave cores free for GUI/OS responsiveness
        max_workers : int, optional
            User-specified maximum worker count (None = auto-determine)
        """
        self.single_thread_threshold = single_thread_threshold_gb
        self.memory_safety_factor = memory_safety_factor
        self.worker_memory_factor = worker_memory_factor
        self.conservative_mode = conservative_mode
        self.max_workers = max_workers
    
    def calculate_chunking_strategy(self, 
                                  file_shape: Tuple[int, int, int, int],
                                  file_path: Optional[str] = None) -> ChunkingResult:
        """
        Calculate optimal chunking strategy for given file
        
        Parameters:
        -----------
        file_shape : tuple
            4D file shape (sy, sx, qy, qx)
        file_path : str, optional
            File path for size calculation (if not provided, estimated from shape)
            
        Returns:
        --------
        ChunkingResult : Complete chunking strategy information
        """
        
        # Calculate file characteristics
        sy, sx, qy, qx = file_shape
        total_frames = sy * sx
        bytes_per_frame = qy * qx * 2  # Assuming uint16
        
        if file_path and os.path.exists(file_path):
            file_size_bytes = os.path.getsize(file_path)
        else:
            file_size_bytes = total_frames * bytes_per_frame
            
        file_size_gb = file_size_bytes / (1024**3)
        
        # Get system resources
        available_memory_gb = self._get_available_memory_gb()
        
        # Determine if single threading is appropriate
        if file_size_gb < self.single_thread_threshold:
            return self._single_thread_strategy(file_shape, file_size_gb, available_memory_gb)
        
        # Calculate optimal worker count
        num_workers = self._calculate_worker_count(available_memory_gb, file_size_gb)

        # DUAL-CONSTRAINT APPROACH: Calculate both memory and data distribution limits
        memory_constraint_gb = self._calculate_memory_constraint_per_worker(available_memory_gb, num_workers)
        data_distribution_constraint_gb = self._calculate_data_distribution_constraint_per_worker(file_size_gb, num_workers)

        # Choose the limiting constraint (the bottleneck)
        target_chunk_size_gb = min(memory_constraint_gb, data_distribution_constraint_gb)
        target_chunk_bytes = target_chunk_size_gb * (1024**3)

        # Apply HDF5 4GB chunk size constraint
        max_chunk_bytes = 4 * (1024**3) - 1024  # Just under 4GB with safety margin
        target_chunk_bytes = min(target_chunk_bytes, max_chunk_bytes)

        # Log constraint analysis for transparency
        constraint_type = "memory" if memory_constraint_gb < data_distribution_constraint_gb else "data distribution"
        print(f"CHUNKING CONSTRAINT ANALYSIS:")
        print(f"  Memory constraint per worker: {memory_constraint_gb:.2f} GB")
        print(f"  Data distribution constraint per worker: {data_distribution_constraint_gb:.2f} GB")
        print(f"  Chosen constraint (bottleneck): {target_chunk_size_gb:.2f} GB ({constraint_type})")
        print(f"  Target chunk size: {target_chunk_bytes / (1024**3):.2f} GB")

        # Calculate optimal chunk dimensions
        chunk_dims, total_chunks, frames_per_chunk = self._find_largest_factor_chunks_under_limit(
            file_shape, target_chunk_bytes, bytes_per_frame
        )
        
        # Calculate performance metrics
        chunk_size_mb = (frames_per_chunk * bytes_per_frame) / (1024**2)
        io_reduction_factor = total_frames // total_chunks if total_chunks > 0 else 1
        estimated_memory_usage_gb = (chunk_size_mb * num_workers) / 1024

        # Calculate actual memory per worker for reporting
        usable_memory_gb = available_memory_gb * self.memory_safety_factor
        memory_per_worker_gb = usable_memory_gb / num_workers
        
        # Determine strategy type
        chunk_sy, chunk_sx = chunk_dims[:2]
        if chunk_sx == 1:
            strategy = ChunkingStrategy.SCAN_LINE
        else:
            strategy = ChunkingStrategy.BLOCK
        
        return ChunkingResult(
            strategy=strategy,
            chunk_dims=chunk_dims,
            num_workers=num_workers,
            available_memory_gb=available_memory_gb,
            memory_per_worker_gb=memory_per_worker_gb,
            chunk_size_mb=chunk_size_mb,
            frames_per_chunk=frames_per_chunk,
            total_chunks=total_chunks,
            io_reduction_factor=io_reduction_factor,
            estimated_memory_usage_gb=estimated_memory_usage_gb,
            file_shape=file_shape,
            file_size_gb=file_size_gb
        )
    
    def generate_chunk_queue(self, chunking_result: ChunkingResult) -> List[ChunkInfo]:
        """
        Generate work queue of chunks for workers to process
        
        Parameters:
        -----------
        chunking_result : ChunkingResult
            Result from calculate_chunking_strategy()
            
        Returns:
        --------
        List[ChunkInfo] : Ordered list of chunks for processing
        """
        sy, sx, qy, qx = chunking_result.file_shape
        chunk_sy, chunk_sx = chunking_result.chunk_dims[:2]
        
        chunks = []
        chunk_id = 0
        
        for start_y in range(0, sy, chunk_sy):
            for start_x in range(0, sx, chunk_sx):
                # Calculate actual chunk size (handle edges)
                actual_chunk_sy = min(chunk_sy, sy - start_y)
                actual_chunk_sx = min(chunk_sx, sx - start_x)
                
                # Create slice objects for input and output
                input_slice = (
                    slice(start_y, start_y + actual_chunk_sy),
                    slice(start_x, start_x + actual_chunk_sx),
                    slice(None),  # Full detector Y
                    slice(None)   # Full detector X
                )
                
                # Output slice is the same for direct copying
                output_slice = input_slice
                
                # Expected shape after processing
                expected_shape = (actual_chunk_sy, actual_chunk_sx, qy, qx)
                
                chunks.append(ChunkInfo(
                    id=chunk_id,
                    input_slice=input_slice,
                    output_slice=output_slice,
                    expected_shape=expected_shape
                ))
                
                chunk_id += 1
        
        return chunks

    def _calculate_memory_constraint_per_worker(self, available_memory_gb: float, num_workers: int) -> float:
        """
        Calculate memory-based chunk size limit per worker

        This ensures we don't exceed available memory even with all workers active.
        """
        usable_memory_gb = available_memory_gb * self.memory_safety_factor  # e.g., 80% of available
        memory_per_worker_gb = usable_memory_gb / num_workers
        return memory_per_worker_gb * self.worker_memory_factor  # e.g., 70% of worker allocation

    def _calculate_data_distribution_constraint_per_worker(self, file_size_gb: float, num_workers: int) -> float:
        """
        Calculate data-distribution-based chunk size for optimal worker utilization

        This ensures all workers have roughly equal amounts of work to do.
        """
        return file_size_gb / num_workers

    def _get_available_memory_gb(self) -> float:
        """Get available system memory in GB"""
        try:
            import psutil
            return psutil.virtual_memory().available / (1024**3)
        except ImportError:
            # Fallback estimate if psutil not available
            # This is very rough - in production should require psutil
            return 4.0  # Conservative 4GB assumption
    
    def _calculate_worker_count(self, available_memory_gb: float, file_size_gb: float) -> int:
        """Calculate optimal number of worker threads"""
        
        logical_cpus = os.cpu_count() or 4
        
        # User override takes precedence
        if self.max_workers is not None:
            return max(1, min(self.max_workers, logical_cpus))
        
        # CPU-based constraint
        if self.conservative_mode:
            cpu_workers = max(1, logical_cpus - 2)  # Leave 2 cores free
        else:
            cpu_workers = max(1, logical_cpus - 1)  # Leave 1 core free
        
        # Memory-based constraint (minimum 500MB per worker)
        min_memory_per_worker = 0.5  # GB
        memory_workers = max(1, int(available_memory_gb / min_memory_per_worker))
        
        # Take the most restrictive constraint
        optimal_workers = min(cpu_workers, memory_workers)
        
        # Cap at reasonable maximum (avoid excessive threading overhead)
        optimal_workers = min(optimal_workers, 16)
        
        return optimal_workers
    
    def _find_largest_factor_chunks_under_limit(self,
                                           file_shape: Tuple[int, int, int, int],
                                           target_chunk_bytes: float,
                                           bytes_per_frame: int) -> Tuple[Tuple[int, int, int, int], int, int]:
        """
        Find the largest factor-based chunks that fit under the size limit

        This finds chunks that:
        1. Divide evenly into the scan dimensions (no partial chunks)
        2. Fit within the target chunk size limit
        3. Are as large as possible (for efficiency) while satisfying 1 & 2

        Returns:
        --------
        tuple : (chunk_dims, total_chunks, frames_per_chunk)
        """
        sy, sx, qy, qx = file_shape
        total_frames = sy * sx
        max_frames_per_chunk = int(target_chunk_bytes // bytes_per_frame)

        # Find all possible rectangular chunk dimensions that divide evenly
        valid_chunks = []

        for chunk_sy in range(1, sy + 1):
            if sy % chunk_sy == 0:  # sy divides evenly
                for chunk_sx in range(1, sx + 1):
                    if sx % chunk_sx == 0:  # sx divides evenly
                        frames_in_chunk = chunk_sy * chunk_sx

                        # Only consider chunks that fit within our limit
                        if frames_in_chunk <= max_frames_per_chunk:
                            chunk_bytes = frames_in_chunk * bytes_per_frame
                            chunks_needed = total_frames // frames_in_chunk
                            valid_chunks.append({
                                'dims': (chunk_sy, chunk_sx, qy, qx),
                                'frames': frames_in_chunk,
                                'bytes': chunk_bytes,
                                'count': chunks_needed
                            })

        if not valid_chunks:
            # Fallback to frame-based if no valid factor chunks found
            print(f"  WARNING: No factor-based chunks found, falling back to single frames")
            return (1, 1, qy, qx), total_frames, 1

        # Choose the largest valid chunks (most efficient processing)
        # This maximizes chunk size while respecting our constraints
        valid_chunks.sort(key=lambda x: x['frames'], reverse=True)
        best_chunk = valid_chunks[0]

        print(f"  Selected chunk size: {best_chunk['frames']:,} frames ({best_chunk['bytes'] / (1024**3):.2f} GB)")
        print(f"  Chunk dimensions: ({best_chunk['dims'][0]}, {best_chunk['dims'][1]})")
        print(f"  Total chunks: {best_chunk['count']}")

        # Worker utilization analysis
        worker_efficiency = min(1.0, best_chunk['count'] / 10)  # Assuming ~10 workers
        print(f"  Worker utilization: {worker_efficiency*100:.0f}% ({min(best_chunk['count'], 10)} of 10 workers active)")

        return best_chunk['dims'], best_chunk['count'], best_chunk['frames']
    
    def _single_thread_strategy(self, 
                              file_shape: Tuple[int, int, int, int],
                              file_size_gb: float,
                              available_memory_gb: float) -> ChunkingResult:
        """Create chunking result for single-threaded processing"""
        
        sy, sx, qy, qx = file_shape
        
        return ChunkingResult(
            strategy=ChunkingStrategy.SINGLE_THREAD,
            chunk_dims=(sy, sx, qy, qx),  # Entire file as one chunk
            num_workers=1,
            available_memory_gb=available_memory_gb,
            memory_per_worker_gb=available_memory_gb,
            chunk_size_mb=(file_size_gb * 1024),
            frames_per_chunk=sy * sx,
            total_chunks=1,
            io_reduction_factor=1,
            estimated_memory_usage_gb=file_size_gb,
            file_shape=file_shape,
            file_size_gb=file_size_gb
        )


def create_adaptive_chunking_strategy(file_shape: Tuple[int, int, int, int],
                                    file_path: Optional[str] = None,
                                    max_workers: Optional[int] = None,
                                    conservative: bool = True) -> ChunkingResult:
    """
    Convenience function to create chunking strategy with default settings
    
    Parameters:
    -----------
    file_shape : tuple
        4D file shape (sy, sx, qy, qx)
    file_path : str, optional
        File path for accurate size calculation
    max_workers : int, optional
        Maximum number of workers (None = auto-determine)
    conservative : bool
        Use conservative settings (leave cores free for system responsiveness)
        
    Returns:
    --------
    ChunkingResult : Complete chunking strategy
    """
    calculator = AdaptiveChunkCalculator(
        max_workers=max_workers,
        conservative_mode=conservative
    )
    
    return calculator.calculate_chunking_strategy(file_shape, file_path)


if __name__ == "__main__":
    # Example usage and testing
    print("=== Adaptive Chunking System Test ===")
    
    # Test different file sizes
    test_cases = [
        # (name, shape, expected_strategy)
        ("Small EELS file", (32, 32, 256, 1024), ChunkingStrategy.SINGLE_THREAD),
        ("Medium 4D STEM", (128, 128, 256, 256), ChunkingStrategy.SCAN_LINE),
        ("Large 4D STEM", (256, 256, 256, 256), ChunkingStrategy.BLOCK),
        ("Very large dataset", (512, 512, 256, 256), ChunkingStrategy.BLOCK),
    ]
    
    for name, shape, expected_strategy in test_cases:
        print(f"\n--- {name}: {shape} ---")
        result = create_adaptive_chunking_strategy(shape)
        
        print(f"Strategy: {result.strategy.value}")
        print(f"Workers: {result.num_workers}")
        print(f"Chunk size: {result.chunk_size_mb:.1f} MB")
        print(f"Total chunks: {result.total_chunks}")
        print(f"I/O reduction: {result.io_reduction_factor}x")
        print(f"Est. memory usage: {result.estimated_memory_usage_gb:.2f} GB")
        
        # Verify expected strategy
        if result.strategy == expected_strategy:
            print("✓ Strategy matches expectation")
        else:
            print(f"⚠ Expected {expected_strategy.value}, got {result.strategy.value}")