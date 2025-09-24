#!/usr/bin/env python3
"""
SmartDataReader - Universal multithreaded large file reader

This module provides a unified interface for reading MIB and EMD files of any size
with automatic memory management, predictive loading, and multithreaded I/O.

Key Features:
- Transparent file size handling (1GB to 1TB+)
- Automatic chunking strategy based on available memory  
- Multithreaded I/O with predictive prefetching
- Future-based non-blocking API
- Memory-safe operation with intelligent caching
"""

import os
import sys
import time
import psutil
import numpy as np
from typing import Tuple, Iterator, Optional, Dict, Any, Union, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from threading import Lock
import h5py

# Import existing MIB/EMD handling functions
from .mib_loader import get_mib_properties, load_mib
from .mib_to_emd_converter import MibToEmdConverter


@dataclass
class ChunkInfo:
    """Information about a data chunk"""
    chunk_id: int
    scan_slice: Tuple[slice, slice]  # (sy_slice, sx_slice)
    detector_slice: Tuple[slice, slice]  # (qy_slice, qx_slice) - usually full detector
    shape: Tuple[int, int, int, int]  # Expected data shape
    memory_mb: float  # Expected memory usage in MB
    priority: str = 'normal'  # 'high', 'normal', 'low'


@dataclass
class FileInfo:
    """Comprehensive file analysis"""
    file_path: str
    file_type: str  # 'mib' or 'emd'
    file_size_gb: float
    shape_4d: Tuple[int, int, int, int]
    dtype: np.dtype
    estimated_memory_gb: float
    scan_size: Optional[Tuple[int, int]] = None  # For MIB files
    metadata: Optional[Dict] = None


class SmartDataReader:
    """
    Universal large file reader with multithreaded I/O and intelligent chunking
    
    This class provides a unified interface for reading MIB and EMD files of any size.
    It automatically detects the optimal chunking strategy and uses multithreaded I/O
    for maximum performance while staying within memory limits.
    """
    
    def __init__(self, 
                 file_path: str,
                 chunk_strategy: str = 'auto',
                 max_memory_gb: Optional[float] = None,
                 max_workers: Optional[int] = None,
                 prefetch_chunks: int = 2,
                 progress_callback: Optional[Callable] = None):
        """
        Initialize SmartDataReader
        
        Parameters:
        -----------
        file_path : str
            Path to MIB or EMD file
        chunk_strategy : str
            'auto', 'memory_safe', 'performance', or 'minimal'
        max_memory_gb : float, optional
            Maximum memory to use (default: 30% of available RAM)
        max_workers : int, optional
            Number of I/O threads (default: min(4, cpu_count))
        prefetch_chunks : int
            Number of chunks to prefetch (default: 2)
        progress_callback : callable, optional
            Progress callback function(current, total, message)
        """
        self.file_path = file_path
        self.chunk_strategy = chunk_strategy
        self.prefetch_chunks = prefetch_chunks
        self.progress_callback = progress_callback
        
        # Analyze file
        self.file_info = self._analyze_file()
        
        # Memory management
        available_memory = psutil.virtual_memory().available / (1024**3)
        self.max_memory_gb = max_memory_gb or (available_memory * 0.3)
        
        # Threading
        self.max_workers = max_workers or min(4, os.cpu_count() or 4)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Chunk planning
        self.chunk_plan = self._create_chunk_plan()
        self.total_chunks = len(self.chunk_plan)
        
        # Cache and state
        self._chunk_cache = {}  # {chunk_id: (data, access_time)}
        self._cache_lock = Lock()
        self._prefetch_futures = {}  # {chunk_id: Future}
        
        self._log(f"SmartDataReader initialized:")
        self._log(f"  File: {os.path.basename(file_path)} ({self.file_info.file_size_gb:.1f} GB)")
        self._log(f"  Type: {self.file_info.file_type.upper()}")
        self._log(f"  Shape: {self.file_info.shape_4d}")
        self._log(f"  Strategy: {chunk_strategy} -> {self.total_chunks} chunks")
        self._log(f"  Memory limit: {self.max_memory_gb:.1f} GB")
        self._log(f"  Workers: {self.max_workers}")
    
    def _analyze_file(self) -> FileInfo:
        """Analyze file and extract metadata"""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
            
        file_size = os.path.getsize(self.file_path)
        file_size_gb = file_size / (1024**3)
        
        # Detect file type
        if self.file_path.lower().endswith('.mib'):
            return self._analyze_mib_file(file_size_gb)
        elif self.file_path.lower().endswith('.emd'):
            return self._analyze_emd_file(file_size_gb)
        else:
            raise ValueError(f"Unsupported file type: {self.file_path}")
    
    def _analyze_mib_file(self, file_size_gb: float) -> FileInfo:
        """Analyze MIB file structure"""
        # Use existing MIB analysis from converter
        converter = MibToEmdConverter()
        metadata = converter.analyze_mib_file(self.file_path)
        
        # Calculate estimated memory for 4D array
        shape_4d = metadata['shape_4d']
        estimated_memory_gb = (np.prod(shape_4d) * 2) / (1024**3)  # uint16 = 2 bytes
        
        return FileInfo(
            file_path=self.file_path,
            file_type='mib',
            file_size_gb=file_size_gb,
            shape_4d=shape_4d,
            dtype=np.uint16,  # MIB files are always uint16
            estimated_memory_gb=estimated_memory_gb,
            scan_size=metadata['scan_size'],
            metadata=metadata
        )
    
    def _analyze_emd_file(self, file_size_gb: float) -> FileInfo:
        """Analyze EMD file structure"""
        with h5py.File(self.file_path, 'r') as f:
            dataset = f['version_1/data/datacubes/datacube_000/data']
            shape_4d = dataset.shape
            dtype = dataset.dtype
            
        estimated_memory_gb = (np.prod(shape_4d) * np.dtype(dtype).itemsize) / (1024**3)
        
        return FileInfo(
            file_path=self.file_path,
            file_type='emd',
            file_size_gb=file_size_gb,
            shape_4d=shape_4d,
            dtype=dtype,
            estimated_memory_gb=estimated_memory_gb
        )
    
    def _create_chunk_plan(self) -> list[ChunkInfo]:
        """Create optimal chunk plan based on file size and strategy"""
        sy, sx, qy, qx = self.file_info.shape_4d
        dtype_size = np.dtype(self.file_info.dtype).itemsize
        
        # Calculate target chunk size based on strategy
        if self.chunk_strategy == 'auto':
            if self.file_info.estimated_memory_gb < (self.max_memory_gb * 0.5):
                # Small file - use large chunks
                target_chunk_mb = min(self.max_memory_gb * 1000 * 0.2, 2000)  # Up to 2GB chunks
            else:
                # Large file - use memory-safe chunks
                target_chunk_mb = min(self.max_memory_gb * 1000 * 0.1, 500)   # Up to 500MB chunks
        elif self.chunk_strategy == 'memory_safe':
            target_chunk_mb = min(self.max_memory_gb * 1000 * 0.05, 200)  # Conservative chunks
        elif self.chunk_strategy == 'performance':
            target_chunk_mb = min(self.max_memory_gb * 1000 * 0.3, 4000)  # Large chunks for speed
        elif self.chunk_strategy == 'minimal':
            target_chunk_mb = 50  # Very small chunks
        else:
            raise ValueError(f"Unknown chunk strategy: {self.chunk_strategy}")
        
        # Calculate optimal chunk dimensions in scan space
        bytes_per_frame = qy * qx * dtype_size
        target_chunk_bytes = target_chunk_mb * (1024**2)
        max_frames_per_chunk = max(1, int(target_chunk_bytes / bytes_per_frame))
        
        # Try to make chunks roughly square in scan dimensions
        chunk_sy = min(sy, max(1, int(np.sqrt(max_frames_per_chunk))))
        chunk_sx = min(sx, max(1, max_frames_per_chunk // chunk_sy))
        
        # Generate chunk plan
        chunks = []
        chunk_id = 0
        
        for start_y in range(0, sy, chunk_sy):
            for start_x in range(0, sx, chunk_sx):
                # Calculate actual chunk size (handle edges)
                actual_chunk_sy = min(chunk_sy, sy - start_y)
                actual_chunk_sx = min(chunk_sx, sx - start_x)
                
                scan_slice = (slice(start_y, start_y + actual_chunk_sy), 
                             slice(start_x, start_x + actual_chunk_sx))
                detector_slice = (slice(None), slice(None))  # Full detector
                
                chunk_shape = (actual_chunk_sy, actual_chunk_sx, qy, qx)
                chunk_memory_mb = (np.prod(chunk_shape) * dtype_size) / (1024**2)
                
                chunks.append(ChunkInfo(
                    chunk_id=chunk_id,
                    scan_slice=scan_slice,
                    detector_slice=detector_slice,
                    shape=chunk_shape,
                    memory_mb=chunk_memory_mb
                ))
                
                chunk_id += 1
        
        self._log(f"Chunk plan: {len(chunks)} chunks, {chunk_memory_mb:.1f} MB each")
        return chunks
    
    def get_chunk(self, chunk_id: int) -> Future[np.ndarray]:
        """
        Get a chunk by ID, returns Future for non-blocking access
        
        Parameters:
        -----------
        chunk_id : int
            Chunk identifier
            
        Returns:
        --------
        Future[np.ndarray] : Future containing chunk data
        """
        if chunk_id >= len(self.chunk_plan):
            raise ValueError(f"Chunk ID {chunk_id} out of range (0-{len(self.chunk_plan)-1})")
        
        # Check cache first
        with self._cache_lock:
            if chunk_id in self._chunk_cache:
                data, _ = self._chunk_cache[chunk_id]
                # Return completed future with cached data
                future = Future()
                future.set_result(data.copy())
                return future
        
        # Check if already being loaded
        if chunk_id in self._prefetch_futures:
            return self._prefetch_futures[chunk_id]
        
        # Submit loading task
        future = self.executor.submit(self._load_chunk, chunk_id)
        self._prefetch_futures[chunk_id] = future
        
        # Start prefetching next chunks
        self._start_prefetch(chunk_id)
        
        return future
    
    def get_chunk_iterator(self) -> Iterator[Tuple[int, Future[np.ndarray]]]:
        """
        Iterate through all chunks with automatic prefetching
        
        Yields:
        -------
        tuple : (chunk_id, future) pairs
        """
        for chunk_id in range(len(self.chunk_plan)):
            yield chunk_id, self.get_chunk(chunk_id)
            
            # Update progress if callback provided
            if self.progress_callback:
                self.progress_callback(chunk_id + 1, len(self.chunk_plan), 
                                     f"Processing chunk {chunk_id + 1}/{len(self.chunk_plan)}")
    
    def _load_chunk(self, chunk_id: int) -> np.ndarray:
        """Load a single chunk from file"""
        chunk_info = self.chunk_plan[chunk_id]
        
        start_time = time.time()
        
        if self.file_info.file_type == 'mib':
            data = self._load_mib_chunk(chunk_info)
        else:  # emd
            data = self._load_emd_chunk(chunk_info)
        
        load_time = time.time() - start_time
        
        # Cache the result
        with self._cache_lock:
            self._chunk_cache[chunk_id] = (data, time.time())
            self._manage_cache()
        
        self._log(f"Loaded chunk {chunk_id}: {data.shape} in {load_time:.2f}s")
        return data
    
    def _load_mib_chunk(self, chunk_info: ChunkInfo) -> np.ndarray:
        """Load chunk from MIB file"""
        # Use existing MIB chunked reader from converter
        converter = MibToEmdConverter()
        
        sy_slice, sx_slice = chunk_info.scan_slice
        chunk_sy, chunk_sx, qy, qx = chunk_info.shape
        
        # Create chunk size tuple for existing reader
        chunk_size = (chunk_sy, chunk_sx, qy, qx)
        
        # Get the chunk using existing chunked reader
        for returned_slice, chunk_data in converter.chunked_mib_reader(
            self.file_path, self.file_info.scan_size, chunk_size
        ):
            # Check if this is the chunk we want
            if (returned_slice[0] == sy_slice and returned_slice[1] == sx_slice):
                return chunk_data
        
        raise RuntimeError(f"Failed to load MIB chunk {chunk_info.chunk_id}")
    
    def _load_emd_chunk(self, chunk_info: ChunkInfo) -> np.ndarray:
        """Load chunk from EMD file"""
        with h5py.File(self.file_path, 'r') as f:
            dataset = f['version_1/data/datacubes/datacube_000/data']
            
            sy_slice, sx_slice = chunk_info.scan_slice
            qy_slice, qx_slice = chunk_info.detector_slice
            
            # Use HDF5 slicing for efficient chunk loading
            chunk_data = dataset[sy_slice, sx_slice, qy_slice, qx_slice]
            
            return np.array(chunk_data)  # Ensure we have a copy
    
    def _start_prefetch(self, current_chunk_id: int):
        """Start prefetching next chunks"""
        for i in range(1, self.prefetch_chunks + 1):
            next_id = current_chunk_id + i
            if (next_id < len(self.chunk_plan) and 
                next_id not in self._prefetch_futures and 
                next_id not in self._chunk_cache):
                
                future = self.executor.submit(self._load_chunk, next_id)
                self._prefetch_futures[next_id] = future
    
    def _manage_cache(self):
        """Manage cache size to stay within memory limits"""
        # Simple LRU eviction based on access time
        current_memory_mb = sum(
            data.nbytes / (1024**2) 
            for data, _ in self._chunk_cache.values()
        )
        
        max_memory_mb = self.max_memory_gb * 1000
        
        if current_memory_mb > max_memory_mb:
            # Sort by access time (oldest first)
            items_by_age = sorted(
                self._chunk_cache.items(),
                key=lambda x: x[1][1]  # Sort by access_time
            )
            
            # Remove oldest items until under limit
            for chunk_id, (data, _) in items_by_age:
                del self._chunk_cache[chunk_id]
                current_memory_mb -= data.nbytes / (1024**2)
                
                if current_memory_mb <= max_memory_mb * 0.8:  # Leave some headroom
                    break
    
    def _log(self, message: str):
        """Log message if verbose"""
        print(f"[SmartDataReader] {message}")
    
    def get_file_info(self) -> FileInfo:
        """Get comprehensive file information"""
        return self.file_info
    
    def get_chunk_info(self, chunk_id: int) -> ChunkInfo:
        """Get information about a specific chunk"""
        if chunk_id >= len(self.chunk_plan):
            raise ValueError(f"Chunk ID {chunk_id} out of range")
        return self.chunk_plan[chunk_id]
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        with self._cache_lock:
            cache_memory_mb = sum(
                data.nbytes / (1024**2) 
                for data, _ in self._chunk_cache.values()
            )
            
        return {
            'cache_memory_mb': cache_memory_mb,
            'cache_items': len(self._chunk_cache),
            'max_memory_gb': self.max_memory_gb,
            'prefetch_futures': len(self._prefetch_futures)
        }
    
    def close(self):
        """Clean up resources"""
        self.executor.shutdown(wait=False)
        with self._cache_lock:
            self._chunk_cache.clear()
        self._prefetch_futures.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Convenience functions for quick access
def create_smart_reader(file_path: str, **kwargs) -> SmartDataReader:
    """Create a SmartDataReader with sensible defaults"""
    return SmartDataReader(file_path, **kwargs)


def get_file_info(file_path: str) -> FileInfo:
    """Quick file analysis without creating full reader"""
    with SmartDataReader(file_path, max_workers=1) as reader:
        return reader.get_file_info()