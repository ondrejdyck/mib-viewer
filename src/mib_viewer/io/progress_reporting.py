#!/usr/bin/env python3
"""
Progress Reporting System for Adaptive Chunking

This module provides comprehensive progress tracking that adapts to variable chunk sizes
and multi-worker processing patterns. It supports both GUI progress bars and detailed
console logging for debugging and monitoring.

Key Features:
- Chunk-based progress tracking (not frame-based)
- Detailed worker activity logging
- Thread-safe progress updates
- Configurable verbosity levels
- Integration with existing GUI progress callbacks
"""

import time
import threading
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

from .adaptive_chunking import ChunkingResult, ChunkInfo


class LogLevel(Enum):
    """Logging verbosity levels"""
    BASIC = 0      # Just strategy setup + progress bar updates
    DETAILED = 1   # + chunk completion messages  
    VERBOSE = 2    # + chunk state transitions (read/process/write)
    DEBUG = 3      # + timing details and performance metrics


class ChunkState(Enum):
    """Chunk processing states for detailed tracking"""
    QUEUED = "queued"
    READING = "reading" 
    PROCESSING = "processing"
    WRITING = "writing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WorkerStats:
    """Statistics for individual worker performance"""
    worker_id: int
    chunks_processed: int = 0
    total_processing_time: float = 0.0
    total_read_time: float = 0.0
    total_write_time: float = 0.0
    bytes_processed: int = 0
    current_chunk_id: Optional[int] = None
    current_state: Optional[ChunkState] = None
    start_time: Optional[float] = None


@dataclass
class ConversionProgress:
    """Overall conversion progress tracking"""
    chunks_completed: int = 0
    total_chunks: int = 0
    bytes_processed: int = 0
    total_bytes: int = 0
    start_time: float = field(default_factory=time.time)
    worker_stats: Dict[int, WorkerStats] = field(default_factory=dict)
    
    @property
    def progress_percent(self) -> int:
        """Calculate overall progress percentage"""
        if self.total_chunks == 0:
            return 0
        return int((self.chunks_completed / self.total_chunks) * 100)
    
    @property
    def elapsed_time(self) -> float:
        """Total elapsed time in seconds"""
        return time.time() - self.start_time
    
    @property
    def chunks_per_second(self) -> float:
        """Processing rate in chunks per second"""
        if self.elapsed_time == 0:
            return 0.0
        return self.chunks_completed / self.elapsed_time
    
    @property
    def estimated_time_remaining(self) -> float:
        """Estimated time remaining in seconds"""
        if self.chunks_per_second == 0:
            return 0.0
        remaining_chunks = self.total_chunks - self.chunks_completed
        return remaining_chunks / self.chunks_per_second


class ProgressReporter:
    """
    Thread-safe progress reporting system for adaptive chunking
    
    This class coordinates progress updates from multiple worker threads and provides
    both GUI progress bar updates and detailed console logging.
    """
    
    def __init__(self,
                 chunking_result: ChunkingResult,
                 progress_callback: Optional[Callable[[int, str], None]] = None,
                 log_callback: Optional[Callable[[str], None]] = None,
                 log_level: LogLevel = LogLevel.BASIC):
        """
        Initialize progress reporter
        
        Parameters:
        -----------
        chunking_result : ChunkingResult
            Result from adaptive chunking calculation
        progress_callback : callable, optional
            GUI progress callback function(progress_percent, status_message)
        log_callback : callable, optional
            Logging callback function(log_message)  
        log_level : LogLevel
            Verbosity level for logging
        """
        self.chunking_result = chunking_result
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.log_level = log_level
        
        # Thread-safe progress tracking
        self.progress = ConversionProgress(
            total_chunks=chunking_result.total_chunks,
            total_bytes=int(chunking_result.file_size_gb * 1024**3)
        )
        self._progress_lock = threading.Lock()
        
        # Initialize worker stats
        for worker_id in range(chunking_result.num_workers):
            self.progress.worker_stats[worker_id] = WorkerStats(worker_id=worker_id)
        
        # Log initial strategy information
        self._log_initial_strategy()
    
    def log_chunk_state_change(self, 
                              chunk_id: int, 
                              new_state: ChunkState, 
                              worker_id: Optional[int] = None,
                              extra_info: str = ""):
        """Log chunk state transitions for debugging"""
        
        if self.log_level.value < LogLevel.VERBOSE.value:
            return
        
        timestamp = time.time() - self.progress.start_time
        worker_part = f" (worker {worker_id})" if worker_id is not None else ""
        
        self._log(f"[{timestamp:6.1f}s] Chunk {chunk_id:3d}: {new_state.value}{worker_part} {extra_info}")
        
        # Update worker stats
        if worker_id is not None:
            with self._progress_lock:
                if worker_id in self.progress.worker_stats:
                    worker_stats = self.progress.worker_stats[worker_id]
                    worker_stats.current_chunk_id = chunk_id
                    worker_stats.current_state = new_state
    
    def start_chunk_processing(self, chunk_id: int, worker_id: int):
        """Mark the start of chunk processing"""
        self.log_chunk_state_change(chunk_id, ChunkState.READING, worker_id)
        
        with self._progress_lock:
            if worker_id in self.progress.worker_stats:
                self.progress.worker_stats[worker_id].start_time = time.time()
    
    def chunk_read_complete(self, chunk_id: int, worker_id: int, read_time: float, bytes_read: int):
        """Mark completion of chunk reading phase"""
        self.log_chunk_state_change(
            chunk_id, ChunkState.PROCESSING, worker_id, 
            f"({bytes_read/(1024**2):.1f} MB in {read_time:.2f}s)"
        )
        
        with self._progress_lock:
            if worker_id in self.progress.worker_stats:
                self.progress.worker_stats[worker_id].total_read_time += read_time
    
    def chunk_processing_complete(self, chunk_id: int, worker_id: int, processing_time: float):
        """Mark completion of chunk processing phase"""
        self.log_chunk_state_change(
            chunk_id, ChunkState.WRITING, worker_id,
            f"(processed in {processing_time:.2f}s)"
        )
        
        with self._progress_lock:
            if worker_id in self.progress.worker_stats:
                self.progress.worker_stats[worker_id].total_processing_time += processing_time
    
    def chunk_write_complete(self, chunk_id: int, worker_id: int, write_time: float):
        """Mark completion of chunk writing phase"""
        self.log_chunk_state_change(
            chunk_id, ChunkState.COMPLETED, worker_id,
            f"(written in {write_time:.2f}s)"
        )
        
        with self._progress_lock:
            if worker_id in self.progress.worker_stats:
                worker_stats = self.progress.worker_stats[worker_id]
                worker_stats.total_write_time += write_time
                worker_stats.chunks_processed += 1
    
    def chunk_completed(self, chunk_id: int, worker_id: int, chunk_bytes: int):
        """Mark overall chunk completion and update progress"""
        
        with self._progress_lock:
            self.progress.chunks_completed += 1
            self.progress.bytes_processed += chunk_bytes
            
            # Update worker stats
            if worker_id in self.progress.worker_stats:
                worker_stats = self.progress.worker_stats[worker_id]
                worker_stats.bytes_processed += chunk_bytes
                worker_stats.current_chunk_id = None
                worker_stats.current_state = None
                
                # Calculate total processing time for this chunk
                if worker_stats.start_time:
                    total_chunk_time = time.time() - worker_stats.start_time
                    worker_stats.start_time = None
        
        # Update GUI progress bar
        self._update_progress_bar()
        
        # Log completion (if detailed logging enabled)
        if self.log_level.value >= LogLevel.DETAILED.value:
            progress_percent = self.progress.progress_percent
            chunks_remaining = self.progress.total_chunks - self.progress.chunks_completed
            
            if chunks_remaining > 0:
                eta = self.progress.estimated_time_remaining
                self._log(f"Chunk {chunk_id} completed by worker {worker_id} "
                         f"({progress_percent}%, {chunks_remaining} remaining, ETA: {eta:.0f}s)")
            else:
                self._log(f"All chunks completed! Total time: {self.progress.elapsed_time:.1f}s")
    
    def chunk_failed(self, chunk_id: int, worker_id: int, error: Exception):
        """Mark chunk processing failure"""
        self.log_chunk_state_change(
            chunk_id, ChunkState.FAILED, worker_id, 
            f"ERROR: {str(error)}"
        )
        
        # Update worker stats
        with self._progress_lock:
            if worker_id in self.progress.worker_stats:
                worker_stats = self.progress.worker_stats[worker_id]
                worker_stats.current_chunk_id = None
                worker_stats.current_state = ChunkState.FAILED
    
    def log_performance_summary(self):
        """Log final performance summary"""
        if self.log_level.value < LogLevel.DETAILED.value:
            return
        
        total_time = self.progress.elapsed_time
        total_chunks = self.progress.chunks_completed
        
        self._log("\n" + "="*60)
        self._log("PERFORMANCE SUMMARY")
        self._log("="*60)
        self._log(f"Total processing time: {total_time:.1f}s")
        self._log(f"Chunks processed: {total_chunks}")
        self._log(f"Average rate: {self.progress.chunks_per_second:.1f} chunks/sec")
        self._log(f"Data processed: {self.progress.bytes_processed / (1024**3):.2f} GB")
        self._log(f"Throughput: {(self.progress.bytes_processed / (1024**2)) / total_time:.1f} MB/s")
        
        # Worker performance breakdown
        if self.log_level.value >= LogLevel.DEBUG.value:
            self._log(f"\nWorker Performance:")
            for worker_id, stats in self.progress.worker_stats.items():
                if stats.chunks_processed > 0:
                    avg_time = (stats.total_read_time + stats.total_processing_time + 
                               stats.total_write_time) / stats.chunks_processed
                    throughput = stats.bytes_processed / (1024**2) / total_time
                    
                    self._log(f"  Worker {worker_id}: {stats.chunks_processed} chunks, "
                             f"{avg_time:.2f}s avg, {throughput:.1f} MB/s")
        
        self._log("="*60)
    
    def get_current_status(self) -> str:
        """Get current processing status string"""
        with self._progress_lock:
            if self.progress.chunks_completed == self.progress.total_chunks:
                return "Conversion completed!"
            
            progress = self.progress.progress_percent
            remaining = self.progress.total_chunks - self.progress.chunks_completed
            
            if self.progress.chunks_per_second > 0:
                eta = self.progress.estimated_time_remaining
                return f"Processing chunk {self.progress.chunks_completed}/{self.progress.total_chunks} ({progress}%, ETA: {eta:.0f}s)"
            else:
                return f"Processing chunk {self.progress.chunks_completed}/{self.progress.total_chunks} ({progress}%)"
    
    def _log_initial_strategy(self):
        """Log the initial chunking strategy information"""
        self._log("=" * 60)
        self._log("ADAPTIVE CHUNKING CONVERSION STRATEGY")
        self._log("=" * 60)
        
        result = self.chunking_result
        self._log(f"File shape: {result.file_shape}")
        self._log(f"File size: {result.file_size_gb:.2f} GB")
        self._log(f"Available memory: {result.available_memory_gb:.1f} GB")
        
        if result.use_single_thread:
            self._log("Strategy: Single-threaded conversion (small file)")
        else:
            self._log(f"Strategy: Multi-threaded conversion ({result.num_workers} workers)")
            self._log(f"Memory per worker: {result.memory_per_worker_gb:.2f} GB")
            self._log(f"Chunk dimensions: {result.chunk_dims}")
            self._log(f"Chunk size: {result.chunk_size_mb:.1f} MB")
            self._log(f"Total chunks: {result.total_chunks}")
            self._log(f"I/O reduction factor: {result.io_reduction_factor}x")
            self._log(f"Estimated memory usage: {result.estimated_memory_usage_gb:.2f} GB")
        
        self._log("=" * 60)
    
    def _update_progress_bar(self):
        """Update GUI progress bar"""
        if self.progress_callback:
            try:
                progress_percent = self.progress.progress_percent
                status_message = self.get_current_status()
                self.progress_callback(progress_percent, status_message)
            except Exception:
                # Don't let progress callback errors break the conversion
                pass
    
    def _log(self, message: str):
        """Send message to log callback and always print to terminal"""
        # Always print to terminal for debugging (especially when GUI crashes)
        print(f"ProgressReporter: {message}")

        # Also send to GUI log callback if available
        if self.log_callback:
            try:
                self.log_callback(message)
            except Exception:
                # Don't let logging errors break the conversion
                pass


def create_progress_reporter(chunking_result: ChunkingResult,
                           progress_callback: Optional[Callable] = None,
                           log_callback: Optional[Callable] = None,
                           verbose: bool = False) -> ProgressReporter:
    """
    Convenience function to create progress reporter with appropriate log level
    
    Parameters:
    -----------
    chunking_result : ChunkingResult
        Chunking strategy result
    progress_callback : callable, optional
        GUI progress update function
    log_callback : callable, optional  
        Logging function
    verbose : bool
        Enable detailed logging
        
    Returns:
    --------
    ProgressReporter : Configured progress reporter
    """
    log_level = LogLevel.DETAILED if verbose else LogLevel.BASIC
    
    return ProgressReporter(
        chunking_result=chunking_result,
        progress_callback=progress_callback,
        log_callback=log_callback,
        log_level=log_level
    )


if __name__ == "__main__":
    # Test progress reporting system
    from .adaptive_chunking import create_adaptive_chunking_strategy
    
    print("=== Progress Reporting System Test ===")
    
    # Create test chunking strategy
    file_shape = (64, 64, 256, 256)
    chunking_result = create_adaptive_chunking_strategy(file_shape)
    
    # Create progress reporter with detailed logging
    def test_progress_callback(percent, message):
        print(f"PROGRESS: {percent}% - {message}")
    
    def test_log_callback(message):
        print(f"LOG: {message}")
    
    reporter = ProgressReporter(
        chunking_result=chunking_result,
        progress_callback=test_progress_callback,
        log_callback=test_log_callback,
        log_level=LogLevel.VERBOSE
    )
    
    # Simulate some chunk processing
    print(f"\nSimulating processing of {chunking_result.total_chunks} chunks...")
    
    chunk_size_bytes = int(chunking_result.chunk_size_mb * 1024 * 1024)
    
    for chunk_id in range(min(3, chunking_result.total_chunks)):
        worker_id = chunk_id % chunking_result.num_workers
        
        # Simulate chunk processing stages
        reporter.start_chunk_processing(chunk_id, worker_id)
        time.sleep(0.1)  # Simulate read time
        
        reporter.chunk_read_complete(chunk_id, worker_id, 0.1, chunk_size_bytes)
        time.sleep(0.05)  # Simulate processing time
        
        reporter.chunk_processing_complete(chunk_id, worker_id, 0.05)
        time.sleep(0.02)  # Simulate write time
        
        reporter.chunk_write_complete(chunk_id, worker_id, 0.02)
        reporter.chunk_completed(chunk_id, worker_id, chunk_size_bytes)
    
    # Log summary
    time.sleep(0.1)
    reporter.log_performance_summary()
    
    print("\n✓ Progress reporting system test completed!")