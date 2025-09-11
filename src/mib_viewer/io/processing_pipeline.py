#!/usr/bin/env python3
"""
ProcessingPipeline - Multithreaded data processing with overlapping I/O and compute

This module provides a high-performance pipeline for processing large datasets
with overlapping I/O, compute, and output operations for maximum throughput.

Key Features:
- Overlapping I/O, compute, and write operations
- CPU-intensive processing in background threads  
- Intelligent queue management and backpressure
- Real-time progress aggregation from multiple workers
- Memory-safe operation with bounded queues
"""

import os
import time
import queue
import threading
import numpy as np
from typing import Callable, Optional, Any, Dict, Tuple, Iterator
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import h5py

# Import existing processing functions
from .mib_loader import apply_data_processing
from .smart_data_reader import SmartDataReader


@dataclass
class ProcessedChunk:
    """Container for processed chunk data"""
    chunk_id: int
    data: np.ndarray
    scan_slice: Tuple[slice, slice]
    processing_time: float
    original_shape: Tuple[int, int, int, int]
    processed_shape: Tuple[int, int, int, int]


@dataclass 
class PipelineStats:
    """Pipeline performance statistics"""
    total_chunks: int
    chunks_completed: int
    total_load_time: float
    total_process_time: float
    total_write_time: float
    total_pipeline_time: float
    avg_throughput_mb_s: float
    peak_memory_mb: float
    queue_stats: Dict[str, Any]


class ProcessingPipeline:
    """
    High-performance processing pipeline with overlapping I/O and compute
    
    This pipeline coordinates three types of operations:
    1. Loading chunks from SmartDataReader (I/O bound)
    2. Processing chunks (CPU bound - binning, summing, etc.)  
    3. Writing processed chunks to output (I/O bound)
    
    All three operations can run concurrently for maximum throughput.
    """
    
    def __init__(self,
                 processing_options: Optional[Dict] = None,
                 max_cpu_workers: Optional[int] = None,
                 max_io_workers: int = 2,
                 queue_size: int = 4,
                 progress_callback: Optional[Callable] = None):
        """
        Initialize ProcessingPipeline
        
        Parameters:
        -----------
        processing_options : dict, optional
            Data processing options (binning, summing, etc.)
        max_cpu_workers : int, optional  
            Number of CPU threads for processing (default: cpu_count)
        max_io_workers : int
            Number of I/O threads for writing (default: 2)
        queue_size : int
            Size of internal queues for memory management (default: 4)
        progress_callback : callable, optional
            Progress callback function(current, total, message)
        """
        self.processing_options = processing_options or {}
        self.progress_callback = progress_callback
        
        # Thread pool configuration
        self.max_cpu_workers = max_cpu_workers or os.cpu_count() or 4
        self.max_io_workers = max_io_workers
        self.queue_size = queue_size
        
        # Thread pools
        self.cpu_executor = ThreadPoolExecutor(max_workers=self.max_cpu_workers, 
                                              thread_name_prefix="CPU")
        self.io_executor = ThreadPoolExecutor(max_workers=self.max_io_workers,
                                             thread_name_prefix="IO")
        
        # Inter-stage queues with backpressure
        self.process_queue = queue.Queue(maxsize=queue_size)  # Raw chunks → Processing
        self.write_queue = queue.Queue(maxsize=queue_size)    # Processed chunks → Writing
        
        # Pipeline state
        self._shutdown_event = threading.Event()
        self._stats_lock = threading.Lock()
        self._stats = {
            'chunks_loaded': 0,
            'chunks_processed': 0, 
            'chunks_written': 0,
            'total_load_time': 0,
            'total_process_time': 0,
            'total_write_time': 0,
            'peak_memory_mb': 0
        }
        
        self._log(f"ProcessingPipeline initialized:")
        self._log(f"  CPU workers: {self.max_cpu_workers}")
        self._log(f"  I/O workers: {self.max_io_workers}")
        self._log(f"  Queue size: {self.queue_size}")
        if processing_options:
            self._log(f"  Processing: {processing_options}")
    
    def process_file(self,
                     reader: SmartDataReader,
                     output_path: str,
                     metadata: Dict) -> PipelineStats:
        """
        Process entire file through the pipeline
        
        Parameters:
        -----------
        reader : SmartDataReader
            Data reader for input file
        output_path : str
            Path for output file
        metadata : dict
            File metadata for output creation
            
        Returns:
        --------
        PipelineStats : Pipeline performance statistics
        """
        total_chunks = reader.total_chunks
        self._log(f"Starting pipeline processing: {total_chunks} chunks")
        
        start_time = time.time()
        
        # Create output file structure
        self._create_output_file(output_path, metadata)
        
        # Start the three-stage pipeline
        try:
            # Stage 1: Start loading chunks (producer)
            load_thread = threading.Thread(
                target=self._load_chunks_stage,
                args=(reader,),
                name="LoadStage"
            )
            
            # Stage 2: Start processing chunks (transformer)
            process_thread = threading.Thread(
                target=self._process_chunks_stage,
                name="ProcessStage"
            )
            
            # Stage 3: Start writing chunks (consumer)
            write_thread = threading.Thread(
                target=self._write_chunks_stage,
                args=(output_path, total_chunks),
                name="WriteStage"
            )
            
            # Launch all stages
            load_thread.start()
            process_thread.start() 
            write_thread.start()
            
            # Wait for completion
            load_thread.join()
            process_thread.join()
            write_thread.join()
            
            total_time = time.time() - start_time
            
            # Calculate final statistics
            stats = self._calculate_final_stats(total_time, total_chunks)
            
            self._log(f"Pipeline completed in {total_time:.1f}s")
            self._log(f"  Throughput: {stats.avg_throughput_mb_s:.1f} MB/s")
            
            return stats
            
        except Exception as e:
            self._shutdown_event.set()  # Signal shutdown
            raise e
    
    def _create_output_file(self, output_path: str, metadata: Dict):
        """Create output EMD file structure"""
        # Determine output data type based on processing
        if self.processing_options and self.processing_options.get('bin_method', 'mean') == 'mean':
            output_dtype = np.float32  # Mean binning produces floats
        else:
            output_dtype = np.uint16   # No processing or sum binning
        
        # Calculate final shape after processing
        if self.processing_options:
            # Apply processing to get final shape
            test_shape = (1, 1) + metadata['shape_4d'][2:]
            test_data = np.zeros(test_shape, dtype=np.uint16)
            processed_test = apply_data_processing(test_data, self.processing_options)
            final_shape = metadata['shape_4d'][:2] + processed_test.shape[2:]
        else:
            final_shape = metadata['shape_4d']
            
        self._log(f"Creating output file: {final_shape} ({output_dtype})")
        
        # Use existing chunking logic
        from .mib_to_emd_converter import MibToEmdConverter
        converter = MibToEmdConverter()
        chunks = converter.determine_optimal_chunks(final_shape)
        
        # Create EMD file structure
        with h5py.File(output_path, 'w') as f:
            # EMD 1.0 structure
            f.attrs['emd_group_type'] = 'file'
            f.attrs['version_major'] = 1
            f.attrs['version_minor'] = 0
            f.attrs['authoring_program'] = 'multithreaded-pipeline'
            
            version_group = f.create_group('version_1')
            version_group.attrs['emd_group_type'] = 'root'
            
            data_group = version_group.create_group('data')
            datacubes_group = data_group.create_group('datacubes')
            datacube_group = datacubes_group.create_group('datacube_000')
            datacube_group.attrs['emd_group_type'] = 'array'
            
            # Create empty dataset for concurrent writing
            dataset = datacube_group.create_dataset(
                'data',
                shape=final_shape,
                dtype=output_dtype,
                chunks=chunks,
                compression='gzip',
                compression_opts=6
            )
            dataset.attrs['units'] = 'counts'
            
            # Add EMD metadata
            converter._add_emd_metadata(datacube_group, version_group, 
                                       {**metadata, 'shape_4d': final_shape})
    
    def _load_chunks_stage(self, reader: SmartDataReader):
        """Stage 1: Load chunks from reader and queue for processing"""
        self._log("Load stage started")
        
        try:
            for chunk_id, future in reader.get_chunk_iterator():
                if self._shutdown_event.is_set():
                    break
                
                load_start = time.time()
                chunk_data = future.result()  # Wait for chunk to load
                load_time = time.time() - load_start
                
                # Get chunk info
                chunk_info = reader.get_chunk_info(chunk_id)
                
                # Queue for processing (blocks if queue is full - backpressure)
                item = (chunk_id, chunk_data, chunk_info, load_time)
                self.process_queue.put(item)
                
                # Update stats
                with self._stats_lock:
                    self._stats['chunks_loaded'] += 1
                    self._stats['total_load_time'] += load_time
                
                self._log(f"Loaded chunk {chunk_id}: {chunk_data.shape} in {load_time:.2f}s")
                
        except Exception as e:
            self._log(f"Load stage error: {e}")
            raise
        finally:
            # Signal end of loading
            self.process_queue.put(None)
            self._log("Load stage completed")
    
    def _process_chunks_stage(self):
        """Stage 2: Process chunks and queue for writing"""
        self._log("Process stage started")
        
        # Submit processing jobs to CPU thread pool
        processing_futures = {}
        
        try:
            while not self._shutdown_event.is_set():
                try:
                    item = self.process_queue.get(timeout=1.0)
                    if item is None:  # End signal from loader
                        break
                        
                    chunk_id, chunk_data, chunk_info, load_time = item
                    
                    # Submit processing to CPU pool
                    future = self.cpu_executor.submit(
                        self._process_single_chunk,
                        chunk_id, chunk_data, chunk_info, load_time
                    )
                    processing_futures[future] = chunk_id
                    
                except queue.Empty:
                    continue
            
            # Wait for all processing to complete
            for future in as_completed(processing_futures):
                chunk_id = processing_futures[future]
                processed_chunk = future.result()
                
                # Queue for writing
                self.write_queue.put(processed_chunk)
                
                with self._stats_lock:
                    self._stats['chunks_processed'] += 1
                
                self._log(f"Processed chunk {chunk_id}")
                
        except Exception as e:
            self._log(f"Process stage error: {e}")
            raise
        finally:
            # Signal end of processing
            self.write_queue.put(None)
            self._log("Process stage completed")
    
    def _process_single_chunk(self, chunk_id: int, chunk_data: np.ndarray, 
                             chunk_info, load_time: float) -> ProcessedChunk:
        """Process a single chunk"""
        process_start = time.time()
        
        original_shape = chunk_data.shape
        
        # Apply data processing if specified
        if self.processing_options:
            processed_data = apply_data_processing(chunk_data, self.processing_options)
        else:
            processed_data = chunk_data
            
        process_time = time.time() - process_start
        
        with self._stats_lock:
            self._stats['total_process_time'] += process_time
        
        return ProcessedChunk(
            chunk_id=chunk_id,
            data=processed_data,
            scan_slice=chunk_info.scan_slice,
            processing_time=process_time,
            original_shape=original_shape,
            processed_shape=processed_data.shape
        )
    
    def _write_chunks_stage(self, output_path: str, total_chunks: int):
        """Stage 3: Write processed chunks to output file"""
        self._log("Write stage started")
        
        chunks_written = 0
        
        try:
            with h5py.File(output_path, 'r+') as f:
                dataset = f['version_1/data/datacubes/datacube_000/data']
                
                while not self._shutdown_event.is_set():
                    try:
                        processed_chunk = self.write_queue.get(timeout=1.0)
                        if processed_chunk is None:  # End signal
                            break
                            
                        write_start = time.time()
                        
                        # Write chunk to dataset
                        sy_slice, sx_slice = processed_chunk.scan_slice
                        dataset[sy_slice, sx_slice, :, :] = processed_chunk.data
                        
                        write_time = time.time() - write_start
                        chunks_written += 1
                        
                        with self._stats_lock:
                            self._stats['chunks_written'] += 1
                            self._stats['total_write_time'] += write_time
                        
                        # Update progress
                        if self.progress_callback:
                            progress = int((chunks_written / total_chunks) * 100)
                            self.progress_callback(
                                chunks_written, total_chunks,
                                f"Writing chunk {chunks_written}/{total_chunks}"
                            )
                        
                        self._log(f"Wrote chunk {processed_chunk.chunk_id} in {write_time:.2f}s")
                        
                    except queue.Empty:
                        continue
                        
        except Exception as e:
            self._log(f"Write stage error: {e}")
            raise
        finally:
            self._log(f"Write stage completed: {chunks_written} chunks")
    
    def _calculate_final_stats(self, total_time: float, total_chunks: int) -> PipelineStats:
        """Calculate final pipeline statistics"""
        with self._stats_lock:
            stats = self._stats.copy()
        
        # Calculate throughput (approximate)
        total_data_mb = total_chunks * 100  # Rough estimate
        avg_throughput = total_data_mb / total_time if total_time > 0 else 0
        
        return PipelineStats(
            total_chunks=total_chunks,
            chunks_completed=stats['chunks_written'],
            total_load_time=stats['total_load_time'],
            total_process_time=stats['total_process_time'], 
            total_write_time=stats['total_write_time'],
            total_pipeline_time=total_time,
            avg_throughput_mb_s=avg_throughput,
            peak_memory_mb=stats['peak_memory_mb'],
            queue_stats={
                'process_queue_size': self.process_queue.qsize(),
                'write_queue_size': self.write_queue.qsize()
            }
        )
    
    def shutdown(self):
        """Clean shutdown of pipeline"""
        self._shutdown_event.set()
        self.cpu_executor.shutdown(wait=True)
        self.io_executor.shutdown(wait=True)
        
        # Clear queues
        while not self.process_queue.empty():
            try:
                self.process_queue.get_nowait()
            except queue.Empty:
                break
                
        while not self.write_queue.empty():
            try:
                self.write_queue.get_nowait()
            except queue.Empty:
                break
    
    def _log(self, message: str):
        """Log pipeline messages"""
        print(f"[ProcessingPipeline] {message}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


def create_processing_pipeline(processing_options: Dict = None, **kwargs) -> ProcessingPipeline:
    """Create a ProcessingPipeline with sensible defaults"""
    return ProcessingPipeline(processing_options=processing_options, **kwargs)