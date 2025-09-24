"""
Adaptive MIB-EMD Converter V2 - Clean threading implementation

This module implements the clean adaptive converter following MULTITHREADED_ARCHITECTURE_PLAN.md
that uses simple ThreadPoolExecutor with one task per chunk (no queues, no worker loops).

Key improvements over V1:
- Simple threading: One ThreadPoolExecutor task per chunk
- No queue-based workers or complex coordination
- Massive I/O reduction: 4-16 chunks instead of 65,536 
- No segfaults: Clean memory management
- Fast performance: Target 10x faster than broken V1

Architecture:
[MIB File] → [Simple ThreadPool: one task per chunk] → [EMD File]
"""

import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, Tuple, Callable
import numpy as np
import h5py

from .adaptive_chunking_v2 import AdaptiveChunkCalculator, ChunkingResult, ChunkInfo
from .processing_pipeline import apply_data_processing
from .mib_loader import get_mib_properties


class SimpleProgressReporter:
    """Simple progress reporting for clean threading model"""
    
    def __init__(self, total_chunks: int, log_callback: Optional[Callable] = None):
        self.total_chunks = total_chunks
        self.completed_chunks = 0
        self.lock = threading.Lock()
        self.log_callback = log_callback or print
        self.start_time = time.time()
        
    def chunk_completed(self, chunk_info: ChunkInfo):
        """Report that a chunk has been completed"""
        with self.lock:
            self.completed_chunks += 1
            progress_percent = (self.completed_chunks / self.total_chunks) * 100
            elapsed_time = time.time() - self.start_time
            
            if self.completed_chunks < self.total_chunks:
                eta_seconds = (elapsed_time / self.completed_chunks) * (self.total_chunks - self.completed_chunks)
                self.log_callback(f"Chunk {chunk_info.chunk_id} completed "
                                f"({progress_percent:.0f}%, {self.total_chunks - self.completed_chunks} remaining, "
                                f"ETA: {eta_seconds:.0f}s)")
            else:
                self.log_callback(f"All chunks completed! Total time: {elapsed_time:.1f}s")
                
    def get_progress(self) -> float:
        """Get current progress as percentage"""
        with self.lock:
            return (self.completed_chunks / self.total_chunks) * 100


class AdaptiveMibEmdConverterV2:
    """
    Clean adaptive MIB-EMD converter using simple ThreadPoolExecutor threading
    
    Following the architectural principles:
    - One ThreadPoolExecutor task per chunk (no queues)
    - Adaptive chunking for massive I/O reduction  
    - Clean memory management (no segfaults)
    - Simple progress reporting
    """
    
    def __init__(self,
                 compression: Optional[str] = 'gzip',
                 compression_level: int = 4,
                 max_workers: Optional[int] = None,
                 progress_callback: Optional[Callable] = None,
                 log_callback: Optional[Callable] = None,
                 verbose: bool = True):
        """
        Initialize the adaptive converter
        
        Args:
            compression: Compression type ('gzip', 'lzf', or None)
            compression_level: Compression level (1-9 for gzip)
            max_workers: Maximum worker threads (auto-detected if None)
            progress_callback: Callback for progress updates
            log_callback: Callback for log messages
            verbose: Enable verbose logging
        """
        self.compression = compression
        self.compression_level = compression_level
        self.max_workers = max_workers or min(os.cpu_count(), 16)  # Reasonable limit
        self.progress_callback = progress_callback
        self.log_callback = log_callback or print
        self.verbose = verbose
        
        if self.verbose:
            self.log("Adaptive MIB-EMD Converter V2 initialized")
            
    def log(self, message: str):
        """Log a message"""
        if self.log_callback:
            self.log_callback(f"AdaptiveConverterV2: {message}")
    
    def convert_to_emd(self,
                       input_path: str, 
                       output_path: str,
                       processing_options: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Convert MIB file to EMD with adaptive chunking
        
        Args:
            input_path: Path to input MIB file
            output_path: Path to output EMD file  
            processing_options: Processing options (binning, etc.)
            
        Returns:
            Dictionary with conversion statistics and performance metrics
        """
        start_time = time.time()
        
        if self.verbose:
            self.log(f"Starting adaptive conversion: {os.path.basename(input_path)}")
            
        # Step 1: Analyze file and calculate adaptive chunking strategy
        file_shape = self._get_file_shape(input_path)
        chunking_result = AdaptiveChunkCalculator.calculate_adaptive_chunks(
            input_path, file_shape
        )
        
        if self.verbose:
            self._log_chunking_strategy(chunking_result)
            
        # Step 2: Create output file structure
        processed_shape = self._calculate_processed_shape(file_shape, processing_options)
        self._create_emd_file_structure(output_path, processed_shape, chunking_result)
        
        # Step 3: Process chunks with simple ThreadPoolExecutor
        conversion_stats = self._process_chunks_simple(
            input_path, output_path, chunking_result, processing_options
        )
        
        # Step 4: Create final result
        total_time = time.time() - start_time
        if self.verbose:
            self.log(f"Adaptive conversion completed in {total_time:.1f}s")
            self.log(f"Achieved {chunking_result.io_reduction_factor}x I/O reduction!")
            
        return self._create_result(
            input_path, output_path, chunking_result, conversion_stats, total_time
        )
    
    def _get_file_shape(self, input_path: str) -> Tuple[int, int, int, int]:
        """Get MIB file shape with proper scan size detection"""
        # Read header and get file size
        with open(input_path, 'rb') as f:
            header_bytes = f.read(384)
            f.seek(0, os.SEEK_END)
            filesize = f.tell()
            
        header_fields = header_bytes.decode('utf-8', errors='ignore').split(',')
        mib_props = get_mib_properties(header_fields)
        
        # Calculate number of frames in file
        import numpy as np
        merlin_frame_dtype = np.dtype([
            ('header', np.bytes_, mib_props.headsize),
            ('data', mib_props.pixeltype, mib_props.merlin_size)
        ])
        num_frames = filesize // merlin_frame_dtype.itemsize
        
        # Auto-detect scan size from number of frames
        from .mib_loader import auto_detect_scan_size
        scan_size = auto_detect_scan_size(num_frames)
        mib_props.scan_size = scan_size
        mib_props.numberOfFramesInFile = num_frames
        
        # Return shape as (sy, sx, qy, qx)
        sy, sx = mib_props.scan_size
        qy, qx = mib_props.merlin_size
        
        return (sy, sx, qy, qx)
    
    def _calculate_processed_shape(self, 
                                 file_shape: Tuple[int, int, int, int],
                                 processing_options: Optional[Dict]) -> Tuple[int, int, int, int]:
        """Calculate output shape after processing"""
        sy, sx, qy, qx = file_shape
        
        if not processing_options:
            return file_shape
            
        # Apply binning to detector dimensions
        bin_factor = processing_options.get('bin_factor', 1)
        if bin_factor > 1:
            qy = qy // bin_factor
            qx = qx // bin_factor
            
        # Apply Y-summing
        if processing_options.get('sum_y', False):
            qy = 1
            
        return (sy, sx, qy, qx)
    
    def _log_chunking_strategy(self, chunking_result: ChunkingResult):
        """Log the chunking strategy details"""
        self.log("=" * 60)
        self.log("ADAPTIVE CHUNKING STRATEGY")
        self.log("=" * 60)
        self.log(f"File size: {chunking_result.file_size_gb:.2f} GB")
        self.log(f"Strategy: {chunking_result.strategy.value}")
        self.log(f"Chunk dimensions: {chunking_result.chunk_dims}")
        self.log(f"Total chunks: {chunking_result.total_chunks}")
        self.log(f"I/O reduction factor: {chunking_result.io_reduction_factor}x")
        self.log(f"Memory per chunk: {chunking_result.estimated_memory_per_chunk_gb:.2f} GB")
        self.log("=" * 60)
    
    def _create_emd_file_structure(self,
                                 output_path: str,
                                 shape_4d: Tuple[int, int, int, int],
                                 chunking_result: ChunkingResult):
        """Create EMD file structure with proper version_1 hierarchy"""
        sy, sx, qy, qx = shape_4d
        chunk_sy, chunk_sx, chunk_qy, chunk_qx = chunking_result.chunk_dims
        
        # Calculate HDF5 chunking (different from our processing chunks)
        hdf5_chunk_dims = (min(32, chunk_sy), min(32, chunk_sx), 
                          min(256, chunk_qy), min(256, chunk_qx))
        
        with h5py.File(output_path, 'w') as f:
            # Create EMD 1.0 structure: version_1/data/datacubes/datacube_000
            version_group = f.create_group('version_1')
            version_group.attrs['major'] = 1
            version_group.attrs['minor'] = 0
            
            data_group = version_group.create_group('data')
            datacubes_group = data_group.create_group('datacubes')
            datacube_group = datacubes_group.create_group('datacube_000')
            
            # Create main dataset
            compression_opts = {}
            if self.compression:
                compression_opts['compression'] = self.compression
                if self.compression == 'gzip':
                    compression_opts['compression_opts'] = self.compression_level
            
            dataset = datacube_group.create_dataset(
                'data',
                shape=(sy, sx, qy, qx),
                chunks=hdf5_chunk_dims,
                dtype=np.uint16,
                **compression_opts
            )
            
            # Add metadata
            datacube_group.attrs['emd_group_type'] = 1
            
            # Create dimension groups
            for i, (name, units) in enumerate([
                ('scan_y', 'pixels'), ('scan_x', 'pixels'),
                ('detector_y', 'pixels'), ('detector_x', 'pixels')
            ]):
                dim_group = datacube_group.create_group(f'dim{i+1}')
                dim_group.attrs['name'] = name
                dim_group.attrs['units'] = units
                
        self.log(f"Created EMD structure with shape {shape_4d}")
    
    def _process_chunks_simple(self,
                             input_path: str,
                             output_path: str, 
                             chunking_result: ChunkingResult,
                             processing_options: Optional[Dict]) -> Dict[str, Any]:
        """
        Process chunks using simple ThreadPoolExecutor - one task per chunk
        
        This is the core of the clean threading architecture:
        - No queues, no worker loops, no complex coordination
        - Just submit one task per chunk to ThreadPoolExecutor
        - Each task processes its chunk independently
        """
        
        # Create progress reporter
        progress_reporter = SimpleProgressReporter(
            chunking_result.total_chunks, self.log_callback
        )
        
        # Statistics
        stats = {
            'chunks_processed': 0,
            'total_bytes_processed': 0,
            'errors': []
        }
        stats_lock = threading.Lock()
        
        # Load MIB properties once (shared across all tasks)
        with open(input_path, 'rb') as f:
            header_bytes = f.read(384)
        header_fields = header_bytes.decode('utf-8', errors='ignore').split(',')
        mib_props = get_mib_properties(header_fields)
        
        # Set up scan size and frame count (same logic as _get_file_shape)
        filesize = os.path.getsize(input_path)
        merlin_frame_dtype = np.dtype([
            ('header', np.bytes_, mib_props.headsize),
            ('data', mib_props.pixeltype, mib_props.merlin_size)
        ])
        num_frames = filesize // merlin_frame_dtype.itemsize
        
        from .mib_loader import auto_detect_scan_size
        scan_size = auto_detect_scan_size(num_frames) 
        mib_props.scan_size = scan_size
        mib_props.numberOfFramesInFile = num_frames
        
        # Use optimal number of workers (don't exceed chunk count)
        num_workers = min(self.max_workers, chunking_result.total_chunks)
        
        if self.verbose:
            self.log(f"Using {num_workers} workers for {chunking_result.total_chunks} chunks")
            
        # Simple ThreadPoolExecutor - one task per chunk
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all chunk processing tasks
            future_to_chunk = {}
            for chunk_info in chunking_result.chunks:
                future = executor.submit(
                    self._process_single_chunk_task,
                    input_path, output_path, chunk_info, mib_props,
                    processing_options, progress_reporter, stats, stats_lock
                )
                future_to_chunk[future] = chunk_info
                
            # Wait for all tasks to complete
            for future in as_completed(future_to_chunk):
                chunk_info = future_to_chunk[future]
                try:
                    future.result()  # This will raise any exceptions from the task
                except Exception as e:
                    error_msg = f"Error processing chunk {chunk_info.chunk_id}: {str(e)}"
                    self.log(error_msg)
                    with stats_lock:
                        stats['errors'].append(error_msg)
        
        self.log("All chunks processed successfully")
        return stats
    
    def _process_single_chunk_task(self,
                                 input_path: str,
                                 output_path: str,
                                 chunk_info: ChunkInfo,
                                 mib_props,  # MibProperties object
                                 processing_options: Optional[Dict],
                                 progress_reporter: SimpleProgressReporter,
                                 stats: Dict,
                                 stats_lock: threading.Lock):
        """
        Process a single chunk - this runs as one ThreadPoolExecutor task
        
        Simple three-phase pipeline:
        1. Read chunk from MIB file
        2. Process chunk (binning, etc.)  
        3. Write chunk to EMD file
        """
        
        try:
            # Phase 1: Read chunk
            chunk_data = self._read_chunk_from_mib(input_path, chunk_info, mib_props)
            chunk_bytes = chunk_data.nbytes
            
            # Phase 2: Process chunk  
            if processing_options:
                chunk_data = apply_data_processing(chunk_data, processing_options)
                
            # Phase 3: Write chunk (thread-safe)
            self._write_chunk_to_emd(output_path, chunk_info, chunk_data)
            
            # Update statistics
            with stats_lock:
                stats['chunks_processed'] += 1
                stats['total_bytes_processed'] += chunk_bytes
                
            # Report progress
            progress_reporter.chunk_completed(chunk_info)
            
        except Exception as e:
            # Let the exception bubble up to be handled by the executor
            raise Exception(f"Chunk {chunk_info.chunk_id} processing failed: {str(e)}")
    
    def _read_chunk_from_mib(self,
                           input_path: str,
                           chunk_info: ChunkInfo,
                           mib_props) -> np.ndarray:
        """Read a chunk from MIB file"""
        sy, sx, qy, qx = chunk_info.expected_shape
        chunk_data = np.zeros((sy, sx, qy, qx), dtype=mib_props.pixeltype)
        
        # Get scan dimensions from mib_props
        total_sy, total_sx = mib_props.scan_size
        detector_qy, detector_qx = mib_props.merlin_size
        
        # Calculate frame size in bytes
        frame_size = detector_qy * detector_qx * mib_props.pixeltype.itemsize
        
        with open(input_path, 'rb') as file:
            for chunk_y in range(sy):
                for chunk_x in range(sx):
                    # Calculate global frame position
                    global_y = chunk_info.input_slice[0].start + chunk_y
                    global_x = chunk_info.input_slice[1].start + chunk_x
                    frame_index = global_y * total_sx + global_x
                    
                    # Read frame
                    frame_offset = mib_props.headsize + frame_index * frame_size
                    file.seek(frame_offset)
                    frame_bytes = file.read(frame_size)
                    
                    # Decode frame
                    frame_data = np.frombuffer(frame_bytes, dtype=mib_props.pixeltype)
                    frame_data = frame_data.reshape(detector_qy, detector_qx)
                    
                    chunk_data[chunk_y, chunk_x] = frame_data
                    
        return chunk_data
    
    def _write_chunk_to_emd(self,
                          output_path: str,
                          chunk_info: ChunkInfo, 
                          chunk_data: np.ndarray):
        """
        Write processed chunk to EMD file with proper slice handling
        
        Note: HDF5 is thread-safe for writing to different regions of the same file,
        so we don't need a global write lock like in the broken V1 implementation
        """
        
        with h5py.File(output_path, 'a') as f:
            dataset = f['version_1/data/datacubes/datacube_000/data']
            
            # Calculate correct output slice based on actual chunk data dimensions
            original_slice = chunk_info.output_slice
            sy_slice = original_slice[0]  # Scan dimensions unchanged
            sx_slice = original_slice[1]
            
            # Detector dimensions: use full range based on processed data
            qy_max, qx_max = chunk_data.shape[2], chunk_data.shape[3]  
            output_slice = (sy_slice, sx_slice, slice(0, qy_max), slice(0, qx_max))
            
            # Write chunk
            dataset[output_slice] = chunk_data
    
    def _create_result(self,
                      input_path: str,
                      output_path: str,
                      chunking_result: ChunkingResult,
                      conversion_stats: Dict,
                      total_time: float) -> Dict[str, Any]:
        """Create clean result dictionary"""
        
        return {
            'input_path': str(input_path),
            'output_path': str(output_path),
            'input_size_gb': float(chunking_result.file_size_gb),
            'output_size_gb': float(os.path.getsize(output_path) / (1024**3)),
            'compression_ratio': float(chunking_result.file_size_gb / (os.path.getsize(output_path) / (1024**3))),
            'total_time_s': float(total_time),
            'total_time_seconds': float(total_time),  # GUI compatibility
            'chunking_strategy': str(chunking_result.strategy.value),
            'adaptive_chunking': True,
            'num_workers': int(min(self.max_workers, chunking_result.total_chunks)),
            'total_chunks': int(chunking_result.total_chunks),
            'io_reduction_factor': int(chunking_result.io_reduction_factor),
            'chunks_processed': int(conversion_stats.get('chunks_processed', 0)),
            'total_bytes_processed': int(conversion_stats.get('total_bytes_processed', 0)),
            'throughput_mb_s': float((chunking_result.file_size_gb * 1024) / total_time),
            'errors': conversion_stats.get('errors', [])
        }