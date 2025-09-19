#!/usr/bin/env python3
"""
Progressive EELS Data Loader

Enables immediate viewing of large EELS datasets (130GB+) by:
1. Initializing zero-filled display array
2. Processing chunks in parallel
3. Populating display array as chunks complete
4. Allowing real-time user interaction with processed regions

Reuses existing converter infrastructure for chunking, file reading, and threading.
"""

import os
import time
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional, Tuple, Any, Dict
from dataclasses import dataclass

# Import existing components for reuse
from .adaptive_chunking import AdaptiveChunkCalculator, ChunkingResult, ChunkInfo
from .mib_loader import get_mib_properties, detect_experiment_type


@dataclass
class ProgressiveLoadingResult:
    """Result of progressive loading initialization"""
    loader: 'ProgressiveEELSLoader'
    data_shape: Tuple[int, int, int, int]  # (sy, sx, 1, energy_channels)
    estimated_time_s: float
    memory_usage_mb: float


class ProgressiveEELSLoader:
    """
    Progressive loader for large EELS datasets

    Provides immediate access to large EELS files by processing chunks in background
    and populating a zero-filled display array as data becomes available.
    """

    def __init__(self, file_path: str, progress_callback: Optional[Callable] = None):
        self.file_path = file_path
        self.progress_callback = progress_callback

        # Initialize state
        self.is_loading = False
        self.chunks_completed = 0
        self.total_chunks = 0
        self.start_time = None

        # Thread management
        self.executor = None
        self.completion_lock = threading.Lock()

        # Initialize MIB properties and detect EELS
        self._initialize_file_properties()

        # Calculate chunking strategy
        self._calculate_chunking_strategy()

        # Initialize zero-filled display array
        self._initialize_display_array()

    def _initialize_file_properties(self):
        """Initialize MIB file properties and validate EELS data"""
        # Read MIB header
        with open(self.file_path, 'rb') as f:
            header_bytes = f.read(384)
        header_fields = header_bytes.decode('utf-8', errors='ignore').split(',')
        self.mib_props = get_mib_properties(header_fields)

        # Get file size for chunking
        file_size = os.path.getsize(self.file_path)

        # Calculate expected data shape
        merlin_frame_dtype = np.dtype([
            ('header', np.bytes_, self.mib_props.headsize),
            ('data', self.mib_props.pixeltype, self.mib_props.merlin_size)
        ])
        total_frames = file_size // merlin_frame_dtype.itemsize

        # Auto-detect scan size (reuse existing logic)
        from .mib_loader import auto_detect_scan_size
        self.scan_size = auto_detect_scan_size(total_frames)
        sy, sx = self.scan_size
        qy, qx = self.mib_props.merlin_size

        self.raw_shape = (sy, sx, qy, qx)

        # Validate this is EELS data
        experiment_type, exp_info = detect_experiment_type(self.raw_shape)
        if experiment_type != "EELS":
            raise ValueError(f"Progressive loading currently only supports EELS data, got: {experiment_type}")

        if not exp_info['can_sum_y']:
            raise ValueError("EELS data appears to already be summed - progressive loading not needed")

        self._log(f"Detected 4D EELS data: {sy}×{sx} scan, {qy}×{qx} detector")
        self._log(f"File size: {file_size / (1024**3):.1f} GB")

    def _calculate_chunking_strategy(self):
        """Calculate optimal chunking strategy using existing converter logic"""
        calculator = AdaptiveChunkCalculator()
        self.chunking_result = calculator.calculate_chunking_strategy(
            file_shape=self.raw_shape,
            file_path=self.file_path
        )

        self.total_chunks = self.chunking_result.total_chunks

        self._log(f"Chunking strategy: {self.chunking_result.chunk_dims}")
        self._log(f"Total chunks: {self.total_chunks}")
        self._log(f"Workers: {self.chunking_result.num_workers}")

    def _initialize_display_array(self):
        """Initialize zero-filled array for progressive display"""
        sy, sx, qy, qx = self.raw_shape

        # Target shape after EELS processing: (sy, sx, 1, energy_channels)
        # Raw: (sy, sx, 1024, 256) -> Reshape: (sy, sx, 256, 1024) -> Y-sum: (sy, sx, 1, 1024)
        # The longer raw dimension becomes the energy dimension after processing
        energy_channels = max(qy, qx)  # 1024 in this case
        self.processed_shape = (sy, sx, 1, energy_channels)

        # Initialize with zeros
        self.processed_data = np.zeros(self.processed_shape, dtype=np.float32)
        self.completion_mask = np.zeros((sy, sx), dtype=bool)  # Track completed regions

        memory_mb = self.processed_data.nbytes / (1024**2)
        self._log(f"Initialized display array: {self.processed_shape}")
        self._log(f"Memory usage: {memory_mb:.1f} MB")

    def start_loading(self) -> ProgressiveLoadingResult:
        """Start progressive loading in background"""
        if self.is_loading:
            raise RuntimeError("Loading already in progress")

        self.is_loading = True
        self.chunks_completed = 0
        self.start_time = time.time()

        # Create chunk queue
        calculator = AdaptiveChunkCalculator()
        chunk_queue = calculator.generate_chunk_queue(self.chunking_result)

        # Start thread pool
        self.executor = ThreadPoolExecutor(max_workers=self.chunking_result.num_workers)

        # Submit all chunks for processing
        for chunk_info in chunk_queue:
            self.executor.submit(self._process_chunk, chunk_info)

        self._log(f"Started progressive loading with {len(chunk_queue)} chunks")

        # Estimate completion time (rough guess based on file size)
        file_size_gb = os.path.getsize(self.file_path) / (1024**3)
        estimated_time_s = file_size_gb * 2.0  # Rough estimate: 2 seconds per GB

        return ProgressiveLoadingResult(
            loader=self,
            data_shape=self.processed_shape,
            estimated_time_s=estimated_time_s,
            memory_usage_mb=self.processed_data.nbytes / (1024**2)
        )

    def _process_chunk(self, chunk_info: ChunkInfo):
        """Process a single chunk: read -> reshape -> sum -> populate"""
        try:
            # 1. Read raw chunk using existing converter logic
            raw_chunk = self._read_chunk_from_mib(chunk_info)

            # 2. Apply EELS detector dimension fix (reshape)
            fixed_chunk = self._apply_eels_reshape(raw_chunk)

            # 3. Sum along detector Y dimension
            summed_chunk = self._apply_y_summing(fixed_chunk)

            # 4. Populate display array
            self._populate_display_array(chunk_info, summed_chunk)

            # 5. Update progress
            self._update_progress()

        except Exception as e:
            self._log(f"Error processing chunk {chunk_info.id}: {str(e)}")

    def _read_chunk_from_mib(self, chunk_info: ChunkInfo) -> np.ndarray:
        """Read chunk from MIB file (adapted from adaptive_converter)"""
        sy_slice, sx_slice = chunk_info.input_slice[:2]
        actual_sy = sy_slice.stop - sy_slice.start
        actual_sx = sx_slice.stop - sx_slice.start
        qy, qx = self.raw_shape[2], self.raw_shape[3]

        # Initialize chunk data array
        chunk_data = np.zeros((actual_sy, actual_sx, qy, qx), dtype=self.mib_props.pixeltype)

        # Calculate merlin frame structure
        merlin_frame_dtype = np.dtype([
            ('header', np.bytes_, self.mib_props.headsize),
            ('data', self.mib_props.pixeltype, self.mib_props.merlin_size)
        ])

        # Read frames for this chunk
        with open(self.file_path, 'rb') as f:
            for chunk_y in range(actual_sy):
                for chunk_x in range(actual_sx):
                    # Calculate global frame position
                    global_y = sy_slice.start + chunk_y
                    global_x = sx_slice.start + chunk_x
                    frame_index = global_y * self.raw_shape[1] + global_x

                    # Seek to frame (using fixed offset logic)
                    frame_offset = frame_index * merlin_frame_dtype.itemsize
                    f.seek(frame_offset)

                    # Read frame
                    frame_bytes = f.read(merlin_frame_dtype.itemsize)
                    if len(frame_bytes) == merlin_frame_dtype.itemsize:
                        frame_record = np.frombuffer(frame_bytes, dtype=merlin_frame_dtype)[0]
                        frame_data = frame_record['data'].reshape(qy, qx)
                        chunk_data[chunk_y, chunk_x] = frame_data

        return chunk_data

    def _apply_eels_reshape(self, chunk_data: np.ndarray) -> np.ndarray:
        """Apply EELS detector dimension reshape fix"""
        sy, sx, current_dy, current_dx = chunk_data.shape
        detector_width, detector_height = self.mib_props.merlin_size

        # Apply reshape fix for EELS data (adapted from mib_loader.py)
        if current_dy != current_dx and current_dy > current_dx:
            # This looks like scrambled EELS data
            if {current_dy, current_dx} == {detector_width, detector_height}:
                # Reshape each detector frame to unscramble
                reshaped_data = np.zeros((sy, sx, current_dx, current_dy), dtype=chunk_data.dtype)
                for scan_y in range(sy):
                    for scan_x in range(sx):
                        frame = chunk_data[scan_y, scan_x, :, :]
                        reshaped_frame = frame.reshape(current_dx, current_dy)
                        reshaped_data[scan_y, scan_x, :, :] = reshaped_frame
                return reshaped_data

        # No reshape needed or safety check failed
        return chunk_data

    def _apply_y_summing(self, chunk_data: np.ndarray) -> np.ndarray:
        """Sum along detector Y dimension to reduce data size and flip energy axis"""
        # Sum along the shorter detector dimension (axis 2 after reshape)
        summed_data = np.sum(chunk_data, axis=2, keepdims=True)

        # Flip energy axis (axis 3) like normal EELS loading does
        return summed_data[:, :, :, ::-1]

    def _populate_display_array(self, chunk_info: ChunkInfo, processed_chunk: np.ndarray):
        """Populate the display array with processed chunk data"""
        sy_slice, sx_slice = chunk_info.input_slice[:2]

        with self.completion_lock:
            # Update display array
            self.processed_data[sy_slice, sx_slice] = processed_chunk

            # Mark region as completed
            self.completion_mask[sy_slice, sx_slice] = True

    def _update_progress(self):
        """Update progress tracking and call progress callback"""
        with self.completion_lock:
            self.chunks_completed += 1

            if self.progress_callback:
                progress_pct = (self.chunks_completed / self.total_chunks) * 100
                elapsed_time = time.time() - self.start_time if self.start_time else 0

                self.progress_callback({
                    'chunks_completed': self.chunks_completed,
                    'total_chunks': self.total_chunks,
                    'progress_percent': progress_pct,
                    'elapsed_time': elapsed_time,
                    'completed_regions': np.sum(self.completion_mask) / self.completion_mask.size * 100
                })

            # Check if loading is complete
            if self.chunks_completed >= self.total_chunks:
                self._on_loading_complete()

    def _on_loading_complete(self):
        """Called when all chunks are processed"""
        self.is_loading = False
        total_time = time.time() - self.start_time if self.start_time else 0
        self._log(f"Progressive loading completed in {total_time:.1f}s")

        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=False)

    def get_processed_data(self) -> np.ndarray:
        """Get current state of processed data array"""
        return self.processed_data.copy()

    def get_completion_mask(self) -> np.ndarray:
        """Get mask showing which regions have been processed"""
        return self.completion_mask.copy()

    def is_region_ready(self, y_slice: slice, x_slice: slice) -> bool:
        """Check if a specific scan region has been processed"""
        return np.all(self.completion_mask[y_slice, x_slice])

    def stop_loading(self):
        """Stop progressive loading"""
        if self.executor:
            self.executor.shutdown(wait=True)
        self.is_loading = False
        self._log("Progressive loading stopped")

    def _log(self, message: str):
        """Log message with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] ProgressiveEELS: {message}")


def should_use_progressive_loading(file_path: str, memory_threshold_gb: float = 4.0) -> bool:
    """
    Determine if progressive loading should be used for a file

    Parameters:
    -----------
    file_path : str
        Path to MIB file
    memory_threshold_gb : float
        Memory threshold in GB. If processed data would exceed this, use progressive loading

    Returns:
    --------
    bool : True if progressive loading recommended
    """
    try:
        # Quick check of file properties
        with open(file_path, 'rb') as f:
            header_bytes = f.read(384)
        header_fields = header_bytes.decode('utf-8', errors='ignore').split(',')
        mib_props = get_mib_properties(header_fields)

        # Calculate expected processed data size
        file_size = os.path.getsize(file_path)
        merlin_frame_dtype = np.dtype([
            ('header', np.bytes_, mib_props.headsize),
            ('data', mib_props.pixeltype, mib_props.merlin_size)
        ])
        total_frames = file_size // merlin_frame_dtype.itemsize

        from .mib_loader import auto_detect_scan_size
        scan_size = auto_detect_scan_size(total_frames)
        sy, sx = scan_size
        qy, qx = mib_props.merlin_size

        # Detect experiment type
        experiment_type, exp_info = detect_experiment_type((sy, sx, qy, qx))

        if experiment_type == "EELS" and exp_info['can_sum_y']:
            # Calculate size after Y-summing
            processed_size_gb = (sy * sx * max(qy, qx) * 4) / (1024**3)  # float32
            return processed_size_gb > memory_threshold_gb

        return False

    except Exception:
        return False  # Default to normal loading if analysis fails