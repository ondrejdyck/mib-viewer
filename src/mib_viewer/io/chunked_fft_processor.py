#!/usr/bin/env python3
"""
Chunked 4D FFT Processor

This module provides memory-efficient 4D FFT computation using detector dimension chunking.
Supports both cropped and full FFT workflows for different analysis needs.

Key Features:
- Memory-bounded processing using detector dimension chunking
- Parallel computation with ThreadPoolExecutor
- Direct EMD file output for efficient storage
- Support for both cropped (BF-focused) and full detector FFT
- Progress reporting and error handling
"""

import os
import time
import threading
import psutil
import numpy as np
import h5py
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any, Tuple, Callable, List, Union
from dataclasses import dataclass
from enum import Enum

# Import adaptive chunking system
from .adaptive_chunking import (
    AdaptiveChunkCalculator, ChunkingResult, ChunkInfo,
    create_adaptive_chunking_strategy
)
from .progress_reporting import (
    ProgressReporter, LogLevel, create_progress_reporter
)


class FFTMode(Enum):
    """FFT computation modes"""
    FULL = "full"        # Process entire detector area
    CROPPED = "cropped"  # Process bright field cropped region


class OptimizationStrategy(Enum):
    """Adaptive optimization strategies based on memory constraints"""
    SCENARIO_A_MULTITHREADED = "multithreaded"  # Both datasets fit in memory
    SCENARIO_B_SINGLE_THREADED = "single_threaded"  # Only one dataset fits
    SCENARIO_C_LAZY_LOADING = "lazy_loading"  # Ultra-large FFT results


@dataclass
class MemoryAnalysis:
    """Memory analysis for adaptive strategy selection"""
    dataset_4d_size_gb: float
    fft_result_size_gb: float
    available_memory_gb: float
    usable_memory_gb: float  # Available * safety factor
    strategy: OptimizationStrategy
    both_fit_in_memory: bool
    fft_fits_alone: bool

    def __post_init__(self):
        """Calculate derived properties"""
        self.total_required_gb = self.dataset_4d_size_gb + self.fft_result_size_gb


@dataclass
class FFTResult:
    """Result of FFT computation with metadata"""
    output_path: str
    mode: FFTMode
    computation_time: float
    chunk_count: int
    original_shape: Tuple[int, int, int, int]
    fft_shape: Tuple[int, int, int, int]
    crop_info: Optional[Dict[str, Any]] = None
    memory_analysis: Optional[MemoryAnalysis] = None
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.SCENARIO_A_MULTITHREADED
    memory_data: Optional[np.ndarray] = None  # In-memory FFT result for immediate display


class Chunked4DFFTProcessor:
    """
    Memory-efficient 4D FFT processor using detector dimension chunking

    This processor can handle arbitrarily large 4D datasets by:
    1. Chunking detector dimensions while preserving scan dimensions
    2. Computing FFT over scan dimensions for each detector chunk
    3. Writing results directly to EMD file without memory accumulation
    4. Supporting both full and cropped FFT workflows
    """

    def __init__(self,
                 conservative_mode: bool = True,
                 max_workers: Optional[int] = None,
                 progress_callback: Optional[Callable] = None):
        """
        Initialize chunked FFT processor

        Parameters:
        -----------
        conservative_mode : bool
            Use conservative memory and CPU settings
        max_workers : int, optional
            Maximum number of worker threads
        progress_callback : callable, optional
            Progress reporting callback function
        """
        self.conservative_mode = conservative_mode
        self.max_workers = max_workers
        self.progress_callback = progress_callback

        # Adaptive optimization settings
        self.memory_safety_factor = 0.8  # Use 80% of available memory
        self.enable_adaptive_optimization = True

        # Background persistence tracking
        self.active_background_threads = []  # Track active background writes
        self.cancellation_event = threading.Event()  # Global cancellation signal

        # Store progress callback for later use
        # Progress reporter will be created when we have chunking_result

        self.chunk_calculator = AdaptiveChunkCalculator(
            conservative_mode=conservative_mode,
            max_workers=max_workers
        )

    def cancel_background_persistence(self, wait_timeout: float = 5.0):
        """
        Cancel all active background persistence operations and wait for cleanup.

        This should be called when the application is closing to ensure:
        1. Background writes are cancelled
        2. Partial FFT data is deleted from HDF5 files
        3. Files are properly closed

        Parameters:
        -----------
        wait_timeout : float
            Maximum seconds to wait for threads to finish cleanup (default: 5.0)
        """
        if not self.active_background_threads:
            return

        self.log(f"⚠️ Cancelling {len(self.active_background_threads)} background persistence thread(s)...")

        # Signal all threads to cancel
        self.cancellation_event.set()

        # Wait for threads to cleanup (with timeout)
        for thread in self.active_background_threads[:]:  # Copy list to avoid modification during iteration
            if thread.is_alive():
                thread.join(timeout=wait_timeout)
                if thread.is_alive():
                    self.log(f"⚠️ Thread did not finish cleanup within {wait_timeout}s", LogLevel.WARNING)

        self.log("✅ All background threads cancelled and cleaned up")

    def calculate_memory_requirements(self, dataset_shape: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """
        Calculate memory requirements for 4D dataset and FFT result

        Parameters:
        -----------
        dataset_shape : tuple
            Shape of 4D dataset (sy, sx, dy, dx)

        Returns:
        --------
        tuple : (dataset_4d_size_gb, fft_result_size_gb)
        """
        sy, sx, dy, dx = dataset_shape
        total_pixels = sy * sx * dy * dx

        # 4D dataset size (uint16 = 2 bytes per pixel)
        dataset_4d_size_gb = (total_pixels * 2) / (1024**3)

        # FFT result size (complex128 = 16 bytes per pixel)
        fft_result_size_gb = (total_pixels * 16) / (1024**3)

        return dataset_4d_size_gb, fft_result_size_gb

    def get_available_memory_gb(self) -> float:
        """Get available system memory in GB"""
        try:
            available_bytes = psutil.virtual_memory().available
            return available_bytes / (1024**3)
        except Exception:
            # Fallback to conservative estimate if psutil fails
            return 8.0  # 8GB default

    def analyze_memory_strategy(self,
                               dataset_shape: Tuple[int, int, int, int],
                               has_dataset_in_memory: bool = False) -> MemoryAnalysis:
        """
        Analyze memory requirements and select optimal strategy

        Parameters:
        -----------
        dataset_shape : tuple
            Shape of 4D dataset (sy, sx, dy, dx)
        has_dataset_in_memory : bool
            Whether 4D dataset is already loaded in memory

        Returns:
        --------
        MemoryAnalysis : Complete memory analysis with strategy recommendation
        """
        dataset_4d_size_gb, fft_result_size_gb = self.calculate_memory_requirements(dataset_shape)
        available_memory_gb = self.get_available_memory_gb()
        usable_memory_gb = available_memory_gb * self.memory_safety_factor

        # Adjust dataset memory requirement if already in memory
        effective_dataset_size = 0.0 if has_dataset_in_memory else dataset_4d_size_gb

        # Determine strategy based on memory constraints
        both_fit = (effective_dataset_size + fft_result_size_gb) <= usable_memory_gb
        fft_fits_alone = fft_result_size_gb <= usable_memory_gb

        if both_fit:
            strategy = OptimizationStrategy.SCENARIO_A_MULTITHREADED
        elif fft_fits_alone:
            strategy = OptimizationStrategy.SCENARIO_B_SINGLE_THREADED
        else:
            strategy = OptimizationStrategy.SCENARIO_C_LAZY_LOADING

        return MemoryAnalysis(
            dataset_4d_size_gb=dataset_4d_size_gb,
            fft_result_size_gb=fft_result_size_gb,
            available_memory_gb=available_memory_gb,
            usable_memory_gb=usable_memory_gb,
            strategy=strategy,
            both_fit_in_memory=both_fit,
            fft_fits_alone=fft_fits_alone
        )

    def log_memory_analysis(self, analysis: MemoryAnalysis, has_dataset_in_memory: bool = False):
        """Log memory analysis details for user information"""
        self.log(f"=== ADAPTIVE FFT MEMORY ANALYSIS ===")
        self.log(f"4D Dataset size: {analysis.dataset_4d_size_gb:.2f} GB")
        self.log(f"FFT result size: {analysis.fft_result_size_gb:.2f} GB")
        self.log(f"Available memory: {analysis.available_memory_gb:.1f} GB")
        self.log(f"Usable memory (80%): {analysis.usable_memory_gb:.1f} GB")

        if has_dataset_in_memory:
            self.log(f"4D dataset already in memory (virtual detectors)")

        self.log(f"Selected strategy: {analysis.strategy.value.upper()}")

        if analysis.strategy == OptimizationStrategy.SCENARIO_A_MULTITHREADED:
            self.log(f"✅ Both datasets fit in memory - using multi-threaded optimization")
        elif analysis.strategy == OptimizationStrategy.SCENARIO_B_SINGLE_THREADED:
            self.log(f"⚠️  FFT result requires memory management - using I/O-optimized processing")
        else:
            self.log(f"🔶 Ultra-large FFT - using lazy loading with representative display")

        self.log(f"====================================")

    def compute_fft(self,
                   input_emd_path: str,
                   output_emd_path: str,
                   mode: FFTMode = FFTMode.FULL,
                   crop_info: Optional[Dict[str, Any]] = None) -> FFTResult:
        """
        Compute 4D FFT using detector dimension chunking

        Parameters:
        -----------
        input_emd_path : str
            Path to input EMD file containing 4D STEM data
        output_emd_path : str
            Path for output EMD file with FFT results
        mode : FFTMode
            FFT computation mode (FULL or CROPPED)
        crop_info : dict, optional
            Crop information for CROPPED mode (required if mode=CROPPED)

        Returns:
        --------
        FFTResult : Computation result with metadata
        """
        start_time = time.time()

        if self.progress_callback:
            self.progress_callback(f"Starting {mode.value} 4D FFT computation")
            self.progress_callback(f"Input: {input_emd_path}")
            self.progress_callback(f"Output: {output_emd_path}")

        try:
            # Step 1: Get input file information
            file_info = self._get_emd_file_info(input_emd_path)
            if self.progress_callback:
                self.progress_callback(f"Input shape: {file_info['shape']}")

            # Step 2: Determine effective FFT shape based on mode
            if mode == FFTMode.CROPPED:
                if crop_info is None:
                    raise ValueError("crop_info required for CROPPED mode")
                fft_shape = self._calculate_cropped_shape(file_info['shape'], crop_info)
                if self.progress_callback:
                    self.progress_callback(f"Cropped FFT shape: {fft_shape}")
            else:
                fft_shape = file_info['shape']
                if self.progress_callback:
                    self.progress_callback(f"Full FFT shape: {fft_shape}")

            # Step 3: Calculate detector chunking strategy
            if self.progress_callback:
                self.progress_callback("Calculating detector chunking strategy...")

            # Calculate chunking strategy for work dimensions (fft_shape)
            chunking_result = create_adaptive_chunking_strategy(
                fft_shape,  # Use effective FFT shape for optimal chunk sizing
                chunk_detector_dims=True   # Key: Use detector chunking for FFT
            )

            # Create progress reporter now that we have chunking_result
            self.progress_reporter = create_progress_reporter(
                chunking_result,
                progress_callback=self.progress_callback,
                verbose=True
            )

            self.log(f"Chunk strategy: {chunking_result.strategy}")
            self.log(f"Chunk dimensions: {chunking_result.chunk_dims}")
            self.log(f"Total chunks: {chunking_result.total_chunks}")
            self.log(f"Workers: {chunking_result.num_workers}")

            # Step 4: Create output EMD file structure
            self._create_output_emd_structure(output_emd_path, fft_shape, mode, crop_info)

            # Step 5: Generate detector chunk queue (work coordinates)
            work_chunks = self.chunk_calculator.generate_chunk_queue(chunking_result)

            # Step 6: Apply coordinate offset for CROPPED mode
            if mode == FFTMode.CROPPED and crop_info is not None:
                dataset_chunks = self._apply_crop_offset_to_chunks(work_chunks, crop_info['bounds'])
            else:
                dataset_chunks = work_chunks

            # Step 7: Process chunks in parallel
            self._process_chunks_parallel(
                input_emd_path, output_emd_path, work_chunks, dataset_chunks, chunking_result
            )

            computation_time = time.time() - start_time
            self.log(f"FFT computation completed in {computation_time:.1f} seconds")

            return FFTResult(
                output_path=output_emd_path,
                mode=mode,
                computation_time=computation_time,
                chunk_count=chunking_result.total_chunks,
                original_shape=file_info['shape'],
                fft_shape=fft_shape,
                crop_info=crop_info
            )

        except Exception as e:
            if self.progress_callback:
                self.progress_callback(f"FFT computation failed: {str(e)}")
            raise

    def compute_fft_from_memory(self,
                               input_data: np.ndarray,
                               output_emd_path: str,
                               mode: FFTMode = FFTMode.FULL,
                               crop_info: Optional[Dict[str, Any]] = None,
                               has_dataset_in_memory: bool = True) -> FFTResult:
        """
        Compute 4D FFT from numpy array already in memory (ADAPTIVE OPTIMIZATION)

        This method implements the adaptive optimization strategy by processing
        data directly from memory rather than reading from disk, providing
        significant performance improvements.

        Parameters:
        -----------
        input_data : np.ndarray
            4D numpy array with shape (sy, sx, dy, dx)
        output_emd_path : str
            Path where FFT results will be written
        mode : FFTMode
            FFT processing mode (FULL or CROPPED)
        crop_info : dict, optional
            Cropping information for CROPPED mode
        has_dataset_in_memory : bool
            Whether the source dataset is already in memory

        Returns:
        --------
        FFTResult : Computation results with adaptive strategy metadata
        """
        start_time = time.time()

        try:
            self.log("=== ADAPTIVE FFT COMPUTATION FROM MEMORY ===")

            # Verify output file is accessible before doing any computation
            self.log(f"Verifying write access to: {output_emd_path}")
            try:
                # Test file access - open in append mode to verify we can write
                with h5py.File(output_emd_path, 'a') as test_f:
                    self.log(f"  File is writable, existing groups: {list(test_f.keys())}")
            except Exception as e:
                raise IOError(f"Cannot access output file for writing: {output_emd_path}\n"
                             f"Error: {str(e)}\n"
                             f"Make sure the file is not open in another program and you have write permissions.")

            # Step 1: Analyze memory strategy
            memory_analysis = self.analyze_memory_strategy(
                input_data.shape, has_dataset_in_memory
            )
            self.log_memory_analysis(memory_analysis, has_dataset_in_memory)

            # Step 2: Select processing approach based on strategy
            if memory_analysis.strategy == OptimizationStrategy.SCENARIO_A_MULTITHREADED:
                result = self._compute_scenario_a_multithreaded(
                    input_data, output_emd_path, mode, crop_info, memory_analysis
                )
            elif memory_analysis.strategy == OptimizationStrategy.SCENARIO_B_SINGLE_THREADED:
                result = self._compute_scenario_b_single_threaded(
                    input_data, output_emd_path, mode, crop_info, memory_analysis
                )
            else:  # SCENARIO_C_LAZY_LOADING
                result = self._compute_scenario_c_lazy_loading(
                    input_data, output_emd_path, mode, crop_info, memory_analysis
                )

            computation_time = time.time() - start_time
            self.log(f"Adaptive FFT computation completed in {computation_time:.1f} seconds")

            # Update result with memory analysis
            result.computation_time = computation_time
            result.memory_analysis = memory_analysis
            result.optimization_strategy = memory_analysis.strategy

            return result

        except Exception as e:
            if self.progress_callback:
                self.progress_callback(f"Adaptive FFT computation failed: {str(e)}")
            raise

    def _compute_scenario_a_multithreaded(self,
                                         input_data: np.ndarray,
                                         output_emd_path: str,
                                         mode: FFTMode,
                                         crop_info: Optional[Dict[str, Any]],
                                         memory_analysis: MemoryAnalysis) -> FFTResult:
        """
        Scenario A: Both datasets fit in memory - Multi-threaded optimization

        This is the optimal scenario where both the 4D dataset and FFT result
        can fit in memory simultaneously. Uses existing multi-threaded chunking
        but processes directly from memory instead of reading from disk.
        """
        self.log("🚀 SCENARIO A: Multi-threaded memory processing")

        # Determine FFT shape based on mode
        if mode == FFTMode.CROPPED and crop_info is not None:
            bounds = crop_info['bounds']
            sy, sx = input_data.shape[:2]
            dy = bounds[1] - bounds[0]  # y_end - y_start
            dx = bounds[3] - bounds[2]  # x_end - x_start
            fft_shape = (sy, sx, dy, dx)
        else:
            fft_shape = input_data.shape

        # Create output EMD file structure
        self._create_output_emd_structure(output_emd_path, fft_shape, mode, crop_info)

        # Use existing chunking logic but optimized for memory
        # For cropped mode, use cropped shape for chunking calculation
        chunking_shape = fft_shape if mode == FFTMode.CROPPED else input_data.shape
        file_info = {'shape': chunking_shape, 'size_gb': memory_analysis.dataset_4d_size_gb}
        chunking_result = self.chunk_calculator.calculate_chunking_strategy(
            file_shape=chunking_shape,
            chunk_detector_dims=True
        )

        # Generate chunk queue
        work_chunks = self.chunk_calculator.generate_chunk_queue(chunking_result)

        # Apply coordinate offset for CROPPED mode
        if mode == FFTMode.CROPPED and crop_info is not None:
            dataset_chunks = self._apply_crop_offset_to_chunks(work_chunks, crop_info['bounds'])
        else:
            dataset_chunks = work_chunks

        # Phase 1: Multi-threaded computation to in-memory array
        fft_result_memory = self._process_chunks_from_memory(
            input_data, output_emd_path, work_chunks, dataset_chunks, chunking_result, fft_shape, mode
        )

        # Phase 2: Return immediately with in-memory data for instant display
        result = FFTResult(
            output_path=output_emd_path,
            mode=mode,
            computation_time=0.0,  # Will be updated by caller
            chunk_count=chunking_result.total_chunks,
            original_shape=input_data.shape,
            fft_shape=fft_shape,
            crop_info=crop_info,
            memory_data=fft_result_memory  # Immediate in-memory access
        )

        # Phase 3: Start background persistence (non-blocking)
        self._start_background_persistence(
            fft_result_memory, output_emd_path, work_chunks, mode, self.progress_callback
        )

        return result

    def _compute_scenario_b_single_threaded(self,
                                           input_data: np.ndarray,
                                           output_emd_path: str,
                                           mode: FFTMode,
                                           crop_info: Optional[Dict[str, Any]],
                                           memory_analysis: MemoryAnalysis) -> FFTResult:
        """
        Scenario B: Only one dataset fits - Single-threaded I/O optimization

        FFT result doesn't fit in memory alongside source data. Uses single-threaded
        processing with large chunks to minimize I/O operations.
        """
        self.log("⚠️  SCENARIO B: Single-threaded I/O optimization")

        # Determine FFT shape
        if mode == FFTMode.CROPPED and crop_info is not None:
            bounds = crop_info['bounds']
            sy, sx = input_data.shape[:2]
            dy = bounds[1] - bounds[0]
            dx = bounds[3] - bounds[2]
            fft_shape = (sy, sx, dy, dx)
        else:
            fft_shape = input_data.shape

        # Create output EMD structure
        self._create_output_emd_structure(output_emd_path, fft_shape, mode, crop_info)

        # Force single-threaded with large chunks for I/O efficiency
        # For cropped mode, use cropped shape for chunking calculation
        chunking_shape = fft_shape if mode == FFTMode.CROPPED else input_data.shape
        chunking_result = self._calculate_io_optimized_chunking(
            chunking_shape, memory_analysis.usable_memory_gb
        )

        # Generate chunk queue
        work_chunks = self.chunk_calculator.generate_chunk_queue(chunking_result)

        # Apply coordinate offset for CROPPED mode
        if mode == FFTMode.CROPPED and crop_info is not None:
            dataset_chunks = self._apply_crop_offset_to_chunks(work_chunks, crop_info['bounds'])
        else:
            dataset_chunks = work_chunks

        # Process with single-threaded optimization
        self._process_chunks_single_threaded_optimized(
            input_data, output_emd_path, work_chunks, dataset_chunks, chunking_result
        )

        return FFTResult(
            output_path=output_emd_path,
            mode=mode,
            computation_time=0.0,  # Will be updated by caller
            chunk_count=chunking_result.total_chunks,
            original_shape=input_data.shape,
            fft_shape=fft_shape,
            crop_info=crop_info
        )

    def _compute_scenario_c_lazy_loading(self,
                                        input_data: np.ndarray,
                                        output_emd_path: str,
                                        mode: FFTMode,
                                        crop_info: Optional[Dict[str, Any]],
                                        memory_analysis: MemoryAnalysis) -> FFTResult:
        """
        Scenario C: Ultra-large FFT - Lazy loading with representative display

        FFT result too large even without source data. Computes and writes FFT
        to disk using chunked approach, then sets up lazy loading architecture.
        """
        self.log("🔶 SCENARIO C: Lazy loading for ultra-large FFT")

        # For now, fall back to Scenario B processing
        # TODO: Implement full lazy loading architecture
        self.log("Note: Using Scenario B processing with lazy loading preparation")

        result = self._compute_scenario_b_single_threaded(
            input_data, output_emd_path, mode, crop_info, memory_analysis
        )

        # Add lazy loading metadata
        result.optimization_strategy = OptimizationStrategy.SCENARIO_C_LAZY_LOADING

        # TODO: Set up lazy loading file handles and representative data

        return result

    def _get_emd_file_info(self, emd_path: str) -> Dict[str, Any]:
        """Get EMD file information"""
        with h5py.File(emd_path, 'r') as f:
            # Standard EMD structure
            data_path = 'version_1/data/datacubes/datacube_000/data'
            if data_path not in f:
                raise ValueError(f"EMD file missing expected dataset: {data_path}")

            dataset = f[data_path]
            shape = dataset.shape
            dtype = dataset.dtype

            return {
                'shape': shape,
                'dtype': dtype,
                'data_path': data_path
            }

    def _calculate_cropped_shape(self, original_shape: Tuple[int, int, int, int],
                               crop_info: Dict[str, Any]) -> Tuple[int, int, int, int]:
        """Calculate shape after bright field cropping"""
        sy, sx, qy, qx = original_shape
        y1, y2, x1, x2 = crop_info['bounds']

        cropped_qy = y2 - y1
        cropped_qx = x2 - x1

        return (sy, sx, cropped_qy, cropped_qx)

    def _get_fft_dataset_path(self, mode: FFTMode) -> str:
        """Get the correct dataset path for FFT results based on mode"""
        mode_suffix = 'cropped' if mode == FFTMode.CROPPED else 'full'
        return f'cached/fft_4d_{mode_suffix}/complex'

    def detect_existing_fft(self, emd_path: str) -> Dict[str, bool]:
        """
        Detect which FFT modes are already computed in the EMD file

        Returns:
        --------
        dict : {'cropped': bool, 'full': bool}
            Whether cropped and/or full FFT results exist
        """
        result = {'cropped': False, 'full': False}

        try:
            with h5py.File(emd_path, 'r') as f:
                # Check for cropped FFT
                cropped_path = 'cached/fft_4d_cropped/complex'
                if cropped_path in f:
                    result['cropped'] = True

                # Check for full FFT
                full_path = 'cached/fft_4d_full/complex'
                if full_path in f:
                    result['full'] = True

        except Exception as e:
            self.log(f"Could not check for existing FFT: {str(e)}", LogLevel.DEBUG)

        return result

    def auto_load_existing_fft(self, emd_path: str, preferred_mode: FFTMode = None) -> Optional[Tuple[np.ndarray, Dict[str, Any], FFTMode]]:
        """
        Auto-load existing FFT data from EMD file

        Parameters:
        -----------
        emd_path : str
            Path to EMD file
        preferred_mode : FFTMode, optional
            Preferred FFT mode to load. If None, loads first available (cropped preferred)

        Returns:
        --------
        tuple : (fft_data, metadata, actual_mode) or None if no FFT found
        """
        existing_fft = self.detect_existing_fft(emd_path)

        # Determine which mode to load
        if preferred_mode and existing_fft.get(preferred_mode.value.lower(), False):
            # Load preferred mode if available
            load_mode = preferred_mode
        elif existing_fft['cropped']:
            # Default to cropped if available
            load_mode = FFTMode.CROPPED
        elif existing_fft['full']:
            # Fallback to full if only full is available
            load_mode = FFTMode.FULL
        else:
            # No FFT data found
            return None

        try:
            # Load the FFT data
            fft_data, metadata = self.load_fft_results(emd_path, load_mode)
            self.log(f"Auto-loaded existing {load_mode.value} FFT from file")
            return fft_data, metadata, load_mode

        except Exception as e:
            self.log(f"Failed to auto-load FFT data: {str(e)}", LogLevel.ERROR)
            return None

    def _create_output_emd_structure(self, output_path: str,
                                   fft_shape: Tuple[int, int, int, int],
                                   mode: FFTMode,
                                   crop_info: Optional[Dict[str, Any]]):
        """Create FFT dataset structure in existing EMD file"""
        sy, sx, qy, qx = fft_shape

        # Determine dataset path based on mode
        mode_suffix = 'cropped' if mode == FFTMode.CROPPED else 'full'
        fft_group_path = f'cached/fft_4d_{mode_suffix}'

        try:
            with h5py.File(output_path, 'a') as f:  # Append mode - don't overwrite existing file
                # Ensure 'cached' parent group exists (needed for fresh EMD files)
                if 'cached' not in f:
                    self.log(f"Creating 'cached' group for FFT storage")
                    f.create_group('cached')

                # Remove existing FFT group if it exists (for recomputation)
                if fft_group_path in f:
                    self.log(f"Removing existing FFT group: {fft_group_path}")
                    del f[fft_group_path]

                # Create mode-specific FFT group
                fft_group = f.create_group(fft_group_path)

                # Create complex dataset for FFT results
                # Use chunked storage for efficient access patterns
                chunk_size = min(sy, 64), min(sx, 64), min(qy, 64), min(qx, 64)

                fft_group.create_dataset(
                    'complex',
                    shape=(sy, sx, qy, qx),
                    dtype=np.complex64,
                    chunks=chunk_size,
                    compression='gzip',
                    compression_opts=1  # Light compression for speed
                )

                # Create metadata group
                metadata_group = fft_group.create_group('metadata')
                metadata_group.attrs['mode'] = mode.value
                metadata_group.attrs['computation_date'] = time.strftime('%Y-%m-%d %H:%M:%S')
                metadata_group.attrs['fft_axes'] = [0, 1]  # Scan dimensions
                metadata_group.attrs['original_shape'] = fft_shape

                if crop_info is not None:
                    crop_group = metadata_group.create_group('crop_info')
                    for key, value in crop_info.items():
                        if isinstance(value, (list, tuple)):
                            crop_group.attrs[key] = np.array(value)
                        else:
                            crop_group.attrs[key] = value

        except Exception as e:
            raise IOError(f"Failed to create FFT structure in EMD file '{output_path}': {str(e)}\n"
                         f"The file may be open in another program or corrupted.")

    def _apply_crop_offset_to_chunks(self, work_chunks: List[ChunkInfo],
                                   crop_bounds: Tuple[int, int, int, int]) -> List[ChunkInfo]:
        """Apply coordinate offset to transform work chunks → dataset chunks"""
        from copy import deepcopy

        y1, y2, x1, x2 = crop_bounds
        y_offset = y1
        x_offset = x1

        dataset_chunks = []

        for work_chunk in work_chunks:
            # Create a copy for dataset coordinates
            dataset_chunk = deepcopy(work_chunk)

            # Transform detector slices: work coordinates → dataset coordinates
            work_qy_slice = work_chunk.input_slice[2]
            work_qx_slice = work_chunk.input_slice[3]

            dataset_qy_slice = slice(
                work_qy_slice.start + y_offset,
                work_qy_slice.stop + y_offset
            )
            dataset_qx_slice = slice(
                work_qx_slice.start + x_offset,
                work_qx_slice.stop + x_offset
            )

            # Update input slice for reading from dataset
            dataset_chunk.input_slice = (
                work_chunk.input_slice[0],  # Scan Y (unchanged)
                work_chunk.input_slice[1],  # Scan X (unchanged)
                dataset_qy_slice,           # Detector Y (offset applied)
                dataset_qx_slice            # Detector X (offset applied)
            )

            # Output slice stays in work coordinates (unchanged)
            dataset_chunk.output_slice = work_chunk.output_slice

            dataset_chunks.append(dataset_chunk)

        return dataset_chunks

    def _process_chunks_parallel(self,
                               input_path: str,
                               output_path: str,
                               work_chunks: List[ChunkInfo],
                               dataset_chunks: List[ChunkInfo],
                               chunking_result: ChunkingResult):
        """Process detector chunks in parallel"""

        self.log(f"Computing 4D FFT chunks: {len(dataset_chunks)} chunks")

        # PHASE 1 OPTIMIZATION: Use shared file handles like adaptive_converter.py
        input_file = None
        output_file = None
        output_dataset = None

        try:
            # Open files ONCE and share handles with workers
            print(f"🔧 I/O OPTIMIZATION: Opening files once for shared access")
            input_file = h5py.File(input_path, 'r')
            input_dataset = input_file['version_1/data/datacubes/datacube_000/data']

            output_file = h5py.File(output_path, 'r+')
            output_dataset = output_file['cached/fft_4d/complex']

            # Thread-safe write lock
            write_lock = threading.Lock()

            max_workers = chunking_result.num_workers
            print(f"🔧 I/O OPTIMIZATION: Using {max_workers} workers with shared file handles")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all chunks to workers with shared file handles
                futures = {
                    executor.submit(
                        self._process_single_chunk_optimized,
                        dataset_chunks[i], work_chunks[i],
                        input_dataset, output_dataset, write_lock
                    ): dataset_chunks[i].id
                    for i in range(len(dataset_chunks))
                }

                # Collect results as they complete
                completed_chunks = 0
                for future in as_completed(futures):
                    chunk_id = futures[future]

                    try:
                        future.result()  # Raises exception if chunk failed
                        completed_chunks += 1

                        # Manual progress reporting
                        progress_percent = int((completed_chunks / len(dataset_chunks)) * 100)
                        if self.progress_callback:
                            self.progress_callback(f"Processed chunk {chunk_id + 1}/{len(dataset_chunks)} ({progress_percent}%)")

                    except Exception as e:
                        self.log(f"Chunk {chunk_id} failed: {str(e)}", LogLevel.DEBUG)
                        raise

        finally:
            # Clean up file handles
            if output_file:
                output_file.close()
            if input_file:
                input_file.close()
            self.log("FFT computation completed")

    def _process_single_chunk(self,
                            dataset_chunk: ChunkInfo,
                            work_chunk: ChunkInfo,
                            input_path: str,
                            output_path: str):
        """Process a single detector chunk: load → FFT → write"""

        chunk_id = dataset_chunk.id

        try:
            # DEBUGGING: Log chunk start
            print(f"🔹 Chunk {chunk_id}: Starting processing...")

            # Load chunk data from input EMD using dataset coordinates
            print(f"🔹 Chunk {chunk_id}: Opening input file {input_path}")
            with h5py.File(input_path, 'r') as f_in:
                data_path = 'version_1/data/datacubes/datacube_000/data'
                print(f"🔹 Chunk {chunk_id}: Loading slice {dataset_chunk.input_slice}")
                chunk_data = f_in[data_path][dataset_chunk.input_slice]
                print(f"🔹 Chunk {chunk_id}: Loaded data shape {chunk_data.shape}")

                # Shape after loading: (sy, sx, chunk_qy, chunk_qx)
                # Scan dimensions are preserved (full sy, sx)
                # Detector dimensions are chunked from the correct region (dataset coordinates)

            # Compute FFT over scan dimensions (axes 0,1)
            # This transforms (sy, sx) → (ky, kx) while preserving detector chunk
            print(f"🔹 Chunk {chunk_id}: Computing FFT...")
            fft_chunk = np.fft.fftshift(
                np.fft.fftn(chunk_data, axes=(0, 1)),
                axes=(0, 1)
            )
            print(f"🔹 Chunk {chunk_id}: FFT complete, result shape {fft_chunk.shape}")

            # Result shape: (ky, kx, chunk_qy, chunk_qx)

            # Write result to output EMD using work coordinates
            print(f"🔹 Chunk {chunk_id}: Opening output file {output_path}")
            with h5py.File(output_path, 'r+') as f_out:
                print(f"🔹 Chunk {chunk_id}: Writing to slice {work_chunk.output_slice}")
                f_out['cached/fft_4d/complex'][work_chunk.output_slice] = fft_chunk
                print(f"🔹 Chunk {chunk_id}: Write complete")

            print(f"✅ Chunk {chunk_id}: Processing completed successfully")

        except Exception as e:
            print(f"❌ Chunk {chunk_id}: Failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _process_single_chunk_optimized(self,
                                       dataset_chunk: ChunkInfo,
                                       work_chunk: ChunkInfo,
                                       input_dataset: Any,
                                       output_dataset: Any,
                                       write_lock: Any):
        """Process a single detector chunk using shared file handles: load → FFT → write"""

        chunk_id = dataset_chunk.id

        try:
            # DEBUGGING: Log chunk start
            print(f"🔹 Chunk {chunk_id}: Starting processing (optimized I/O)...")

            # Load chunk data from shared input dataset (NO FILE OPEN/CLOSE)
            print(f"🔹 Chunk {chunk_id}: Reading slice {dataset_chunk.input_slice}")
            chunk_data = input_dataset[dataset_chunk.input_slice]
            print(f"🔹 Chunk {chunk_id}: Loaded data shape {chunk_data.shape}")

            # Compute FFT over scan dimensions (axes 0,1)
            # This transforms (sy, sx) → (ky, kx) while preserving detector chunk
            print(f"🔹 Chunk {chunk_id}: Computing FFT...")
            fft_chunk = np.fft.fftshift(
                np.fft.fftn(chunk_data, axes=(0, 1)),
                axes=(0, 1)
            )
            print(f"🔹 Chunk {chunk_id}: FFT complete, result shape {fft_chunk.shape}")

            # Write result using shared output dataset with thread-safe lock
            print(f"🔹 Chunk {chunk_id}: Writing to slice {work_chunk.output_slice}")
            with write_lock:  # Thread-safe write access
                output_dataset[work_chunk.output_slice] = fft_chunk
            print(f"🔹 Chunk {chunk_id}: Write complete")

            print(f"✅ Chunk {chunk_id}: Processing completed successfully (optimized)")

        except Exception as e:
            print(f"❌ Chunk {chunk_id}: Failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _process_chunks_from_memory(self,
                                   input_data: np.ndarray,
                                   output_path: str,
                                   work_chunks: List[ChunkInfo],
                                   dataset_chunks: List[ChunkInfo],
                                   chunking_result: ChunkingResult,
                                   fft_shape: Tuple[int, int, int, int],
                                   mode: FFTMode) -> np.ndarray:
        """
        Process chunks directly from memory using optimized multi-threading
        Phase 1: Multi-threaded computation to in-memory array (no I/O contention)
        """
        self.log(f"🚀 Processing {len(dataset_chunks)} chunks from memory to memory (multi-threaded)")

        # Phase 1: Allocate in-memory result array
        fft_result_memory = np.zeros(fft_shape, dtype=np.complex128)
        self.log(f"📊 Allocated {fft_result_memory.nbytes / (1024**3):.2f} GB for in-memory FFT result")

        # Thread-safe write lock for memory array
        write_lock = threading.Lock()

        max_workers = chunking_result.num_workers
        self.log(f"Using {max_workers} workers for memory-to-memory processing")

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all chunks to workers writing to memory array
                futures = {
                    executor.submit(
                        self._process_single_chunk_memory_to_memory,
                        dataset_chunks[i], work_chunks[i],
                        input_data, fft_result_memory, write_lock
                    ): dataset_chunks[i].id
                    for i in range(len(dataset_chunks))
                }

                # Collect results as they complete
                completed_chunks = 0
                for future in as_completed(futures):
                    chunk_id = futures[future]

                    try:
                        future.result()  # Raises exception if chunk failed
                        completed_chunks += 1

                        # Progress reporting for computation phase
                        progress_percent = int((completed_chunks / len(dataset_chunks)) * 100)
                        if self.progress_callback:
                            self.progress_callback(f"✅ Computed chunk {chunk_id + 1}/{len(dataset_chunks)} ({progress_percent}%)")

                    except Exception as e:
                        self.log(f"Chunk {chunk_id} failed: {str(e)}", LogLevel.DEBUG)
                        raise

        except Exception as e:
            self.log(f"Memory-to-memory processing failed: {str(e)}", LogLevel.ERROR)
            raise

        self.log(f"🎉 Memory-to-memory computation completed successfully")
        return fft_result_memory

    def _start_background_persistence(self,
                                    memory_data: np.ndarray,
                                    output_path: str,
                                    work_chunks: List[ChunkInfo],
                                    mode: FFTMode,
                                    progress_callback: Optional[Callable[[str], None]] = None):
        """
        Phase 3: Start background single-threaded chunked persistence of in-memory data to disk
        Uses the same chunk structure as computation for consistency and progress tracking
        """
        import threading

        def background_write():
            fft_group_path = f'cached/fft_4d_{mode.value}'
            f = None
            cancelled = False

            try:
                self.log(f"💾 Starting chunked background persistence ({len(work_chunks)} chunks)...")

                # Open file with error handling
                try:
                    f = h5py.File(output_path, 'r+')
                except Exception as e:
                    raise IOError(f"Cannot open EMD file for writing: {output_path}\nError: {str(e)}")

                dataset_path = self._get_fft_dataset_path(mode)
                self.log(f"💾 Looking for dataset: {dataset_path}")

                if dataset_path not in f:
                    available_paths = list(f.keys())
                    raise KeyError(f"Dataset path '{dataset_path}' not found in file.\n"
                                 f"Available top-level paths: {available_paths}")

                output_dataset = f[dataset_path]

                # Write one chunk at a time with cancellation checks
                total_chunks = len(work_chunks)

                for i, chunk in enumerate(work_chunks):
                    # Check for cancellation before each chunk
                    if self.cancellation_event.is_set():
                        cancelled = True
                        self.log(f"⚠️ Background write cancelled at chunk {i+1}/{total_chunks}")
                        if progress_callback:
                            progress_callback(f"⚠️ Background: Cancelled, cleaning up...")
                        break

                    # Write this chunk from memory to disk
                    chunk_data = memory_data[chunk.output_slice]
                    output_dataset[chunk.output_slice] = chunk_data

                    # Progress update
                    progress_percent = int(((i + 1) / total_chunks) * 100)
                    if progress_callback:
                        progress_callback(f"💾 Background: Writing chunk {i+1}/{total_chunks} ({progress_percent}%)")

                    self.log(f"💾 Chunk {i+1}/{total_chunks} written to disk ({progress_percent}%)")

                if cancelled:
                    # Clean up: delete partially written FFT data
                    self.log(f"🧹 Cleaning up: Deleting partial FFT group {fft_group_path}")
                    if fft_group_path in f:
                        del f[fft_group_path]
                    self.log(f"✅ Cleanup complete - EMD file restored to previous state")
                    if progress_callback:
                        progress_callback("✅ Background: Cancelled and cleaned up")
                else:
                    if progress_callback:
                        progress_callback("✅ Background: Save completed")
                    self.log("✅ Background persistence to disk completed successfully")

            except Exception as e:
                error_msg = f"❌ Background persistence failed: {str(e)}"
                self.log(error_msg, LogLevel.ERROR)
                if progress_callback:
                    progress_callback(error_msg)

            finally:
                # Always close file properly
                if f is not None:
                    try:
                        f.close()
                        self.log("📁 HDF5 file closed properly")
                    except:
                        pass

                # Remove from active threads list
                try:
                    self.active_background_threads.remove(persistence_thread)
                except:
                    pass

        # Start background thread and track it
        persistence_thread = threading.Thread(target=background_write, daemon=True)
        self.active_background_threads.append(persistence_thread)
        persistence_thread.start()

        return persistence_thread

    def _process_single_chunk_memory_to_memory(self,
                                             dataset_chunk: ChunkInfo,
                                             work_chunk: ChunkInfo,
                                             input_data: np.ndarray,
                                             output_array: np.ndarray,
                                             write_lock: Any):
        """Process a single chunk from memory to memory (no I/O)"""

        chunk_id = dataset_chunk.id

        try:
            # Load chunk data from memory using dataset coordinates (with crop offset)
            chunk_data = input_data[dataset_chunk.input_slice]

            # Compute FFT over scan dimensions (axes 0,1)
            fft_chunk = np.fft.fftshift(
                np.fft.fftn(chunk_data, axes=(0, 1)),
                axes=(0, 1)
            )

            # Write result to memory array with thread-safe lock
            with write_lock:
                output_array[work_chunk.output_slice] = fft_chunk

        except Exception as e:
            print(f"❌ Memory chunk {chunk_id}: Failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _process_single_chunk_from_memory(self,
                                         dataset_chunk: ChunkInfo,
                                         work_chunk: ChunkInfo,
                                         input_data: np.ndarray,
                                         output_dataset: Any,
                                         write_lock: Any):
        """Process a single chunk directly from memory array"""

        chunk_id = dataset_chunk.id

        try:
            print(f"🔹 Chunk {chunk_id}: Processing from memory (optimized)...")

            # For memory processing, load from full dataset using dataset coordinates
            # then apply cropping logic during FFT processing
            chunk_data = input_data[dataset_chunk.input_slice]
            print(f"🔹 Chunk {chunk_id}: Loaded from memory, shape {chunk_data.shape}")

            # Compute FFT over scan dimensions (axes 0,1)
            fft_chunk = np.fft.fftshift(
                np.fft.fftn(chunk_data, axes=(0, 1)),
                axes=(0, 1)
            )
            print(f"🔹 Chunk {chunk_id}: FFT complete, result shape {fft_chunk.shape}")

            # Write result using shared output dataset with thread-safe lock
            with write_lock:
                output_dataset[work_chunk.output_slice] = fft_chunk
            print(f"✅ Chunk {chunk_id}: Memory processing completed")

        except Exception as e:
            print(f"❌ Chunk {chunk_id}: Failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _calculate_io_optimized_chunking(self,
                                        dataset_shape: Tuple[int, int, int, int],
                                        usable_memory_gb: float) -> ChunkingResult:
        """
        Calculate chunking strategy optimized for I/O efficiency (Scenario B)
        Uses largest possible chunks that fit in memory, single-threaded
        """
        self.log("Calculating I/O-optimized chunking strategy (large chunks, single-threaded)")

        # Force single-threaded with maximum chunk size
        # Override the adaptive calculator to prioritize I/O efficiency
        calculator = AdaptiveChunkCalculator(
            conservative_mode=True,
            max_workers=1  # Force single-threaded
        )

        # Calculate with focus on minimizing chunk count
        result = calculator.calculate_chunking_strategy(
            file_shape=dataset_shape,
            chunk_detector_dims=True
        )

        self.log(f"I/O-optimized strategy: {result.total_chunks} chunks, {result.chunk_size_mb:.1f} MB each")
        return result

    def _process_chunks_single_threaded_optimized(self,
                                                 input_data: np.ndarray,
                                                 output_path: str,
                                                 work_chunks: List[ChunkInfo],
                                                 dataset_chunks: List[ChunkInfo],
                                                 chunking_result: ChunkingResult):
        """
        Process chunks single-threaded with large chunks for I/O efficiency
        (Scenario B implementation)
        """
        self.log(f"Processing {len(dataset_chunks)} chunks single-threaded (I/O optimized)")

        output_file = None
        output_dataset = None

        try:
            # Open output file ONCE
            output_file = h5py.File(output_path, 'r+')
            output_dataset = output_file['cached/fft_4d/complex']

            # Process chunks sequentially (no threading, no lock needed)
            for i, (dataset_chunk, work_chunk) in enumerate(zip(dataset_chunks, work_chunks)):
                chunk_id = dataset_chunk.id

                try:
                    # Process chunk directly from memory
                    chunk_data = input_data[dataset_chunk.input_slice]

                    # Compute FFT
                    fft_chunk = np.fft.fftshift(
                        np.fft.fftn(chunk_data, axes=(0, 1)),
                        axes=(0, 1)
                    )

                    # Write directly (no lock needed for single-threaded)
                    output_dataset[work_chunk.output_slice] = fft_chunk

                    # Progress reporting
                    progress_percent = int(((i + 1) / len(dataset_chunks)) * 100)
                    if self.progress_callback:
                        self.progress_callback(f"Processed chunk {i + 1}/{len(dataset_chunks)} ({progress_percent}%)")

                    print(f"✅ Chunk {chunk_id}: Single-threaded processing completed")

                except Exception as e:
                    print(f"❌ Chunk {chunk_id}: Failed with error: {e}")
                    raise

        finally:
            if output_file:
                output_file.close()


    def load_fft_results(self, fft_emd_path: str, mode: FFTMode) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load cached FFT results from EMD file

        Parameters:
        -----------
        fft_emd_path : str
            Path to EMD file with cached FFT results
        mode : FFTMode
            FFT mode (cropped or full) to determine correct dataset path

        Returns:
        --------
        tuple : (fft_complex_data, metadata)
        """
        with h5py.File(fft_emd_path, 'r') as f:
            # Get correct dataset path
            dataset_path = self._get_fft_dataset_path(mode)
            mode_suffix = 'cropped' if mode == FFTMode.CROPPED else 'full'
            metadata_path = f'cached/fft_4d_{mode_suffix}/metadata'

            # Load complex FFT data
            fft_data = f[dataset_path][:]

            # Load metadata
            metadata = {}
            meta_group = f[metadata_path]
            for key in meta_group.attrs:
                metadata[key] = meta_group.attrs[key]

            # Load crop info if available
            if 'crop_info' in meta_group:
                crop_info = {}
                crop_group = meta_group['crop_info']
                for key in crop_group.attrs:
                    crop_info[key] = crop_group.attrs[key]
                metadata['crop_info'] = crop_info

            return fft_data, metadata

    def compute_amplitude_phase(self, fft_complex: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute amplitude and phase from complex FFT data

        Parameters:
        -----------
        fft_complex : ndarray
            Complex FFT data

        Returns:
        --------
        tuple : (amplitude, phase)
        """
        amplitude = np.abs(fft_complex)
        phase = np.angle(fft_complex)
        return amplitude, phase

    def log(self, message: str, level: LogLevel = LogLevel.BASIC):
        """Log message through progress reporter or progress callback"""
        if hasattr(self, 'progress_reporter') and self.progress_reporter:
            # Use the internal _log method of ProgressReporter
            self.progress_reporter._log(message)
        elif self.progress_callback:
            self.progress_callback(message)


def create_chunked_fft_processor(conservative: bool = True,
                               max_workers: Optional[int] = None,
                               progress_callback: Optional[Callable] = None) -> Chunked4DFFTProcessor:
    """
    Convenience function to create FFT processor with default settings

    Parameters:
    -----------
    conservative : bool
        Use conservative settings for memory and CPU usage
    max_workers : int, optional
        Maximum number of worker threads
    progress_callback : callable, optional
        Progress reporting callback

    Returns:
    --------
    Chunked4DFFTProcessor : Configured FFT processor
    """
    return Chunked4DFFTProcessor(
        conservative_mode=conservative,
        max_workers=max_workers,
        progress_callback=progress_callback
    )


if __name__ == "__main__":
    # Example usage
    print("=== Chunked 4D FFT Processor Test ===")

    # This would typically be run with actual EMD files
    # processor = create_chunked_fft_processor(progress_callback=print)
    #
    # # Full FFT
    # result = processor.compute_fft(
    #     "input.emd",
    #     "output_full_fft.emd",
    #     mode=FFTMode.FULL
    # )
    #
    # # Cropped FFT
    # crop_info = {'bounds': [64, 192, 64, 192]}  # Example crop bounds
    # result = processor.compute_fft(
    #     "input.emd",
    #     "output_cropped_fft.emd",
    #     mode=FFTMode.CROPPED,
    #     crop_info=crop_info
    # )

    print("FFT processor module loaded successfully")