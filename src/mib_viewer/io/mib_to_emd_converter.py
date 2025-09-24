#!/usr/bin/env python3
"""
MIB to EMD Converter

Converts Quantum Detectors MIB files to EMD 1.0 format (HDF5) with optimal
compression for 4D STEM datasets. Compatible with py4DSTEM and STEMTooL.

Usage:
    python mib_to_emd_converter.py input.mib output.h5 [options]
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Callable
import warnings

import numpy as np
import h5py
import emdfile
from tqdm import tqdm
import psutil

# Import our MIB and EMD loading functions
try:
    # Try relative import (when run as module)
    from .mib_loader import load_mib, load_emd, get_mib_properties, auto_detect_scan_size, MibProperties, apply_data_processing
except ImportError:
    # Fall back for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from io.mib_loader import load_mib, load_emd, get_mib_properties, auto_detect_scan_size, MibProperties, apply_data_processing

class MibToEmdConverter:
    """
    Converter for MIB and EMD files to processed EMD 1.0 format with py4DSTEM compatibility
    Supports both MIB → EMD conversion and EMD → EMD processing (binning, Y-summing)
    """
    
    def __init__(self, compression='gzip', compression_level=6, chunk_size=None, log_callback=None, progress_callback=None):
        """
        Initialize converter with compression settings
        
        Parameters:
        -----------
        compression : str
            Compression algorithm ('gzip', 'szip', 'lzf', or None)
        compression_level : int
            Compression level (1-9 for gzip, ignored for others)
        chunk_size : tuple or None
            HDF5 chunk size (sy, sx, qy, qx). If None, auto-determined.
        log_callback : callable or None
            Function to call for logging messages. Should accept (message, level) parameters.
        progress_callback : callable or None
            Function to call for progress updates. Should accept (progress_percent, status_message) parameters.
        """
        self.compression = compression
        self.compression_level = compression_level if compression == 'gzip' else None
        self.chunk_size = chunk_size
        self.log_callback = log_callback
        self.progress_callback = progress_callback
    
    def log(self, message, level="INFO"):
        """Log message to both terminal and GUI (if callback provided)"""
        # Always print to terminal for backward compatibility
        print(message)
        
        # Also send to GUI if callback is provided
        if self.log_callback:
            try:
                self.log_callback(message, level)
            except Exception:
                # Don't let GUI logging errors break the conversion
                pass
    
    def update_progress(self, progress_percent, status_message=""):
        """Update progress via callback if provided"""
        if self.progress_callback:
            try:
                self.progress_callback(progress_percent, status_message)
            except Exception:
                # Don't let progress callback errors break the conversion
                pass
    
    def should_use_chunked_mode(self, file_path: str, data_shape: Optional[Tuple] = None, safety_factor: float = 0.5) -> bool:
        """
        Determine if file requires chunked processing based on memory constraints
        
        Parameters:
        -----------
        file_path : str
            Path to input file
        data_shape : tuple, optional
            Shape of data if known (to estimate memory usage)
        safety_factor : float
            Fraction of available memory to use as threshold (default: 0.5 = 50%)
            
        Returns:
        --------
        bool : True if chunked processing should be used
        """
        try:
            # Get available system memory
            available_memory = psutil.virtual_memory().available
            
            # Estimate memory needed for data
            if data_shape is not None:
                # Calculate memory needed for processing (data + processed data + overhead)
                bytes_per_element = 2  # uint16
                data_size = np.prod(data_shape) * bytes_per_element
                # Processing overhead: original + processed + compression buffers
                estimated_memory_needed = data_size * 3
            else:
                # Fall back to file size estimation
                file_size = os.path.getsize(file_path)
                # For compressed files, assume 3x expansion + processing overhead
                estimated_memory_needed = file_size * 6
            
            use_chunked = estimated_memory_needed > (available_memory * safety_factor)
            
            if use_chunked:
                self.log(f"Memory check: Need ~{estimated_memory_needed / (1024**3):.1f} GB, "
                        f"Available: {available_memory / (1024**3):.1f} GB")
                self.log("Using chunked processing mode for memory safety")
            else:
                self.log(f"Memory check: Need ~{estimated_memory_needed / (1024**3):.1f} GB, "
                        f"Available: {available_memory / (1024**3):.1f} GB")
                self.log("Using standard in-memory processing")
            
            return use_chunked
            
        except Exception as e:
            self.log(f"Memory detection failed: {e}, defaulting to chunked mode", "WARNING")
            return True  # Default to safe chunked mode
    
    def detect_file_type(self, file_path: str) -> str:
        """
        Detect whether input file is MIB or EMD format
        
        Parameters:
        -----------
        file_path : str
            Path to input file
            
        Returns:
        --------
        str : 'mib' or 'emd'
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.mib':
            return 'mib'
        elif file_ext in ['.emd', '.h5', '.hdf5']:
            # Check if it's a valid EMD file by trying to access structure
            try:
                with h5py.File(file_path, 'r') as f:
                    if 'version_1' in f and 'data' in f['version_1']:
                        return 'emd'
            except:
                pass
            # Fallback - assume EMD based on extension
            return 'emd'
        else:
            # Default to MIB if unknown extension
            return 'mib'
    
    def analyze_emd_file(self, emd_path: str) -> Dict[str, Any]:
        """
        Analyze EMD file to extract metadata and dimensions
        
        Parameters:
        -----------
        emd_path : str
            Path to input EMD file
            
        Returns:
        --------
        dict : Metadata dictionary with file properties
        """
        self.log(f"Analyzing EMD file: {os.path.basename(emd_path)}")
        
        with h5py.File(emd_path, 'r') as f:
            # Get file size
            filesize = os.path.getsize(emd_path)
            
            # Navigate to data
            datacube = f['version_1/data/datacubes/datacube_000']
            data_shape = datacube['data'].shape
            data_dtype = datacube['data'].dtype
            
            # Extract existing metadata if available
            metadata = {
                'original_filename': os.path.basename(emd_path),
                'filesize_gb': filesize / (1024**3),
                'shape_4d': data_shape,
                'pixel_dtype': str(data_dtype),
                'num_frames': data_shape[0] * data_shape[1] if len(data_shape) >= 2 else 0,
                'scan_size': (data_shape[0], data_shape[1]) if len(data_shape) >= 2 else (1, 1),
                'detector_size': (data_shape[2], data_shape[3]) if len(data_shape) >= 4 else (1, 1),
            }
            
            # Try to extract original metadata from microscope group
            try:
                microscope_group = f['version_1/metadata/microscope']
                for key, value in microscope_group.attrs.items():
                    # Convert bytes to string if needed
                    if isinstance(value, bytes):
                        value = value.decode('utf-8')
                    elif isinstance(value, np.ndarray):
                        value = value.tolist()
                    metadata[key] = value
            except KeyError:
                # No microscope metadata available
                pass
        
        self.log(f"  Shape: {metadata['shape_4d']}")
        self.log(f"  Size: {metadata['filesize_gb']:.2f} GB")
        self.log(f"  Frames: {metadata['num_frames']:,}")
        self.log(f"  Detector: {metadata['detector_size']}")
        
        return metadata
        
    def analyze_mib_file(self, mib_path: str) -> Dict[str, Any]:
        """
        Analyze MIB file to extract metadata and dimensions
        
        Returns:
        --------
        dict : Metadata dictionary with file properties
        """
        self.log(f"Analyzing MIB file: {os.path.basename(mib_path)}")
        
        # Read header
        with open(mib_path, 'rb') as f:
            header_bytes = f.read(384)
            f.seek(0, os.SEEK_END)
            filesize = f.tell()
        
        header_fields = header_bytes.decode().split(',')
        props = get_mib_properties(header_fields)
        
        # Calculate dimensions
        merlin_frame_dtype = np.dtype([
            ('header', np.bytes_, props.headsize),
            ('data', props.pixeltype, props.merlin_size)
        ])
        num_frames = filesize // merlin_frame_dtype.itemsize
        scan_size = auto_detect_scan_size(num_frames)
        
        # Extract metadata from header
        metadata = {
            'original_filename': os.path.basename(mib_path),
            'filesize_gb': filesize / (1024**3),
            'detector_size': props.merlin_size,
            'scan_size': scan_size,
            'num_frames': num_frames,
            'pixel_dtype': str(props.pixeltype),
            'dynamic_range': props.dyn_range,
            'header_size': props.headsize,
            'detector_mode': 'Single' if props.single else 'Quad',
            'raw_format': props.raw,
            'shape_4d': (scan_size[1], scan_size[0], props.merlin_size[1], props.merlin_size[0])
        }
        
        # Parse additional metadata from header if available
        if len(header_fields) > 9:
            try:
                metadata['acquisition_time'] = header_fields[9] if header_fields[9] else 'Unknown'
                metadata['dwell_time_s'] = float(header_fields[10]) if header_fields[10] else None
                metadata['acceleration_voltage_kv'] = float(header_fields[14]) if len(header_fields) > 14 and header_fields[14] else None
            except (ValueError, IndexError):
                pass
        
        self.log(f"  Shape: {metadata['shape_4d']}")
        self.log(f"  Size: {metadata['filesize_gb']:.2f} GB")
        self.log(f"  Frames: {metadata['num_frames']:,}")
        self.log(f"  Detector: {metadata['detector_size']}")
        
        return metadata
    
    def calculate_optimal_chunk_size(self, file_shape: Tuple[int, int, int, int], available_memory: int, processing_factor: int = 3) -> Tuple[int, int, int, int]:
        """
        Calculate optimal chunk size using factor-based approach for even load balancing

        This algorithm:
        1. Preserves full detector dimensions (no chunking in detector space)
        2. Uses only factors that divide evenly into scan dimensions
        3. Targets optimal chunks per worker for good parallelization
        4. Validates memory constraints per chunk

        Parameters:
        -----------
        file_shape : tuple
            4D data shape (sy, sx, qy, qx)
        available_memory : int
            Available memory in bytes
        processing_factor : int
            Memory multiplication factor for processing overhead (default: 3)

        Returns:
        --------
        tuple : Optimal chunk size (chunk_sy, chunk_sx, qy, qx)
        """
        import psutil

        sy, sx, qy, qx = file_shape
        bytes_per_element = 2  # uint16

        # Step 1: Calculate target frames per worker
        total_frames = sy * sx
        available_workers = max(1, psutil.cpu_count() - 2)  # Leave 2 cores for system
        target_frames_per_worker = total_frames // available_workers

        self.log(f"Total frames: {total_frames:,}, Workers: {available_workers}, Target per worker: {target_frames_per_worker:,}")

        # Step 2: Find all factors that divide evenly into scan dimensions
        def get_factors(n):
            return [i for i in range(1, n + 1) if n % i == 0]

        sy_factors = get_factors(sy)
        sx_factors = get_factors(sx)

        # Step 3: Find best factor combination that is ≤ target frames per worker (memory safety priority)
        best_chunk = None
        best_frames_per_chunk = 0  # Track largest chunk that fits under target
        best_total_chunks = 0

        for chunk_sy in sy_factors:
            for chunk_sx in sx_factors:
                frames_per_chunk = chunk_sy * chunk_sx

                # Calculate total number of chunks this would create
                total_chunks = (sy // chunk_sy) * (sx // chunk_sx)

                # Memory validation: ensure chunk fits in per-worker budget
                bytes_per_frame = qy * qx * bytes_per_element
                chunk_memory = frames_per_chunk * bytes_per_frame * processing_factor
                per_worker_memory = available_memory // available_workers

                if chunk_memory > per_worker_memory:
                    continue  # Skip chunks that are too large

                # MEMORY SAFETY: Only consider chunks ≤ target (never exceed target)
                if frames_per_chunk <= target_frames_per_worker:
                    # Among valid chunks, choose the largest one (closest to target from below)
                    if frames_per_chunk > best_frames_per_chunk:
                        best_frames_per_chunk = frames_per_chunk
                        best_chunk = (chunk_sy, chunk_sx, qy, qx)
                        best_total_chunks = total_chunks

        # Step 4: Fallback to frame-based chunking if no valid factors found
        if best_chunk is None:
            self.log("Warning: No valid factor-based chunks found, falling back to frame-based chunking")
            best_chunk = (1, 1, qy, qx)
            best_total_chunks = total_frames

        # Step 5: Log results
        chunk_sy, chunk_sx, _, _ = best_chunk
        frames_per_chunk = chunk_sy * chunk_sx
        chunk_memory = frames_per_chunk * qy * qx * bytes_per_element * processing_factor

        self.log(f"Memory-safe chunk size: {best_chunk}")
        self.log(f"Frames per chunk: {frames_per_chunk:,} (target was {target_frames_per_worker:,})")
        self.log(f"Total chunks: {best_total_chunks}")
        self.log(f"Estimated chunk memory: {chunk_memory / (1024**2):.1f} MB")
        self.log(f"Load distribution: {best_total_chunks} chunks across {available_workers} workers")

        # Log safety information
        if frames_per_chunk < target_frames_per_worker:
            safety_margin = target_frames_per_worker - frames_per_chunk
            self.log(f"Memory safety: Using {safety_margin:,} fewer frames per chunk for safety")

        return best_chunk
    
    def chunked_mib_reader(self, mib_path: str, scan_size: Tuple[int, int], chunk_size: Tuple[int, int, int, int]):
        """
        Generator for reading MIB file in chunks
        
        Parameters:
        -----------
        mib_path : str
            Path to MIB file
        scan_size : tuple
            Scan dimensions (sy, sx)
        chunk_size : tuple
            Chunk dimensions (chunk_sy, chunk_sx, qy, qx)
            
        Yields:
        -------
        tuple : (chunk_slice, chunk_data) where chunk_slice is the location in the full array
        """
        sy, sx = scan_size
        chunk_sy, chunk_sx, qy, qx = chunk_size
        
        # Load MIB properties once
        with open(mib_path, 'rb') as f:
            header_bytes = f.read(384)
        header_fields = header_bytes.decode().split(',')
        props = get_mib_properties(header_fields)
        
        # Calculate merlin frame structure
        merlin_frame_dtype = np.dtype([
            ('header', np.bytes_, props.headsize),
            ('data', props.pixeltype, props.merlin_size)
        ])
        
        # Iterate through chunks
        for start_y in range(0, sy, chunk_sy):
            for start_x in range(0, sx, chunk_sx):
                # Calculate actual chunk size (handle edges)
                actual_chunk_sy = min(chunk_sy, sy - start_y)
                actual_chunk_sx = min(chunk_sx, sx - start_x)
                
                # Create chunk slice info
                chunk_slice = (
                    slice(start_y, start_y + actual_chunk_sy),
                    slice(start_x, start_x + actual_chunk_sx),
                    slice(None),
                    slice(None)
                )
                
                # Load chunk data
                chunk_data = np.zeros((actual_chunk_sy, actual_chunk_sx, qy, qx), dtype=props.pixeltype)
                
                # Read frame by frame for this chunk
                with open(mib_path, 'rb') as f:
                    for chunk_y in range(actual_chunk_sy):
                        for chunk_x in range(actual_chunk_sx):
                            # Calculate global frame index
                            global_y = start_y + chunk_y
                            global_x = start_x + chunk_x
                            frame_idx = global_y * sx + global_x
                            
                            # Seek to frame position
                            frame_offset = frame_idx * merlin_frame_dtype.itemsize
                            f.seek(frame_offset)
                            
                            # Read and parse frame
                            frame_bytes = f.read(merlin_frame_dtype.itemsize)
                            if len(frame_bytes) == merlin_frame_dtype.itemsize:
                                frame = np.frombuffer(frame_bytes, dtype=merlin_frame_dtype)[0]
                                frame_data = np.array(frame['data']).reshape(props.merlin_size)
                                chunk_data[chunk_y, chunk_x] = frame_data
                
                yield chunk_slice, chunk_data
    
    def chunked_emd_reader(self, emd_path: str, chunk_size: Tuple[int, int, int, int]):
        """
        Generator for reading EMD file in chunks
        
        Parameters:
        -----------
        emd_path : str
            Path to EMD file
        chunk_size : tuple
            Chunk dimensions (chunk_sy, chunk_sx, qy, qx)
            
        Yields:
        -------
        tuple : (chunk_slice, chunk_data) where chunk_slice is the location in the full array
        """
        chunk_sy, chunk_sx, qy, qx = chunk_size
        
        with h5py.File(emd_path, 'r') as f:
            dataset = f['version_1/data/datacubes/datacube_000/data']
            sy, sx, data_qy, data_qx = dataset.shape
            
            # Iterate through chunks
            for start_y in range(0, sy, chunk_sy):
                for start_x in range(0, sx, chunk_sx):
                    # Calculate actual chunk size (handle edges)
                    actual_chunk_sy = min(chunk_sy, sy - start_y)
                    actual_chunk_sx = min(chunk_sx, sx - start_x)
                    
                    # Create chunk slice info
                    chunk_slice = (
                        slice(start_y, start_y + actual_chunk_sy),
                        slice(start_x, start_x + actual_chunk_sx),
                        slice(None),
                        slice(None)
                    )
                    
                    # Read chunk from HDF5 (memory-mapped, efficient)
                    chunk_data = dataset[chunk_slice]
                    
                    yield chunk_slice, chunk_data
    
    def determine_optimal_chunks(self, shape_4d: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Determine optimal chunk size based on data shape and threading strategy

        Uses factor-based chunking for optimal worker load balancing and parallelization.
        Preserves full detector dimensions and distributes work evenly across available workers.

        Parameters:
        -----------
        shape_4d : tuple
            4D data shape (sy, sx, qy, qx)

        Returns:
        --------
        tuple : Optimal chunk size
        """
        if self.chunk_size is not None:
            return self.chunk_size

        # Use the improved factor-based algorithm
        import psutil
        available_memory = psutil.virtual_memory().available
        return self.calculate_optimal_chunk_size(shape_4d, available_memory)
    
    def convert_to_emd(self, input_path: str, output_path: str, metadata_extra: Optional[Dict] = None, processing_options: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Convert MIB file to EMD 1.0 format or process existing EMD file
        
        Parameters:
        -----------
        input_path : str
            Path to input MIB or EMD file
        output_path : str
            Path to output EMD/HDF5 file
        metadata_extra : dict, optional
            Additional metadata to include
        processing_options : dict, optional
            Data processing options (binning, Y-summing)
            
        Returns:
        --------
        dict : Conversion statistics
        """
        # Detect input file type
        file_type = self.detect_file_type(input_path)
        self.log(f"\nDetected file type: {file_type.upper()}")
        self.log(f"Starting processing: {os.path.basename(input_path)} → {os.path.basename(output_path)}")
        self.update_progress(5, f"Analyzing {file_type.upper()} file...")
        
        # Analyze input file based on type
        if file_type == 'mib':
            metadata = self.analyze_mib_file(input_path)
        else:  # emd
            metadata = self.analyze_emd_file(input_path)
            
        if metadata_extra:
            metadata.update(metadata_extra)
        
        # Check if chunked processing is needed
        use_chunked = self.should_use_chunked_mode(input_path, metadata['shape_4d'])
        
        if use_chunked:
            return self._convert_chunked(input_path, output_path, file_type, metadata, processing_options)
        else:
            return self._convert_in_memory(input_path, output_path, file_type, metadata, processing_options)
    
    def _convert_in_memory(self, input_path: str, output_path: str, file_type: str, metadata: Dict, processing_options: Optional[Dict] = None) -> Dict[str, Any]:
        """Standard in-memory conversion for smaller files"""
        # Load data based on file type
        self.log(f"\nLoading {file_type.upper()} data...")
        self.update_progress(10, f"Loading {file_type.upper()} data...")
        start_time = time.time()
        
        if file_type == 'mib':
            data_4d = load_mib(input_path)
        else:  # emd
            data_4d = load_emd(input_path)
        
        load_time = time.time() - start_time
        self.log(f"  Loaded in {load_time:.1f}s")
        
        return self._process_and_write_data(data_4d, input_path, output_path, metadata, processing_options, load_time, is_chunked=False)
    
    def _convert_chunked(self, input_path: str, output_path: str, file_type: str, metadata: Dict, processing_options: Optional[Dict] = None) -> Dict[str, Any]:
        """Chunked conversion for large files that don't fit in memory"""
        self.log(f"\nUsing chunked processing for large {file_type.upper()} file...")
        self.update_progress(10, f"Preparing chunked processing...")
        
        # Calculate optimal chunk size based on available memory
        available_memory = psutil.virtual_memory().available
        chunk_size = self.calculate_optimal_chunk_size(metadata['shape_4d'], available_memory)
        
        sy, sx, qy, qx = metadata['shape_4d']
        total_chunks = ((sy + chunk_size[0] - 1) // chunk_size[0]) * ((sx + chunk_size[1] - 1) // chunk_size[1])
        
        self.log(f"Processing {total_chunks} chunks of size {chunk_size}")
        self.update_progress(15, f"Processing {total_chunks} chunks...")
        
        # Apply processing to shape metadata if needed
        if processing_options:
            original_shape = metadata['shape_4d']
            # Create a dummy small array to test processing and get final shape
            test_data = np.zeros((1, 1, qy, qx), dtype=np.uint16)
            processed_test = apply_data_processing(test_data, processing_options)
            final_qy, final_qx = processed_test.shape[2], processed_test.shape[3]
            metadata['shape_4d'] = (sy, sx, final_qy, final_qx)
            metadata['original_shape_4d'] = original_shape
            if 'processing_applied' not in metadata:
                metadata['processing_applied'] = processing_options.copy()
            self.log(f"Processing will change shape: {original_shape} → {metadata['shape_4d']}")
        
        # Create output EMD file structure first
        hdf5_chunks = self.determine_optimal_chunks(metadata['shape_4d'])
        
        compression_kwargs = {}
        if self.compression:
            compression_kwargs['compression'] = self.compression
            if self.compression_level is not None:
                compression_kwargs['compression_opts'] = self.compression_level
        
        self.update_progress(20, "Creating output file structure...")
        
        start_time = time.time()
        
        with h5py.File(output_path, 'w') as f:
            # Create EMD 1.0 structure (same as before)
            f.attrs['emd_group_type'] = 'file'
            f.attrs['version_major'] = 1
            f.attrs['version_minor'] = 0
            f.attrs['authoring_program'] = 'mib-to-emd-converter'
            
            version_group = f.create_group('version_1')
            version_group.attrs['emd_group_type'] = 'root'
            
            data_group = version_group.create_group('data')
            datacubes_group = data_group.create_group('datacubes')
            datacube_group = datacubes_group.create_group('datacube_000')
            datacube_group.attrs['emd_group_type'] = 'array'
            
            # Determine output dtype based on processing
            if processing_options and processing_options.get('bin_method', 'mean') == 'mean':
                # Mean binning produces float values - use appropriate dtype
                output_dtype = np.float32  # More memory efficient than float64
                self.log(f"  Using float32 dtype for mean binning")
            else:
                # No processing or sum binning - use original uint16
                output_dtype = np.uint16
            
            # Create empty dataset for chunked writing
            self.log(f"  Creating chunked dataset: {metadata['shape_4d']} ({output_dtype})")
            dataset = datacube_group.create_dataset(
                'data', 
                shape=metadata['shape_4d'],
                dtype=output_dtype,
                chunks=hdf5_chunks,
                **compression_kwargs
            )
            dataset.attrs['units'] = 'counts'
            
            # Process and write chunks
            chunk_count = 0
            if file_type == 'mib':
                chunk_reader = self.chunked_mib_reader(input_path, (sy, sx), chunk_size)
            else:  # emd
                chunk_reader = self.chunked_emd_reader(input_path, chunk_size)
            
            for chunk_slice, chunk_data in chunk_reader:
                # Apply processing to chunk if specified
                if processing_options:
                    chunk_data = apply_data_processing(chunk_data, processing_options)
                
                # Write chunk to output dataset
                dataset[chunk_slice] = chunk_data
                
                # Update progress
                chunk_count += 1
                progress = 20 + int((chunk_count / total_chunks) * 70)  # 20% to 90%
                self.update_progress(progress, f"Processing chunk {chunk_count}/{total_chunks}")
                
                # Force garbage collection to manage memory
                del chunk_data
                
            # Add remaining EMD structure (dimensions, metadata, etc.)
            self._add_emd_metadata(datacube_group, version_group, metadata)
        
        write_time = time.time() - start_time
        load_time = 0  # No bulk loading time for chunked processing
        
        self.update_progress(100, "Chunked conversion completed!")
        
        return self._calculate_conversion_stats(input_path, output_path, load_time, write_time)
    
    def _add_emd_metadata(self, datacube_group, version_group, metadata: Dict):
        """Add EMD metadata structure to the file"""
        # Create dimension datasets
        sy, sx, qy, qx = metadata['shape_4d']
        
        # Real space dimensions (scan coordinates)
        datacube_group.create_dataset('dim1', data=np.arange(sy))
        datacube_group.create_dataset('dim2', data=np.arange(sx))
        datacube_group['dim1'].attrs['name'] = 'scan_y'
        datacube_group['dim1'].attrs['units'] = 'pixel'
        datacube_group['dim2'].attrs['name'] = 'scan_x'  
        datacube_group['dim2'].attrs['units'] = 'pixel'
        
        # Reciprocal space dimensions (detector coordinates)
        datacube_group.create_dataset('dim3', data=np.arange(qy))
        datacube_group.create_dataset('dim4', data=np.arange(qx))
        datacube_group['dim3'].attrs['name'] = 'detector_y'
        datacube_group['dim3'].attrs['units'] = 'pixel'
        datacube_group['dim4'].attrs['name'] = 'detector_x'
        datacube_group['dim4'].attrs['units'] = 'pixel'
        
        # Add dimension convention documentation
        datacube_group.attrs['dimension_order'] = 'scan_y, scan_x, detector_y, detector_x'
        datacube_group.attrs['dimension_convention'] = 'MIB Viewer format: EELS energy in detector_x dimension'
        
        # Metadata group
        metadata_group = version_group.create_group('metadata')
        microscope_group = metadata_group.create_group('microscope')
        
        # Store acquisition metadata
        for key, value in metadata.items():
            if value is not None:
                try:
                    if isinstance(value, (str, int, float)):
                        microscope_group.attrs[key] = value
                    elif isinstance(value, (list, tuple)):
                        microscope_group.attrs[key] = list(value)
                except Exception:
                    # Skip metadata that can't be stored as HDF5 attributes
                    pass
        
        # Log group
        log_group = version_group.create_group('log')
        log_group.attrs['conversion_date'] = time.strftime('%Y-%m-%d %H:%M:%S')
        log_group.attrs['converter_version'] = '1.0'
        log_group.attrs['compression_algorithm'] = self.compression or 'none'
        if self.compression_level:
            log_group.attrs['compression_level'] = self.compression_level
    
    def _calculate_conversion_stats(self, input_path: str, output_path: str, load_time: float, write_time: float) -> Dict[str, Any]:
        """Calculate conversion statistics"""
        input_size = os.path.getsize(input_path)
        output_size = os.path.getsize(output_path)
        compression_ratio = input_size / output_size
        
        stats = {
            'input_size_gb': input_size / (1024**3),
            'output_size_gb': output_size / (1024**3),
            'compression_ratio': compression_ratio,
            'load_time_s': load_time,
            'write_time_s': write_time,
            'total_time_s': load_time + write_time
        }
        
        self.log(f"\nConversion completed!")
        self.log(f"  Input size: {stats['input_size_gb']:.2f} GB")
        self.log(f"  Output size: {stats['output_size_gb']:.2f} GB") 
        self.log(f"  Compression ratio: {stats['compression_ratio']:.1f}x")
        self.log(f"  Total time: {stats['total_time_s']:.1f}s")
        
        return stats
    
    def _process_and_write_data(self, data_4d: np.ndarray, input_path: str, output_path: str, metadata: Dict, processing_options: Optional[Dict], load_time: float, is_chunked: bool = False) -> Dict[str, Any]:
        """Common data processing and writing logic for in-memory conversion"""
        self.update_progress(30, "Applying data processing..." if processing_options else "Preparing data...")
        
        # Apply data processing if specified
        if processing_options:
            self.log("\nApplying data processing...")
            processing_start = time.time()
            
            # Log what processing is being applied
            if processing_options.get('sum_y', False):
                self.log("  Summing in Y direction (EELS processing)")
            if processing_options.get('bin_factor', 1) > 1:
                bin_factor = processing_options['bin_factor']
                bin_method = processing_options.get('bin_method', 'mean')
                self.log(f"  Applying {bin_factor}x{bin_factor} binning ({bin_method})")
            
            # Apply processing
            original_shape = data_4d.shape
            data_4d = apply_data_processing(data_4d, processing_options)
            processed_shape = data_4d.shape
            
            processing_time = time.time() - processing_start
            self.log(f"  Processing completed in {processing_time:.1f}s")
            self.log(f"  Shape: {original_shape} → {processed_shape}")
            
            # Update metadata to reflect processed shape
            metadata['shape_4d'] = processed_shape
            metadata['original_shape_4d'] = original_shape
            if 'processing_applied' not in metadata:
                metadata['processing_applied'] = processing_options.copy()
        
        # Determine chunking based on final processed shape
        chunks = self.determine_optimal_chunks(metadata['shape_4d'])
        
        # Create EMD file
        self.log(f"\nCreating EMD file with {self.compression} compression...")
        start_time = time.time()
        
        # Prepare compression kwargs
        compression_kwargs = {}
        if self.compression:
            compression_kwargs['compression'] = self.compression
            if self.compression_level is not None:
                compression_kwargs['compression_opts'] = self.compression_level
        
        with h5py.File(output_path, 'w') as f:
            # Create EMD 1.0 root structure
            f.attrs['emd_group_type'] = 'file'
            f.attrs['version_major'] = 1
            f.attrs['version_minor'] = 0
            f.attrs['authoring_program'] = 'mib-to-emd-converter'
            
            # Create py4DSTEM compatible structure
            version_group = f.create_group('version_1')
            version_group.attrs['emd_group_type'] = 'root'
            
            # Data group
            data_group = version_group.create_group('data')
            datacubes_group = data_group.create_group('datacubes')
            datacube_group = datacubes_group.create_group('datacube_000')
            datacube_group.attrs['emd_group_type'] = 'array'
            
            # Create main dataset with compression and progress bar
            self.log(f"  Writing 4D dataset: {metadata['shape_4d']}")
            dataset = datacube_group.create_dataset(
                'data', 
                data=data_4d,
                chunks=chunks,
                **compression_kwargs
            )
            dataset.attrs['units'] = 'counts'
            
            # Create dimension datasets
            sy, sx, qy, qx = metadata['shape_4d']
            
            # Real space dimensions (scan coordinates)
            datacube_group.create_dataset('dim1', data=np.arange(sy))
            datacube_group.create_dataset('dim2', data=np.arange(sx))
            datacube_group['dim1'].attrs['name'] = 'scan_y'
            datacube_group['dim1'].attrs['units'] = 'pixel'
            datacube_group['dim2'].attrs['name'] = 'scan_x'  
            datacube_group['dim2'].attrs['units'] = 'pixel'
            
            # Reciprocal space dimensions (detector coordinates)
            datacube_group.create_dataset('dim3', data=np.arange(qy))
            datacube_group.create_dataset('dim4', data=np.arange(qx))
            datacube_group['dim3'].attrs['name'] = 'detector_y'
            datacube_group['dim3'].attrs['units'] = 'pixel'
            datacube_group['dim4'].attrs['name'] = 'detector_x'
            datacube_group['dim4'].attrs['units'] = 'pixel'
            
            # Add dimension convention documentation
            datacube_group.attrs['dimension_order'] = 'scan_y, scan_x, detector_y, detector_x'
            datacube_group.attrs['dimension_convention'] = 'MIB Viewer format: EELS energy in detector_x dimension'
            
            # Metadata group
            metadata_group = version_group.create_group('metadata')
            microscope_group = metadata_group.create_group('microscope')
            
            # Store acquisition metadata
            for key, value in metadata.items():
                if value is not None:
                    try:
                        if isinstance(value, (str, int, float)):
                            microscope_group.attrs[key] = value
                        elif isinstance(value, (list, tuple)):
                            microscope_group.attrs[key] = list(value)
                    except Exception:
                        # Skip metadata that can't be stored as HDF5 attributes
                        pass
            
            # Log group
            log_group = version_group.create_group('log')
            log_group.attrs['conversion_date'] = time.strftime('%Y-%m-%d %H:%M:%S')
            log_group.attrs['converter_version'] = '1.0'
            log_group.attrs['compression_algorithm'] = self.compression or 'none'
            if self.compression_level:
                log_group.attrs['compression_level'] = self.compression_level
        
        write_time = time.time() - start_time
        
        # Calculate statistics
        input_size = os.path.getsize(input_path)
        output_size = os.path.getsize(output_path)
        compression_ratio = input_size / output_size
        
        stats = {
            'input_size_gb': input_size / (1024**3),
            'output_size_gb': output_size / (1024**3),
            'compression_ratio': compression_ratio,
            'load_time_s': load_time,
            'write_time_s': write_time,
            'total_time_s': load_time + write_time
        }
        
        self.log(f"\nConversion completed!")
        self.log(f"  Input size: {stats['input_size_gb']:.2f} GB")
        self.log(f"  Output size: {stats['output_size_gb']:.2f} GB") 
        self.log(f"  Compression ratio: {stats['compression_ratio']:.1f}x")
        self.log(f"  Total time: {stats['total_time_s']:.1f}s")
        
        return stats

def main():
    """Command line interface for MIB/EMD to EMD conversion and processing"""
    parser = argparse.ArgumentParser(
        description='Convert MIB files to EMD 1.0 format or process existing EMD files with binning/Y-summing',
        epilog='Examples:\n  python mib_to_emd_converter.py data.mib data.emd\n  python mib_to_emd_converter.py input.emd processed.emd --bin-factor 2 --sum-y',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('input_file', help='Input MIB or EMD file path')
    parser.add_argument('output_emd', help='Output EMD file path (.emd extension recommended)')
    
    # Processing options
    parser.add_argument('--bin-factor', type=int, choices=[1, 2, 4, 8, 16], default=1,
                       help='Binning factor for detector dimensions (default: 1, no binning)')
    parser.add_argument('--bin-method', choices=['mean', 'sum'], default='mean',
                       help='Binning method (default: mean)')
    parser.add_argument('--sum-y', action='store_true',
                       help='Sum in Y direction for EELS processing')
    parser.add_argument('--compression', choices=['gzip', 'szip', 'lzf', 'none'], 
                       default='gzip', help='Compression algorithm (default: gzip)')
    parser.add_argument('--compression-level', type=int, choices=range(1, 10),
                       default=6, help='Compression level for gzip (1-9, default: 6)')
    parser.add_argument('--chunks', nargs=4, type=int, metavar=('SY', 'SX', 'QY', 'QX'),
                       help='Custom chunk size (default: auto-determined)')
    parser.add_argument('--force', action='store_true',
                       help='Overwrite output file if it exists')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)
    
    # Check output file
    if os.path.exists(args.output_emd) and not args.force:
        print(f"Error: Output file exists: {args.output_emd}")
        print("Use --force to overwrite")
        sys.exit(1)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output_emd) or '.', exist_ok=True)
    
    # Setup processing options
    processing_options = {}
    if args.bin_factor > 1:
        processing_options['bin_factor'] = args.bin_factor
        processing_options['bin_method'] = args.bin_method
    if args.sum_y:
        processing_options['sum_y'] = True
    
    # Setup converter
    compression = args.compression if args.compression != 'none' else None
    chunk_size = tuple(args.chunks) if args.chunks else None
    
    converter = MibToEmdConverter(
        compression=compression,
        compression_level=args.compression_level,
        chunk_size=chunk_size
    )
    
    try:
        # Convert/process file
        stats = converter.convert_to_emd(args.input_file, args.output_emd, 
                                        processing_options=processing_options if processing_options else None)
        
        # Success message  
        input_type = "MIB" if args.input_file.lower().endswith('.mib') else "EMD"
        operation = "converted" if input_type == "MIB" else "processed"
        print(f"\n[SUCCESS] Successfully {operation} {os.path.basename(args.input_file)} to EMD format!")
        print(f"   Saved: {args.output_emd}")
        print(f"   Size reduction: {stats['compression_ratio']:.1f}x")
        
        if processing_options:
            print(f"   Processing applied: {', '.join(f'{k}={v}' for k, v in processing_options.items())}")
        
    except KeyboardInterrupt:
        print("\n[CANCELLED] Conversion cancelled by user")
        # Clean up partial file
        if os.path.exists(args.output_emd):
            os.remove(args.output_emd)
        sys.exit(1)
        
    except Exception as e:
        operation = "conversion" if args.input_file.lower().endswith('.mib') else "processing"
        print(f"\n[ERROR] {operation.capitalize()} failed: {str(e)}")
        # Clean up partial file
        if os.path.exists(args.output_emd):
            os.remove(args.output_emd)
        sys.exit(1)

if __name__ == "__main__":
    main()