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
from typing import Optional, Tuple, Dict, Any
import warnings

import numpy as np
import h5py
import emdfile
from tqdm import tqdm

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
    
    def __init__(self, compression='gzip', compression_level=6, chunk_size=None, log_callback=None):
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
        """
        self.compression = compression
        self.compression_level = compression_level if compression == 'gzip' else None
        self.chunk_size = chunk_size
        self.log_callback = log_callback
    
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
    
    def determine_optimal_chunks(self, shape_4d: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Determine optimal chunk size based on data shape and threading strategy
        
        Uses frame-based chunking (1, 1, qy, qx) for optimal threading performance.
        Compression benchmark shows <5% penalty vs larger chunks but enables 
        perfect parallelization for future multithreading.
        
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
        
        sy, sx, qy, qx = shape_4d
        
        # Frame-based chunking for threading optimization
        # Each chunk = one detector frame at one scan position
        chunk_sy = 1
        chunk_sx = 1 
        chunk_qy = qy  # Full detector height
        chunk_qx = qx  # Full detector width
        
        # Handle edge cases for processed data
        if chunk_qy == 0 or chunk_qx == 0:
            # Fallback to minimal valid chunks
            chunk_qy = max(1, qy)
            chunk_qx = max(1, qx)
        
        chunks = (chunk_sy, chunk_sx, chunk_qy, chunk_qx)
        
        # Calculate chunk size for logging
        bytes_per_pixel = 2  # uint16
        chunk_mb = (chunk_sy * chunk_sx * chunk_qy * chunk_qx * bytes_per_pixel) / (1024**2)
        
        self.log(f"  Frame-based chunks: {chunks} (~{chunk_mb:.2f} MB per chunk)")
        self.log(f"  Threading-optimized for future multithreading")
        return chunks
    
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
        
        # Analyze input file based on type
        if file_type == 'mib':
            metadata = self.analyze_mib_file(input_path)
        else:  # emd
            metadata = self.analyze_emd_file(input_path)
            
        if metadata_extra:
            metadata.update(metadata_extra)
        
        # Load data based on file type
        self.log(f"\nLoading {file_type.upper()} data...")
        start_time = time.time()
        
        if file_type == 'mib':
            data_4d = load_mib(input_path, metadata['scan_size'])
        else:  # emd
            data_4d = load_emd(input_path)
        
        load_time = time.time() - start_time
        self.log(f"  Loaded in {load_time:.1f}s")
        
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