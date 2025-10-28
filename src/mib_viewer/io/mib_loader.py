#!/usr/bin/env python3
"""
MIB File Loading Utilities

Contains functions for loading and parsing Quantum Detectors MIB files.
Extracted from the original GUI code for reusability across the application.
"""

import os
import numpy as np
from typing import Optional, Tuple


class MibProperties:
    """Container for MIB file properties and metadata"""
    def __init__(self):
        self.path = ''
        self.scan_size = (1, 1)
        self.xy = 1
        self.merlin_size = (515, 515)
        self.headsize = 384
        self.pixeltype = np.dtype('>u2')  # Default: 12-bit processed data
        self.offset = 0
        self.raw = False
        self.quad = False 
        self.single = True
        self.dyn_range = '12-bit'
        self.numberOfFramesInFile = 1
        self.detectorgeometry = '1x1'


def get_mib_properties(head):
    """Parse header of a MIB data and return object containing frame parameters"""
    fp = MibProperties()
    
    # Read detector size - QD gives (width, height)
    fp.merlin_size = (int(head[4]), int(head[5]))
    
    # Test if RAW
    if head[6] == 'R64':
        fp.raw = True
    
    if '2x2' in head[7]:
        fp.detectorgeometry = '2x2'
    if 'Nx1' in head[7]:
        fp.detectorgeometry = 'Nx1'
    
    # Test if single
    if head[2] == '00384':
        fp.single = True
    # Test if quad and read full quad header
    if head[2] == '00768':
        fp.headsize = 768
        fp.quad = True
        fp.single = False
    
    # Set bit-depths for processed data
    if not fp.raw:
        if head[6] == 'U08':
            fp.pixeltype = np.dtype('uint8')
            fp.dyn_range = '1 or 6-bit'
        if head[6] == 'U16':
            fp.pixeltype = np.dtype('>u2')
            fp.dyn_range = '12-bit'
        if head[6] == 'U32':
            fp.pixeltype = np.dtype('>u4')
            fp.dyn_range = '24-bit'
    
    return fp


def auto_detect_scan_size(num_frames):
    """Automatically detect scan size from number of frames"""
    # Try to find the best square or rectangular arrangement
    # Priority: square > rectangular with reasonable aspect ratio
    
    # First try perfect square
    sqrt_frames = int(np.sqrt(num_frames))
    if sqrt_frames * sqrt_frames == num_frames:
        return (sqrt_frames, sqrt_frames)
    
    # Try common rectangular arrangements
    factors = []
    for i in range(1, int(np.sqrt(num_frames)) + 1):
        if num_frames % i == 0:
            factors.append((i, num_frames // i))
    
    # Find the most square-like arrangement (closest to 1:1 aspect ratio)
    if factors:
        best_ratio = float('inf')
        best_size = factors[-1]
        for w, h in factors:
            ratio = max(w, h) / min(w, h)  # Aspect ratio
            if ratio < best_ratio:
                best_ratio = ratio
                best_size = (w, h)
        return best_size
    
    # Fallback: assume 1D scan
    return (num_frames, 1)


def load_mib(path_buffer):
    """Load Quantum Detectors MIB file from a path."""

    # Read header from the start of the file
    try:
        with open(path_buffer, 'rb') as f:
            head = f.read(384).decode().split(',')
            f.seek(0, os.SEEK_END)
            filesize = f.tell()
    except:
        raise ValueError('File does not contain MIB header')

    # Parse header info
    mib_prop = get_mib_properties(head)
    mib_prop.path = path_buffer

    # Find the size of the data
    merlin_frame_dtype = np.dtype([
        ('header', np.bytes_, mib_prop.headsize),
        ('data', mib_prop.pixeltype, mib_prop.merlin_size)
    ])
    mib_prop.numberOfFramesInFile = filesize // merlin_frame_dtype.itemsize

    # Always auto-detect scan size from actual frame count
    scan_size = auto_detect_scan_size(mib_prop.numberOfFramesInFile)
    print(f"Auto-detected scan size: {scan_size[0]}x{scan_size[1]} from {mib_prop.numberOfFramesInFile} frames")

    mib_prop.scan_size = scan_size
    if type(scan_size) == int:
        mib_prop.xy = scan_size
    if type(scan_size) == tuple:
        mib_prop.xy = scan_size[0] * scan_size[1]
    
    if mib_prop.raw:
        raise ValueError('RAW MIB data not supported.')
    
    # Load processed MIB file
    data = np.memmap(
        mib_prop.path,
        dtype=merlin_frame_dtype,
        offset=mib_prop.offset,
        shape=mib_prop.scan_size
    )
    
    # Fix detector dimension scrambling issue
    # QD gives merlin_size as (width, height) but detector data may be scrambled during memmap loading
    # Instead of transpose, reshape the detector frames to correct orientation
    raw_data = data['data']
    sy, sx = raw_data.shape[:2]  # Scan dimensions
    detector_width, detector_height = mib_prop.merlin_size  # Original (width, height) from MIB header
    
    # Current detector shape as loaded by memmap
    current_dy, current_dx = raw_data.shape[2:4]
    
    print(f"MIB detector size from header: {detector_width}×{detector_height} (width×height)")
    print(f"Loaded detector shape: {current_dy}×{current_dx}")
    
    # Apply reshape fix ONLY for EELS data that appears to be scrambled
    # Safe condition: only reshape when we have rectangular detector (EELS) with wrong orientation
    if current_dy != current_dx and current_dy > current_dx:
        # This looks like EELS data where dy > dx (likely scrambled)
        # For EELS, we expect dy < dx (shorter × longer dimensions)
        
        # Additional safety check: dimensions should match the header values
        if {current_dy, current_dx} == {detector_width, detector_height}:
            print(f"EELS reshape fix: detected scrambled detector dimensions")
            print(f"  Current: {current_dy}×{current_dx} (dy > dx, likely scrambled)")
            print(f"  Header: {detector_width}×{detector_height} (width×height)")
            print(f"  Reshaping: ({current_dy}, {current_dx}) → ({current_dx}, {current_dy})")
            
            # Reshape each detector frame to swap dimensions back
            reshaped_data = np.zeros((sy, sx, current_dx, current_dy), dtype=raw_data.dtype)
            for scan_y in range(sy):
                for scan_x in range(sx):
                    # Reshape the detector frame to unscramble
                    frame = raw_data[scan_y, scan_x, :, :]  # Shape: (dy, dx) - scrambled
                    reshaped_frame = frame.reshape(current_dx, current_dy)  # Unscramble to (dx, dy)
                    reshaped_data[scan_y, scan_x, :, :] = reshaped_frame
            
            print(f"  Result: {reshaped_data.shape[2]}×{reshaped_data.shape[3]} (now dy < dx for EELS)")
            return reshaped_data
        else:
            print(f"Detector dimensions don't match header - skipping reshape for safety")
            print(f"  Loaded: {current_dy}×{current_dx}, Header: {detector_width}×{detector_height}")
            return raw_data
    else:
        # Data appears to be in correct orientation (4D STEM or properly oriented EELS)
        print("Detector orientation appears correct - no reshaping needed")
        return raw_data


def load_emd(path_buffer):
    """Load EMD 1.0 format file created by our MIB to EMD converter.
    
    Parameters:
    -----------
    path_buffer : str
        Path to the EMD file
        
    Returns:
    --------
    numpy.ndarray
        4D array with shape (sy, sx, qy, qx) - same format as load_mib()
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required to load EMD files. Install with: pip install h5py")
    
    try:
        with h5py.File(path_buffer, 'r') as f:
            # Navigate to the data in EMD 1.0 structure
            # Structure: /version_1/data/datacubes/datacube_000/data
            if 'version_1' not in f:
                raise ValueError("Not a valid EMD 1.0 file - missing version_1 group")
            
            version_group = f['version_1']
            if 'data' not in version_group or 'datacubes' not in version_group['data']:
                raise ValueError("Not a valid EMD 1.0 file - missing data/datacubes structure")
            
            datacubes = version_group['data']['datacubes']
            if 'datacube_000' not in datacubes:
                raise ValueError("Not a valid EMD 1.0 file - missing datacube_000")
            
            datacube = datacubes['datacube_000']
            if 'data' not in datacube:
                raise ValueError("Not a valid EMD 1.0 file - missing data dataset")
            
            # Load the 4D data - it's memory-mapped so this is efficient
            data = datacube['data'][:]
            
            # Verify expected 4D shape
            if len(data.shape) != 4:
                raise ValueError(f"Expected 4D data, got {len(data.shape)}D with shape {data.shape}")
            
            # EMD files store data that's already been processed by load_mib() 
            # which means the detector dimensions are already transposed correctly
            # No additional transpose needed - return data as-is
            return data
            
    except OSError as e:
        if "Unable to open file" in str(e):
            raise ValueError(f"Cannot open EMD file: {path_buffer}")
        else:
            raise ValueError(f"Error reading EMD file: {str(e)}")
    except Exception as e:
        raise ValueError(f"Failed to load EMD file: {str(e)}")


def load_data_file(path_buffer):
    """Universal loader that detects file type and loads appropriately.

    Parameters:
    -----------
    path_buffer : str
        Path to the data file (.mib or .emd)

    Returns:
    --------
    numpy.ndarray
        4D array with shape (sy, sx, qy, qx)
    """
    path_str = str(path_buffer).lower()
    
    if path_str.endswith('.emd'):
        return load_emd(path_buffer)
    elif path_str.endswith('.mib'):
        return load_mib(path_buffer)
    else:
        # Try to auto-detect based on file content
        try:
            # Try EMD first (safer - won't corrupt memory if wrong)
            return load_emd(path_buffer)
        except:
            try:
                # Fall back to MIB
                return load_mib(path_buffer)
            except:
                raise ValueError(f"Cannot determine file type or load data from: {path_buffer}")


def detect_experiment_type(shape):
    """Detect experiment type (EELS vs 4D STEM) based on data shape
    
    Parameters:
    -----------
    shape : tuple
        4D data shape (scan_y, scan_x, detector_y, detector_x)
        
    Returns:
    --------
    tuple : (experiment_type, processing_info)
        experiment_type: "EELS", "4D_STEM", or "UNKNOWN"
        processing_info: dict with recommended processing options
    """
    sy, sx, dy, dx = shape
    
    # EELS detection: detector dimensions are not equal (rectangular detector)
    if dy != dx:
        # Determine if we can sum in Y direction (need 2D detector, not already summed to 1D)
        can_sum_y = min(dy, dx) > 1  # Can only sum if both dimensions > 1
        
        return "EELS", {
            'can_sum_y': can_sum_y,
            'can_bin': False,        # EELS doesn't typically need binning
            'recommended_processing': 'sum_y' if can_sum_y else 'none',
            'detector_type': f'EELS spectrometer ({dy}×{dx})',
            'processing_note': 'Sum in Y direction to reduce data size' if can_sum_y else 'Already summed'
        }
    
    # 4D STEM detection: square detector (equal dimensions)
    elif dy == dx:
        valid_factors = get_valid_bin_factors(dy)  # Use actual detector size
        detector_name = "Quad detector" if dy >= 512 else "Single detector"
        
        return "4D_STEM", {
            'can_sum_y': False,      # 4D STEM shouldn't sum Y
            'can_bin': True,
            'valid_bin_factors': valid_factors,
            'recommended_processing': 'bin_2x2',
            'detector_type': f'{detector_name} ({dy}×{dx})',
            'processing_note': 'Binning reduces file size and noise'
        }
    
    # Unknown/unsupported configuration
    else:
        return "UNKNOWN", {
            'can_sum_y': False,
            'can_bin': False,
            'recommended_processing': 'none',
            'detector_type': f'Unknown detector ({dy}×{dx})',
            'processing_note': 'Unsupported detector configuration'
        }


def get_valid_bin_factors(detector_size):
    """Get binning factors that evenly divide detector dimension
    
    Parameters:
    -----------
    detector_size : int
        Detector dimension (256, 512, etc.)
        
    Returns:
    --------
    list : Valid binning factors
    """
    # Common factors that work well for typical detector sizes
    candidate_factors = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    
    # Return only factors that divide evenly and don't exceed detector size
    return [f for f in candidate_factors if f <= detector_size and detector_size % f == 0]


def apply_data_processing(data_4d, processing_options):
    """Apply EELS Y-summing and/or 4D binning to data

    Parameters:
    -----------
    data_4d : numpy.ndarray
        4D data array (scan_y, scan_x, detector_y, detector_x)
    processing_options : dict
        Processing options including:
        - 'sum_y': bool, whether to sum in Y direction
        - 'bin_factor': int, binning factor for detector dimensions
        - 'bin_method': str, 'mean' or 'sum'

    Returns:
    --------
    numpy.ndarray : Processed data
    """
    original_shape = data_4d.shape

    # Detect if this is EELS data (rectangular detector with dy < dx after unscrambling)
    sy, sx, dy, dx = data_4d.shape
    is_eels = (dy != dx and dy < dx)

    # Apply binning first (if requested)
    if processing_options.get('bin_factor', 1) > 1:
        data_4d = apply_binning(data_4d,
                               processing_options['bin_factor'],
                               processing_options.get('bin_method', 'mean'))
        print(f"Applied {processing_options['bin_factor']}x{processing_options['bin_factor']} binning: "
              f"{original_shape} → {data_4d.shape}")

    # Apply Y-summing second (if requested)
    if processing_options.get('sum_y', False):
        data_4d = np.sum(data_4d, axis=2, keepdims=True)
        print(f"Applied Y-summing: {data_4d.shape}")

    # CRITICAL: For EELS data from MIB files, ALWAYS flip energy axis
    # MIB files store EELS with backward energy axis, must be corrected for EMD
    if is_eels:
        data_4d = data_4d[:, :, :, ::-1]
        print(f"Applied energy axis flip for EELS data (MIB → EMD conversion)")

    return data_4d


def apply_binning(data_4d, bin_factor, method='mean'):
    """Bin 4D data by given factor in detector dimensions
    
    Parameters:
    -----------
    data_4d : numpy.ndarray
        4D data array (scan_y, scan_x, detector_y, detector_x)
    bin_factor : int
        Binning factor (must divide detector dimensions evenly)
    method : str
        'mean' or 'sum' for binning operation
        
    Returns:
    --------
    numpy.ndarray : Binned data
    """
    sy, sx, qy, qx = data_4d.shape
    
    # Validate binning factor
    if qy % bin_factor != 0 or qx % bin_factor != 0:
        raise ValueError(f"Binning factor {bin_factor} does not divide evenly into "
                        f"detector dimensions ({qy}×{qx})")
    
    # Calculate output dimensions
    new_qy = qy // bin_factor
    new_qx = qx // bin_factor
    
    # Reshape for binning: (sy, sx, new_qy, bin_factor, new_qx, bin_factor)
    binned_data = data_4d.reshape(sy, sx, new_qy, bin_factor, new_qx, bin_factor)
    
    # Apply binning operation along the bin_factor axes (3 and 5)
    if method == 'mean':
        return binned_data.mean(axis=(3, 5))
    elif method == 'sum':
        return binned_data.sum(axis=(3, 5))
    else:
        raise ValueError(f"Unknown binning method: {method}. Use 'mean' or 'sum'.")


def calculate_processed_size(original_shape, processing_options):
    """Calculate the shape and size after processing
    
    Parameters:
    -----------
    original_shape : tuple
        Original 4D shape (scan_y, scan_x, detector_y, detector_x)
    processing_options : dict
        Processing options
        
    Returns:
    --------
    tuple : (new_shape, size_reduction_factor)
    """
    sy, sx, qy, qx = original_shape
    
    # Apply binning to detector dimensions
    bin_factor = processing_options.get('bin_factor', 1)
    if bin_factor > 1:
        qy = qy // bin_factor
        qx = qx // bin_factor
    
    # Apply Y-summing
    if processing_options.get('sum_y', False):
        qy = 1  # Y dimension becomes 1
    
    new_shape = (sy, sx, qy, qx)
    
    # Calculate size reduction factor
    original_size = sy * sx * original_shape[2] * original_shape[3]
    new_size = sy * sx * qy * qx
    reduction_factor = original_size / new_size if new_size > 0 else 1
    
    return new_shape, reduction_factor


def get_data_file_info(path_buffer):
    """Get basic information about a data file without loading the full dataset.
    
    Parameters:
    -----------
    path_buffer : str
        Path to the data file (.mib or .emd)
        
    Returns:
    --------
    dict
        Dictionary with file information including shape, size, type, and processing options
    """
    path_str = str(path_buffer).lower()
    
    if path_str.endswith('.emd'):
        try:
            import h5py
            with h5py.File(path_buffer, 'r') as f:
                datacube = f['version_1/data/datacubes/datacube_000']
                data_shape = datacube['data'].shape
                file_size = os.path.getsize(path_buffer)
                
                # Detect experiment type and add processing recommendations
                experiment_type, processing_info = detect_experiment_type(data_shape)
                
                return {
                    'file_type': 'EMD 1.0',
                    'shape': data_shape,
                    'size_bytes': file_size,
                    'size_gb': file_size / (1024**3),
                    'compressed': True,  # EMD files are typically compressed
                    'compatible': True,
                    'experiment_type': experiment_type,
                    'processing_options': processing_info
                }
        except Exception as e:
            return {'file_type': 'EMD', 'error': str(e), 'compatible': False}
            
    elif path_str.endswith('.mib'):
        try:
            # Read just the header for MIB files
            with open(path_buffer, 'rb') as f:
                head = f.read(384).decode().split(',')
                f.seek(0, os.SEEK_END)
                filesize = f.tell()
            
            mib_prop = get_mib_properties(head)
            num_frames = filesize // (mib_prop.headsize + np.prod(mib_prop.merlin_size) * 2)
            scan_size = auto_detect_scan_size(num_frames)
            data_shape = (scan_size[1], scan_size[0], mib_prop.merlin_size[1], mib_prop.merlin_size[0])
            
            # Detect experiment type and add processing recommendations
            experiment_type, processing_info = detect_experiment_type(data_shape)
            
            return {
                'file_type': 'MIB',
                'shape': data_shape,
                'size_bytes': filesize,
                'size_gb': filesize / (1024**3),
                'compressed': False,
                'scan_size': scan_size,
                'detector_size': mib_prop.merlin_size,
                'compatible': True,
                'experiment_type': experiment_type,
                'processing_options': processing_info
            }
        except Exception as e:
            return {'file_type': 'MIB', 'error': str(e), 'compatible': False}
    
    else:
        return {'file_type': 'Unknown', 'error': 'Unsupported file extension', 'compatible': False}


def walk_emd_structure(filename):
    """
    Walk through complete EMD file structure and extract all metadata.

    Parameters:
    -----------
    filename : str
        Path to EMD file

    Returns:
    --------
    dict
        Complete file structure with nested groups, datasets, and attributes
        Format: {
            'file_info': {...},
            'structure': {...},  # Nested dict mirroring HDF5 structure
            'flat_items': [...]  # Flat list for tree widget population
        }
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required to inspect EMD files. Install with: pip install h5py")

    try:
        with h5py.File(filename, 'r') as f:
            # File-level info
            import os
            file_info = {
                'filename': os.path.basename(filename),
                'filepath': filename,
                'filesize_bytes': os.path.getsize(filename),
                'filesize_mb': os.path.getsize(filename) / (1024 * 1024),
                'hdf5_version': f.attrs.get('version_major', 'unknown'),
            }

            # Extract file-level attributes
            file_attrs = {}
            for key, value in f.attrs.items():
                file_attrs[key] = _convert_hdf5_value(value)
            file_info['attributes'] = file_attrs

            # Walk the complete structure
            structure = {}
            flat_items = []

            def _walk_group(group, path='/', level=0):
                """Recursively walk HDF5 group structure"""

                # Add this group to flat list
                group_info = {
                    'path': path,
                    'name': path.split('/')[-1] or 'Root',
                    'type': 'group',
                    'level': level,
                    'attributes': {},
                    'children': []
                }

                # Extract group attributes
                for key, value in group.attrs.items():
                    group_info['attributes'][key] = _convert_hdf5_value(value)

                flat_items.append(group_info)

                # Create structure entry
                current_dict = {}
                structure_path = path.strip('/').split('/') if path != '/' else []

                # Walk through items in this group
                for name in group.keys():
                    item = group[name]
                    item_path = f"{path.rstrip('/')}/{name}" if path != '/' else f"/{name}"

                    if isinstance(item, h5py.Group):
                        # Recursively process subgroups
                        substructure = _walk_group(item, item_path, level + 1)
                        current_dict[name] = substructure
                        group_info['children'].append(name)

                    elif isinstance(item, h5py.Dataset):
                        # Process dataset
                        dataset_info = {
                            'path': item_path,
                            'name': name,
                            'type': 'dataset',
                            'level': level + 1,
                            'shape': item.shape,
                            'dtype': str(item.dtype),
                            'size_bytes': item.size * item.dtype.itemsize,
                            'size_mb': (item.size * item.dtype.itemsize) / (1024 * 1024),
                            'chunks': item.chunks,
                            'compression': item.compression,
                            'attributes': {}
                        }

                        # Extract dataset attributes
                        for key, value in item.attrs.items():
                            dataset_info['attributes'][key] = _convert_hdf5_value(value)

                        # Add statistics for small datasets
                        if dataset_info['size_mb'] < 100:  # Only for datasets < 100MB
                            try:
                                data_sample = item[:]
                                if data_sample.size > 0:
                                    dataset_info['statistics'] = {
                                        'min': float(data_sample.min()),
                                        'max': float(data_sample.max()),
                                        'mean': float(data_sample.mean()),
                                        'std': float(data_sample.std())
                                    }
                            except:
                                pass

                        flat_items.append(dataset_info)
                        group_info['children'].append(name)
                        current_dict[name] = dataset_info

                return current_dict

            # Start walking from root
            structure = _walk_group(f)

            return {
                'file_info': file_info,
                'structure': structure,
                'flat_items': flat_items
            }

    except Exception as e:
        raise ValueError(f"Failed to walk EMD structure: {str(e)}")


def _convert_hdf5_value(value):
    """Convert HDF5 attribute value to Python-serializable format"""
    try:
        if isinstance(value, bytes):
            return value.decode('utf-8', errors='ignore')
        elif isinstance(value, np.ndarray):
            if value.size == 1:
                return value.item()
            else:
                return value.tolist()
        elif hasattr(value, 'item'):
            return value.item()
        else:
            return value
    except:
        return str(value)