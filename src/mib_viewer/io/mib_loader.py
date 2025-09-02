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
    
    # Read detector size
    fp.merlin_size = (int(head[4]), int(head[5]))
    
    # Test if RAW
    if head[6] == 'R64':
        fp.raw = True
    
    if head[7].endswith('2x2'):
        fp.detectorgeometry = '2x2'
    if head[7].endswith('Nx1G'):
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


def load_mib(path_buffer, scan_size=None):
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
    
    # Auto-detect scan size if not provided
    if scan_size is None:
        scan_size = auto_detect_scan_size(mib_prop.numberOfFramesInFile)
        print(f"Auto-detected scan size: {scan_size[0]}x{scan_size[1]} from {mib_prop.numberOfFramesInFile} frames")
    
    mib_prop.scan_size = scan_size
    if type(scan_size) == int:
        mib_prop.xy = scan_size
    if type(scan_size) == tuple:
        mib_prop.xy = scan_size[0] * scan_size[1]
    
    if mib_prop.xy > mib_prop.numberOfFramesInFile:
        raise ValueError(f"Requested number of frames: {mib_prop.xy} exceeds available frames: {mib_prop.numberOfFramesInFile}")
    
    if mib_prop.raw:
        raise ValueError('RAW MIB data not supported.')
    
    # Load processed MIB file
    data = np.memmap(
        mib_prop.path,
        dtype=merlin_frame_dtype,
        offset=mib_prop.offset,
        shape=mib_prop.scan_size
    )
    
    return data['data']


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
            
            return data
            
    except OSError as e:
        if "Unable to open file" in str(e):
            raise ValueError(f"Cannot open EMD file: {path_buffer}")
        else:
            raise ValueError(f"Error reading EMD file: {str(e)}")
    except Exception as e:
        raise ValueError(f"Failed to load EMD file: {str(e)}")


def load_data_file(path_buffer, scan_size=None):
    """Universal loader that detects file type and loads appropriately.
    
    Parameters:
    -----------
    path_buffer : str
        Path to the data file (.mib or .emd)
    scan_size : tuple, optional
        Scan size for MIB files (ignored for EMD files)
        
    Returns:
    --------
    numpy.ndarray
        4D array with shape (sy, sx, qy, qx)
    """
    path_str = str(path_buffer).lower()
    
    if path_str.endswith('.emd'):
        return load_emd(path_buffer)
    elif path_str.endswith('.mib'):
        return load_mib(path_buffer, scan_size)
    else:
        # Try to auto-detect based on file content
        try:
            # Try EMD first (safer - won't corrupt memory if wrong)
            return load_emd(path_buffer)
        except:
            try:
                # Fall back to MIB
                return load_mib(path_buffer, scan_size)
            except:
                raise ValueError(f"Cannot determine file type or load data from: {path_buffer}")


def get_data_file_info(path_buffer):
    """Get basic information about a data file without loading the full dataset.
    
    Parameters:
    -----------
    path_buffer : str
        Path to the data file (.mib or .emd)
        
    Returns:
    --------
    dict
        Dictionary with file information including shape, size, and type
    """
    path_str = str(path_buffer).lower()
    
    if path_str.endswith('.emd'):
        try:
            import h5py
            with h5py.File(path_buffer, 'r') as f:
                datacube = f['version_1/data/datacubes/datacube_000']
                data_shape = datacube['data'].shape
                file_size = os.path.getsize(path_buffer)
                
                return {
                    'file_type': 'EMD 1.0',
                    'shape': data_shape,
                    'size_bytes': file_size,
                    'size_gb': file_size / (1024**3),
                    'compressed': True,  # EMD files are typically compressed
                    'compatible': True
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
            
            return {
                'file_type': 'MIB',
                'shape': (scan_size[1], scan_size[0], mib_prop.merlin_size[1], mib_prop.merlin_size[0]),
                'size_bytes': filesize,
                'size_gb': filesize / (1024**3),
                'compressed': False,
                'scan_size': scan_size,
                'detector_size': mib_prop.merlin_size,
                'compatible': True
            }
        except Exception as e:
            return {'file_type': 'MIB', 'error': str(e), 'compatible': False}
    
    else:
        return {'file_type': 'Unknown', 'error': 'Unsupported file extension', 'compatible': False}