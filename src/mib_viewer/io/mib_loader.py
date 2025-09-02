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