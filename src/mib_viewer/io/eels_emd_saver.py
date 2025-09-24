#!/usr/bin/env python3
"""
EELS EMD Saver

Saves processed EELS data from progressive loading as EMD 1.0 format.
Reuses existing EMD writing infrastructure for compatibility with py4DSTEM and STEMTooL.
"""

import os
import time
import numpy as np
import h5py
from typing import Optional, Callable
from pathlib import Path

# Import MIB properties for metadata extraction
try:
    from .mib_loader import get_mib_properties
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from io.mib_loader import get_mib_properties


class EelsEmdSaver:
    """
    Save processed EELS data as EMD 1.0 format

    Designed specifically for progressive loading output:
    - Input: (scan_y, scan_x, 1, energy_channels) processed EELS data
    - Output: EMD 1.0 HDF5 file compatible with py4DSTEM/STEMTooL
    """

    def __init__(self, compression='gzip', compression_level=6):
        """
        Initialize saver with compression settings

        Parameters:
        -----------
        compression : str
            Compression algorithm ('gzip', 'szip', 'lzf', or None)
        compression_level : int
            Compression level (1-9 for gzip)
        """
        self.compression = compression
        self.compression_level = compression_level

    def save_eels_data(self, eels_data: np.ndarray, original_mib_path: str,
                       output_path: str, progress_callback: Optional[Callable] = None) -> bool:
        """
        Save processed EELS data as EMD file

        Parameters:
        -----------
        eels_data : np.ndarray
            Processed EELS data with shape (scan_y, scan_x, 1, energy_channels)
        original_mib_path : str
            Path to original MIB file for metadata extraction
        output_path : str
            Output EMD file path
        progress_callback : callable, optional
            Callback function for progress updates

        Returns:
        --------
        bool : True if save successful, False otherwise
        """
        try:
            if progress_callback:
                progress_callback("Extracting metadata from original MIB file...")

            # Extract metadata from original MIB file
            mib_metadata = self._extract_mib_metadata(original_mib_path)

            if progress_callback:
                progress_callback("Creating EMD file structure...")

            # Validate EELS data shape
            if len(eels_data.shape) != 4:
                raise ValueError(f"Expected 4D EELS data, got shape {eels_data.shape}")

            sy, sx, detector_y, energy_channels = eels_data.shape
            if detector_y != 1:
                raise ValueError(f"Expected Y-summed EELS data (detector_y=1), got {detector_y}")

            # Create EMD file
            self._create_emd_file(eels_data, output_path, mib_metadata, progress_callback)

            if progress_callback:
                progress_callback("Verifying EMD file integrity...")

            # Verify the created file
            self._verify_emd_file(output_path, eels_data.shape)

            if progress_callback:
                progress_callback("Save completed successfully!")

            return True

        except Exception as e:
            if progress_callback:
                progress_callback(f"Error saving EMD file: {str(e)}")
            raise

    def _extract_mib_metadata(self, mib_path: str) -> dict:
        """Extract metadata from original MIB file"""
        try:
            # Read MIB header
            with open(mib_path, 'rb') as f:
                header_bytes = f.read(384)
            header_fields = header_bytes.decode('utf-8', errors='ignore').split(',')
            mib_props = get_mib_properties(header_fields)

            # Get file stats
            file_size = os.path.getsize(mib_path)

            return {
                'source_file': os.path.basename(mib_path),
                'source_path': mib_path,
                'file_size': file_size,
                'mib_properties': mib_props,
                'processing_info': {
                    'method': 'progressive_loading',
                    'y_summed': True,
                    'energy_axis_flipped': True,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
            }
        except Exception as e:
            # Return minimal metadata if MIB reading fails
            return {
                'source_file': os.path.basename(mib_path) if mib_path else 'unknown',
                'source_path': mib_path or '',
                'error': f"Could not read MIB metadata: {str(e)}",
                'processing_info': {
                    'method': 'progressive_loading',
                    'y_summed': True,
                    'energy_axis_flipped': True,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
            }

    def _create_emd_file(self, eels_data: np.ndarray, output_path: str,
                        metadata: dict, progress_callback: Optional[Callable] = None):
        """Create EMD 1.0 file with EELS data"""
        sy, sx, detector_y, energy_channels = eels_data.shape

        # Setup compression options
        compression_opts = {}
        if self.compression:
            compression_opts['compression'] = self.compression
            if self.compression == 'gzip':
                compression_opts['compression_opts'] = self.compression_level

        with h5py.File(output_path, 'w') as f:
            if progress_callback:
                progress_callback("Writing EMD structure...")

            # Create EMD 1.0 root structure (following adaptive_converter pattern)
            f.attrs['emd_group_type'] = 'file'
            f.attrs['version_major'] = 1
            f.attrs['version_minor'] = 0

            # Create version_1 group
            version_group = f.create_group('version_1')
            version_group.attrs['major'] = 1
            version_group.attrs['minor'] = 0

            # Create proper EMD structure: version_1/data/datacubes/datacube_000
            data_group = version_group.create_group('data')
            datacubes_group = data_group.create_group('datacubes')
            datacube_group = datacubes_group.create_group('datacube_000')

            # Add metadata attributes
            datacube_group.attrs['emd_group_type'] = 1  # Datacube
            datacube_group.attrs['data_type'] = 'EELS'
            datacube_group.attrs['source_file'] = metadata['source_file']
            datacube_group.attrs['processing_method'] = 'progressive_loading'

            if progress_callback:
                progress_callback("Writing EELS data...")

            # Create main dataset - use float32 for processed EELS data
            dataset = datacube_group.create_dataset(
                'data',
                data=eels_data,
                chunks=(min(sy, 64), min(sx, 64), 1, min(energy_channels, 256)),
                dtype=np.float32,
                **compression_opts
            )

            if progress_callback:
                progress_callback("Writing dimension metadata...")

            # Create dimension datasets following EMD 1.0 spec
            dim_group = datacube_group.create_group('dim1')
            dim_group.attrs['name'] = 'scan_y'
            dim_group.attrs['units'] = 'pixels'
            dim_group.create_dataset('dim1', data=np.arange(sy))

            dim_group = datacube_group.create_group('dim2')
            dim_group.attrs['name'] = 'scan_x'
            dim_group.attrs['units'] = 'pixels'
            dim_group.create_dataset('dim2', data=np.arange(sx))

            dim_group = datacube_group.create_group('dim3')
            dim_group.attrs['name'] = 'detector_y'
            dim_group.attrs['units'] = 'pixels'
            dim_group.create_dataset('dim3', data=np.array([0]))  # Y-summed dimension

            dim_group = datacube_group.create_group('dim4')
            dim_group.attrs['name'] = 'energy'
            dim_group.attrs['units'] = 'channels'
            dim_group.create_dataset('dim4', data=np.arange(energy_channels))

            # Add processing metadata
            if 'processing_info' in metadata:
                proc_group = datacube_group.create_group('processing')
                for key, value in metadata['processing_info'].items():
                    proc_group.attrs[key] = value

            # Add original MIB properties if available
            if 'mib_properties' in metadata:
                mib_group = datacube_group.create_group('original_mib')
                mib_props = metadata['mib_properties']
                try:
                    mib_group.attrs['detector_width'] = mib_props.merlin_size[0]
                    mib_group.attrs['detector_height'] = mib_props.merlin_size[1]
                    mib_group.attrs['pixel_type'] = str(mib_props.pixeltype)
                    mib_group.attrs['head_size'] = mib_props.headsize
                except:
                    pass  # Skip if MIB properties are incomplete

    def _verify_emd_file(self, file_path: str, expected_shape: tuple):
        """Verify the created EMD file is valid"""
        try:
            with h5py.File(file_path, 'r') as f:
                # Check EMD structure
                assert 'version_1' in f, "Missing version_1 group"
                assert 'data' in f['version_1'], "Missing data group"
                assert 'datacubes' in f['version_1/data'], "Missing datacubes group"
                assert 'datacube_000' in f['version_1/data/datacubes'], "Missing datacube_000 group"

                # Check dataset
                dataset = f['version_1/data/datacubes/datacube_000/data']
                assert dataset.shape == expected_shape, f"Shape mismatch: {dataset.shape} vs {expected_shape}"

                # Check attributes
                datacube_group = f['version_1/data/datacubes/datacube_000']
                assert datacube_group.attrs.get('data_type') == 'EELS', "Missing EELS data type"

                # Verify data is not all zeros (basic sanity check)
                sample_data = dataset[0, 0, 0, :100]  # Sample first 100 energy channels
                assert np.any(sample_data > 0), "Data appears to be all zeros"

        except Exception as e:
            raise RuntimeError(f"EMD file verification failed: {str(e)}")

    def get_estimated_file_size(self, eels_data: np.ndarray) -> float:
        """Estimate output file size in MB"""
        uncompressed_size = eels_data.nbytes
        # Rough compression estimate (gzip typically achieves 3-5x on EELS data)
        compression_factor = 4.0 if self.compression == 'gzip' else 1.0

        # Add overhead for metadata (~10MB)
        overhead = 10 * 1024 * 1024

        estimated_size = (uncompressed_size / compression_factor) + overhead
        return estimated_size / (1024 * 1024)  # Convert to MB