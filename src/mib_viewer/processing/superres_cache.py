#!/usr/bin/env python3
"""
Super-Resolution Cache File Manager

Manages large intermediate arrays (bigFT, correlations) in separate HDF5 cache files
to keep the main EMD file small while enabling resume capability.

Key features:
- Separate cache file per reconstruction run
- Chunked I/O for memory efficiency
- Resume capability with progress tracking
- Metadata storage for validation
"""

import os
import h5py
import numpy as np
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass


@dataclass
class CacheMetadata:
    """Metadata about cached computation"""
    linked_emd_path: str
    timestamp: str
    step_completed: int  # 0=none, 1=detection, 2=bigft, 3=correlations, 4=shifts, 5=reconstruction
    original_data_shape: Tuple[int, int, int, int]
    cropped_data_shape: Tuple[int, int, int, int]
    crop_bounds: Tuple[int, int, int, int]  # (y1, y2, x1, x2)
    bf_center: Tuple[float, float]  # (cy, cx)
    bf_radius: int
    reference_smoothing: float
    version: str = "1.0"


class SuperResCacheManager:
    """
    Manages HDF5 cache files for super-resolution intermediate data

    Cache file structure:
    /bigFT/
        /data [sy, sx, crop_dy, crop_dx] complex128
        /metadata/...
    /correlations/
        /data [sy, sx, crop_dy, crop_dx] float64
        /metadata/...
    /metadata/
        step_completed: int
        ... (all CacheMetadata fields)
    """

    def __init__(self,
                 emd_path: str,
                 timestamp: Optional[str] = None,
                 create_new: bool = True):
        """
        Initialize cache manager

        Parameters
        ----------
        emd_path : str
            Path to main EMD file
        timestamp : str, optional
            Timestamp for cache file naming. If None, uses current time
        create_new : bool
            If True, creates new cache file. If False, opens existing
        """
        self.emd_path = str(Path(emd_path).resolve())
        self.emd_basename = Path(emd_path).stem
        self.emd_dir = Path(emd_path).parent

        # Generate timestamp if not provided
        if timestamp is None:
            self.timestamp = time.strftime('%Y%m%d_%H%M%S')
        else:
            self.timestamp = timestamp

        # Construct cache file path
        self.cache_filename = f"{self.emd_basename}_superres_cache_{self.timestamp}.h5"
        self.cache_path = str(self.emd_dir / self.cache_filename)

        self.h5file = None
        self._metadata = None

        if create_new and not os.path.exists(self.cache_path):
            # Will be created in create_cache_file()
            pass
        elif os.path.exists(self.cache_path):
            # Open existing cache file
            self.h5file = h5py.File(self.cache_path, 'r+')
            self._load_metadata()
        else:
            raise FileNotFoundError(f"Cache file not found: {self.cache_path}")

    def create_cache_file(self,
                         crop_info: Dict[str, Any],
                         original_shape: Tuple[int, int, int, int],
                         cropped_shape: Tuple[int, int, int, int],
                         reference_smoothing: float = 0.5):
        """
        Create new cache file with proper structure

        Parameters
        ----------
        crop_info : dict
            Dictionary with keys: 'center', 'radius', 'bounds'
        original_shape : tuple
            Original 4D data shape (sy, sx, dy, dx)
        cropped_shape : tuple
            Cropped data shape (sy, sx, crop_dy, crop_dx)
        reference_smoothing : float
            Gaussian sigma for reference image
        """
        if self.h5file is not None:
            raise RuntimeError("Cache file already created or opened")

        print(f"Creating cache file: {self.cache_path}")

        # Create new HDF5 file
        self.h5file = h5py.File(self.cache_path, 'w')

        sy, sx, crop_dy, crop_dx = cropped_shape

        # Create bigFT group and dataset
        bigft_group = self.h5file.create_group('bigFT')

        # Use chunking for efficient I/O
        chunk_shape = (min(sy, 32), min(sx, 32), min(crop_dy, 32), min(crop_dx, 32))

        bigft_group.create_dataset(
            'data',
            shape=cropped_shape,
            dtype=np.complex128,
            chunks=chunk_shape,
            compression='gzip',
            compression_opts=1  # Light compression for speed
        )

        # bigFT metadata
        bigft_meta = bigft_group.create_group('metadata')
        bigft_meta.attrs['crop_bounds'] = crop_info['bounds']
        bigft_meta.attrs['bf_center'] = crop_info['center']
        bigft_meta.attrs['bf_radius'] = crop_info['radius']
        bigft_meta.attrs['reference_smoothing'] = reference_smoothing
        bigft_meta.attrs['computation_date'] = time.strftime('%Y-%m-%d %H:%M:%S')

        # Create correlations group and dataset
        corr_group = self.h5file.create_group('correlations')

        corr_group.create_dataset(
            'data',
            shape=cropped_shape,
            dtype=np.float64,
            chunks=chunk_shape,
            compression='gzip',
            compression_opts=1
        )

        # Correlations metadata
        corr_meta = corr_group.create_group('metadata')
        corr_meta.attrs['reference_pixel'] = [crop_dy // 2, crop_dx // 2]
        corr_meta.attrs['normalization_applied'] = False

        # Root metadata
        meta_group = self.h5file.create_group('metadata')
        meta_group.attrs['linked_emd_path'] = self.emd_path
        meta_group.attrs['timestamp'] = self.timestamp
        meta_group.attrs['step_completed'] = 1  # Step 1 (detection) complete at cache creation
        meta_group.attrs['original_data_shape'] = original_shape
        meta_group.attrs['cropped_data_shape'] = cropped_shape
        meta_group.attrs['crop_bounds'] = crop_info['bounds']
        meta_group.attrs['bf_center'] = crop_info['center']
        meta_group.attrs['bf_radius'] = crop_info['radius']
        meta_group.attrs['reference_smoothing'] = reference_smoothing
        meta_group.attrs['version'] = "1.0"

        self.h5file.flush()

        # Load metadata into memory
        self._load_metadata()

        print(f"✅ Cache file created: {self.cache_path}")
        print(f"   bigFT shape: {cropped_shape}, dtype: complex128")
        print(f"   correlations shape: {cropped_shape}, dtype: float64")

        # Calculate sizes
        bigft_size_gb = np.prod(cropped_shape) * 16 / (1024**3)  # complex128 = 16 bytes
        corr_size_gb = np.prod(cropped_shape) * 8 / (1024**3)   # float64 = 8 bytes
        print(f"   bigFT size: {bigft_size_gb:.2f} GB")
        print(f"   correlations size: {corr_size_gb:.2f} GB")
        print(f"   Total cache size: {bigft_size_gb + corr_size_gb:.2f} GB")

    def _load_metadata(self):
        """Load metadata from cache file into memory"""
        if self.h5file is None:
            raise RuntimeError("Cache file not open")

        meta = self.h5file['metadata']

        self._metadata = CacheMetadata(
            linked_emd_path=str(meta.attrs['linked_emd_path']),
            timestamp=str(meta.attrs['timestamp']),
            step_completed=int(meta.attrs['step_completed']),
            original_data_shape=tuple(meta.attrs['original_data_shape']),
            cropped_data_shape=tuple(meta.attrs['cropped_data_shape']),
            crop_bounds=tuple(meta.attrs['crop_bounds']),
            bf_center=tuple(meta.attrs['bf_center']),
            bf_radius=int(meta.attrs['bf_radius']),
            reference_smoothing=float(meta.attrs['reference_smoothing']),
            version=str(meta.attrs.get('version', '1.0'))
        )

    def get_metadata(self) -> CacheMetadata:
        """Get cache metadata"""
        if self._metadata is None:
            self._load_metadata()
        return self._metadata

    def get_step_completed(self) -> int:
        """Get the last completed step number"""
        if self.h5file is None:
            raise RuntimeError("Cache file not open")
        return int(self.h5file['metadata'].attrs['step_completed'])

    def set_step_completed(self, step: int):
        """
        Mark a step as completed

        Parameters
        ----------
        step : int
            Step number (1-5)
        """
        if self.h5file is None:
            raise RuntimeError("Cache file not open")

        self.h5file['metadata'].attrs['step_completed'] = step
        self.h5file.flush()

        if self._metadata is not None:
            self._metadata.step_completed = step

        print(f"✅ Marked Step {step} as completed in cache")

    def write_bigft(self, bigft_data: np.ndarray, output_slice: Tuple[slice, ...]):
        """
        Write bigFT chunk to cache

        Parameters
        ----------
        bigft_data : ndarray
            Complex FFT data chunk
        output_slice : tuple of slices
            Where to write in the full array (sy_slice, sx_slice, dy_slice, dx_slice)
        """
        if self.h5file is None:
            raise RuntimeError("Cache file not open")

        self.h5file['bigFT/data'][output_slice] = bigft_data
        # Note: flush called periodically, not every write

    def read_bigft(self, input_slice: Optional[Tuple[slice, ...]] = None) -> np.ndarray:
        """
        Read bigFT from cache

        Parameters
        ----------
        input_slice : tuple of slices, optional
            Which chunk to read. If None, reads entire array

        Returns
        -------
        ndarray : complex128
            bigFT data
        """
        if self.h5file is None:
            raise RuntimeError("Cache file not open")

        if 'bigFT' not in self.h5file:
            raise KeyError("bigFT data not found in cache file")

        if input_slice is None:
            return self.h5file['bigFT/data'][:]
        else:
            return self.h5file['bigFT/data'][input_slice]

    def write_correlations(self, corr_data: np.ndarray, output_slice: Tuple[slice, ...]):
        """
        Write correlations chunk to cache

        Parameters
        ----------
        corr_data : ndarray
            Correlation data chunk
        output_slice : tuple of slices
            Where to write in the full array
        """
        if self.h5file is None:
            raise RuntimeError("Cache file not open")

        self.h5file['correlations/data'][output_slice] = corr_data
        # Note: flush called periodically, not every write

    def read_correlations(self, input_slice: Optional[Tuple[slice, ...]] = None) -> np.ndarray:
        """
        Read correlations from cache

        Parameters
        ----------
        input_slice : tuple of slices, optional
            Which chunk to read. If None, reads entire array

        Returns
        -------
        ndarray : float64
            Correlation data
        """
        if self.h5file is None:
            raise RuntimeError("Cache file not open")

        if 'correlations' not in self.h5file:
            raise KeyError("correlations data not found in cache file")

        if input_slice is None:
            return self.h5file['correlations/data'][:]
        else:
            return self.h5file['correlations/data'][input_slice]

    def has_bigft(self) -> bool:
        """Check if bigFT data exists in cache"""
        if self.h5file is None:
            return False
        return 'bigFT' in self.h5file

    def has_correlations(self) -> bool:
        """Check if correlations data exists in cache"""
        if self.h5file is None:
            return False
        return 'correlations' in self.h5file

    def delete_bigft(self):
        """
        Delete bigFT data to save space (~34 GB)

        Call this after Step 3 completes if space is limited.
        Removes the ability to recompute correlations without recalculating bigFT.
        """
        if self.h5file is None:
            raise RuntimeError("Cache file not open")

        if 'bigFT' not in self.h5file:
            print("⚠️  bigFT already deleted or never created")
            return

        # Get size before deletion
        shape = self.h5file['bigFT/data'].shape
        size_gb = np.prod(shape) * 16 / (1024**3)

        # Delete the group
        del self.h5file['bigFT']
        self.h5file.flush()

        print(f"✅ Deleted bigFT from cache (saved {size_gb:.2f} GB)")
        print(f"   Note: Cannot recompute correlations without bigFT")

    def flush(self):
        """Flush pending writes to disk"""
        if self.h5file is not None:
            self.h5file.flush()

    def close(self):
        """Close cache file"""
        if self.h5file is not None:
            self.h5file.close()
            self.h5file = None
            print(f"📁 Closed cache file: {self.cache_filename}")

    def get_cache_size_gb(self) -> float:
        """Get total cache file size in GB"""
        if not os.path.exists(self.cache_path):
            return 0.0
        return os.path.getsize(self.cache_path) / (1024**3)

    @classmethod
    def from_existing(cls, cache_path: str) -> 'SuperResCacheManager':
        """
        Open existing cache file

        Parameters
        ----------
        cache_path : str
            Path to existing cache file

        Returns
        -------
        SuperResCacheManager
            Manager instance with opened cache file
        """
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cache file not found: {cache_path}")

        # Parse timestamp from filename
        # Format: {basename}_superres_cache_{timestamp}.h5
        filename = Path(cache_path).stem
        parts = filename.split('_superres_cache_')

        if len(parts) != 2:
            raise ValueError(f"Invalid cache filename format: {filename}")

        timestamp = parts[1]

        # Get EMD path from cache metadata
        with h5py.File(cache_path, 'r') as f:
            emd_path = str(f['metadata'].attrs['linked_emd_path'])

        # Create manager instance
        manager = cls(emd_path, timestamp=timestamp, create_new=False)

        return manager

    @staticmethod
    def find_cache_files(emd_path: str) -> List[str]:
        """
        Find all cache files associated with an EMD file

        Parameters
        ----------
        emd_path : str
            Path to EMD file

        Returns
        -------
        list of str
            Paths to cache files, sorted by timestamp (newest first)
        """
        emd_path = Path(emd_path).resolve()
        emd_basename = emd_path.stem
        emd_dir = emd_path.parent

        # Search pattern: {basename}_superres_cache_*.h5
        pattern = f"{emd_basename}_superres_cache_*.h5"

        cache_files = list(emd_dir.glob(pattern))

        # Sort by modification time (newest first)
        cache_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        return [str(p) for p in cache_files]

    @staticmethod
    def get_cache_info(cache_path: str) -> Dict[str, Any]:
        """
        Get information about a cache file without fully opening it

        Parameters
        ----------
        cache_path : str
            Path to cache file

        Returns
        -------
        dict
            Information about cache: step_completed, size_gb, timestamp, etc.
        """
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cache file not found: {cache_path}")

        size_gb = os.path.getsize(cache_path) / (1024**3)

        with h5py.File(cache_path, 'r') as f:
            meta = f['metadata']

            info = {
                'cache_path': cache_path,
                'filename': Path(cache_path).name,
                'size_gb': size_gb,
                'step_completed': int(meta.attrs['step_completed']),
                'timestamp': str(meta.attrs['timestamp']),
                'linked_emd_path': str(meta.attrs['linked_emd_path']),
                'cropped_data_shape': tuple(meta.attrs['cropped_data_shape']),
                'has_bigft': 'bigFT' in f,
                'has_correlations': 'correlations' in f,
            }

        return info

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
        return False


# Utility functions

def cleanup_cache_file(cache_path: str) -> bool:
    """
    Delete a cache file

    Parameters
    ----------
    cache_path : str
        Path to cache file to delete

    Returns
    -------
    bool
        True if deleted successfully
    """
    try:
        if os.path.exists(cache_path):
            size_gb = os.path.getsize(cache_path) / (1024**3)
            os.remove(cache_path)
            print(f"🗑️  Deleted cache file: {Path(cache_path).name} ({size_gb:.2f} GB)")
            return True
        else:
            print(f"⚠️  Cache file not found: {cache_path}")
            return False
    except Exception as e:
        print(f"❌ Failed to delete cache file: {e}")
        return False


def cleanup_all_cache_files(emd_path: str) -> int:
    """
    Delete all cache files for an EMD file

    Parameters
    ----------
    emd_path : str
        Path to EMD file

    Returns
    -------
    int
        Number of files deleted
    """
    cache_files = SuperResCacheManager.find_cache_files(emd_path)

    deleted = 0
    for cache_file in cache_files:
        if cleanup_cache_file(cache_file):
            deleted += 1

    return deleted


if __name__ == "__main__":
    # Example usage and testing
    print("=== SuperResCacheManager Test ===\n")

    # Simulate cache creation
    test_emd_path = "/path/to/mydata.emd"
    test_crop_info = {
        'center': (128.5, 127.3),
        'radius': 32,
        'bounds': (96, 160, 95, 159)  # y1, y2, x1, x2
    }
    test_original_shape = (256, 256, 256, 256)
    test_cropped_shape = (256, 256, 64, 64)

    print("Example: Creating cache file")
    print(f"EMD: {test_emd_path}")
    print(f"Cropped shape: {test_cropped_shape}")
    print(f"\nCache would be created at:")

    # Don't actually create file in test
    manager = SuperResCacheManager(test_emd_path, timestamp="20250107_153045", create_new=True)
    print(f"  {manager.cache_path}")
    print(f"\nCache file operations:")
    print(f"  - create_cache_file(): Initialize structure")
    print(f"  - write_bigft(): Write FFT chunks")
    print(f"  - read_bigft(): Read FFT chunks")
    print(f"  - write_correlations(): Write correlation chunks")
    print(f"  - read_correlations(): Read correlation chunks")
    print(f"  - set_step_completed(): Track progress")
    print(f"  - delete_bigft(): Save ~34 GB")
    print(f"  - close(): Clean shutdown")
    print(f"\n✅ Cache manager module loaded successfully")
