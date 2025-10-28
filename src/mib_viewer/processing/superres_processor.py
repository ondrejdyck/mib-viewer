"""4D STEM Super-Resolution Bright Field Reconstruction

Implementation of the super-resolution algorithm that exploits shifted images
from different detector pixels to achieve resolution beyond the probe size limit.

Algorithm reference: 4D-STEM-Super-Resolution-Algorithm.md
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple, Optional, Callable
import time

# Import adaptive chunking and cache manager
from ..io.adaptive_chunking import AdaptiveChunkCalculator, ChunkingResult
from .superres_cache import SuperResCacheManager


class SuperResProcessor:
    """Processor for super-resolution bright field reconstruction from 4D STEM data."""

    def __init__(self, progress_callback: Optional[Callable[[str], None]] = None):
        """Initialize the super-resolution processor.

        Parameters
        ----------
        progress_callback : callable, optional
            Function to call with progress updates (takes string message)
        """
        self.progress_callback = progress_callback

    def log(self, message: str):
        """Log a message via progress callback if available."""
        if self.progress_callback:
            self.progress_callback(message)
        else:
            print(message)

    def find_bf_center(self, data_4d: np.ndarray,
                       scan_pos: Tuple[int, int] = (0, 0),
                       threshold_fraction: float = 0.1) -> Tuple[float, float]:
        """Find bright field disk center using thresholded center of mass.

        Parameters
        ----------
        data_4d : ndarray, shape (scan_y, scan_x, det_y, det_x)
            4D STEM dataset
        scan_pos : tuple of int
            Scan position to use for center finding (default: (0, 0))
        threshold_fraction : float
            Fraction of maximum intensity to use as threshold (default: 0.1)
            Pixels below this are excluded from COM calculation

        Returns
        -------
        center_y, center_x : float
            Center of mass coordinates in detector space
        """
        self.log(f"Finding BF center from scan position {scan_pos}...")

        # Extract single diffraction pattern
        frame = data_4d[scan_pos[0], scan_pos[1], :, :]

        # Apply threshold to isolate BF disk
        threshold = threshold_fraction * np.max(frame)
        masked_frame = np.where(frame > threshold, frame, 0)

        # Create coordinate grids
        det_y, det_x = frame.shape
        v, u = np.meshgrid(np.arange(det_y), np.arange(det_x), indexing='ij')

        # Center of mass calculation on masked (thresholded) data
        sum_intensity = np.sum(masked_frame)
        if sum_intensity > 0:
            mean_y = np.sum(masked_frame * v) / sum_intensity
            mean_x = np.sum(masked_frame * u) / sum_intensity
        else:
            # Fallback to detector center if threshold too high
            mean_y = det_y / 2.0
            mean_x = det_x / 2.0
            self.log(f"  Warning: No pixels above threshold, using detector center")

        self.log(f"  BF center found at: ({mean_y:.2f}, {mean_x:.2f})")
        return mean_y, mean_x

    def crop_to_bf_region(self, data_4d: np.ndarray,
                          center_y: float, center_x: float,
                          radius: int = 32) -> np.ndarray:
        """Crop 4D dataset to bright field region.

        Parameters
        ----------
        data_4d : ndarray, shape (scan_y, scan_x, det_y, det_x)
            Full 4D STEM dataset
        center_y, center_x : float
            BF disk center coordinates
        radius : int
            Half-width of region to extract (default: 32)

        Returns
        -------
        cropped : ndarray, shape (scan_y, scan_x, 2*radius, 2*radius)
            Cropped dataset focused on BF disk
        """
        self.log(f"Cropping to BF region (radius={radius})...")

        cy, cx = int(center_y), int(center_x)

        # Clip bounds to ensure valid crop region
        det_y, det_x = data_4d.shape[2], data_4d.shape[3]
        y1 = max(0, cy - radius)
        y2 = min(det_y, cy + radius)
        x1 = max(0, cx - radius)
        x2 = min(det_x, cx + radius)

        cropped = data_4d[:, :, y1:y2, x1:x2]

        self.log(f"  Cropped shape: {cropped.shape}")
        self.log(f"  Crop bounds: y=[{y1}:{y2}], x=[{x1}:{x2}]")
        return cropped

    def compute_bigft_to_cache(self,
                               data_4d_full,  # Memory-mapped or array-like
                               crop_bounds: Tuple[int, int, int, int],  # (y1, y2, x1, x2)
                               cache_manager: SuperResCacheManager,
                               reference_smoothing: float = 0.5) -> np.ndarray:
        """
        Compute 4D FFT over scan dimensions and write to cache file.

        Uses adaptive chunking to handle memory constraints:
        - Works directly with memory-mapped data (never loads full array)
        - Chunks detector dimensions to fit in memory
        - Writes directly to cache file

        Parameters
        ----------
        data_4d_full : array-like (memory-mapped), shape (scan_y, scan_x, det_y, det_x)
            Full 4D STEM dataset (memory-mapped, not loaded into RAM)
        crop_bounds : tuple of int
            Detector crop bounds (y1, y2, x1, x2) for BF region
        cache_manager : SuperResCacheManager
            Cache file manager for writing bigFT
        reference_smoothing : float
            Gaussian sigma for smoothing reference image (default: 0.5)

        Returns
        -------
        reference_image : ndarray, shape (scan_y, scan_x)
            The reference image (for diagnostics and FFT display)
        """
        self.log("=" * 60)
        self.log("Computing 4D FFT over scan dimensions")
        self.log("=" * 60)
        t0 = time.time()

        # Get shapes
        sy, sx = data_4d_full.shape[0], data_4d_full.shape[1]
        y1, y2, x1, x2 = crop_bounds
        crop_dy = y2 - y1
        crop_dx = x2 - x1
        
        cropped_shape = (sy, sx, crop_dy, crop_dx)
        w_y, w_x = crop_dy // 2, crop_dx // 2  # Center detector pixel in cropped coords

        # Create reference image (load only central detector pixel - minimal memory)
        self.log(f"Creating reference from central detector pixel (σ={reference_smoothing})...")
        central_pixel_data = data_4d_full[:, :, y1 + w_y, x1 + w_x]  # Shape: (sy, sx)
        central_image = gaussian_filter(central_pixel_data, reference_smoothing).astype(float)

        # Analyze memory requirements
        self.log("Analyzing memory requirements...")
        dataset_size_gb = (sy * sx * crop_dy * crop_dx * 2) / (1024**3)  # uint16 input
        bigft_size_gb = (sy * sx * crop_dy * crop_dx * 16) / (1024**3)   # complex128 output

        self.log(f"  Cropped dataset size: {dataset_size_gb:.2f} GB")
        self.log(f"  bigFT size: {bigft_size_gb:.2f} GB")
        self.log(f"  Crop bounds: y=[{y1}:{y2}], x=[{x1}:{x2}]")

        # Use adaptive chunking with detector dimension chunking
        self.log("Calculating chunking strategy...")
        chunk_calculator = AdaptiveChunkCalculator(
            conservative_mode=True,
            max_workers=1  # Force single-threaded for I/O efficiency
        )

        chunking_result = chunk_calculator.calculate_chunking_strategy(
            file_shape=cropped_shape,
            chunk_detector_dims=True  # Chunk detector dimensions, preserve scan dims
        )

        self.log(f"  Strategy: {chunking_result.strategy}")
        self.log(f"  Chunk dimensions: {chunking_result.chunk_dims}")
        self.log(f"  Total chunks: {chunking_result.total_chunks}")
        self.log(f"  Chunk size: {chunking_result.chunk_size_mb:.1f} MB")

        # Generate chunk queue
        chunks = chunk_calculator.generate_chunk_queue(chunking_result)

        # Process chunks: FFT and write to cache
        self.log(f"Processing {len(chunks)} detector chunks...")

        for i, chunk in enumerate(chunks):
            # Extract chunk from ORIGINAL data with crop offset applied
            # chunk.input_slice is relative to cropped coords, need to offset to full coords
            sy_slice, sx_slice, dy_slice, dx_slice = chunk.input_slice
            
            # Adjust detector slices to account for crop offset
            dy_start = dy_slice.start + y1 if dy_slice.start is not None else y1
            dy_stop = (dy_slice.stop + y1) if dy_slice.stop is not None else y2
            dx_start = dx_slice.start + x1 if dx_slice.start is not None else x1
            dx_stop = (dx_slice.stop + x1) if dx_slice.stop is not None else x2
            
            # Load chunk from memory-mapped data (only this chunk enters RAM)
            chunk_data = data_4d_full[sy_slice, sx_slice, dy_start:dy_stop, dx_start:dx_stop]

            # Compute FFT over scan dimensions (axes 0, 1)
            # Input: (sy, sx, chunk_dy, chunk_dx)
            # Output: (sy, sx, chunk_dy, chunk_dx) complex128
            bigft_chunk = np.fft.fft2(chunk_data, axes=(0, 1)).astype(np.complex128)

            # Write to cache using output_slice (work coordinates)
            cache_manager.write_bigft(bigft_chunk, chunk.output_slice)

            # Free memory immediately
            del chunk_data, bigft_chunk

            # Progress reporting
            progress_pct = int(((i + 1) / len(chunks)) * 100)
            self.log(f"  Chunk {i+1}/{len(chunks)} ({progress_pct}%): wrote to cache")

            # Periodic flush
            if (i + 1) % 10 == 0:
                cache_manager.flush()

        # Final flush
        cache_manager.flush()

        # Mark Step 2 as completed
        cache_manager.set_step_completed(2)

        elapsed = time.time() - t0
        self.log("=" * 60)
        self.log(f"4D FFT computation completed in {elapsed:.1f}s")
        self.log(f"bigFT written to cache: {cache_manager.cache_filename}")
        self.log(f"Cache size: {cache_manager.get_cache_size_gb():.2f} GB")
        self.log("=" * 60)

        return central_image

    def compute_correlations_from_cache(self,
                                        data_4d_full,  # Memory-mapped
                                        crop_bounds: Tuple[int, int, int, int],
                                        cache_manager: SuperResCacheManager,
                                        reference_image: np.ndarray) -> None:
        """
        Compute cross-correlations from cached bigFT and write to cache.

        This reads bigFT in chunks, computes correlations, normalizes, and writes back to cache.
        Never loads the full arrays into memory.

        Parameters
        ----------
        data_4d_full : array-like (memory-mapped)
            Full 4D STEM dataset (for normalization computation)
        crop_bounds : tuple of int
            Detector crop bounds (y1, y2, x1, x2)
        cache_manager : SuperResCacheManager
            Cache manager with bigFT data
        reference_image : ndarray, shape (scan_y, scan_x)
            Reference image (central detector pixel, smoothed)
        """
        self.log("=" * 60)
        self.log("Computing cross-correlations from cached bigFT")
        self.log("=" * 60)
        t0 = time.time()

        # Get metadata
        metadata = cache_manager.get_metadata()
        sy, sx, dy, dx = metadata.cropped_data_shape
        y1, y2, x1, x2 = crop_bounds

        # Compute reference FFT (small, fits in memory)
        self.log("Computing reference FFT...")
        central_slice = np.fft.fft2(reference_image)

        # Compute normalization factors (per detector pixel)
        # norms = np.sum(central_image**2) * np.sum(data_4d_cropped**2, axis=(0, 1))
        self.log("Computing normalization factors...")
        norm_ref = np.sum(reference_image**2)
        
        # Compute sum of squares for each detector pixel in chunks
        norms = np.zeros((dy, dx), dtype=np.float64)
        
        # Use adaptive chunking for detector dimensions
        chunk_calculator = AdaptiveChunkCalculator(
            conservative_mode=True,
            max_workers=1
        )

        chunking_result = chunk_calculator.calculate_chunking_strategy(
            file_shape=(sy, sx, dy, dx),
            chunk_detector_dims=True
        )

        chunks = chunk_calculator.generate_chunk_queue(chunking_result)
        
        for i, chunk in enumerate(chunks):
            # Extract chunk from original data with crop offset
            sy_slice, sx_slice, dy_slice, dx_slice = chunk.input_slice
            
            dy_start = dy_slice.start + y1 if dy_slice.start is not None else y1
            dy_stop = (dy_slice.stop + y1) if dy_slice.stop is not None else y2
            dx_start = dx_slice.start + x1 if dx_slice.start is not None else x1
            dx_stop = (dx_slice.stop + x1) if dx_slice.stop is not None else x2
            
            chunk_data = data_4d_full[sy_slice, sx_slice, dy_start:dy_stop, dx_start:dx_stop]
            
            # Sum of squares over scan dimensions
            chunk_norms = np.sum(chunk_data.astype(np.float64)**2, axis=(0, 1))
            
            # Store in correct position
            _, _, dy_out, dx_out = chunk.output_slice
            norms[dy_out, dx_out] = chunk_norms
            
            del chunk_data
        
        # Complete normalization: multiply by reference norm
        norms = norm_ref * norms
        norms = np.where(norms < 1e-9, 1, norms)  # Avoid division by zero
        
        self.log(f"  Normalization factors computed, shape: {norms.shape}")

        # Now process correlations
        self.log(f"Processing {len(chunks)} detector chunks for correlations...")

        for i, chunk in enumerate(chunks):
            # Read bigFT chunk from cache
            bigft_chunk = cache_manager.read_bigft(chunk.input_slice)

            # Multiply by conjugate of reference (broadcast)
            corr_chunk = bigft_chunk * np.conj(central_slice)[:, :, np.newaxis, np.newaxis]

            # IFFT back to real space
            corr_chunk = np.fft.ifft2(corr_chunk, axes=(0, 1))
            
            # Square and take absolute value (MUST match process_full order!)
            corr_chunk = corr_chunk ** 2  # Square first (complex)
            corr_chunk = np.abs(corr_chunk)  # Then absolute value

            # NOTE: Do NOT apply fftshift here! It will be applied when reading
            # the full correlation array. Shifting individual chunks breaks the
            # spatial correspondence with the slice coordinates.

            # Normalize using precomputed norms
            _, _, dy_out, dx_out = chunk.output_slice
            chunk_norms = norms[dy_out, dx_out]
            corr_chunk = corr_chunk / chunk_norms  # Broadcast division

            # Write to cache (without fftshift)
            cache_manager.write_correlations(corr_chunk, chunk.output_slice)

            # Free memory
            del bigft_chunk, corr_chunk

            # Progress reporting
            progress_pct = int(((i + 1) / len(chunks)) * 100)
            self.log(f"  Chunk {i+1}/{len(chunks)} ({progress_pct}%): wrote correlations to cache")

            # Periodic flush
            if (i + 1) % 10 == 0:
                cache_manager.flush()

        # Final flush
        cache_manager.flush()

        # Mark Step 3 as completed
        cache_manager.set_step_completed(3)

        elapsed = time.time() - t0
        self.log("=" * 60)
        self.log(f"Correlations computed in {elapsed:.1f}s")
        self.log("=" * 60)

    def compute_cross_correlations(self,
                                   data_4d_cropped: np.ndarray,
                                   reference_smoothing: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Compute cross-correlations between all detector pixels and reference.

        This is the core of the algorithm:
        1. FFT over scan dimensions for all detector pixels
        2. Create smoothed reference from central detector pixel
        3. Compute cross-correlation in Fourier space (multiply by conjugate)
        4. IFFT back to real space
        5. Take squared magnitude as correlation strength

        MEMORY OPTIMIZED: Reuses arrays to avoid keeping multiple large copies in memory.

        Parameters
        ----------
        data_4d_cropped : ndarray, shape (scan_y, scan_x, det_y, det_x)
            Cropped 4D dataset centered on BF disk
        reference_smoothing : float
            Gaussian sigma for smoothing reference image (default: 0.5)

        Returns
        -------
        correlations : ndarray, shape (scan_y, scan_x, det_y, det_x)
            Cross-correlation maps for each detector pixel
        reference_image : ndarray, shape (scan_y, scan_x)
            The reference image used (for diagnostics)
        """
        self.log("Computing cross-correlations...")
        t0 = time.time()

        sy, sx, dy, dx = data_4d_cropped.shape
        w_y, w_x = dy // 2, dx // 2  # Center detector pixel

        # Step 1: FFT over scan dimensions for all detector pixels
        self.log("  Step 1/5: FFT over scan dimensions...")
        result = np.fft.fft2(data_4d_cropped, axes=(0, 1))  # complex128

        # Step 2: Create reference from central detector pixel with smoothing
        self.log(f"  Step 2/5: Creating reference (central pixel with σ={reference_smoothing})...")
        central_image = gaussian_filter(data_4d_cropped[:, :, w_y, w_x], reference_smoothing).astype(float)
        central_slice = np.fft.fft2(central_image)

        # Step 3: Cross-correlation in Fourier space (IN-PLACE to save memory)
        # Multiply by conjugate of reference (broadcast to all detector pixels)
        self.log("  Step 3/5: Cross-correlation in Fourier space...")
        result *= np.conj(central_slice)[:, :, np.newaxis, np.newaxis]

        # Step 4: IFFT back to real space (reuse 'result' variable)
        self.log("  Step 4/5: IFFT to real space...")
        result = np.fft.ifft2(result, axes=(0, 1))  # Still complex128
        result = result ** 2  # Square
        result = np.abs(result)  # Take absolute value -> now float64

        # Step 5: fftshift
        self.log("  Step 5/5: fftshift and normalization...")
        result = np.fft.fftshift(np.abs(result), axes=(0, 1))

        # Normalize by image energies (compute normalization factor efficiently)
        norm_factor = np.sum(central_image**2)
        norms = norm_factor * np.sum(data_4d_cropped**2, axis=(0, 1))
        # Avoid division by zero
        norms = np.where(norms < 1e-9, 1, norms)
        result /= norms  # In-place division

        elapsed = time.time() - t0
        self.log(f"  Cross-correlation completed in {elapsed:.2f}s")

        return result, central_image

    def find_shift_maps(self,
                       correlations: np.ndarray,
                       subpixel_refine: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find shift maps from cross-correlation results.

        For each detector pixel, finds the peak in its correlation map,
        which indicates the shift of that pixel's image relative to the reference.

        Parameters
        ----------
        correlations : ndarray, shape (scan_y, scan_x, det_y, det_x)
            Cross-correlation maps
        subpixel_refine : bool
            Whether to use quadratic fitting for sub-pixel accuracy (default: True)

        Returns
        -------
        xm_sub : ndarray, shape (det_y, det_x)
            X (scan_y direction) shift for each detector pixel
        ym_sub : ndarray, shape (det_y, det_x)
            Y (scan_x direction) shift for each detector pixel
        im_sub : ndarray, shape (det_y, det_x)
            Correlation strength at peak (quality metric)
        """
        self.log("Finding shift maps...")
        t0 = time.time()

        # Apply fftshift to center the correlations (if not already done)
        # This is needed when correlations come from cache without fftshift applied
        correlations = np.fft.fftshift(correlations, axes=(0, 1))

        # Step 1: Find integer maximum for each detector pixel (matches Cell 14)
        self.log("  Finding integer peak positions...")
        maxarg = np.argmax(correlations.reshape((correlations.shape[0]*correlations.shape[1],
                                                  correlations.shape[2], correlations.shape[3])), axis=0)
        XSy, XSx = np.unravel_index(maxarg, (correlations.shape[0], correlations.shape[1]))

        if not subpixel_refine:
            # Extract quality map
            i_idx, j_idx = np.meshgrid(np.arange(correlations.shape[2]),
                                       np.arange(correlations.shape[3]), indexing='ij')
            im_sub = correlations[XSy, XSx, i_idx, j_idx]
            elapsed = time.time() - t0
            self.log(f"  Shift maps found (integer only) in {elapsed:.2f}s")
            return XSy.astype(float), XSx.astype(float), im_sub

        # Step 2: Sub-pixel refinement using quadratic fit (matches Cell 18 EXACTLY)
        self.log("  Performing sub-pixel quadratic refinement...")
        xm_sub = np.zeros(XSx.shape, dtype=float)
        ym_sub = np.zeros(XSx.shape, dtype=float)
        im_sub = np.zeros(XSx.shape, dtype=float)

        for i in range(xm_sub.shape[0]):
            for j in range(xm_sub.shape[1]):
                u = XSy[i, j]
                v = XSx[i, j]
                ym = correlations[u, v, i, j]

                if (1 <= u < correlations.shape[0]-1) and (1 <= v < correlations.shape[1]-1):
                    dx = correlations[u, v+1, i, j] - correlations[u, v-1, i, j]
                    dy = correlations[u+1, v, i, j] - correlations[u-1, v, i, j]
                    dxx = correlations[u, v+1, i, j] - 2*ym + correlations[u, v-1, i, j]
                    dyy = correlations[u+1, v, i, j] - 2*ym + correlations[u-1, v, i, j]

                    # Exactly as in notebook
                    denom_x = dxx if dxx <= 1e-3 else 1e6
                    denom_y = dyy if dyy <= 1e-3 else 1e6

                    xm_sub[i, j] = u - dy / denom_y / 2
                    ym_sub[i, j] = v - dx / denom_x / 2
                    im_sub[i, j] = ym - 0*(dx*dx/8/denom_x + dy*dy/8/denom_y)
                else:
                    # Outside valid region
                    xm_sub[i, j] = u
                    ym_sub[i, j] = v
                    im_sub[i, j] = ym

        elapsed = time.time() - t0
        self.log(f"  Shift maps found with sub-pixel refinement in {elapsed:.2f}s")

        return xm_sub, ym_sub, im_sub

    def reconstruct_superres_from_memmap(self,
                                         data_4d_full,  # Memory-mapped
                                         crop_bounds: Tuple[int, int, int, int],
                                         xm_sub: np.ndarray,
                                         ym_sub: np.ndarray,
                                         im_sub: np.ndarray,
                                         fac: int = 4,
                                         pad: int = 16,
                                         lims: int = 16,
                                         thresh: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct super-resolution image from memory-mapped data.

        Parameters
        ----------
        data_4d_full : array-like (memory-mapped)
            Full 4D STEM dataset
        crop_bounds : tuple of int
            Detector crop bounds (y1, y2, x1, x2)
        xm_sub, ym_sub : ndarray (det_y, det_x)
            Sub-pixel shift estimates
        im_sub : ndarray (det_y, det_x)
            Quality metric
        fac : int
            Upscaling factor (default: 4)
        pad : int
            Padding (default: 16)
        lims : int
            Detector radius (default: 16)
        thresh : float
            Quality threshold (default: 0.8)

        Returns
        -------
        big : ndarray
            Super-resolved image
        norm : ndarray
            Normalization map
        """
        self.log(f"Reconstructing super-resolution image ({fac}x)...")
        t0 = time.time()

        y1, y2, x1, x2 = crop_bounds
        sy, sx = data_4d_full.shape[0], data_4d_full.shape[1]
        crop_dy = y2 - y1
        crop_dx = x2 - x1

        # Exactly as in notebook
        x, y = np.meshgrid(np.arange(sx), np.arange(sy))
        big = np.zeros((sy*fac+pad, sx*fac+pad), dtype=float)
        norm = np.zeros_like(big)

        center_x = crop_dy // 2
        center_y = crop_dx // 2

        pixels_used = 0
        pixels_skipped = 0

        # Adjust lims to not exceed detector bounds
        max_lims_y = min(lims, center_x, crop_dy - center_x - 1)
        max_lims_x = min(lims, center_y, crop_dx - center_y - 1)
        
        if max_lims_y < lims or max_lims_x < lims:
            self.log(f"  Warning: Detector region too small for requested radius {lims}")
            self.log(f"  Using reduced radius: y={max_lims_y}, x={max_lims_x}")
            self.log(f"  Detector size: {crop_dy}×{crop_dx}, center: ({center_x}, {center_y})")

        for i in range(-max_lims_x, max_lims_x + 1):
            for j in range(-max_lims_y, max_lims_y + 1):
                # Check bounds in shift map arrays
                shift_y_idx = center_x + j
                shift_x_idx = center_y + i
                
                if (shift_y_idx < 0 or shift_y_idx >= crop_dy or 
                    shift_x_idx < 0 or shift_x_idx >= crop_dx):
                    continue
                
                # Extract the slice for the current shift (load only one detector pixel at a time)
                det_y_idx = y1 + shift_y_idx
                det_x_idx = x1 + shift_x_idx
                tmp1 = data_4d_full[:, :, det_y_idx, det_x_idx].astype(float)

                # Use xm_sub and ym_sub for shifts
                x_shift = ym_sub[shift_y_idx, shift_x_idx] - sx / 2 - pad/2/fac
                y_shift = xm_sub[shift_y_idx, shift_x_idx] - sy / 2 - pad/2/fac
                fit_qual = im_sub[shift_y_idx, shift_x_idx]

                if fit_qual > thresh:
                    xcos = (x - x_shift) * fac
                    ycos = (y - y_shift) * fac

                    # Clip the computed coordinates to remain within bounds
                    xcos = np.clip(np.round(xcos).astype(int), 0, big.shape[1] - 1)
                    ycos = np.clip(np.round(ycos).astype(int), 0, big.shape[0] - 1)

                    # Update the transformation and normalization arrays
                    big[ycos, xcos] += tmp1
                    norm[ycos, xcos] += 1
                    pixels_used += 1
                else:
                    pixels_skipped += 1

        # Normalize
        invalid = np.where(norm < 1)
        big[invalid] = 0
        valid = np.where(norm > 0)
        big[valid] = big[valid] / norm[valid]

        elapsed = time.time() - t0
        self.log(f"  Reconstruction complete in {elapsed:.2f}s")
        self.log(f"  Detector pixels used: {pixels_used}")
        self.log(f"  Detector pixels skipped (quality < {thresh}): {pixels_skipped}")
        self.log(f"  Output shape: {big.shape}")

        return big, norm

    def reconstruct_superres(self,
                            trim4d: np.ndarray,
                            xm_sub: np.ndarray,
                            ym_sub: np.ndarray,
                            im_sub: np.ndarray,
                            fac: int = 4,
                            pad: int = 16,
                            lims: int = 16,
                            thresh: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct super-resolution image - EXACT copy of notebook Cell 19.

        Parameters
        ----------
        trim4d : ndarray (scan_y, scan_x, det_y, det_x)
            Cropped 4D STEM dataset
        xm_sub, ym_sub : ndarray (det_y, det_x)
            Sub-pixel shift estimates
        im_sub : ndarray (det_y, det_x)
            Quality metric
        fac : int
            Upscaling factor (default: 4)
        pad : int
            Padding (default: 16)
        lims : int
            Detector radius (default: 16)
        thresh : float
            Quality threshold (default: 0.8)

        Returns
        -------
        big : ndarray
            Super-resolved image
        norm : ndarray
            Normalization map
        """
        self.log(f"Reconstructing super-resolution image ({fac}x)...")
        t0 = time.time()

        # Exactly as in notebook
        x, y = np.meshgrid(np.arange(trim4d.shape[1]), np.arange(trim4d.shape[0]))
        big = np.zeros((trim4d.shape[0]*fac+pad, trim4d.shape[1]*fac+pad), dtype=float)
        norm = np.zeros_like(big)

        center_x = trim4d.shape[2] // 2
        center_y = trim4d.shape[3] // 2

        pixels_used = 0
        pixels_skipped = 0

        for i in range(-lims, lims):
            for j in range(-lims, lims):
                # Extract the slice for the current shift
                tmp1 = trim4d[:, :, center_x + j, center_y + i].astype(float)

                # Use xm_sub and ym_sub for shifts
                x_shift = ym_sub[center_x + j, center_y + i] - trim4d.shape[1] / 2 - pad/2/fac
                y_shift = xm_sub[center_x + j, center_y + i] - trim4d.shape[0] / 2 - pad/2/fac
                fit_qual = im_sub[center_x + j, center_y + i]

                if fit_qual > thresh:
                    xcos = (x - x_shift) * fac
                    ycos = (y - y_shift) * fac

                    # Clip the computed coordinates to remain within bounds
                    xcos = np.clip(np.round(xcos).astype(int), 0, big.shape[1] - 1)
                    ycos = np.clip(np.round(ycos).astype(int), 0, big.shape[0] - 1)

                    # Update the transformation and normalization arrays
                    big[ycos, xcos] += tmp1
                    norm[ycos, xcos] += 1
                    pixels_used += 1
                else:
                    pixels_skipped += 1

        # Normalize
        invalid = np.where(norm < 1)
        big[invalid] = 0
        valid = np.where(norm > 0)
        big[valid] = big[valid] / norm[valid]

        elapsed = time.time() - t0
        self.log(f"  Reconstruction complete in {elapsed:.2f}s")
        self.log(f"  Detector pixels used: {pixels_used}")
        self.log(f"  Detector pixels skipped (quality < {thresh}): {pixels_skipped}")
        self.log(f"  Output shape: {big.shape}")

        return big, norm

    def process_full(self,
                    data_4d: np.ndarray,
                    bf_center: Optional[Tuple[float, float]] = None,
                    bf_radius: int = 32,
                    reference_smoothing: float = 0.5,
                    upscale_factor: int = 4,
                    detector_radius: int = 16,
                    quality_threshold: float = 0.8) -> dict:
        """Complete super-resolution reconstruction pipeline.

        Parameters
        ----------
        data_4d : ndarray, shape (scan_y, scan_x, det_y, det_x)
            4D STEM dataset
        bf_center : tuple of float, optional
            BF disk center (y, x). If None, auto-detect using center of mass
        bf_radius : int
            Radius for cropping BF region (default: 32)
        reference_smoothing : float
            Gaussian smoothing for reference image (default: 0.5)
        upscale_factor : int
            Resolution improvement factor (default: 4)
        detector_radius : int
            Detector pixels to use, ±radius from center (default: 16)
        quality_threshold : float
            Minimum correlation quality (default: 0.8)

        Returns
        -------
        results : dict
            Dictionary containing:
            - 'superres_image': Super-resolved bright field image
            - 'standard_bf': Standard BF image (central detector pixel)
            - 'shift_y': Vertical shift map
            - 'shift_x': Horizontal shift map
            - 'quality': Quality/correlation strength map
            - 'normalization': Contribution count map
            - 'reference_image': Reference image used
            - 'bf_center': BF center used (y, x)
        """
        self.log("=" * 60)
        self.log("4D STEM Super-Resolution Reconstruction")
        self.log("=" * 60)

        # Auto-detect BF center if not provided
        if bf_center is None:
            bf_center = self.find_bf_center(data_4d)
        else:
            self.log(f"Using provided BF center: {bf_center}")

        # Crop to BF region
        data_cropped = self.crop_to_bf_region(data_4d, bf_center[0], bf_center[1], bf_radius)

        # Compute cross-correlations
        correlations, reference_image = self.compute_cross_correlations(
            data_cropped, reference_smoothing
        )

        # Find shift maps (returns xm_sub, ym_sub, im_sub like notebook)
        xm_sub, ym_sub, im_sub = self.find_shift_maps(correlations, subpixel_refine=True)

        # Reconstruct super-resolution image
        superres, norm = self.reconstruct_superres(
            data_cropped, xm_sub, ym_sub, im_sub,
            fac=upscale_factor,
            lims=detector_radius,
            thresh=quality_threshold
        )

        # Create standard BF for comparison (central detector pixel)
        w_y = data_cropped.shape[2] // 2
        w_x = data_cropped.shape[3] // 2
        standard_bf = data_cropped[:, :, w_y, w_x]

        self.log("=" * 60)
        self.log("Reconstruction complete!")
        self.log("=" * 60)

        return {
            'superres_image': superres,
            'standard_bf': standard_bf,
            'shift_y': xm_sub,  # Keep old key names for compatibility
            'shift_x': ym_sub,
            'quality': im_sub,
            'normalization': norm,
            'reference_image': reference_image,
            'bf_center': bf_center,
            'data_cropped': data_cropped,  # For diagnostics
        }
