#!/usr/bin/env python3
"""Full ADF super-resolution using pairwise alignment approach.

This script implements the complete ADF super-resolution pipeline:
1. Extract annular region pixels
2. Use pairwise alignment to create composite reference (1 iteration)
3. Align all annular pixels to this reference
4. Reconstruct super-resolution ADF image
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from src.mib_viewer.processing.superres_processor import SuperResProcessor
import time


def shift_image_fourier(image: np.ndarray, shift_y: float, shift_x: float) -> np.ndarray:
    """Shift an image by sub-pixel amounts using Fourier shift theorem."""
    sy, sx = image.shape
    
    # Create frequency grids
    fy = np.fft.fftfreq(sy)
    fx = np.fft.fftfreq(sx)
    fy_grid, fx_grid = np.meshgrid(fy, fx, indexing='ij')
    
    # FFT of image
    image_fft = np.fft.fft2(image)
    
    # Apply phase shift
    phase_shift = np.exp(-2j * np.pi * (shift_y * fy_grid + shift_x * fx_grid))
    shifted_fft = image_fft * phase_shift
    
    # IFFT back
    shifted = np.fft.ifft2(shifted_fft).real
    
    return shifted


def find_pairwise_shift(pixel_A: np.ndarray, pixel_B: np.ndarray, 
                       smoothing: float = 0.5) -> tuple:
    """Find relative shift between two pixel images."""
    sy, sx = pixel_A.shape
    
    # Smooth reference
    ref_smoothed = gaussian_filter(pixel_A, smoothing).astype(float)
    
    # FFT correlation
    ref_fft = np.fft.fft2(ref_smoothed)
    pixel_B_fft = np.fft.fft2(pixel_B.astype(float))
    
    corr = pixel_B_fft * np.conj(ref_fft)
    corr = np.fft.ifft2(corr)
    corr = corr ** 2
    corr = np.abs(corr)
    corr = np.fft.fftshift(corr)
    
    # Normalize
    norm_factor = np.sum(ref_smoothed**2) * np.sum(pixel_B**2)
    if norm_factor > 1e-9:
        corr = corr / norm_factor
    
    # Find peak with subpixel refinement
    peak_idx = np.argmax(corr)
    peak_y, peak_x = np.unravel_index(peak_idx, corr.shape)
    
    if 1 <= peak_y < sy-1 and 1 <= peak_x < sx-1:
        c = corr[peak_y, peak_x]
        dx = corr[peak_y, peak_x+1] - corr[peak_y, peak_x-1]
        dy = corr[peak_y+1, peak_x] - corr[peak_y-1, peak_x]
        dxx = corr[peak_y, peak_x+1] - 2*c + corr[peak_y, peak_x-1]
        dyy = corr[peak_y+1, peak_x] - 2*c + corr[peak_y-1, peak_x]
        
        denom_x = dxx if dxx <= 1e-3 else 1e6
        denom_y = dyy if dyy <= 1e-3 else 1e6
        
        shift_y_sub = peak_y - dy / denom_y / 2
        shift_x_sub = peak_x - dx / denom_x / 2
        quality = c
    else:
        shift_y_sub = float(peak_y)
        shift_x_sub = float(peak_x)
        quality = corr[peak_y, peak_x]
    
    # Convert to centered coordinates
    shift_y = shift_y_sub - sy / 2
    shift_x = shift_x_sub - sx / 2
    
    return shift_y, shift_x, quality


def get_annular_pixel_coords(center_y: float, center_x: float, 
                             inner_radius: int, outer_radius: int) -> list:
    """Get list of detector pixel coordinates in annular region."""
    cy, cx = int(center_y), int(center_x)
    
    coords = []
    # Scan a square region that contains the annulus
    for dy in range(-outer_radius, outer_radius + 1):
        for dx in range(-outer_radius, outer_radius + 1):
            r = np.sqrt(dy**2 + dx**2)
            if inner_radius <= r <= outer_radius:
                coords.append((cy + dy, cx + dx))
    
    return coords


def test_full_adf_superres(emd_path: str,
                          bf_center: tuple = (127.6, 126.4),
                          inner_radius: int = 35,
                          outer_radius: int = 55,
                          smoothing: float = 0.5,
                          damping: float = 0.5):
    """Run full ADF super-resolution with pairwise alignment."""
    
    print("\n" + "=" * 70)
    print("Full ADF Super-Resolution with Pairwise Alignment")
    print("=" * 70 + "\n")
    
    # Load data (smaller subset for memory)
    print(f"Loading data from: {emd_path}")
    with h5py.File(emd_path, 'r') as f:
        data_4d = f['version_1/data/datacubes/datacube_000/data'][:128, :128, :, :]
    
    print(f"  Loaded 4D data shape: {data_4d.shape}")
    sy, sx = data_4d.shape[0], data_4d.shape[1]
    
    # Get annular pixel coordinates
    print(f"\nFinding annular pixels (r={inner_radius}-{outer_radius})...")
    annular_coords = get_annular_pixel_coords(bf_center[0], bf_center[1], 
                                              inner_radius, outer_radius)
    print(f"  Found {len(annular_coords)} pixels in annular region")
    
    # Step 1: Initialize with one opposing pair
    print("\n--- Step 1: Initialize with opposing pair ---")
    
    # Find left/right pair at mid-radius
    mid_radius = (inner_radius + outer_radius) // 2
    cy, cx = int(bf_center[0]), int(bf_center[1])
    
    right_coord = (cy, cx + mid_radius)
    left_coord = (cy, cx - mid_radius)
    
    print(f"  Right pixel: {right_coord}")
    print(f"  Left pixel: {left_coord}")
    
    right_img = data_4d[:, :, right_coord[0], right_coord[1]].astype(float)
    left_img = data_4d[:, :, left_coord[0], left_coord[1]].astype(float)
    
    # Find relative shift
    t0 = time.time()
    shift_y, shift_x, quality = find_pairwise_shift(right_img, left_img, smoothing)
    print(f"  Relative shift: ({shift_y:.2f}, {shift_x:.2f}), quality: {quality:.3f}")
    
    # Move each halfway
    shift_amount_y = shift_y * damping / 2
    shift_amount_x = shift_x * damping / 2
    
    right_aligned = shift_image_fourier(right_img, -shift_amount_y, -shift_amount_x)
    left_aligned = shift_image_fourier(left_img, shift_amount_y, shift_amount_x)
    
    # Create initial reference
    reference = (right_aligned + left_aligned) / 2
    reference = gaussian_filter(reference, smoothing)
    
    print(f"  Created initial reference from pair")
    print(f"  Time: {time.time() - t0:.2f}s")
    
    # Step 2: Align all annular pixels to this reference (ONE ITERATION)
    print("\n--- Step 2: Align all annular pixels to reference ---")
    print(f"  Processing {len(annular_coords)} pixels...")
    
    # We'll build a 4D array with just the annular pixels
    # But we need to track which detector position each corresponds to
    n_pixels = len(annular_coords)
    
    # Extract all annular pixel images
    annular_images = np.zeros((sy, sx, n_pixels), dtype=float)
    for i, (py, px) in enumerate(annular_coords):
        annular_images[:, :, i] = data_4d[:, :, py, px].astype(float)
        
        if (i + 1) % 100 == 0:
            print(f"    Extracted {i+1}/{n_pixels} pixels...")
    
    print(f"  Extracted all pixel images")
    
    # Now correlate each with the reference
    print(f"  Computing correlations with reference...")
    
    # We can reuse the SuperResProcessor correlation logic!
    # Create a fake 4D array where detector dims are (1, n_pixels)
    data_fake_4d = annular_images.reshape(sy, sx, 1, n_pixels)
    
    processor = SuperResProcessor()
    
    # Use the existing correlation function but with custom reference
    # We need to replicate the correlation logic with our custom reference
    print("  FFT over scan dimensions...")
    t0 = time.time()
    bigFT = np.fft.fft2(data_fake_4d, axes=(0, 1))
    
    print("  FFT of reference...")
    ref_fft = np.fft.fft2(reference)
    
    print("  Cross-correlation in Fourier space...")
    result = bigFT * np.conj(ref_fft)[:, :, np.newaxis, np.newaxis]
    
    print("  IFFT to real space...")
    result = np.fft.ifft2(result, axes=(0, 1))
    result = result ** 2
    result = np.abs(result)
    
    print("  fftshift and normalization...")
    result = np.fft.fftshift(result, axes=(0, 1))
    
    # Normalize
    norm_factor = np.sum(reference**2)
    norms = norm_factor * np.sum(data_fake_4d**2, axis=(0, 1))
    norms = np.where(norms < 1e-9, 1, norms)
    correlations = result / norms
    
    print(f"  Correlations computed in {time.time() - t0:.2f}s")
    
    # Step 3: Find shift maps
    print("\n--- Step 3: Find shift maps ---")
    t0 = time.time()
    
    # Reshape correlations to (sy, sx, 1, n_pixels) for find_shift_maps
    xm_sub, ym_sub, im_sub = processor.find_shift_maps(correlations, subpixel_refine=True)
    
    # Reshape back to 1D arrays (one per pixel)
    xm_sub = xm_sub.flatten()
    ym_sub = ym_sub.flatten()
    im_sub = im_sub.flatten()
    
    print(f"  Shift maps found in {time.time() - t0:.2f}s")
    print(f"  Quality range: [{im_sub.min():.3f}, {im_sub.max():.3f}]")
    print(f"  Quality mean: {im_sub.mean():.3f}")
    
    # Step 4: Reconstruct super-resolution image
    print("\n--- Step 4: Reconstruct super-resolution ADF ---")
    
    # We need to reconstruct using the annular pixels
    # This is similar to reconstruct_superres but we iterate over our annular_coords
    
    fac = 4
    pad = 16
    thresh = 0.3  # Lower threshold for ADF
    
    x, y = np.meshgrid(np.arange(sx), np.arange(sy))
    big = np.zeros((sy*fac+pad, sx*fac+pad), dtype=float)
    norm = np.zeros_like(big)
    
    pixels_used = 0
    pixels_skipped = 0
    
    t0 = time.time()
    for i in range(n_pixels):
        fit_qual = im_sub[i]
        
        if fit_qual > thresh:
            # Get the pixel image
            tmp1 = annular_images[:, :, i]
            
            # Get shifts (centered)
            x_shift = ym_sub[i] - sx / 2 - pad/2/fac
            y_shift = xm_sub[i] - sy / 2 - pad/2/fac
            
            # Compute shifted coordinates
            xcos = (x - x_shift) * fac
            ycos = (y - y_shift) * fac
            
            # Clip to bounds
            xcos = np.clip(np.round(xcos).astype(int), 0, big.shape[1] - 1)
            ycos = np.clip(np.round(ycos).astype(int), 0, big.shape[0] - 1)
            
            # Accumulate
            big[ycos, xcos] += tmp1
            norm[ycos, xcos] += 1
            pixels_used += 1
        else:
            pixels_skipped += 1
        
        if (i + 1) % 100 == 0:
            print(f"    Processed {i+1}/{n_pixels} pixels...")
    
    # Normalize
    invalid = np.where(norm < 1)
    big[invalid] = 0
    valid = np.where(norm > 0)
    big[valid] = big[valid] / norm[valid]
    
    print(f"  Reconstruction complete in {time.time() - t0:.2f}s")
    print(f"  Pixels used: {pixels_used}/{n_pixels}")
    print(f"  Pixels skipped (quality < {thresh}): {pixels_skipped}")
    
    # Create standard ADF for comparison (sum over annular region)
    print("\n--- Creating standard ADF for comparison ---")
    standard_adf = np.sum(annular_images, axis=2)
    
    # Visualize results
    print("\n--- Generating visualization ---")
    visualize_results(big, standard_adf, reference, xm_sub, ym_sub, im_sub, 
                     annular_coords, bf_center, sy, sx, inner_radius, outer_radius)
    
    print("\n" + "=" * 70)
    print("ADF Super-Resolution Complete!")
    print("=" * 70)
    
    return {
        'superres': big,
        'standard_adf': standard_adf,
        'reference': reference,
        'shift_y': xm_sub,
        'shift_x': ym_sub,
        'quality': im_sub,
        'normalization': norm,
        'annular_coords': annular_coords,
    }


def visualize_results(superres, standard_adf, reference, shift_y, shift_x, quality,
                     annular_coords, bf_center, sy, sx, inner_radius, outer_radius):
    """Create comprehensive visualization."""
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Row 0: Main images
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(standard_adf, cmap='gray')
    ax.set_title(f'Standard ADF\n(Sum of annular pixels)\n{standard_adf.shape}')
    ax.axis('off')
    
    ax = fig.add_subplot(gs[0, 1:3])
    ax.imshow(superres, cmap='gray')
    upscale = superres.shape[0] / standard_adf.shape[0]
    ax.set_title(f'Super-Resolution ADF ({upscale:.1f}x)\n{superres.shape}')
    ax.axis('off')
    
    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(reference, cmap='gray')
    ax.set_title(f'Pairwise Reference\n{reference.shape}')
    ax.axis('off')
    
    # Row 1: Quality and shift statistics
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(quality, bins=50, edgecolor='black')
    ax.axvline(quality.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {quality.mean():.3f}')
    ax.set_xlabel('Correlation Quality')
    ax.set_ylabel('Count')
    ax.set_title('Quality Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Shift magnitudes
    shift_mag = np.sqrt((shift_y - sy/2)**2 + (shift_x - sx/2)**2)
    
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(shift_mag, bins=50, edgecolor='black')
    ax.axvline(shift_mag.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {shift_mag.mean():.2f}')
    ax.set_xlabel('Shift Magnitude (pixels)')
    ax.set_ylabel('Count')
    ax.set_title('Shift Magnitude Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Shift map visualization (2D scatter)
    ax = fig.add_subplot(gs[1, 2:])
    
    # Convert annular coords to relative positions
    cy, cx = int(bf_center[0]), int(bf_center[1])
    rel_y = np.array([coord[0] - cy for coord in annular_coords])
    rel_x = np.array([coord[1] - cx for coord in annular_coords])
    
    # Plot shifts as colors
    scatter = ax.scatter(rel_x, rel_y, c=quality, cmap='viridis', s=20, alpha=0.7)
    
    # Draw annular region
    circle_inner = plt.Circle((0, 0), inner_radius, fill=False, color='r', linewidth=2, linestyle='--')
    circle_outer = plt.Circle((0, 0), outer_radius, fill=False, color='r', linewidth=2, linestyle='--')
    ax.add_patch(circle_inner)
    ax.add_patch(circle_outer)
    
    ax.set_xlabel('Detector X (relative to center)')
    ax.set_ylabel('Detector Y (relative to center)')
    ax.set_title('Quality Map (Detector Space)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Quality')
    
    # Row 2: Shift vector field (subsampled)
    ax = fig.add_subplot(gs[2, :2])
    
    # Subsample for clarity
    step = max(1, len(annular_coords) // 100)
    
    shift_y_centered = shift_y - sy / 2
    shift_x_centered = shift_x - sx / 2
    
    ax.quiver(rel_x[::step], rel_y[::step], 
             shift_x_centered[::step], shift_y_centered[::step],
             quality[::step], cmap='viridis', scale=50, alpha=0.7)
    
    circle_inner = plt.Circle((0, 0), inner_radius, fill=False, color='r', linewidth=2, linestyle='--')
    circle_outer = plt.Circle((0, 0), outer_radius, fill=False, color='r', linewidth=2, linestyle='--')
    ax.add_patch(circle_inner)
    ax.add_patch(circle_outer)
    
    ax.set_xlabel('Detector X (relative to center)')
    ax.set_ylabel('Detector Y (relative to center)')
    ax.set_title('Shift Vectors (Detector Space, subsampled)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Statistics text
    ax = fig.add_subplot(gs[2, 2:])
    ax.axis('off')
    
    stats_text = f"""
    STATISTICS:
    
    Annular Region:
      Inner radius: {inner_radius} pixels
      Outer radius: {outer_radius} pixels
      Total pixels: {len(annular_coords)}
    
    Quality:
      Min:    {quality.min():.3f}
      Max:    {quality.max():.3f}
      Mean:   {quality.mean():.3f}
      Median: {np.median(quality):.3f}
    
    Shift Magnitude:
      Min:    {shift_mag.min():.2f} pixels
      Max:    {shift_mag.max():.2f} pixels
      Mean:   {shift_mag.mean():.2f} pixels
      Median: {np.median(shift_mag):.2f} pixels
    
    Super-Resolution:
      Input size:  {standard_adf.shape}
      Output size: {superres.shape}
      Upscale:     {superres.shape[0] / standard_adf.shape[0]:.1f}x
    """
    
    ax.text(0.1, 0.5, stats_text, fontfamily='monospace', fontsize=10,
           verticalalignment='center')
    
    output_path = 'full_adf_superres_test.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved visualization to: {output_path}")


if __name__ == '__main__':
    emd_path = "/media/o2d/data/ORNL Dropbox/Ondrej Dyck/Code/mib-viewer-project/SS_a7_2ms_256x256 4D_16nmFoV -20nmFocus.emd"
    
    results = test_full_adf_superres(
        emd_path,
        bf_center=(127.6, 126.4),
        inner_radius=35,
        outer_radius=55,
        smoothing=0.5,
        damping=0.5
    )
    
    print("\n✅ Full ADF super-resolution test complete!")
    print("Check full_adf_superres_test.png for results.")
    print("\nKey questions:")
    print("  1. Does the super-res ADF look sharper than standard ADF?")
    print("  2. Are the shift patterns smooth and reasonable?")
    print("  3. What's the quality distribution like?")
    print("  4. Does this approach work for ADF super-resolution?")
