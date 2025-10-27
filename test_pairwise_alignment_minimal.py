#!/usr/bin/env python3
"""Test pairwise alignment approach for ADF super-resolution.

This script tests the idea of aligning opposing detector pixels to each other
iteratively, without needing a sharp reference image.

Algorithm:
1. Select 4 pixels from annular region (top, right, bottom, left)
2. Align left/right pair by moving each halfway toward the other
3. Create initial reference from aligned pair
4. Align top/bottom to this reference
5. Update reference with all 4 pixels
6. Iterate until convergence
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from typing import Tuple, List, Dict


def find_pairwise_shift(pixel_A: np.ndarray, 
                       pixel_B: np.ndarray,
                       smoothing: float = 0.5) -> Tuple[float, float, float]:
    """Find relative shift between two detector pixel images using cross-correlation.
    
    This reuses the core correlation logic from SuperResProcessor but for just
    two pixel images instead of a full 4D dataset.
    
    Parameters
    ----------
    pixel_A : ndarray, shape (scan_y, scan_x)
        First pixel image (will be used as reference)
    pixel_B : ndarray, shape (scan_y, scan_x)
        Second pixel image (will be correlated against A)
    smoothing : float
        Gaussian smoothing sigma for reference
        
    Returns
    -------
    shift_y : float
        Vertical shift (in pixels)
    shift_x : float
        Horizontal shift (in pixels)
    quality : float
        Correlation quality at peak
    """
    sy, sx = pixel_A.shape
    
    # Smooth the reference (pixel_A)
    ref_smoothed = gaussian_filter(pixel_A, smoothing).astype(float)
    
    # FFT of reference
    ref_fft = np.fft.fft2(ref_smoothed)
    
    # FFT of pixel_B
    pixel_B_fft = np.fft.fft2(pixel_B.astype(float))
    
    # Cross-correlation in Fourier space
    corr = pixel_B_fft * np.conj(ref_fft)
    
    # IFFT back to real space
    corr = np.fft.ifft2(corr)
    corr = corr ** 2
    corr = np.abs(corr)
    
    # fftshift to center
    corr = np.fft.fftshift(corr)
    
    # Normalize
    norm_factor = np.sum(ref_smoothed**2) * np.sum(pixel_B**2)
    if norm_factor > 1e-9:
        corr = corr / norm_factor
    
    # Find peak (integer)
    peak_idx = np.argmax(corr)
    peak_y, peak_x = np.unravel_index(peak_idx, corr.shape)
    
    # Subpixel refinement (quadratic fit)
    if 1 <= peak_y < sy-1 and 1 <= peak_x < sx-1:
        # Get neighboring values
        c = corr[peak_y, peak_x]
        dx = corr[peak_y, peak_x+1] - corr[peak_y, peak_x-1]
        dy = corr[peak_y+1, peak_x] - corr[peak_y-1, peak_x]
        dxx = corr[peak_y, peak_x+1] - 2*c + corr[peak_y, peak_x-1]
        dyy = corr[peak_y+1, peak_x] - 2*c + corr[peak_y-1, peak_x]
        
        # Quadratic fit (same logic as SuperResProcessor)
        denom_x = dxx if dxx <= 1e-3 else 1e6
        denom_y = dyy if dyy <= 1e-3 else 1e6
        
        shift_y_sub = peak_y - dy / denom_y / 2
        shift_x_sub = peak_x - dx / denom_x / 2
        quality = c
    else:
        shift_y_sub = float(peak_y)
        shift_x_sub = float(peak_x)
        quality = corr[peak_y, peak_x]
    
    # Convert to centered coordinates (shift relative to center)
    shift_y = shift_y_sub - sy / 2
    shift_x = shift_x_sub - sx / 2
    
    return shift_y, shift_x, quality


def shift_image(image: np.ndarray, shift_y: float, shift_x: float) -> np.ndarray:
    """Shift an image by sub-pixel amounts using Fourier shift theorem.
    
    Parameters
    ----------
    image : ndarray
        Image to shift
    shift_y, shift_x : float
        Shift amounts in pixels (can be sub-pixel)
        
    Returns
    -------
    shifted : ndarray
        Shifted image
    """
    sy, sx = image.shape
    
    # Create frequency grids
    fy = np.fft.fftfreq(sy)
    fx = np.fft.fftfreq(sx)
    fy_grid, fx_grid = np.meshgrid(fy, fx, indexing='ij')
    
    # FFT of image
    image_fft = np.fft.fft2(image)
    
    # Apply phase shift in Fourier space
    # Shift theorem: F{f(x-a)} = exp(-2πi*a*ξ) * F{f(x)}
    phase_shift = np.exp(-2j * np.pi * (shift_y * fy_grid + shift_x * fx_grid))
    shifted_fft = image_fft * phase_shift
    
    # IFFT back to real space
    shifted = np.fft.ifft2(shifted_fft).real
    
    return shifted


def align_four_pixels_iteratively(pixel_images: List[np.ndarray],
                                  pixel_labels: List[str],
                                  max_iterations: int = 10,
                                  convergence_threshold: float = 0.1,
                                  damping: float = 0.5,
                                  smoothing: float = 0.5) -> Dict:
    """Iteratively align 4 detector pixels using pairwise correlations.
    
    Algorithm:
    1. Start with left/right pair (indices 1 and 3)
    2. Find their relative shift
    3. Move each halfway toward the other
    4. Create initial reference from these two
    5. Align top/bottom (indices 0 and 2) to this reference
    6. Update reference with all 4 pixels
    7. Repeat until shifts converge
    
    Parameters
    ----------
    pixel_images : list of ndarray
        4 pixel images in order: [top, right, bottom, left]
    pixel_labels : list of str
        Labels for the 4 pixels
    max_iterations : int
        Maximum iterations
    convergence_threshold : float
        Stop when max shift change < this value (pixels)
    damping : float
        Damping factor (0.5 = move halfway, 1.0 = full shift)
    smoothing : float
        Gaussian smoothing for correlations
        
    Returns
    -------
    results : dict
        Contains:
        - 'aligned_images': List of aligned images
        - 'reference': Final composite reference
        - 'shifts': Final shift values for each pixel
        - 'history': List of iteration diagnostics
    """
    print("\n" + "=" * 70)
    print("Iterative Pairwise Alignment of 4 Detector Pixels")
    print("=" * 70)
    
    # Initialize with copies
    aligned = [img.copy().astype(float) for img in pixel_images]
    shifts = [(0.0, 0.0) for _ in range(4)]  # (shift_y, shift_x) for each pixel
    history = []
    
    # Iteration 0: Align left/right pair
    print("\n--- Iteration 0: Initialize with left/right pair ---")
    
    # Indices: 0=top, 1=right, 2=bottom, 3=left
    right_idx = 1
    left_idx = 3
    
    # Find relative shift between right and left
    shift_y, shift_x, quality = find_pairwise_shift(
        aligned[right_idx], aligned[left_idx], smoothing
    )
    
    print(f"  Right-Left relative shift: ({shift_y:.2f}, {shift_x:.2f}), quality: {quality:.3f}")
    
    # Move each halfway toward the other (with damping)
    # Right moves by -shift/2, Left moves by +shift/2
    shift_amount_y = shift_y * damping / 2
    shift_amount_x = shift_x * damping / 2
    
    aligned[right_idx] = shift_image(aligned[right_idx], -shift_amount_y, -shift_amount_x)
    aligned[left_idx] = shift_image(aligned[left_idx], shift_amount_y, shift_amount_x)
    
    shifts[right_idx] = (-shift_amount_y, -shift_amount_x)
    shifts[left_idx] = (shift_amount_y, shift_amount_x)
    
    # Create initial reference from these two
    reference = (aligned[right_idx] + aligned[left_idx]) / 2
    reference = gaussian_filter(reference, smoothing)
    
    print(f"  Created initial reference from right/left pair")
    
    # Record iteration 0
    history.append({
        'iteration': 0,
        'shifts': shifts.copy(),
        'reference': reference.copy(),
        'aligned_images': [img.copy() for img in aligned],
        'max_shift_change': np.sqrt(shift_amount_y**2 + shift_amount_x**2),
    })
    
    # Main iteration loop
    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")
        
        old_shifts = shifts.copy()
        
        # Align all 4 pixels to current reference
        for i, label in enumerate(pixel_labels):
            # Find shift relative to reference
            shift_y, shift_x, quality = find_pairwise_shift(
                reference, pixel_images[i], smoothing
            )
            
            # Apply damped shift
            shift_amount_y = shift_y * damping
            shift_amount_x = shift_x * damping
            
            # Update aligned image (shift from ORIGINAL, not accumulated)
            total_shift_y = shifts[i][0] + shift_amount_y
            total_shift_x = shifts[i][1] + shift_amount_x
            
            aligned[i] = shift_image(pixel_images[i], total_shift_y, total_shift_x)
            shifts[i] = (total_shift_y, total_shift_x)
            
            print(f"  {label:6s}: shift=({shift_y:6.2f}, {shift_x:6.2f}), "
                  f"total=({total_shift_y:6.2f}, {total_shift_x:6.2f}), quality={quality:.3f}")
        
        # Update reference (average of all 4 aligned pixels)
        reference = np.mean(aligned, axis=0)
        reference = gaussian_filter(reference, smoothing)
        
        # Check convergence (max change in any shift)
        max_shift_change = 0
        for i in range(4):
            dy = shifts[i][0] - old_shifts[i][0]
            dx = shifts[i][1] - old_shifts[i][1]
            shift_change = np.sqrt(dy**2 + dx**2)
            max_shift_change = max(max_shift_change, shift_change)
        
        print(f"  Max shift change: {max_shift_change:.4f} pixels")
        
        # Record iteration
        history.append({
            'iteration': iteration,
            'shifts': shifts.copy(),
            'reference': reference.copy(),
            'aligned_images': [img.copy() for img in aligned],
            'max_shift_change': max_shift_change,
        })
        
        # Check convergence
        if max_shift_change < convergence_threshold:
            print(f"\n✅ Converged after {iteration} iterations!")
            break
    else:
        print(f"\n⚠️  Reached max iterations ({max_iterations}) without convergence")
    
    print("=" * 70)
    
    return {
        'aligned_images': aligned,
        'reference': reference,
        'shifts': shifts,
        'history': history,
    }


def test_pairwise_alignment(emd_path: str, 
                           bf_center: Tuple[float, float] = (127.6, 126.4),
                           test_radius: int = 45):
    """Test pairwise alignment on 4 pixels from annular region.
    
    Parameters
    ----------
    emd_path : str
        Path to EMD file
    bf_center : tuple
        (y, x) coordinates of BF disk center
    test_radius : int
        Radius at which to sample the 4 test pixels
    """
    print("\n" + "=" * 70)
    print("Testing Pairwise Alignment for ADF Super-Resolution")
    print("4-Pixel Minimal Test")
    print("=" * 70 + "\n")
    
    # Load data
    print(f"Loading data from: {emd_path}")
    with h5py.File(emd_path, 'r') as f:
        # Load smaller subset to save memory (128x128 scan instead of 256x256)
        data_4d = f['version_1/data/datacubes/datacube_000/data'][:128, :128, :, :]
    
    print(f"  Loaded 4D data shape: {data_4d.shape}")
    print(f"  BF center: ({bf_center[0]:.1f}, {bf_center[1]:.1f})")
    
    # Select 4 pixels at test_radius from center
    # Angles: 0° (top), 90° (right), 180° (bottom), 270° (left)
    cy, cx = int(bf_center[0]), int(bf_center[1])
    
    angles_deg = [0, 90, 180, 270]
    angles_rad = [np.deg2rad(a) for a in angles_deg]
    labels = ['Top', 'Right', 'Bottom', 'Left']
    
    pixel_coords = []
    for angle in angles_rad:
        # Convert polar to Cartesian (note: y-axis points down in images)
        dy = -test_radius * np.cos(angle)  # Negative because y increases downward
        dx = test_radius * np.sin(angle)
        
        pixel_y = int(cy + dy)
        pixel_x = int(cx + dx)
        pixel_coords.append((pixel_y, pixel_x))
    
    print(f"\nSelected 4 pixels at radius {test_radius}:")
    for label, coord in zip(labels, pixel_coords):
        print(f"  {label:6s}: detector pixel ({coord[0]}, {coord[1]})")
    
    # Extract the 4 pixel images
    pixel_images = []
    for py, px in pixel_coords:
        pixel_img = data_4d[:, :, py, px].astype(float)
        pixel_images.append(pixel_img)
    
    print(f"\nExtracted pixel images, shape: {pixel_images[0].shape}")
    
    # Run iterative alignment
    results = align_four_pixels_iteratively(
        pixel_images,
        labels,
        max_iterations=10,
        convergence_threshold=0.1,
        damping=0.5,
        smoothing=0.5
    )
    
    # Visualize results
    visualize_alignment_results(results, labels, pixel_images)
    
    return results


def visualize_alignment_results(results: Dict, labels: List[str], 
                                original_images: List[np.ndarray]):
    """Create comprehensive visualization of alignment results."""
    
    history = results['history']
    n_iterations = len(history)
    
    # Create figure with multiple rows
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 6, hspace=0.4, wspace=0.3)
    
    # Row 0: Original pixel images
    for i, (label, img) in enumerate(zip(labels, original_images)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img, cmap='gray')
        ax.set_title(f'{label}\n(Original)')
        ax.axis('off')
    
    # Row 0, col 4: Convergence plot
    ax_conv = fig.add_subplot(gs[0, 4:])
    iterations = [h['iteration'] for h in history]
    shift_changes = [h['max_shift_change'] for h in history]
    ax_conv.plot(iterations, shift_changes, 'o-', linewidth=2, markersize=8)
    ax_conv.axhline(0.1, color='r', linestyle='--', label='Convergence threshold')
    ax_conv.set_xlabel('Iteration')
    ax_conv.set_ylabel('Max Shift Change (pixels)')
    ax_conv.set_title('Convergence History')
    ax_conv.grid(True, alpha=0.3)
    ax_conv.legend()
    
    # Row 1: Aligned images after iteration 0 (initial left/right alignment)
    iter0 = history[0]
    for i, (label, img) in enumerate(zip(labels, iter0['aligned_images'])):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(img, cmap='gray')
        shift_y, shift_x = iter0['shifts'][i]
        ax.set_title(f'{label}\nIter 0: ({shift_y:.1f}, {shift_x:.1f})')
        ax.axis('off')
    
    # Row 1, col 4: Reference after iteration 0
    ax = fig.add_subplot(gs[1, 4])
    ax.imshow(iter0['reference'], cmap='gray')
    ax.set_title('Reference\n(Iter 0)')
    ax.axis('off')
    
    # Row 2: Final aligned images
    final = history[-1]
    for i, (label, img) in enumerate(zip(labels, final['aligned_images'])):
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(img, cmap='gray')
        shift_y, shift_x = final['shifts'][i]
        ax.set_title(f'{label}\nFinal: ({shift_y:.1f}, {shift_x:.1f})')
        ax.axis('off')
    
    # Row 2, col 4: Final reference
    ax = fig.add_subplot(gs[2, 4])
    ax.imshow(final['reference'], cmap='gray')
    ax.set_title(f'Final Reference\n(Iter {final["iteration"]})')
    ax.axis('off')
    
    # Row 2, col 5: Difference (final - iter0 reference)
    ax = fig.add_subplot(gs[2, 5])
    diff = final['reference'] - iter0['reference']
    im = ax.imshow(diff, cmap='RdBu_r', vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
    ax.set_title('Reference Change\n(Final - Iter0)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Row 3: Shift vectors visualization
    ax = fig.add_subplot(gs[3, :3])
    
    # Plot shift vectors
    angles_deg = [0, 90, 180, 270]
    colors = ['red', 'green', 'blue', 'orange']
    
    for i, (label, angle, color) in enumerate(zip(labels, angles_deg, colors)):
        # Plot evolution of shifts over iterations
        shift_y_history = [h['shifts'][i][0] for h in history]
        shift_x_history = [h['shifts'][i][1] for h in history]
        
        ax.plot(shift_x_history, shift_y_history, 'o-', 
                color=color, label=label, linewidth=2, markersize=8)
        
        # Mark start and end
        ax.plot(shift_x_history[0], shift_y_history[0], 'o', 
                color=color, markersize=12, markerfacecolor='white', markeredgewidth=2)
        ax.plot(shift_x_history[-1], shift_y_history[-1], 's', 
                color=color, markersize=12)
    
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Shift X (pixels)')
    ax.set_ylabel('Shift Y (pixels)')
    ax.set_title('Shift Evolution (○ = start, □ = end)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Row 3, col 3-5: Shift magnitude over iterations
    ax = fig.add_subplot(gs[3, 3:])
    
    for i, (label, color) in enumerate(zip(labels, colors)):
        shift_magnitudes = []
        for h in history:
            sy, sx = h['shifts'][i]
            mag = np.sqrt(sy**2 + sx**2)
            shift_magnitudes.append(mag)
        
        ax.plot(iterations, shift_magnitudes, 'o-', 
                color=color, label=label, linewidth=2, markersize=6)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Shift Magnitude (pixels)')
    ax.set_title('Shift Magnitude vs Iteration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save figure
    output_path = 'pairwise_alignment_test.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved visualization to: {output_path}")
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("Results Summary:")
    print("=" * 70)
    print(f"Converged in {final['iteration']} iterations")
    print(f"Final max shift change: {final['max_shift_change']:.4f} pixels")
    print("\nFinal shifts:")
    for label, shift in zip(labels, final['shifts']):
        sy, sx = shift
        mag = np.sqrt(sy**2 + sx**2)
        print(f"  {label:6s}: ({sy:6.2f}, {sx:6.2f}) pixels, magnitude: {mag:.2f}")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    emd_path = "/media/o2d/data/ORNL Dropbox/Ondrej Dyck/Code/mib-viewer-project/SS_a7_2ms_256x256 4D_16nmFoV -20nmFocus.emd"
    
    # Known BF center from previous analysis
    bf_center = (127.6, 126.4)
    
    # Test at radius 45 (middle of annular region: 35-55)
    results = test_pairwise_alignment(
        emd_path,
        bf_center=bf_center,
        test_radius=45
    )
    
    print("\n✅ Pairwise alignment test complete!")
    print("Check pairwise_alignment_test.png for results.")
    print("\nKey questions to answer:")
    print("  1. Did the alignment converge?")
    print("  2. Do the aligned images look sharper/more similar?")
    print("  3. Does the final reference look sharp?")
    print("  4. Are the shift patterns physically reasonable?")
