#!/usr/bin/env python3
"""Test super-resolution using annular dark field detector region."""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from src.mib_viewer.processing.superres_processor import SuperResProcessor


def crop_to_annular_region(data_4d, center_y, center_x, inner_radius=35, outer_radius=55):
    """Crop 4D dataset to annular region (for ADF).
    
    Parameters
    ----------
    data_4d : ndarray, shape (scan_y, scan_x, det_y, det_x)
        Full 4D STEM dataset
    center_y, center_x : float
        BF disk center coordinates
    inner_radius : int
        Inner radius of annular region (default: 35, just outside BF disk)
    outer_radius : int
        Outer radius of annular region (default: 55)
    
    Returns
    -------
    cropped : ndarray
        Cropped dataset containing annular region
    mask : ndarray
        Boolean mask showing which pixels are in the annulus
    crop_bounds : tuple
        (y1, y2, x1, x2) bounds used for cropping
    """
    print(f"Cropping to annular region: inner_r={inner_radius}, outer_r={outer_radius}")
    
    cy, cx = int(center_y), int(center_x)
    
    # Create a square region that contains the annulus
    det_y, det_x = data_4d.shape[2], data_4d.shape[3]
    y1 = max(0, cy - outer_radius)
    y2 = min(det_y, cy + outer_radius)
    x1 = max(0, cx - outer_radius)
    x2 = min(det_x, cx + outer_radius)
    
    print(f"  Crop bounds: y=[{y1}:{y2}], x=[{x1}:{x2}]")
    
    cropped = data_4d[:, :, y1:y2, x1:x2]
    
    # Create mask for annular region
    dy, dx = cropped.shape[2], cropped.shape[3]
    yy, xx = np.meshgrid(np.arange(dy), np.arange(dx), indexing='ij')
    
    # Distance from center (in cropped coordinates)
    center_y_crop = cy - y1
    center_x_crop = cx - x1
    r = np.sqrt((yy - center_y_crop)**2 + (xx - center_x_crop)**2)
    
    # Mask: keep only pixels in annular region
    mask = (r >= inner_radius) & (r <= outer_radius)
    
    print(f"  Cropped shape: {cropped.shape}")
    print(f"  Pixels in annulus: {np.sum(mask)} / {mask.size}")
    
    # Apply mask (set pixels outside annulus to zero)
    # Note: We keep the full cropped array but zero out non-annular pixels
    # This preserves the spatial relationships for the correlation algorithm
    cropped_masked = cropped.copy()
    cropped_masked[:, :, ~mask] = 0
    
    return cropped_masked, mask, (y1, y2, x1, x2)


def compute_virtual_adf_reference(data_4d, mask, crop_bounds, reference_smoothing=0.5):
    """Compute virtual ADF reference image by averaging over annular region.
    
    Parameters
    ----------
    data_4d : ndarray, shape (scan_y, scan_x, det_y, det_x)
        Full 4D dataset
    mask : ndarray, shape (crop_dy, crop_dx)
        Boolean mask indicating annular pixels
    crop_bounds : tuple
        (y1, y2, x1, x2) detector crop bounds
    reference_smoothing : float
        Gaussian smoothing sigma
        
    Returns
    -------
    virtual_adf : ndarray, shape (scan_y, scan_x)
        Virtual ADF image (sum over annular region)
    """
    from scipy.ndimage import gaussian_filter
    
    y1, y2, x1, x2 = crop_bounds
    
    # Extract the cropped detector region
    cropped_data = data_4d[:, :, y1:y2, x1:x2]
    
    # Sum over all pixels in the annular region
    # mask is (crop_dy, crop_dx), we want to sum over those pixels
    virtual_adf = np.sum(cropped_data[:, :, mask], axis=2)
    
    # Apply smoothing (same as BF reference)
    virtual_adf = gaussian_filter(virtual_adf, reference_smoothing).astype(float)
    
    return virtual_adf


def test_annular_superres(emd_path, bf_center=(127.6, 126.4), 
                          inner_radius=35, outer_radius=55):
    """Test super-resolution on annular detector region.
    
    Parameters
    ----------
    emd_path : str
        Path to EMD file
    bf_center : tuple
        (y, x) coordinates of BF disk center
    inner_radius : int
        Inner radius of annular region
    outer_radius : int
        Outer radius of annular region
    """
    print("\n" + "=" * 70)
    print("Testing Super-Resolution on Annular Dark Field Region")
    print("Using Virtual ADF Reference")
    print("=" * 70 + "\n")
    
    # Load data
    print(f"Loading data from: {emd_path}")
    with h5py.File(emd_path, 'r') as f:
        data_4d = f['version_1/data/datacubes/datacube_000/data'][:]
    
    print(f"  Loaded 4D data shape: {data_4d.shape}")
    print(f"  BF center: ({bf_center[0]:.1f}, {bf_center[1]:.1f})")
    
    # Crop to annular region (ADF)
    data_annular, mask, crop_bounds = crop_to_annular_region(
        data_4d, bf_center[0], bf_center[1], 
        inner_radius=inner_radius, 
        outer_radius=outer_radius
    )
    
    # Compute virtual ADF reference image
    print("\nComputing virtual ADF reference (sum over annular region)...")
    virtual_adf_ref = compute_virtual_adf_reference(
        data_4d, mask, crop_bounds, reference_smoothing=0.5
    )
    print(f"  Virtual ADF shape: {virtual_adf_ref.shape}")
    print(f"  Virtual ADF range: [{virtual_adf_ref.min():.0f}, {virtual_adf_ref.max():.0f}]")
    
    # Run super-resolution using custom reference
    print("\nRunning super-resolution on annular region...")
    processor = SuperResProcessor()
    
    # We need to use the lower-level functions to inject our custom reference
    # First, compute cross-correlations with our virtual ADF reference
    from scipy.ndimage import gaussian_filter
    
    print("Computing cross-correlations with virtual ADF reference...")
    correlations, _ = processor.compute_cross_correlations(
        data_annular,
        reference_smoothing=0.5  # This will be overridden
    )
    
    # Actually, we need to do this manually to use our custom reference
    # Let me recompute with the virtual reference
    sy, sx, dy, dx = data_annular.shape
    
    # FFT over scan dimensions
    print("  FFT over scan dimensions...")
    bigFT = np.fft.fft2(data_annular, axes=(0, 1))
    
    # FFT of virtual reference
    print("  FFT of virtual ADF reference...")
    ref_fft = np.fft.fft2(virtual_adf_ref)
    
    # Cross-correlation in Fourier space
    print("  Cross-correlation in Fourier space...")
    result = bigFT * np.conj(ref_fft)[:, :, np.newaxis, np.newaxis]
    
    # IFFT back to real space
    print("  IFFT to real space...")
    result = np.fft.ifft2(result, axes=(0, 1))
    result = result ** 2
    result = np.abs(result)
    
    # fftshift
    print("  fftshift and normalization...")
    result = np.fft.fftshift(result, axes=(0, 1))
    
    # Normalize
    norm_factor = np.sum(virtual_adf_ref**2)
    norms = norm_factor * np.sum(data_annular**2, axis=(0, 1))
    norms = np.where(norms < 1e-9, 1, norms)
    correlations = result / norms
    
    # Find shift maps
    print("Finding shift maps...")
    xm_sub, ym_sub, im_sub = processor.find_shift_maps(
        correlations, subpixel_refine=True
    )
    
    # Reconstruct super-resolution image
    print("Reconstructing super-resolution image...")
    superres, norm = processor.reconstruct_superres(
        data_annular, xm_sub, ym_sub, im_sub,
        fac=4,
        lims=16,
        thresh=0.1  # Lower threshold for ADF
    )
    
    # Package results
    results = {
        'superres_image': superres,
        'standard_bf': virtual_adf_ref,  # Use virtual ADF as "standard"
        'shift_y': xm_sub,
        'shift_x': ym_sub,
        'quality': im_sub,
        'normalization': norm,
        'reference_image': virtual_adf_ref,
        'data_cropped': data_annular,
        'bf_center': bf_center
    }
    
    # Visualize results
    print("\nGenerating visualizations...")
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Row 1: Detector visualization and standard images
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(mask, cmap='gray')
    ax1.set_title(f'Annular Detector Mask\n(r={inner_radius}-{outer_radius} pixels)')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    # Show full detector with annular region marked
    detector_image = data_4d[128, 128, :, :]
    ax2.imshow(detector_image, cmap='gray')
    circle1 = plt.Circle((bf_center[1], bf_center[0]), inner_radius, 
                         fill=False, color='r', linewidth=2, label='Inner')
    circle2 = plt.Circle((bf_center[1], bf_center[0]), outer_radius, 
                         fill=False, color='g', linewidth=2, label='Outer')
    ax2.add_patch(circle1)
    ax2.add_patch(circle2)
    ax2.set_title('Detector with Annular Region')
    ax2.legend()
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(results['standard_bf'], cmap='gray')
    ax3.set_title(f"Standard ADF\n{results['standard_bf'].shape}")
    ax3.axis('off')
    
    # Row 2: Super-res and shift maps
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.imshow(results['superres_image'], cmap='gray')
    upscale = results['superres_image'].shape[0] / results['standard_bf'].shape[0]
    ax4.set_title(f"Super-Resolution ADF ({upscale:.1f}x)\n{results['superres_image'].shape}")
    ax4.axis('off')
    
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.imshow(results['quality'], cmap='viridis')
    ax5.set_title('Quality Map')
    ax5.axis('off')
    plt.colorbar(ax5.images[0], ax=ax5, fraction=0.046)
    
    # Row 3: Shift maps
    sy, sx = results['data_cropped'].shape[:2]
    shift_y_centered = results['shift_y'] - sy // 2
    shift_x_centered = results['shift_x'] - sx // 2
    
    ax6 = fig.add_subplot(gs[2, 0])
    im6 = ax6.imshow(shift_y_centered, cmap='RdBu_r', vmin=-20, vmax=20)
    ax6.set_title('Shift Map (Y) - Centered')
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046)
    
    ax7 = fig.add_subplot(gs[2, 1])
    im7 = ax7.imshow(shift_x_centered, cmap='RdBu_r', vmin=-20, vmax=20)
    ax7.set_title('Shift Map (X) - Centered')
    ax7.axis('off')
    plt.colorbar(im7, ax=ax7, fraction=0.046)
    
    # Shift vector field
    ax8 = fig.add_subplot(gs[2, 2])
    step = 4  # Show every 4th vector
    y_grid, x_grid = np.meshgrid(
        np.arange(0, shift_y_centered.shape[0], step),
        np.arange(0, shift_y_centered.shape[1], step),
        indexing='ij'
    )
    dy = shift_y_centered[::step, ::step]
    dx = shift_x_centered[::step, ::step]
    ax8.quiver(x_grid, y_grid, dx, dy, scale=50)
    ax8.set_title('Shift Vectors (subsampled)')
    ax8.set_aspect('equal')
    ax8.invert_yaxis()
    
    output_path = 'superres_annular_test.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved visualization to: {output_path}")
    
    # Print statistics
    print("\n" + "=" * 70)
    print("Results Summary:")
    print("=" * 70)
    print(f"Annular region: inner_r={inner_radius}, outer_r={outer_radius}")
    print(f"Pixels in annulus: {np.sum(mask)}")
    print(f"\nStandard ADF shape: {results['standard_bf'].shape}")
    print(f"Super-res ADF shape: {results['superres_image'].shape}")
    print(f"Upscaling factor: {upscale:.1f}x")
    
    print(f"\nShift map Y (centered): [{shift_y_centered.min():.2f}, {shift_y_centered.max():.2f}]")
    print(f"Shift map X (centered): [{shift_x_centered.min():.2f}, {shift_x_centered.max():.2f}]")
    
    print(f"\nQuality: [{results['quality'].min():.3f}, {results['quality'].max():.3f}]")
    print(f"Quality mean: {results['quality'].mean():.3f}")
    
    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)
    
    return results


if __name__ == '__main__':
    emd_path = "/media/o2d/data/ORNL Dropbox/Ondrej Dyck/Code/mib-viewer-project/SS_a7_2ms_256x256 4D_16nmFoV -20nmFocus.emd"
    
    # Known BF center from previous analysis
    bf_center = (127.6, 126.4)
    
    # BF disk radius is 32, so let's use annular region from 35-55
    results = test_annular_superres(
        emd_path, 
        bf_center=bf_center,
        inner_radius=35,
        outer_radius=55
    )
    
    print("\n✅ Annular super-resolution test complete!")
    print("Check superres_annular_test.png for results.")
    print("\nThis tests whether super-resolution works on ADF detector regions.")
    print("If successful, we could get super-res Z-contrast imaging!")
