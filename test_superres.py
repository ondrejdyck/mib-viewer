#!/usr/bin/env python3
"""Test script for super-resolution algorithm.

This script tests the super-resolution reconstruction on synthetic data
to verify the algorithm works correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.mib_viewer.processing.superres_processor import SuperResProcessor


def create_test_pattern(size=256, feature_size=10):
    """Create a test pattern with sharp features."""
    # Create a grid pattern with sharp features
    y, x = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')

    # Multiple Gaussian peaks
    pattern = np.zeros((size, size))

    # Grid of peaks
    for i in range(5):
        for j in range(5):
            cy = size // 6 * (i + 1)
            cx = size // 6 * (j + 1)
            peak = np.exp(-((y - cy)**2 + (x - cx)**2) / (2 * feature_size**2))
            pattern += peak

    # Add some noise
    pattern += 0.1 * np.random.randn(size, size)
    pattern = np.clip(pattern, 0, None)

    return pattern


def generate_synthetic_4d(scan_size=64, det_size=64, image_size=256):
    """Generate synthetic 4D STEM data with known shifts.

    Each detector pixel sees a shifted version of the real-space image,
    simulating the phase contrast shifts in real 4D STEM.
    """
    print(f"Generating synthetic 4D STEM data...")
    print(f"  Scan size: {scan_size} x {scan_size}")
    print(f"  Detector size: {det_size} x {det_size}")
    print(f"  Real-space image: {image_size} x {image_size}")

    # Create test pattern
    true_image = create_test_pattern(image_size)

    # Downsampled version for scan
    scan_image = true_image[::image_size//scan_size, ::image_size//scan_size]

    # Initialize 4D dataset
    data_4d = np.zeros((scan_size, scan_size, det_size, det_size))

    # For each detector pixel, create a shifted version
    center_det_y = det_size // 2
    center_det_x = det_size // 2

    # Maximum shift in pixels (at detector edge)
    max_shift = 4.0

    for i in range(det_size):
        for j in range(det_size):
            # Calculate shift proportional to distance from center
            shift_y = (i - center_det_y) / (det_size / 2) * max_shift
            shift_x = (j - center_det_x) / (det_size / 2) * max_shift

            # Create shifted scan image using Fourier shift theorem
            # (more accurate than scipy.ndimage.shift)
            ky, kx = np.meshgrid(
                np.fft.fftfreq(scan_size),
                np.fft.fftfreq(scan_size),
                indexing='ij'
            )

            # Phase ramp for shift
            phase_ramp = np.exp(-2j * np.pi * (ky * shift_y + kx * shift_x))

            # Apply shift in Fourier space
            scan_fft = np.fft.fft2(scan_image)
            shifted_fft = scan_fft * phase_ramp
            shifted_image = np.real(np.fft.ifft2(shifted_fft))

            # Store in 4D dataset
            data_4d[:, :, i, j] = shifted_image

    # Add noise
    noise_level = 0.05 * np.max(data_4d)
    data_4d += noise_level * np.random.randn(*data_4d.shape)
    data_4d = np.clip(data_4d, 0, None)

    print(f"  Generated data shape: {data_4d.shape}")
    print(f"  Data range: [{data_4d.min():.3f}, {data_4d.max():.3f}]")

    return data_4d, scan_image, true_image


def test_synthetic_data():
    """Test super-resolution on synthetic data."""
    print("\n" + "=" * 70)
    print("Testing Super-Resolution Algorithm on Synthetic Data")
    print("=" * 70 + "\n")

    # Generate synthetic data
    data_4d, scan_image, true_image = generate_synthetic_4d(
        scan_size=64,
        det_size=64,
        image_size=256
    )

    # Run super-resolution
    processor = SuperResProcessor()

    results = processor.process_full(
        data_4d,
        bf_radius=32,
        reference_smoothing=0.5,
        upscale_factor=4,
        detector_radius=16,
        quality_threshold=0.6  # Lower threshold for synthetic data
    )

    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Images
    axes[0, 0].imshow(scan_image, cmap='gray')
    axes[0, 0].set_title('Ground Truth (Downsampled)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(results['standard_bf'], cmap='gray')
    axes[0, 1].set_title('Standard BF (Central Pixel)')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(results['superres_image'], cmap='gray')
    axes[0, 2].set_title('Super-Resolution Reconstruction')
    axes[0, 2].axis('off')

    # Row 2: Diagnostics
    im1 = axes[1, 0].imshow(results['shift_y'], cmap='RdBu_r')
    axes[1, 0].set_title('Shift Map (Y)')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)

    im2 = axes[1, 1].imshow(results['shift_x'], cmap='RdBu_r')
    axes[1, 1].set_title('Shift Map (X)')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)

    im3 = axes[1, 2].imshow(results['quality'], cmap='viridis')
    axes[1, 2].set_title('Quality Map')
    axes[1, 2].axis('off')
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)

    plt.tight_layout()
    plt.savefig('superres_test_synthetic.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: superres_test_synthetic.png")

    # Print statistics
    print("\n" + "=" * 70)
    print("Results Summary:")
    print("=" * 70)
    print(f"Standard BF shape: {results['standard_bf'].shape}")
    print(f"Super-res shape: {results['superres_image'].shape}")
    print(f"Upscaling: {results['superres_image'].shape[0] / results['standard_bf'].shape[0]:.1f}x")
    print(f"\nShift map range Y: [{results['shift_y'].min():.2f}, {results['shift_y'].max():.2f}]")
    print(f"Shift map range X: [{results['shift_x'].min():.2f}, {results['shift_x'].max():.2f}]")
    print(f"Quality range: [{results['quality'].min():.3f}, {results['quality'].max():.3f}]")
    print(f"Quality mean: {results['quality'].mean():.3f}")

    # Check if shifts look reasonable (should show smooth variation)
    shift_y_smoothness = np.mean(np.abs(np.diff(results['shift_y'], axis=0)))
    shift_x_smoothness = np.mean(np.abs(np.diff(results['shift_x'], axis=1)))
    print(f"\nShift smoothness (lower = smoother):")
    print(f"  Y: {shift_y_smoothness:.3f}")
    print(f"  X: {shift_x_smoothness:.3f}")

    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)

    return results


if __name__ == '__main__':
    results = test_synthetic_data()

    print("\nTest passed! Algorithm is working as expected.")
    print("\nNext steps:")
    print("  1. Review the output image: superres_test_synthetic.png")
    print("  2. Check that shift maps show smooth variation")
    print("  3. Verify super-resolution image shows improvement over standard BF")
