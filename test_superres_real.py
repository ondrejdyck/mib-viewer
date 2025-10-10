#!/usr/bin/env python3
"""Test super-resolution algorithm on real 4D STEM data."""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from src.mib_viewer.processing.superres_processor import SuperResProcessor


def load_4d_data(emd_path):
    """Load 4D STEM data from EMD file."""
    print(f"Loading data from: {emd_path}")

    with h5py.File(emd_path, 'r') as f:
        # Find the 4D datacube
        datacube_path = 'version_1/data/datacubes/datacube_000/data'

        if datacube_path in f:
            data_4d = f[datacube_path][:]
            print(f"  Loaded 4D data shape: {data_4d.shape}")
            print(f"  Data type: {data_4d.dtype}")
            print(f"  Data range: [{data_4d.min()}, {data_4d.max()}]")
            return data_4d
        else:
            print(f"  ERROR: Could not find datacube at {datacube_path}")
            print(f"  Available paths:")
            def print_structure(name, obj):
                print(f"    {name}")
            f.visititems(print_structure)
            return None


def test_real_data(emd_path):
    """Test super-resolution on real 4D STEM data."""
    print("\n" + "=" * 70)
    print("Testing Super-Resolution Algorithm on Real 4D STEM Data")
    print("=" * 70 + "\n")

    # Load data
    data_4d = load_4d_data(emd_path)
    if data_4d is None:
        return None

    # Run super-resolution
    processor = SuperResProcessor()

    print("\nRunning super-resolution reconstruction...")
    results = processor.process_full(
        data_4d,
        bf_radius=32,
        reference_smoothing=0.5,
        upscale_factor=4,
        detector_radius=16,
        quality_threshold=0.7
    )

    # Visualize results
    fig = plt.figure(figsize=(18, 12))

    # Create grid: 3 rows x 3 columns
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Row 1: Main images
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(results['standard_bf'], cmap='gray')
    ax1.set_title(f"Standard BF (Central Pixel)\n{results['standard_bf'].shape}")
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1:])
    ax2.imshow(results['superres_image'], cmap='gray')
    upscale = results['superres_image'].shape[0] / results['standard_bf'].shape[0]
    ax2.set_title(f"Super-Resolution Reconstruction ({upscale:.1f}x)\n{results['superres_image'].shape}")
    ax2.axis('off')

    # Row 2: Shift maps (centered like notebook visualization)
    sy, sx = results['data_cropped'].shape[:2]

    ax3 = fig.add_subplot(gs[1, 0])
    shift_y_centered = results['shift_y'] - sy // 2
    im3 = ax3.imshow(shift_y_centered, cmap='RdBu_r', vmin=-20, vmax=20)
    ax3.set_title('Shift Map (Y) - Centered')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    ax4 = fig.add_subplot(gs[1, 1])
    shift_x_centered = results['shift_x'] - sx // 2
    im4 = ax4.imshow(shift_x_centered, cmap='RdBu_r', vmin=-20, vmax=20)
    ax4.set_title('Shift Map (X) - Centered')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)

    ax5 = fig.add_subplot(gs[1, 2])
    im5 = ax5.imshow(results['quality'], cmap='viridis')
    ax5.set_title('Quality Map')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046)

    # Row 3: Additional diagnostics
    ax6 = fig.add_subplot(gs[2, 0])
    im6 = ax6.imshow(results['normalization'], cmap='hot')
    ax6.set_title('Contribution Count')
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046)

    ax7 = fig.add_subplot(gs[2, 1])
    ax7.imshow(results['reference_image'], cmap='gray')
    ax7.set_title('Reference Image (Smoothed Central)')
    ax7.axis('off')

    # Shift vector field (subsampled for clarity)
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
    ax8.set_title('Shift Vectors (centered, subsampled)')
    ax8.set_aspect('equal')
    ax8.invert_yaxis()

    output_path = 'superres_test_real.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved visualization to: {output_path}")

    # Print statistics
    print("\n" + "=" * 70)
    print("Results Summary:")
    print("=" * 70)
    print(f"Input 4D shape: {data_4d.shape}")
    print(f"BF center used: ({results['bf_center'][0]:.2f}, {results['bf_center'][1]:.2f})")
    print(f"\nStandard BF shape: {results['standard_bf'].shape}")
    print(f"Super-res shape: {results['superres_image'].shape}")
    print(f"Upscaling factor: {results['superres_image'].shape[0] / results['standard_bf'].shape[0]:.1f}x")

    sy_stats, sx_stats = results['data_cropped'].shape[:2]
    shift_y_centered_stats = results['shift_y'] - sy_stats // 2
    shift_x_centered_stats = results['shift_x'] - sx_stats // 2

    print(f"\nShift map Y (centered): [{shift_y_centered_stats.min():.2f}, {shift_y_centered_stats.max():.2f}]")
    print(f"Shift map X (centered): [{shift_x_centered_stats.min():.2f}, {shift_x_centered_stats.max():.2f}]")

    print(f"\nQuality: [{results['quality'].min():.3f}, {results['quality'].max():.3f}]")
    print(f"Quality mean: {results['quality'].mean():.3f}")
    print(f"Quality median: {np.median(results['quality']):.3f}")

    print(f"\nContribution count: [{results['normalization'].min():.0f}, {results['normalization'].max():.0f}]")
    print(f"Average contributions per pixel: {results['normalization'][results['normalization'] > 0].mean():.1f}")

    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)

    return results


if __name__ == '__main__':
    emd_path = "/media/o2d/data/ORNL Dropbox/Ondrej Dyck/Code/mib-viewer-project/SS_a7_2ms_256x256 4D_16nmFoV -20nmFocus.emd"

    results = test_real_data(emd_path)

    if results is not None:
        print("\n✅ Test passed! Check superres_test_real.png for results.")
        print("\nNext steps:")
        print("  1. Review the super-resolution image quality")
        print("  2. Check that shift maps show smooth, physically reasonable patterns")
        print("  3. Verify quality map shows high correlation values")
        print("  4. Assess whether we're ready for GUI integration")
