#!/usr/bin/env python3
"""Compare BF and ADF shift maps to validate pairwise alignment approach.

This script:
1. Runs BF super-resolution (known to work)
2. Extracts BF shifts at the same 4 detector positions we tested
3. Compares them to the ADF pairwise alignment shifts
4. Validates whether the pairwise approach gives physically reasonable results
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from src.mib_viewer.processing.superres_processor import SuperResProcessor


def compare_bf_adf_shifts(emd_path: str,
                          bf_center: tuple = (127.6, 126.4),
                          test_radius: int = 45):
    """Compare BF and ADF shifts at the same detector positions."""
    
    print("\n" + "=" * 70)
    print("Comparing BF and ADF Shift Maps")
    print("=" * 70 + "\n")
    
    # Load data (smaller subset)
    print(f"Loading data from: {emd_path}")
    with h5py.File(emd_path, 'r') as f:
        data_4d = f['version_1/data/datacubes/datacube_000/data'][:128, :128, :, :]
    
    print(f"  Loaded 4D data shape: {data_4d.shape}")
    
    # Run BF super-resolution
    print("\nRunning BF super-resolution...")
    processor = SuperResProcessor()
    
    bf_results = processor.process_full(
        data_4d,
        bf_center=bf_center,
        bf_radius=32,
        reference_smoothing=0.5,
        upscale_factor=4,
        detector_radius=16,
        quality_threshold=0.7
    )
    
    print("  BF super-resolution complete!")
    
    # Get BF shift maps
    bf_shift_y = bf_results['shift_y']  # Shape: (det_y, det_x) in cropped coords
    bf_shift_x = bf_results['shift_x']
    bf_quality = bf_results['quality']
    
    # The BF results are in cropped coordinates (64x64 centered on BF disk)
    # We need to map our test pixel positions to these cropped coordinates
    
    cy, cx = int(bf_center[0]), int(bf_center[1])
    bf_radius = 32
    
    # Cropped region bounds
    y1 = cy - bf_radius
    x1 = cx - bf_radius
    
    # Our 4 test pixels in full detector coordinates
    angles_deg = [0, 90, 180, 270]
    angles_rad = [np.deg2rad(a) for a in angles_deg]
    labels = ['Top', 'Right', 'Bottom', 'Left']
    
    pixel_coords_full = []
    for angle in angles_rad:
        dy = -test_radius * np.cos(angle)
        dx = test_radius * np.sin(angle)
        pixel_y = int(cy + dy)
        pixel_x = int(cx + dx)
        pixel_coords_full.append((pixel_y, pixel_x))
    
    # Convert to cropped coordinates
    pixel_coords_cropped = []
    for py, px in pixel_coords_full:
        py_crop = py - y1
        px_crop = px - x1
        pixel_coords_cropped.append((py_crop, px_crop))
    
    print("\nPixel positions:")
    print("  Label  | Full Coords | Cropped Coords")
    print("  " + "-" * 45)
    for label, full, crop in zip(labels, pixel_coords_full, pixel_coords_cropped):
        print(f"  {label:6s} | ({full[0]:3d}, {full[1]:3d}) | ({crop[0]:3d}, {crop[1]:3d})")
    
    # Extract BF shifts at these positions
    sy, sx = data_4d.shape[0], data_4d.shape[1]
    
    print("\nBF Shifts (centered):")
    print("  Label  | Shift Y | Shift X | Quality | Magnitude")
    print("  " + "-" * 55)
    
    bf_shifts_at_pixels = []
    for label, (py_crop, px_crop) in zip(labels, pixel_coords_cropped):
        # Check if position is within cropped region
        if 0 <= py_crop < bf_shift_y.shape[0] and 0 <= px_crop < bf_shift_x.shape[1]:
            shift_y = bf_shift_y[py_crop, px_crop] - sy / 2
            shift_x = bf_shift_x[py_crop, px_crop] - sx / 2
            quality = bf_quality[py_crop, px_crop]
            magnitude = np.sqrt(shift_y**2 + shift_x**2)
            
            bf_shifts_at_pixels.append((shift_y, shift_x, quality))
            print(f"  {label:6s} | {shift_y:7.2f} | {shift_x:7.2f} | {quality:7.3f} | {magnitude:7.2f}")
        else:
            print(f"  {label:6s} | OUT OF BOUNDS")
            bf_shifts_at_pixels.append((np.nan, np.nan, np.nan))
    
    # ADF shifts from pairwise alignment (from previous test)
    # These are the final shifts we got
    adf_shifts = [
        (0.02, 0.10),   # Top
        (-0.03, -0.03), # Right
        (-0.01, 0.19),  # Bottom
        (0.02, 0.07),   # Left
    ]
    
    print("\nADF Shifts (from pairwise alignment):")
    print("  Label  | Shift Y | Shift X | Magnitude")
    print("  " + "-" * 45)
    
    for label, (shift_y, shift_x) in zip(labels, adf_shifts):
        magnitude = np.sqrt(shift_y**2 + shift_x**2)
        print(f"  {label:6s} | {shift_y:7.2f} | {shift_x:7.2f} | {magnitude:7.2f}")
    
    # Compare BF vs ADF
    print("\n" + "=" * 70)
    print("Comparison: BF vs ADF Shifts")
    print("=" * 70)
    print("  Label  | BF Shift    | ADF Shift   | Difference  | Angle Diff")
    print("  " + "-" * 70)
    
    for label, bf_shift, adf_shift in zip(labels, bf_shifts_at_pixels, adf_shifts):
        if not np.isnan(bf_shift[0]):
            bf_y, bf_x, _ = bf_shift
            adf_y, adf_x = adf_shift
            
            diff_y = bf_y - adf_y
            diff_x = bf_x - adf_x
            diff_mag = np.sqrt(diff_y**2 + diff_x**2)
            
            # Angle difference
            bf_angle = np.arctan2(bf_x, bf_y) * 180 / np.pi
            adf_angle = np.arctan2(adf_x, adf_y) * 180 / np.pi
            angle_diff = bf_angle - adf_angle
            
            print(f"  {label:6s} | ({bf_y:6.2f},{bf_x:6.2f}) | ({adf_y:6.2f},{adf_x:6.2f}) | "
                  f"{diff_mag:6.2f} px | {angle_diff:7.1f}°")
    
    # Visualize comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 0: BF shift maps
    sy_center, sx_center = sy // 2, sx // 2
    
    ax = axes[0, 0]
    shift_y_centered = bf_shift_y - sy_center
    im = ax.imshow(shift_y_centered, cmap='RdBu_r', vmin=-20, vmax=20)
    ax.set_title('BF Shift Map (Y)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Mark the 4 test pixels
    for (py_crop, px_crop), label in zip(pixel_coords_cropped, labels):
        if 0 <= py_crop < bf_shift_y.shape[0] and 0 <= px_crop < bf_shift_x.shape[1]:
            ax.plot(px_crop, py_crop, 'go', markersize=12, markerfacecolor='none', markeredgewidth=2)
            ax.text(px_crop, py_crop-3, label, color='green', ha='center', fontweight='bold')
    
    ax = axes[0, 1]
    shift_x_centered = bf_shift_x - sx_center
    im = ax.imshow(shift_x_centered, cmap='RdBu_r', vmin=-20, vmax=20)
    ax.set_title('BF Shift Map (X)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    for (py_crop, px_crop), label in zip(pixel_coords_cropped, labels):
        if 0 <= py_crop < bf_shift_y.shape[0] and 0 <= px_crop < bf_shift_x.shape[1]:
            ax.plot(px_crop, py_crop, 'go', markersize=12, markerfacecolor='none', markeredgewidth=2)
    
    ax = axes[0, 2]
    im = ax.imshow(bf_quality, cmap='viridis')
    ax.set_title('BF Quality Map')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    for (py_crop, px_crop), label in zip(pixel_coords_cropped, labels):
        if 0 <= py_crop < bf_shift_y.shape[0] and 0 <= px_crop < bf_shift_x.shape[1]:
            ax.plot(px_crop, py_crop, 'go', markersize=12, markerfacecolor='none', markeredgewidth=2)
    
    # Row 1: Comparison plots
    ax = axes[1, 0]
    # Plot BF shifts as vectors
    for label, bf_shift, color in zip(labels, bf_shifts_at_pixels, ['red', 'green', 'blue', 'orange']):
        if not np.isnan(bf_shift[0]):
            bf_y, bf_x, _ = bf_shift
            ax.arrow(0, 0, bf_x, bf_y, head_width=0.5, head_length=0.3, 
                    fc=color, ec=color, linewidth=2, label=f'{label} (BF)', alpha=0.7)
    
    ax.set_xlabel('Shift X (pixels)')
    ax.set_ylabel('Shift Y (pixels)')
    ax.set_title('BF Shift Vectors')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal')
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(0, color='k', linestyle='-', alpha=0.3)
    
    ax = axes[1, 1]
    # Plot ADF shifts as vectors
    for label, adf_shift, color in zip(labels, adf_shifts, ['red', 'green', 'blue', 'orange']):
        adf_y, adf_x = adf_shift
        ax.arrow(0, 0, adf_x, adf_y, head_width=0.05, head_length=0.03,
                fc=color, ec=color, linewidth=2, label=f'{label} (ADF)', alpha=0.7)
    
    ax.set_xlabel('Shift X (pixels)')
    ax.set_ylabel('Shift Y (pixels)')
    ax.set_title('ADF Shift Vectors (Pairwise)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal')
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(0, color='k', linestyle='-', alpha=0.3)
    
    ax = axes[1, 2]
    # Overlay both
    for label, bf_shift, adf_shift, color in zip(labels, bf_shifts_at_pixels, adf_shifts, 
                                                   ['red', 'green', 'blue', 'orange']):
        if not np.isnan(bf_shift[0]):
            bf_y, bf_x, _ = bf_shift
            adf_y, adf_x = adf_shift
            
            # BF as solid arrow
            ax.arrow(0, 0, bf_x, bf_y, head_width=0.5, head_length=0.3,
                    fc=color, ec=color, linewidth=2, alpha=0.5, linestyle='-')
            
            # ADF as dashed arrow
            ax.arrow(0, 0, adf_x, adf_y, head_width=0.05, head_length=0.03,
                    fc='none', ec=color, linewidth=2, linestyle='--')
            
            # Label
            ax.text(bf_x, bf_y, label, color=color, fontweight='bold')
    
    ax.set_xlabel('Shift X (pixels)')
    ax.set_ylabel('Shift Y (pixels)')
    ax.set_title('BF (solid) vs ADF (dashed)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    output_path = 'bf_adf_shift_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved comparison to: {output_path}")
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == '__main__':
    emd_path = "/media/o2d/data/ORNL Dropbox/Ondrej Dyck/Code/mib-viewer-project/SS_a7_2ms_256x256 4D_16nmFoV -20nmFocus.emd"
    
    compare_bf_adf_shifts(
        emd_path,
        bf_center=(127.6, 126.4),
        test_radius=45
    )
    
    print("\nKey observations to look for:")
    print("  1. Are BF and ADF shifts similar in magnitude?")
    print("  2. Do they point in similar directions?")
    print("  3. If different, is it systematic or random?")
    print("  4. This tells us if pairwise alignment captures real physics!")
