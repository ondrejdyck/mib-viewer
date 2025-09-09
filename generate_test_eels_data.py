#!/usr/bin/env python3
"""
Generate fake 4D EELS dataset for testing
Creates realistic synthetic EELS data with energy loss peaks
"""

import numpy as np
import h5py
import os
from pathlib import Path

def create_fake_eels_data(scan_shape=(64, 64), detector_shape=(256, 1024), 
                         energy_range=(0, 2048), peaks=None):
    """
    Create synthetic 4D EELS dataset
    
    Parameters:
    -----------
    scan_shape : tuple
        (scan_y, scan_x) dimensions
    detector_shape : tuple  
        (detector_y, detector_x) where detector_x is energy axis
    energy_range : tuple
        (min_energy, max_energy) in eV
    peaks : list of dicts
        Energy loss peaks with format:
        [{'energy': 285, 'width': 5, 'intensity': 1000}, ...]
    """
    if peaks is None:
        # Default peaks: Zero loss, Carbon K-edge, Plasmon
        peaks = [
            {'energy': 0, 'width': 2, 'intensity': 5000},      # Zero loss peak
            {'energy': 285, 'width': 8, 'intensity': 1200},    # Carbon K-edge
            {'energy': 20, 'width': 4, 'intensity': 800},      # Plasmon
        ]
    
    sy, sx = scan_shape
    dy, dx = detector_shape
    
    print(f"Generating 4D EELS data: {sy}×{sx} scan, {dy}×{dx} detector")
    print(f"Energy range: {energy_range[0]}-{energy_range[1]} eV")
    
    # Create energy axis (maps to detector_x)
    energy_axis = np.linspace(energy_range[0], energy_range[1], dx)
    
    # Initialize 4D array
    data_4d = np.zeros((sy, sx, dy, dx), dtype=np.uint16)
    
    # Add background (decreasing with energy)
    background = 100 * np.exp(-energy_axis / 200)  # Exponential background
    background = np.maximum(background, 10)  # Minimum background
    
    # Create spatial variations (simulate sample thickness/composition changes)
    # Thickness varies across scan
    thickness_map = 0.8 + 0.4 * np.sin(np.linspace(0, 4*np.pi, sy))[:, None]
    thickness_map = thickness_map * (0.9 + 0.2 * np.sin(np.linspace(0, 6*np.pi, sx))[None, :])
    
    # Composition variation (affects peak intensities)
    carbon_map = 0.5 + 0.5 * np.cos(np.linspace(0, 3*np.pi, sy))[:, None]
    carbon_map = carbon_map * (0.7 + 0.3 * np.cos(np.linspace(0, 5*np.pi, sx))[None, :])
    
    for scan_y in range(sy):
        for scan_x in range(sx):
            # Get local thickness and composition
            thickness = thickness_map[scan_y, scan_x]
            carbon_content = carbon_map[scan_y, scan_x]
            
            # Start with background scaled by thickness
            spectrum = background * thickness
            
            # Add peaks with spatial variation
            for peak in peaks:
                energy = peak['energy']
                width = peak['width']
                base_intensity = peak['intensity']
                
                # Apply spatial modulation
                if energy == 285:  # Carbon K-edge
                    intensity = base_intensity * carbon_content * thickness
                elif energy == 0:  # Zero loss
                    intensity = base_intensity * thickness
                else:  # Other peaks
                    intensity = base_intensity * thickness * 0.8
                
                # Create Gaussian peak
                gaussian = intensity * np.exp(-((energy_axis - energy) / width)**2)
                spectrum += gaussian
            
            # Add noise
            noise = np.random.poisson(spectrum * 0.1)  # Poisson noise
            spectrum += noise
            
            # Ensure positive values and convert to uint16
            spectrum = np.maximum(spectrum, 0)
            spectrum = np.clip(spectrum, 0, 65535).astype(np.uint16)
            
            # Fill all detector Y positions with the same spectrum
            # (simulating Y-summing would collapse this to 1D)
            data_4d[scan_y, scan_x, :, :] = spectrum[None, :]
    
    print(f"Generated data with peaks at: {[p['energy'] for p in peaks]} eV")
    return data_4d, energy_axis, peaks

def save_as_emd(data_4d, filename, energy_axis=None, metadata=None):
    """Save 4D data as EMD 1.0 format"""
    print(f"Saving as EMD: {filename}")
    
    with h5py.File(filename, 'w') as f:
        # EMD 1.0 structure
        f.attrs['emd_group_type'] = 'file'
        f.attrs['version_major'] = 1
        f.attrs['version_minor'] = 0
        f.attrs['authoring_program'] = 'fake-eels-generator'
        
        version_group = f.create_group('version_1')
        version_group.attrs['emd_group_type'] = 'root'
        
        # Data group
        data_group = version_group.create_group('data')
        datacubes_group = data_group.create_group('datacubes')
        datacube_group = datacubes_group.create_group('datacube_000')
        datacube_group.attrs['emd_group_type'] = 'array'
        
        # Main dataset
        dataset = datacube_group.create_dataset('data', data=data_4d, 
                                               compression='gzip', compression_opts=6)
        dataset.attrs['units'] = 'counts'
        
        # Dimension datasets
        sy, sx, dy, dx = data_4d.shape
        datacube_group.create_dataset('dim1', data=np.arange(sy))
        datacube_group.create_dataset('dim2', data=np.arange(sx))
        datacube_group.create_dataset('dim3', data=np.arange(dy))
        
        if energy_axis is not None:
            datacube_group.create_dataset('dim4', data=energy_axis)
        else:
            datacube_group.create_dataset('dim4', data=np.arange(dx))
        
        # Dimension attributes
        datacube_group['dim1'].attrs['name'] = 'scan_y'
        datacube_group['dim1'].attrs['units'] = 'pixel'
        datacube_group['dim2'].attrs['name'] = 'scan_x'
        datacube_group['dim2'].attrs['units'] = 'pixel'
        datacube_group['dim3'].attrs['name'] = 'detector_y'
        datacube_group['dim3'].attrs['units'] = 'pixel'
        datacube_group['dim4'].attrs['name'] = 'energy'
        datacube_group['dim4'].attrs['units'] = 'eV'
        
        # Metadata
        metadata_group = version_group.create_group('metadata')
        metadata_group.attrs['experiment_type'] = 'EELS'
        metadata_group.attrs['detector_type'] = f'Synthetic EELS ({dy}×{dx})'
        metadata_group.attrs['is_synthetic'] = True
        
        if metadata:
            for key, value in metadata.items():
                metadata_group.attrs[key] = value

def main():
    """Generate test datasets"""
    
    # Create output directory
    output_dir = Path("test_data")
    output_dir.mkdir(exist_ok=True)
    
    print("=== Generating Fake 4D EELS Test Data ===\n")
    
    # Generate main test dataset (64x64, 256x1024)
    print("1. Main test dataset (64×64 scan, 256×1024 detector)")
    data_4d, energy_axis, peaks = create_fake_eels_data(
        scan_shape=(64, 64),
        detector_shape=(256, 1024),
        energy_range=(0, 2048)
    )
    
    # Save as EMD
    emd_filename = output_dir / "fake_4d_eels_64x64_256x1024.emd"
    save_as_emd(data_4d, emd_filename, energy_axis, {
        'scan_size': (64, 64),
        'detector_size': (256, 1024),
        'peaks': str(peaks)
    })
    
    print(f"✓ Saved: {emd_filename}")
    print(f"  Shape: {data_4d.shape}")
    print(f"  Size: {os.path.getsize(emd_filename) / 1024**2:.1f} MB")
    
    # Generate smaller test dataset for quick testing
    print("\n2. Small test dataset (16×16 scan, 256×1024 detector)")
    data_4d_small, energy_axis_small, _ = create_fake_eels_data(
        scan_shape=(16, 16),
        detector_shape=(256, 1024),
        energy_range=(0, 2048)
    )
    
    emd_filename_small = output_dir / "fake_4d_eels_16x16_256x1024.emd"
    save_as_emd(data_4d_small, emd_filename_small, energy_axis_small, {
        'scan_size': (16, 16),
        'detector_size': (256, 1024),
        'peaks': str(peaks)
    })
    
    print(f"✓ Saved: {emd_filename_small}")
    print(f"  Shape: {data_4d_small.shape}")
    print(f"  Size: {os.path.getsize(emd_filename_small) / 1024**2:.1f} MB")
    
    # Generate opposite orientation for testing transpose logic
    print("\n3. Transposed detector dataset (64×64 scan, 1024×256 detector)")
    data_4d_transposed, energy_axis_t, _ = create_fake_eels_data(
        scan_shape=(64, 64),
        detector_shape=(1024, 256),
        energy_range=(0, 2048)
    )
    
    emd_filename_t = output_dir / "fake_4d_eels_64x64_1024x256.emd"
    save_as_emd(data_4d_transposed, emd_filename_t, energy_axis_t, {
        'scan_size': (64, 64),
        'detector_size': (1024, 256),
        'peaks': str(peaks)
    })
    
    print(f"✓ Saved: {emd_filename_t}")
    print(f"  Shape: {data_4d_transposed.shape}")
    print(f"  Size: {os.path.getsize(emd_filename_t) / 1024**2:.1f} MB")
    
    print(f"\n=== Test datasets created in {output_dir}/ ===")
    print("\nThese datasets contain:")
    print("- Zero loss peak at 0 eV")
    print("- Plasmon peak at 20 eV") 
    print("- Carbon K-edge at 285 eV")
    print("- Spatial variations in thickness and composition")
    print("- Realistic noise and background")

if __name__ == "__main__":
    main()