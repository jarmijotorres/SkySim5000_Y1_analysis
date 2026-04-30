#!/usr/bin/env python
"""
Test script for volume complete sample creation
This creates a small test dataset to verify the functionality
"""

import numpy as np
import h5py
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.volume_complete_samples import (
    calculate_absolute_magnitude_limit,
    create_volume_complete_sample,
    calculate_derived_quantities
)
from data_loader.loader import DataLoader
from utils.constants import COSMOLOGY


def create_test_data(n_galaxies=10000, output_file='test_skysim5000.h5'):
    """
    Create a test dataset mimicking SkySim5000 structure
    """
    print("Creating test dataset...")
    
    # Generate mock data
    np.random.seed(42)
    
    # Redshifts uniformly distributed
    z = np.random.uniform(0.0, 1.2, n_galaxies)
    
    # Stellar masses log-normal
    log_mass = np.random.normal(10.0, 0.5, n_galaxies)
    stellar_mass = 10**log_mass
    
    # Absolute magnitudes correlated with mass
    M_r = -20.0 - 2.5 * (log_mass - 10.0) + np.random.normal(0, 0.3, n_galaxies)
    M_g = M_r - 0.5 + np.random.normal(0, 0.1, n_galaxies)  # g-r ~ 0.5
    M_u = M_g - 1.2 + np.random.normal(0, 0.15, n_galaxies)  # u-g ~ 1.2
    M_i = M_r + 0.3 + np.random.normal(0, 0.1, n_galaxies)   # r-i ~ 0.3
    M_z = M_i + 0.2 + np.random.normal(0, 0.1, n_galaxies)   # i-z ~ 0.2
    
    # UV luminosity (simplified)
    sed_uv = 10**(0.4 * (-M_u - 25.0)) * 1e10  # Rough conversion
    
    # Dust extinction
    A_v = np.random.exponential(0.5, n_galaxies)
    A_v = np.clip(A_v, 0, 3)
    
    # Save to HDF5
    with h5py.File(output_file, 'w') as f:
        grp = f.create_group('SkySim5000_small')
        
        grp.create_dataset('redshift_true', data=z)
        grp.create_dataset('stellar_mass', data=stellar_mass)
        grp.create_dataset('Mag_true_u_lsst_z0', data=M_u)
        grp.create_dataset('Mag_true_g_lsst_z0', data=M_g)
        grp.create_dataset('Mag_true_r_lsst_z0', data=M_r)
        grp.create_dataset('Mag_true_i_lsst_z0', data=M_i)
        grp.create_dataset('Mag_true_z_lsst_z0', data=M_z)
        grp.create_dataset('sed_1246_306_disk_no_host_extinction', data=sed_uv)
        grp.create_dataset('A_v_disk', data=A_v)
    
    print(f"Created test dataset with {n_galaxies} galaxies: {output_file}")
    return output_file


def test_magnitude_limit_calculation():
    """Test the magnitude limit calculation"""
    print("\nTesting magnitude limit calculation...")
    
    # Test cases
    test_cases = [
        (25.0, 0.35),  # Low redshift
        (25.0, 0.70),  # Medium redshift
        (25.0, 1.00),  # High redshift
    ]
    
    for app_mag, z in test_cases:
        abs_mag = calculate_absolute_magnitude_limit(app_mag, z)
        dist_mod = COSMOLOGY.distmod(z).value
        print(f"  z={z:.2f}: m_lim={app_mag:.1f} -> M_lim={abs_mag:.2f} (DM={dist_mod:.2f})")


def test_volume_complete_sample(data_file):
    """Test creating volume complete samples"""
    print("\nTesting volume complete sample creation...")
    
    # Load data
    loader = DataLoader(data_file)
    data = loader.load_data()
    
    # Test for one redshift bin
    z_min, z_max = 0.35, 0.70
    
    print(f"\nCreating sample for z=[{z_min}, {z_max}]...")
    subset_loader, mask, abs_mag_limit = create_volume_complete_sample(
        loader, z_min, z_max, band='r', apparent_mag_limit=25.0, verbose=True
    )
    
    # Calculate derived quantities
    print("\nCalculating derived quantities...")
    results = calculate_derived_quantities(subset_loader, verbose=True)
    
    # Print summary statistics
    print("\nSample Statistics:")
    print(f"  Number of galaxies: {len(results['redshift_true']):,}")
    print(f"  Redshift range: {results['redshift_true'].min():.3f} - {results['redshift_true'].max():.3f}")
    print(f"  log(M*) range: {results['log_stellar_mass'].min():.2f} - {results['log_stellar_mass'].max():.2f}")
    
    if 'colour_gr' in results:
        print(f"  (g-r) color range: {np.nanmin(results['colour_gr']):.2f} - {np.nanmax(results['colour_gr']):.2f}")
    
    if 'log_sfr_uv_corr' in results:
        valid_sfr = np.isfinite(results['log_sfr_uv_corr'])
        if valid_sfr.any():
            print(f"  log(SFR) range: {results['log_sfr_uv_corr'][valid_sfr].min():.2f} - {results['log_sfr_uv_corr'][valid_sfr].max():.2f}")
    
    if 'metallicity_fmr' in results:
        valid_z = np.isfinite(results['metallicity_fmr'])
        if valid_z.any():
            print(f"  Metallicity range: {results['metallicity_fmr'][valid_z].min():.2f} - {results['metallicity_fmr'][valid_z].max():.2f}")
    
    return results


def main():
    """Run all tests"""
    print("="*60)
    print("Volume Complete Sample Test Suite")
    print("="*60)
    
    # Create test data
    test_file = 'test_skysim5000.h5'
    if not Path(test_file).exists():
        create_test_data(output_file=test_file)
    else:
        print(f"Using existing test file: {test_file}")
    
    # Run tests
    test_magnitude_limit_calculation()
    results = test_volume_complete_sample(test_file)
    
    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)
    
    # Cleanup option
    response = input("\nDelete test file? (y/n): ")
    if response.lower() == 'y':
        Path(test_file).unlink()
        print("Test file deleted.")


if __name__ == "__main__":
    main()