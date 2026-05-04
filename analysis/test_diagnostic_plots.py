#!/usr/bin/env python
"""
Test script to verify the diagnostic plots modifications
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.run_volume_complete_analysis import (
    plot_cmd_with_marginals,
    plot_sfr_distributions_comparison,
    plot_metallicity_distribution,
    save_all_redshift_bins_hdf5
)


def create_mock_data(n_galaxies=10000, z_min=0.0, z_max=0.35):
    """Create mock galaxy data for testing"""
    np.random.seed(42)
    
    results = {}
    
    # Basic properties
    results['redshift_true'] = np.random.uniform(z_min, z_max, n_galaxies)
    results['stellar_mass'] = 10**(np.random.normal(10, 0.5, n_galaxies))
    results['log_stellar_mass'] = np.log10(results['stellar_mass'])
    
    # Magnitudes
    results['Mag_abs_r'] = np.random.normal(-20, 1.5, n_galaxies)
    
    # Colors - create bimodal distribution
    red_fraction = 0.3
    n_red = int(n_galaxies * red_fraction)
    n_blue = n_galaxies - n_red
    
    color_red = np.random.normal(0.8, 0.1, n_red)
    color_blue = np.random.normal(0.3, 0.15, n_blue)
    results['colour_gr'] = np.concatenate([color_red, color_blue])
    np.random.shuffle(results['colour_gr'])
    
    # SFR
    results['log_sfr_uv_corr'] = np.random.normal(0.5, 1.0, n_galaxies)
    results['sfr_uv_corr'] = 10**results['log_sfr_uv_corr']
    results['log_ssfr'] = results['log_sfr_uv_corr'] - results['log_stellar_mass']
    results['ssfr'] = 10**results['log_ssfr']
    
    # Metallicity
    results['metallicity_fmr'] = np.random.normal(8.5, 0.3, n_galaxies)
    
    # Star-forming classification
    results['is_star_forming'] = results['log_ssfr'] > -11
    
    # Metadata
    results['z_min'] = z_min
    results['z_max'] = z_max
    results['abs_mag_limit'] = -18.5
    
    return results


def test_diagnostic_plots():
    """Test all diagnostic plot functions"""
    print("Testing diagnostic plots...")
    
    # Create output directory
    output_dir = Path("analysis/test_plots")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Define redshift bins
    z_bins = [(0.0, 0.35), (0.35, 0.7), (0.7, 1.0)]
    
    # Create mock data for each redshift bin
    all_results = []
    for z_min, z_max in z_bins:
        print(f"\nCreating mock data for z=[{z_min:.2f}, {z_max:.2f}]")
        results = create_mock_data(n_galaxies=5000, z_min=z_min, z_max=z_max)
        all_results.append(results)
        
        # Test individual plots
        print("  Testing CMD with marginals...")
        plot_cmd_with_marginals(results, z_min, z_max, output_dir)
        
        print("  Testing metallicity distribution...")
        plot_metallicity_distribution(results, z_min, z_max, output_dir)
    
    # Test combined plots
    print("\nTesting SFR/sSFR distributions comparison...")
    plot_sfr_distributions_comparison(all_results, z_bins, output_dir)
    
    # Test HDF5 saving
    print("\nTesting HDF5 saving...")
    hdf5_file = save_all_redshift_bins_hdf5(all_results, z_bins, output_dir)
    
    # Verify HDF5 file
    import h5py
    print(f"\nVerifying HDF5 file: {hdf5_file}")
    with h5py.File(hdf5_file, 'r') as f:
        print(f"  Number of redshift bins: {f.attrs['n_redshift_bins']}")
        print("  Groups in file:")
        for group_name in f.keys():
            grp = f[group_name]
            print(f"    {group_name}: {grp.attrs['n_galaxies']} galaxies")
            print(f"      Datasets: {list(grp.keys())[:5]}...")
    
    print("\n✓ All tests completed successfully!")
    print(f"  Test plots saved to: {output_dir}")


if __name__ == "__main__":
    test_diagnostic_plots()