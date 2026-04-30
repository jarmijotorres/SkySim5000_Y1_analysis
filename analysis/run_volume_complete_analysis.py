#!/usr/bin/env python
"""
Example script to run the volume complete sample analysis
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.volume_complete_samples import (
    create_volume_complete_sample,
    calculate_derived_quantities,
    save_volume_complete_catalog,
    plot_sample_properties
)
from data_loader.loader import DataLoader


def run_analysis(data_file_path):
    """
    Run the volume complete sample analysis
    
    Parameters
    ----------
    data_file_path : str
        Path to the SkySim5000 HDF5 file
    """
    print("="*60)
    print("Volume Complete Sample Analysis")
    print("="*60)
    
    # Configuration
    output_dir = Path("analysis/volume_complete_catalogs")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Redshift bins
    z_bins = [(0.0, 0.35), (0.35, 0.7), (0.7, 1.0)]
    
    # Survey parameters
    apparent_mag_limit = 25.0  # r-band limit
    band = 'r'
    
    # Load data
    print(f"\nLoading data from: {data_file_path}")
    
    columns_to_load = [
        'redshift_true',
        'stellar_mass',
        'sed_1246_306_disk_no_host_extinction',
        'A_v_disk',
        'Mag_true_u_lsst_z0',
        'Mag_true_g_lsst_z0',
        'Mag_true_r_lsst_z0',
        'Mag_true_i_lsst_z0',
        'Mag_true_z_lsst_z0'
    ]
    
    loader = DataLoader(data_file_path)
    data = loader.load_data(columns=columns_to_load)
    print(f"Loaded {len(loader.get_redshifts()):,} galaxies")
    
    # Process each redshift bin
    for z_min, z_max in z_bins:
        print(f"\n{'='*60}")
        print(f"Processing redshift bin: z = [{z_min:.2f}, {z_max:.2f}]")
        print(f"{'='*60}")
        
        # Create volume complete sample
        subset_loader, mask, abs_mag_limit = create_volume_complete_sample(
            loader, z_min, z_max, band=band, 
            apparent_mag_limit=apparent_mag_limit
        )
        
        # Calculate derived quantities
        results = calculate_derived_quantities(subset_loader)
        
        # Add sample metadata
        results['z_min'] = z_min
        results['z_max'] = z_max
        results['abs_mag_limit'] = abs_mag_limit
        
        # Save catalog
        filepath = save_volume_complete_catalog(
            results, z_min, z_max, output_dir
        )
        
        # Create diagnostic plots
        plot_sample_properties(results, z_min, z_max, output_dir)
        
        print(f"\nCompleted processing for z=[{z_min:.2f}, {z_max:.2f}]")
        print(f"Output saved to: {filepath}")
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"All catalogs saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python run_volume_complete_analysis.py <path_to_skysim5000.h5>")
        print("\nExample:")
        print("  python run_volume_complete_analysis.py /data/skysim5000/skysim5000.h5")
        sys.exit(1)
    
    data_file = sys.argv[1]
    
    # Check if file exists
    if not Path(data_file).exists():
        print(f"Error: Data file not found: {data_file}")
        sys.exit(1)
    
    # Run the analysis
    run_analysis(data_file)