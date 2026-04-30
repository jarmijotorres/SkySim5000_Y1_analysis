#!/usr/bin/env python
"""
Script to create volume complete samples for different redshift bins
from the SkySim5000 galaxy catalog.

This script:
1. Loads the SkySim5000 data
2. Defines 3 redshift bins: [0.0-0.35], [0.35-0.7], [0.7-1.0]
3. Finds magnitude limits at zmax for volume completeness
4. Calculates inferred quantities (colours, SFR, metallicity)
5. Saves each redshift slice as a new catalogue
"""

import numpy as np
import pandas as pd
import h5py
from pathlib import Path
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader.loader import DataLoader
from sfr_calculator.sfr_calc import SFRCalculator
from metallicity.metallicity_calc import MetallicityCalculator
from colour_analysis.colour_analyzer import ColourAnalysis
from utils.constants import COSMOLOGY
import matplotlib.pyplot as plt


def calculate_absolute_magnitude_limit(apparent_mag_limit, z, cosmology=COSMOLOGY):
    """
    Calculate absolute magnitude limit from apparent magnitude limit at given redshift
    
    Parameters
    ----------
    apparent_mag_limit : float
        Apparent magnitude limit
    z : float
        Redshift
    cosmology : astropy.cosmology
        Cosmology to use
        
    Returns
    -------
    float
        Absolute magnitude limit
    """
    # Distance modulus
    dist_mod = cosmology.distmod(z).value
    
    # K-correction approximation (simplified - in reality would need SED fitting)
    # For r-band, approximate K-correction as K(z) ~ -0.5 * z for z < 1
    k_correction = -0.5 * z
    
    # Absolute magnitude limit
    abs_mag_limit = apparent_mag_limit - dist_mod - k_correction
    
    return abs_mag_limit


def create_volume_complete_sample(data_loader, z_min, z_max, band='r', 
                                 apparent_mag_limit=25.0, verbose=True):
    """
    Create volume complete sample for a redshift bin
    
    Parameters
    ----------
    data_loader : DataLoader
        Loaded data
    z_min : float
        Minimum redshift
    z_max : float
        Maximum redshift
    band : str
        Photometric band for magnitude limit
    apparent_mag_limit : float
        Survey apparent magnitude limit
    verbose : bool
        Print progress messages
        
    Returns
    -------
    dict
        Volume complete sample data
    """
    if verbose:
        print(f"\nCreating volume complete sample for z=[{z_min:.2f}, {z_max:.2f}]")
    
    # Get redshifts
    z_true = data_loader.get_redshifts(true=True)
    
    # Select galaxies in redshift bin
    z_mask = (z_true >= z_min) & (z_true < z_max)
    n_total = z_mask.sum()
    
    if verbose:
        print(f"  Total galaxies in redshift bin: {n_total:,}")
    
    # Calculate absolute magnitude limit at z_max
    abs_mag_limit = calculate_absolute_magnitude_limit(apparent_mag_limit, z_max)
    
    if verbose:
        print(f"  Absolute magnitude limit at z_max={z_max:.2f}: M_{band} < {abs_mag_limit:.2f}")
    
    # Get absolute magnitudes
    abs_mags = data_loader.get_magnitudes(band=band, z0=True)
    
    # Apply magnitude cut for volume completeness
    mag_mask = abs_mags < abs_mag_limit
    
    # Combined mask
    final_mask = z_mask & mag_mask
    n_complete = final_mask.sum()
    
    if verbose:
        print(f"  Volume complete sample size: {n_complete:,} ({100*n_complete/n_total:.1f}%)")
    
    # Create subset data loader
    subset_loader = data_loader.apply_selection(final_mask)
    
    return subset_loader, final_mask, abs_mag_limit


def calculate_derived_quantities(data_loader, verbose=True):
    """
    Calculate derived quantities (colours, SFR, metallicity) for the sample
    
    Parameters
    ----------
    data_loader : DataLoader
        Data loader with selected sample
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        Dictionary with all derived quantities
    """
    if verbose:
        print("  Calculating derived quantities...")
    
    results = {}
    
    # Get basic properties
    results['redshift_true'] = data_loader.get_redshifts(true=True)
    results['stellar_mass'] = data_loader.get_stellar_masses()
    results['log_stellar_mass'] = np.log10(results['stellar_mass'])
    
    # Calculate colours
    if verbose:
        print("    - Calculating colours")
    
    colour_analyzer = ColourAnalysis(data_loader)
    
    # Common colours
    for bands in [('u', 'r'), ('g', 'r'), ('r', 'i'), ('g', 'i')]:
        colour_name = f'{bands[0]}{bands[1]}'
        try:
            results[f'colour_{colour_name}'] = data_loader.get_colors(bands)
        except:
            if verbose:
                print(f"      Warning: Could not calculate {bands[0]}-{bands[1]} colour")
    
    # Calculate SFR
    if verbose:
        print("    - Calculating star formation rates")
    
    sfr_calc = SFRCalculator()
    
    # Get UV luminosity
    sed_uv = data_loader.get_sed_data()
    L_UV_erg = sfr_calc.calculate_uv_luminosity(sed_uv)
    
    # Calculate SFR (without dust correction)
    log_sfr_obs, sfr_obs = sfr_calc.calculate_sfr_uv(L_UV_erg)
    
    # Apply dust correction
    A_v = data_loader.get_dust_extinction()
    log_sfr_corr = sfr_calc.apply_dust_correction(log_sfr_obs, A_v)
    sfr_corr = 10**log_sfr_corr
    
    results['sfr_uv_obs'] = sfr_obs
    results['log_sfr_uv_obs'] = log_sfr_obs
    results['sfr_uv_corr'] = sfr_corr
    results['log_sfr_uv_corr'] = log_sfr_corr
    results['A_v'] = A_v
    
    # Calculate sSFR
    ssfr, log_ssfr = sfr_calc.calculate_ssfr(sfr_corr, results['stellar_mass'])
    results['ssfr'] = ssfr
    results['log_ssfr'] = log_ssfr
    
    # Classify star-forming vs quenched
    sf_class = sfr_calc.classify_sfms(log_ssfr)
    results['is_star_forming'] = sf_class['star_forming']
    results['is_quenched'] = sf_class['quenched']
    
    # Calculate metallicity
    if verbose:
        print("    - Calculating metallicities")
    
    metal_calc = MetallicityCalculator()
    
    # FMR metallicity
    Z_fmr, mu_fmr = metal_calc.fmr_metallicity(
        results['log_stellar_mass'], 
        results['log_sfr_uv_corr']
    )
    results['metallicity_fmr'] = Z_fmr
    results['fmr_mu'] = mu_fmr
    
    # Mass-metallicity relation
    Z_mzr = metal_calc.mass_metallicity_relation(
        results['log_stellar_mass'],
        relation='tremonti2004'
    )
    results['metallicity_mzr'] = Z_mzr
    
    # Get magnitudes for all bands
    for band in ['u', 'g', 'r', 'i', 'z']:
        try:
            results[f'Mag_abs_{band}'] = data_loader.get_magnitudes(band=band, z0=True)
        except:
            if verbose:
                print(f"      Warning: Could not get {band}-band magnitude")
    
    return results


def save_volume_complete_catalog(results, z_min, z_max, output_dir, verbose=True):
    """
    Save volume complete catalog to file
    
    Parameters
    ----------
    results : dict
        Dictionary with all galaxy properties
    z_min : float
        Minimum redshift of bin
    z_max : float
        Maximum redshift of bin
    output_dir : str or Path
        Output directory
    verbose : bool
        Print progress
        
    Returns
    -------
    Path
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create filename
    filename = f"skysim5000_volume_complete_z{z_min:.2f}-{z_max:.2f}.h5"
    filepath = output_dir / filename
    
    if verbose:
        print(f"  Saving catalog to: {filepath}")
    
    # Convert to pandas DataFrame for easier handling
    df = pd.DataFrame(results)
    
    # Save to HDF5
    with h5py.File(filepath, 'w') as f:
        # Add metadata
        f.attrs['z_min'] = z_min
        f.attrs['z_max'] = z_max
        f.attrs['n_galaxies'] = len(df)
        f.attrs['description'] = f'Volume complete sample for z=[{z_min:.2f}, {z_max:.2f}]'
        
        # Create main group
        grp = f.create_group('data')
        
        # Save each column
        for col in df.columns:
            data = df[col].values
            if data.dtype == bool:
                data = data.astype(np.uint8)
            grp.create_dataset(col, data=data, compression='gzip')
    
    # Also save as CSV for convenience
    csv_filepath = filepath.with_suffix('.csv')
    df.to_csv(csv_filepath, index=False)
    
    if verbose:
        print(f"  Also saved as CSV: {csv_filepath}")
        print(f"  Sample contains {len(df):,} galaxies")
    
    return filepath


def plot_sample_properties(results, z_min, z_max, output_dir):
    """
    Create diagnostic plots for the volume complete sample
    """
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Volume Complete Sample: z=[{z_min:.2f}, {z_max:.2f}]', fontsize=14)
    
    # 1. Stellar mass distribution
    ax = axes[0, 0]
    ax.hist(results['log_stellar_mass'], bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel(r'$\log(M_*/M_\odot)$')
    ax.set_ylabel('Number of galaxies')
    ax.set_title('Stellar Mass Distribution')
    
    # 2. Color-magnitude diagram
    ax = axes[0, 1]
    if 'colour_gr' in results and 'Mag_abs_r' in results:
        valid = np.isfinite(results['colour_gr']) & np.isfinite(results['Mag_abs_r'])
        ax.hexbin(results['Mag_abs_r'][valid], results['colour_gr'][valid], 
                  gridsize=30, cmap='Blues', mincnt=1)
        ax.set_xlabel(r'$M_r$')
        ax.set_ylabel(r'$(g-r)$')
        ax.set_title('Color-Magnitude Diagram')
        ax.invert_xaxis()
    
    # 3. SFR vs stellar mass
    ax = axes[1, 0]
    valid = np.isfinite(results['log_stellar_mass']) & np.isfinite(results['log_sfr_uv_corr'])
    ax.hexbin(results['log_stellar_mass'][valid], results['log_sfr_uv_corr'][valid],
              gridsize=30, cmap='Blues', mincnt=1)
    ax.set_xlabel(r'$\log(M_*/M_\odot)$')
    ax.set_ylabel(r'$\log(\mathrm{SFR}/M_\odot\,\mathrm{yr}^{-1})$')
    ax.set_title('Star Formation Main Sequence')
    
    # 4. Metallicity vs stellar mass
    ax = axes[1, 1]
    valid = np.isfinite(results['log_stellar_mass']) & np.isfinite(results['metallicity_fmr'])
    ax.hexbin(results['log_stellar_mass'][valid], results['metallicity_fmr'][valid],
              gridsize=30, cmap='Blues', mincnt=1)
    ax.set_xlabel(r'$\log(M_*/M_\odot)$')
    ax.set_ylabel(r'$12 + \log(\mathrm{O/H})$')
    ax.set_title('Mass-Metallicity Relation (FMR)')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / f'volume_complete_z{z_min:.2f}-{z_max:.2f}_properties.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved diagnostic plot: {plot_file}")


def main():
    """
    Main function to create volume complete samples
    """
    print("="*60)
    print("Creating Volume Complete Samples for SkySim5000")
    print("="*60)
    
    # Configuration
    data_file = "/path/to/skysim5000.h5"  # Update this path
    output_dir = Path("analysis/volume_complete_catalogs")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Redshift bins
    z_bins = [(0.0, 0.35), (0.35, 0.7), (0.7, 1.0)]
    
    # Survey parameters
    apparent_mag_limit = 25.0  # r-band limit
    band = 'r'
    
    # Check if data file exists
    if not Path(data_file).exists():
        print(f"\nError: Data file not found: {data_file}")
        print("Please update the 'data_file' path in the script to point to your SkySim5000 HDF5 file.")
        return
    
    # Load all necessary columns
    print(f"\nLoading data from: {data_file}")
    
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
    
    loader = DataLoader(data_file)
    data = loader.load_data(columns=columns_to_load)
    print(f"Loaded {len(loader.get_redshifts()):,} galaxies")
    
    # Process each redshift bin
    all_samples = {}
    
    for z_min, z_max in z_bins:
        print("\n" + "-"*60)
        
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
        
        # Store for summary
        all_samples[f'z{z_min:.2f}-{z_max:.2f}'] = {
            'n_galaxies': len(results['redshift_true']),
            'abs_mag_limit': abs_mag_limit,
            'filepath': filepath
        }
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for bin_name, info in all_samples.items():
        print(f"\n{bin_name}:")
        print(f"  Galaxies: {info['n_galaxies']:,}")
        print(f"  M_r limit: {info['abs_mag_limit']:.2f}")
        print(f"  File: {info['filepath'].name}")
    
    print(f"\nAll catalogs saved to: {output_dir}")
    print("\nDone!")


if __name__ == "__main__":
    main()