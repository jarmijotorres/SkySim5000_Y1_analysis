#!/usr/bin/env python
"""
Example script to run the volume complete sample analysis
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import h5py

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.volume_complete_samples import (
    create_volume_complete_sample,
    calculate_derived_quantities,
    save_volume_complete_catalog,
    plot_sample_properties
)
from data_loader.loader import DataLoader
from utils.plotting import set_plot_style, add_colorbar


def plot_cmd_with_marginals(results, z_min, z_max, output_dir):
    """
    Create colour-magnitude diagram with marginal distributions
    Using 2D histogram colored by log N of galaxies
    
    Parameters
    ----------
    results : dict
        Dictionary with galaxy properties
    z_min : float
        Minimum redshift
    z_max : float
        Maximum redshift
    output_dir : Path
        Output directory
    """
    output_dir = Path(output_dir)
    
    # Set plotting style
    set_plot_style()
    
    # Get data
    if 'colour_gr' not in results or 'Mag_abs_r' not in results:
        print("  Warning: Cannot create CMD plot - missing g-r color or M_r data")
        return
    
    color_gr = results['colour_gr']
    mag_r = results['Mag_abs_r']
    stellar_mass = results['stellar_mass']
    log_stellar_mass = results['log_stellar_mass']
    
    # Remove invalid values
    valid = np.isfinite(color_gr) & np.isfinite(mag_r) & np.isfinite(stellar_mass)
    color_gr = color_gr[valid]
    mag_r = mag_r[valid]
    stellar_mass = stellar_mass[valid]
    log_stellar_mass = log_stellar_mass[valid]
    
    # Create figure with two separate CMDs
    fig = plt.figure(figsize=(16, 8))
    
    # Define grid for subplots with marginals
    # Remove right marginal for second plot
    gs = gridspec.GridSpec(2, 4, figure=fig,
                          width_ratios=[1, 3, 0.1, 3],
                          height_ratios=[1, 3],
                          hspace=0.05, wspace=0.3)
    
    # First CMD colored by log N
    ax_cmd1 = fig.add_subplot(gs[1, 1])
    ax_hist_top1 = fig.add_subplot(gs[0, 1], sharex=ax_cmd1)
    ax_hist_right1 = fig.add_subplot(gs[1, 0], sharey=ax_cmd1)
    
    # Second CMD colored by log N (stellar mass binned)
    ax_cmd2 = fig.add_subplot(gs[1, 3])
    ax_hist_top2 = fig.add_subplot(gs[0, 3], sharex=ax_cmd2)
    
    # Create 2D histogram for first CMD (M_r)
    hist1, xedges1, yedges1 = np.histogram2d(mag_r, color_gr, bins=50)
    # Add small value to avoid log(0)
    hist1_log = np.log10(hist1.T + 1)
    
    # Plot as color map
    im1 = ax_cmd1.imshow(hist1_log, origin='lower', aspect='auto',
                         extent=[xedges1[0], xedges1[-1], yedges1[0], yedges1[-1]],
                         cmap='viridis', interpolation='nearest')
    ax_cmd1.set_xlabel(r'$M_r$', fontsize=14)
    ax_cmd1.set_ylabel(r'$(g-r)$', fontsize=14)
    ax_cmd1.set_title(r'CMD colored by $\log N$ (galaxies)', fontsize=14)
    ax_cmd1.invert_xaxis()
    ax_cmd1.set_xlim(-16, -24)
    ax_cmd1.set_ylim(-0.2, 1.5)
    
    # Add colorbar for log N
    cbar1 = plt.colorbar(im1, ax=ax_cmd1, pad=0.02)
    cbar1.set_label(r'$\log_{10}(N + 1)$', fontsize=12)
    
    # Marginal distributions for first CMD (using histtype='step')
    ax_hist_top1.hist(mag_r, bins=50, histtype='step', color='steelblue',
                      linewidth=2, density=True)
    ax_hist_top1.set_ylabel('Density', fontsize=10)
    ax_hist_top1.set_xlim(ax_cmd1.get_xlim())
    plt.setp(ax_hist_top1.get_xticklabels(), visible=False)
    
    ax_hist_right1.hist(color_gr, bins=50, orientation='horizontal',
                       histtype='step', color='steelblue', linewidth=2, density=True)
    ax_hist_right1.set_xlabel('Density', fontsize=10)
    ax_hist_right1.set_ylim(ax_cmd1.get_ylim())
    plt.setp(ax_hist_right1.get_yticklabels(), visible=False)
    ax_hist_right1.invert_xaxis()
    
    # Create 2D histogram for second CMD (M_star)
    # Use stellar mass instead of M_r for x-axis
    hist2, xedges2, yedges2 = np.histogram2d(log_stellar_mass, color_gr, bins=50)
    hist2_log = np.log10(hist2.T + 1)
    
    # Plot as color map
    im2 = ax_cmd2.imshow(hist2_log, origin='lower', aspect='auto',
                         extent=[xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]],
                         cmap='viridis', interpolation='nearest')
    ax_cmd2.set_xlabel(r'$\log(M_*/M_\odot)$', fontsize=14)
    ax_cmd2.set_ylabel(r'$(g-r)$', fontsize=14)
    ax_cmd2.set_title(r'CMD colored by $\log N$ (galaxies)', fontsize=14)
    ax_cmd2.set_xlim(8, 12)
    ax_cmd2.set_ylim(-0.2, 1.5)
    
    # Add colorbar for log N
    cbar2 = plt.colorbar(im2, ax=ax_cmd2, pad=0.02)
    cbar2.set_label(r'$\log_{10}(N + 1)$', fontsize=12)
    
    # Marginal distribution for second CMD (top only)
    ax_hist_top2.hist(log_stellar_mass, bins=50, histtype='step', color='tomato',
                      linewidth=2, density=True)
    ax_hist_top2.set_ylabel('Density', fontsize=10)
    ax_hist_top2.set_xlim(ax_cmd2.get_xlim())
    plt.setp(ax_hist_top2.get_xticklabels(), visible=False)
    
    # Overall title
    fig.suptitle(f'Color-Magnitude Diagrams with Marginal Distributions\n'
                 f'Volume Complete Sample: z=[{z_min:.2f}, {z_max:.2f}]',
                 fontsize=16, y=0.98)
    
    # Save plot
    plot_file = output_dir / f'cmd_marginals_z{z_min:.2f}-{z_max:.2f}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved CMD with marginals plot: {plot_file}")


def plot_sfr_distributions_comparison(all_results, z_bins, output_dir):
    """
    Create SFR and sSFR 1D distributions for all redshift bins in a single plot
    
    Parameters
    ----------
    all_results : list of dict
        List of results dictionaries for each redshift bin
    z_bins : list of tuples
        List of (z_min, z_max) tuples
    output_dir : Path
        Output directory
    """
    output_dir = Path(output_dir)
    
    # Set plotting style
    set_plot_style()
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Define colors for each redshift bin
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot 1: log SFR (dust corrected) - left panel
    for i, (results, (z_min, z_max)) in enumerate(zip(all_results, z_bins)):
        if 'log_sfr_uv_corr' in results:
            data = results['log_sfr_uv_corr']
            valid = np.isfinite(data)
            ax1.hist(data[valid], bins=50, histtype='step', density=True,
                    label=f'z=[{z_min:.2f}, {z_max:.2f}]',
                    color=colors[i % len(colors)],
                    linewidth=2, alpha=0.8)
    
    ax1.set_xlabel(r'$\log(\mathrm{SFR}/M_\odot\,\mathrm{yr}^{-1})$', fontsize=14)
    ax1.set_ylabel('Normalized Density', fontsize=14)
    ax1.set_title('Star Formation Rate Distribution', fontsize=16)
    ax1.legend(loc='upper right', fontsize=12)
    ax1.set_xlim(-3, 3)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: log sSFR - right panel
    for i, (results, (z_min, z_max)) in enumerate(zip(all_results, z_bins)):
        if 'log_ssfr' in results:
            data = results['log_ssfr']
            valid = np.isfinite(data)
            ax2.hist(data[valid], bins=50, histtype='step', density=True,
                    label=f'z=[{z_min:.2f}, {z_max:.2f}]',
                    color=colors[i % len(colors)],
                    linewidth=2, alpha=0.8)
    
    ax2.set_xlabel(r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$', fontsize=14)
    ax2.set_ylabel('Normalized Density', fontsize=14)
    ax2.set_title('Specific Star Formation Rate Distribution', fontsize=16)
    ax2.legend(loc='upper left', fontsize=12)
    ax2.set_xlim(-12, -8)
    ax2.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle('Star Formation Rate Distributions Across Redshift', fontsize=18, y=1.02)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / 'sfr_ssfr_distributions_all_redshifts.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Saved SFR/sSFR distributions comparison plot: {plot_file}")


def plot_metallicity_distribution(results, z_min, z_max, output_dir):
    """
    Create 1D metallicity distribution plot with log-scale y-axis
    
    Parameters
    ----------
    results : dict
        Dictionary with galaxy properties
    z_min : float
        Minimum redshift
    z_max : float
        Maximum redshift
    output_dir : Path
        Output directory
    """
    output_dir = Path(output_dir)
    
    # Set plotting style
    set_plot_style()
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Get metallicity data
    if 'metallicity_fmr' in results:
        metallicity = results['metallicity_fmr']
        valid = np.isfinite(metallicity)
        metallicity_valid = metallicity[valid]
        
        # Plot histogram
        counts, bins, patches = ax.hist(metallicity_valid, bins=50,
                                       range=(-5, 15),
                                       histtype='step',
                                       color='darkblue',
                                       linewidth=2,
                                       label='FMR metallicity')
        
        # Set log scale for y-axis
        ax.set_yscale('log')
        
        # Labels and title
        ax.set_xlabel(r'$12 + \log(\mathrm{O/H})$', fontsize=14)
        ax.set_ylabel('Number of galaxies', fontsize=14)
        ax.set_title(f'Metallicity Distribution\nz=[{z_min:.2f}, {z_max:.2f}]', fontsize=16)
        
        # Set x-axis limits
        ax.set_xlim(-5, 15)
        
        # Add grid
        ax.grid(True, alpha=0.3, which='both')
        
        # Add statistics text
        stats_text = f'N = {len(metallicity_valid):,}\n'
        stats_text += f'Mean = {np.mean(metallicity_valid):.2f}\n'
        stats_text += f'Median = {np.median(metallicity_valid):.2f}\n'
        stats_text += f'Std = {np.std(metallicity_valid):.2f}'
        
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        # Save plot
        plot_file = output_dir / f'metallicity_distribution_z{z_min:.2f}-{z_max:.2f}.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved metallicity distribution plot: {plot_file}")
    else:
        print("  Warning: Cannot create metallicity plot - missing metallicity data")


def save_all_redshift_bins_hdf5(all_results, z_bins, output_dir):
    """
    Save all redshift bins data in a single HDF5 file
    
    Parameters
    ----------
    all_results : list of dict
        List of results dictionaries for each redshift bin
    z_bins : list of tuples
        List of (z_min, z_max) tuples
    output_dir : Path
        Output directory
    """
    output_dir = Path(output_dir)
    
    # Create filename
    filename = "skysim5000_all_redshift_bins.h5"
    filepath = output_dir / filename
    
    print(f"\nSaving all redshift bins to: {filepath}")
    
    with h5py.File(filepath, 'w') as f:
        # Add global metadata
        f.attrs['n_redshift_bins'] = len(z_bins)
        f.attrs['description'] = 'Volume complete samples for all redshift bins'
        from datetime import datetime
        f.attrs['creation_date'] = datetime.now().isoformat()
        
        # Save each redshift bin
        for i, (results, (z_min, z_max)) in enumerate(zip(all_results, z_bins)):
            # Create group for this redshift bin
            grp_name = f'z{z_min:.2f}_{z_max:.2f}'
            grp = f.create_group(grp_name)
            
            # Add metadata for this bin
            grp.attrs['z_min'] = z_min
            grp.attrs['z_max'] = z_max
            grp.attrs['n_galaxies'] = len(results['redshift_true'])
            if 'abs_mag_limit' in results:
                grp.attrs['abs_mag_limit'] = results['abs_mag_limit']
            
            # Save data arrays
            for key, value in results.items():
                if key in ['z_min', 'z_max', 'abs_mag_limit']:
                    continue  # Skip metadata already saved as attributes
                
                try:
                    data = np.array(value)
                    if data.dtype == bool:
                        data = data.astype(np.uint8)
                    grp.create_dataset(key, data=data, compression='gzip')
                except Exception as e:
                    print(f"    Warning: Could not save {key}: {e}")
    
    print(f"  Successfully saved all redshift bins to HDF5")
    return filepath


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
    
    # Store results for all redshift bins
    all_results = []
    
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
        
        # Create original diagnostic plots
        plot_sample_properties(results, z_min, z_max, output_dir)
        
        # Create CMD with marginal distributions
        plot_cmd_with_marginals(results, z_min, z_max, output_dir)
        
        # Create metallicity distribution plot
        plot_metallicity_distribution(results, z_min, z_max, output_dir)
        
        # Store results for comparison plots
        all_results.append(results)
        
        print(f"\nCompleted processing for z=[{z_min:.2f}, {z_max:.2f}]")
        print(f"Output saved to: {filepath}")
    
    # Create SFR distributions comparison plot
    print("\nCreating SFR/sSFR distributions comparison plot...")
    plot_sfr_distributions_comparison(all_results, z_bins, output_dir)
    
    # Save all redshift bins in one HDF5 file
    hdf5_filepath = save_all_redshift_bins_hdf5(all_results, z_bins, output_dir)
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"All catalogs and plots saved to: {output_dir}")
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