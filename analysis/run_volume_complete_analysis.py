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
    Colored by both M_r and stellar mass
    
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
    gs = gridspec.GridSpec(2, 4, figure=fig, 
                          width_ratios=[1, 3, 1, 3], 
                          height_ratios=[1, 3],
                          hspace=0.05, wspace=0.3)
    
    # First CMD colored by M_r
    ax_cmd1 = fig.add_subplot(gs[1, 1])
    ax_hist_top1 = fig.add_subplot(gs[0, 1], sharex=ax_cmd1)
    ax_hist_right1 = fig.add_subplot(gs[1, 0], sharey=ax_cmd1)
    
    # Second CMD colored by stellar mass
    ax_cmd2 = fig.add_subplot(gs[1, 3])
    ax_hist_top2 = fig.add_subplot(gs[0, 3], sharex=ax_cmd2)
    ax_hist_right2 = fig.add_subplot(gs[1, 2], sharey=ax_cmd2)
    
    # Plot first CMD (colored by M_r)
    scatter1 = ax_cmd1.scatter(mag_r, color_gr, c=mag_r, 
                              cmap='viridis_r', s=1, alpha=0.5,
                              rasterized=True)
    ax_cmd1.set_xlabel(r'$M_r$', fontsize=14)
    ax_cmd1.set_ylabel(r'$(g-r)$', fontsize=14)
    ax_cmd1.set_title(r'CMD colored by $M_r$', fontsize=14)
    ax_cmd1.invert_xaxis()
    ax_cmd1.set_xlim(-16, -24)
    ax_cmd1.set_ylim(-0.2, 1.5)
    
    # Add colorbar for M_r
    cbar1 = plt.colorbar(scatter1, ax=ax_cmd1, pad=0.02)
    cbar1.set_label(r'$M_r$', fontsize=12)
    
    # Marginal distributions for first CMD
    ax_hist_top1.hist(mag_r, bins=50, color='steelblue', alpha=0.7, density=True)
    ax_hist_top1.set_ylabel('Density', fontsize=10)
    ax_hist_top1.set_xlim(ax_cmd1.get_xlim())
    plt.setp(ax_hist_top1.get_xticklabels(), visible=False)
    
    ax_hist_right1.hist(color_gr, bins=50, orientation='horizontal', 
                       color='steelblue', alpha=0.7, density=True)
    ax_hist_right1.set_xlabel('Density', fontsize=10)
    ax_hist_right1.set_ylim(ax_cmd1.get_ylim())
    plt.setp(ax_hist_right1.get_yticklabels(), visible=False)
    ax_hist_right1.invert_xaxis()
    
    # Plot second CMD (colored by stellar mass)
    scatter2 = ax_cmd2.scatter(mag_r, color_gr, c=log_stellar_mass, 
                              cmap='plasma', s=1, alpha=0.5,
                              rasterized=True)
    ax_cmd2.set_xlabel(r'$M_r$', fontsize=14)
    ax_cmd2.set_ylabel(r'$(g-r)$', fontsize=14)
    ax_cmd2.set_title(r'CMD colored by Stellar Mass', fontsize=14)
    ax_cmd2.invert_xaxis()
    ax_cmd2.set_xlim(-16, -24)
    ax_cmd2.set_ylim(-0.2, 1.5)
    
    # Add colorbar for stellar mass
    cbar2 = plt.colorbar(scatter2, ax=ax_cmd2, pad=0.02)
    cbar2.set_label(r'$\log(M_*/M_\odot)$', fontsize=12)
    
    # Marginal distributions for second CMD
    ax_hist_top2.hist(mag_r, bins=50, color='tomato', alpha=0.7, density=True)
    ax_hist_top2.set_ylabel('Density', fontsize=10)
    ax_hist_top2.set_xlim(ax_cmd2.get_xlim())
    plt.setp(ax_hist_top2.get_xticklabels(), visible=False)
    
    ax_hist_right2.hist(color_gr, bins=50, orientation='horizontal', 
                       color='tomato', alpha=0.7, density=True)
    ax_hist_right2.set_xlabel('Density', fontsize=10)
    ax_hist_right2.set_ylim(ax_cmd2.get_ylim())
    plt.setp(ax_hist_right2.get_yticklabels(), visible=False)
    ax_hist_right2.invert_xaxis()
    
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
    Create SFR 1D distributions for all redshift bins in a single plot
    
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
    
    # Create figure with subplots for different SFR measurements
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Define colors for each redshift bin
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot 1: log SFR (observed)
    ax = axes[0]
    for i, (results, (z_min, z_max)) in enumerate(zip(all_results, z_bins)):
        if 'log_sfr_uv_obs' in results:
            data = results['log_sfr_uv_obs']
            valid = np.isfinite(data)
            ax.hist(data[valid], bins=50, alpha=0.6, density=True,
                   label=f'z=[{z_min:.2f}, {z_max:.2f}]',
                   color=colors[i % len(colors)], 
                   edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel(r'$\log(\mathrm{SFR}_{\mathrm{UV,obs}}/M_\odot\,\mathrm{yr}^{-1})$', fontsize=12)
    ax.set_ylabel('Normalized Density', fontsize=12)
    ax.set_title('UV SFR Distribution (Observed)', fontsize=14)
    ax.legend(loc='upper right')
    ax.set_xlim(-3, 3)
    
    # Plot 2: log SFR (dust corrected)
    ax = axes[1]
    for i, (results, (z_min, z_max)) in enumerate(zip(all_results, z_bins)):
        if 'log_sfr_uv_corr' in results:
            data = results['log_sfr_uv_corr']
            valid = np.isfinite(data)
            ax.hist(data[valid], bins=50, alpha=0.6, density=True,
                   label=f'z=[{z_min:.2f}, {z_max:.2f}]',
                   color=colors[i % len(colors)],
                   edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel(r'$\log(\mathrm{SFR}_{\mathrm{UV,corr}}/M_\odot\,\mathrm{yr}^{-1})$', fontsize=12)
    ax.set_ylabel('Normalized Density', fontsize=12)
    ax.set_title('UV SFR Distribution (Dust Corrected)', fontsize=14)
    ax.legend(loc='upper right')
    ax.set_xlim(-3, 3)
    
    # Plot 3: log sSFR
    ax = axes[2]
    for i, (results, (z_min, z_max)) in enumerate(zip(all_results, z_bins)):
        if 'log_ssfr' in results:
            data = results['log_ssfr']
            valid = np.isfinite(data)
            ax.hist(data[valid], bins=50, alpha=0.6, density=True,
                   label=f'z=[{z_min:.2f}, {z_max:.2f}]',
                   color=colors[i % len(colors)],
                   edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel(r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$', fontsize=12)
    ax.set_ylabel('Normalized Density', fontsize=12)
    ax.set_title('Specific SFR Distribution', fontsize=14)
    ax.legend(loc='upper left')
    ax.set_xlim(-12, -8)
    
    # Plot 4: Statistics summary
    ax = axes[3]
    ax.axis('off')
    
    # Create statistics table
    stats_text = "SFR Statistics Summary\n" + "="*40 + "\n\n"
    
    for i, (results, (z_min, z_max)) in enumerate(zip(all_results, z_bins)):
        stats_text += f"z=[{z_min:.2f}, {z_max:.2f}]:\n"
        
        if 'log_sfr_uv_corr' in results:
            sfr_data = results['log_sfr_uv_corr']
            valid = np.isfinite(sfr_data)
            sfr_valid = sfr_data[valid]
            
            stats_text += f"  N galaxies: {len(sfr_valid):,}\n"
            stats_text += f"  Mean log(SFR): {np.mean(sfr_valid):.2f}\n"
            stats_text += f"  Median log(SFR): {np.median(sfr_valid):.2f}\n"
            stats_text += f"  Std log(SFR): {np.std(sfr_valid):.2f}\n"
            
            # Star-forming fraction
            if 'is_star_forming' in results:
                sf_frac = np.sum(results['is_star_forming']) / len(results['is_star_forming'])
                stats_text += f"  Star-forming fraction: {sf_frac:.2%}\n"
        
        stats_text += "\n"
    
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    # Overall title
    fig.suptitle('Star Formation Rate Distributions Across Redshift', fontsize=16)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / 'sfr_distributions_all_redshifts.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Saved SFR distributions comparison plot: {plot_file}")


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
        
        # Store results for comparison plots
        all_results.append(results)
        
        print(f"\nCompleted processing for z=[{z_min:.2f}, {z_max:.2f}]")
        print(f"Output saved to: {filepath}")
    
    # Create SFR distributions comparison plot
    print("\nCreating SFR distributions comparison plot...")
    plot_sfr_distributions_comparison(all_results, z_bins, output_dir)
    
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