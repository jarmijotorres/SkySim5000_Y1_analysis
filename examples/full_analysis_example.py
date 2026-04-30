"""
Complete example of SkySim5000 analysis pipeline

This script demonstrates:
1. Loading data from HDF5 file
2. Calculating luminosity functions
3. Analyzing galaxy colors
4. Computing star formation rates
5. Estimating metallicities
6. Creating publication-quality plots
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import our analysis modules
from skysim5000_analysis import (
    DataLoader, 
    LuminosityFunction,
    ColourAnalysis,
    SFRCalculator,
    MetallicityCalculator
)
from skysim5000_analysis.utils import set_plot_style, save_figure


def main():
    """Run complete analysis pipeline"""
    
    # Set consistent plotting style
    set_plot_style()
    
    # Define paths
    data_path = Path("/path/to/SkySim5000.hdf5")  # Update this path
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    print("SkySim5000 Analysis Pipeline")
    print("=" * 50)
    
    # =========================================================================
    # 1. Load Data
    # =========================================================================
    print("\n1. Loading data...")
    
    # Define columns to load
    columns = [
        'Mag_true_r_lsst_z0',
        'Mag_true_u_lsst_z0', 
        'Mag_true_g_lsst_z0',
        'Mag_true_i_lsst_z0',
        'mag_r_lsst',
        'stellar_mass',
        'redshift_true',
        'sed_1246_306_disk_no_host_extinction',
        'A_v_disk'
    ]
    
    # Load data
    loader = DataLoader(data_path)
    data = loader.load_data(columns=columns)
    loader.summary()
    
    # Get key arrays
    mag_r = loader.get_magnitudes('r', z0=True)
    redshifts = loader.get_redshifts(true=True)
    stellar_mass = loader.get_stellar_masses()
    
    # Apply selection cuts
    z_min, z_max = 0.25, 1.0
    mag_lim = 25.0
    
    selection_mask = (
        (redshifts >= z_min) & 
        (redshifts <= z_max) &
        (loader.get_magnitudes('r', z0=False) < mag_lim)
    )
    
    print(f"\nSelected {selection_mask.sum():,} / {len(selection_mask):,} galaxies")
    print(f"Redshift range: {z_min} < z < {z_max}")
    print(f"Magnitude limit: r < {mag_lim}")
    
    # =========================================================================
    # 2. Luminosity Function
    # =========================================================================
    print("\n2. Calculating luminosity function...")
    
    lf_calc = LuminosityFunction()
    
    # Calculate Vmax
    area_deg2 = 9.0 * 13.0  # Example survey area
    vmax = lf_calc.calculate_vmax(
        mag_r[selection_mask],
        redshifts[selection_mask],
        mag_lim, z_min, z_max, area_deg2
    )
    
    # Calculate LF
    mag_centers, phi, phi_err = lf_calc.calculate_lf(
        mag_r[selection_mask], 
        vmax,
        mag_bins=np.arange(-25.5, -15.0, 0.5)
    )
    
    # Fit Schechter function
    schechter_params = lf_calc.fit_schechter(mag_centers, phi, phi_err)
    if schechter_params:
        print(f"  Schechter fit:")
        print(f"    φ* = {schechter_params['phi_star']:.3e} ± {schechter_params['phi_star_err']:.3e}")
        print(f"    M* = {schechter_params['M_star']:.2f} ± {schechter_params['M_star_err']:.2f}")
        print(f"    α  = {schechter_params['alpha']:.2f} ± {schechter_params['alpha_err']:.2f}")
    
    # Plot LF
    fig, ax = plt.subplots(figsize=(8, 6))
    
    valid = np.isfinite(phi)
    ax.errorbar(mag_centers[valid], phi[valid], yerr=phi_err[valid],
                fmt='o', capsize=3, label='1/Vmax')
    
    ax.set_yscale('log')
    ax.invert_xaxis()
    ax.set_xlabel(r'$M_r$')
    ax.set_ylabel(r'$\phi$ [Mpc$^{-3}$ mag$^{-1}$]')
    ax.set_title('r-band Luminosity Function')
    ax.legend()
    
    save_figure(fig, output_dir / 'luminosity_function.png')
    plt.close()
    
    # =========================================================================
    # 3. Color Analysis
    # =========================================================================
    print("\n3. Analyzing galaxy colors...")
    
    color_analyzer = ColourAnalysis()
    
    # Calculate colors
    g_r = loader.get_colors(('g', 'r'))
    u_g = loader.get_colors(('u', 'g'))
    
    # Apply color selection
    color_mask = selection_mask & (g_r > 0)
    
    # Classify red/blue galaxies
    color_cut = 0.5
    red_mask, blue_mask = color_analyzer.classify_red_blue(g_r[color_mask], color_cut)
    red_fraction = red_mask.mean()
    
    print(f"  Red galaxy fraction: {red_fraction:.1%}")
    print(f"  Blue galaxy fraction: {(1-red_fraction):.1%}")
    
    # Create CMD
    fig = color_analyzer.plot_cmd(
        mag_r[color_mask],
        g_r[color_mask],
        xlabel=r'$M_r$',
        ylabel=r'$(g-r)$',
        title='Color-Magnitude Diagram',
        colour_cut=color_cut
    )
    save_figure(fig, output_dir / 'cmd.png')
    plt.close()
    
    # Test for bimodality
    bimodality = color_analyzer.bimodality_test(g_r[color_mask])
    print(f"  Color distribution is {'bimodal' if bimodality['is_bimodal'] else 'unimodal'}")
    
    # =========================================================================
    # 4. Star Formation Rates
    # =========================================================================
    print("\n4. Calculating star formation rates...")
    
    sfr_calc = SFRCalculator()
    
    # Get UV data
    sed_uv = loader.get_sed_data()
    A_v = loader.get_dust_extinction()
    
    # Calculate UV luminosity
    L_UV = sfr_calc.calculate_uv_luminosity(sed_uv[color_mask])
    
    # Calculate SFR (no dust correction)
    log_sfr_obs, sfr_obs = sfr_calc.calculate_sfr_uv(L_UV)
    
    # Apply dust correction
    log_sfr_corr = sfr_calc.apply_dust_correction(
        log_sfr_obs, 
        A_v[color_mask]
    )
    
    # Calculate sSFR
    sfr_corr = 10**log_sfr_corr
    ssfr, log_ssfr = sfr_calc.calculate_ssfr(sfr_corr, stellar_mass[color_mask])
    
    # Classify star-forming vs quenched
    sf_class = sfr_calc.classify_sfms(log_ssfr)
    print(f"  Star-forming fraction: {sf_class['sf_fraction']:.1%}")
    print(f"  Quenched fraction: {sf_class['quenched_fraction']:.1%}")
    
    # Plot SFR-Mass relation
    fig, ax = plt.subplots(figsize=(8, 6))
    
    valid_sfr = np.isfinite(log_sfr_corr) & (stellar_mass[color_mask] > 0)
    
    h = ax.hist2d(
        np.log10(stellar_mass[color_mask][valid_sfr]),
        log_sfr_corr[valid_sfr],
        bins=50, cmap='magma'
    )
    plt.colorbar(h[3], ax=ax, label='N galaxies')
    
    # Add SFMS line
    mass_range = np.linspace(8, 12, 100)
    sfms = sfr_calc.sfms_relation(mass_range)
    ax.plot(mass_range, sfms, 'c--', lw=2, label='SFMS (Elbaz+07)')
    
    ax.set_xlabel(r'$\log_{10}(M_*/M_\odot)$')
    ax.set_ylabel(r'$\log_{10}$(SFR [M$_\odot$ yr$^{-1}$])')
    ax.set_title('Star-Forming Main Sequence')
    ax.legend()
    
    save_figure(fig, output_dir / 'sfms.png')
    plt.close()
    
    # =========================================================================
    # 5. Metallicity
    # =========================================================================
    print("\n5. Calculating metallicities...")
    
    met_calc = MetallicityCalculator()
    
    # Calculate FMR metallicity
    log_mstar = np.log10(stellar_mass[color_mask][valid_sfr])
    Z_fmr, mu = met_calc.fmr_metallicity(log_mstar, log_sfr_corr[valid_sfr])
    
    # Get statistics
    Z_stats = met_calc.metallicity_distribution_stats(Z_fmr)
    print(f"  Mean metallicity: {Z_stats['mean']:.2f}")
    print(f"  Solar metallicity fraction: {Z_stats['solar_fraction']:.1%}")
    
    # Plot mass-metallicity relation
    fig, ax = plt.subplots(figsize=(8, 6))
    
    h = ax.hexbin(log_mstar, Z_fmr, gridsize=50, cmap='magma', mincnt=1)
    plt.colorbar(h, ax=ax, label='N galaxies')
    
    ax.axhline(8.69, color='cyan', ls='--', label=r'Solar $Z_\odot$')
    ax.set_xlabel(r'$\log_{10}(M_*/M_\odot)$')
    ax.set_ylabel(r'$12 + \log_{10}$(O/H)')
    ax.set_title('Mass-Metallicity Relation')
    ax.legend()
    
    save_figure(fig, output_dir / 'mzr.png')
    plt.close()
    
    # =========================================================================
    # 6. Summary Statistics
    # =========================================================================
    print("\n6. Summary Statistics")
    print("=" * 50)
    print(f"Total galaxies analyzed: {color_mask.sum():,}")
    print(f"Redshift range: {redshifts[color_mask].min():.2f} - {redshifts[color_mask].max():.2f}")
    print(f"Magnitude range: {mag_r[color_mask].min():.2f} - {mag_r[color_mask].max():.2f}")
    print(f"Stellar mass range: {np.log10(stellar_mass[color_mask].min()):.2f} - {np.log10(stellar_mass[color_mask].max()):.2f}")
    
    print("\nAnalysis complete! Results saved to:", output_dir)


if __name__ == "__main__":
    main()