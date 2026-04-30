# Volume Complete Sample Analysis

This directory contains scripts for creating volume complete galaxy samples from the SkySim5000 catalog.

## Overview

The volume complete sample analysis creates galaxy catalogs that are complete above a certain absolute magnitude limit for different redshift bins. This ensures that the samples are not biased by the survey's apparent magnitude limit at different redshifts.

## Scripts

### 1. `volume_complete_samples.py`

Main module containing all the functions for creating volume complete samples. This script:

- Loads the SkySim5000 galaxy catalog
- Defines 3 redshift bins: [0.0-0.35], [0.35-0.7], [0.7-1.0]
- Calculates the absolute magnitude limit at z_max for each bin
- Selects galaxies brighter than this limit (volume complete)
- Calculates derived quantities:
  - Colors (u-r, g-r, r-i, g-i)
  - Star formation rates (UV-based, dust-corrected)
  - Specific star formation rates
  - Metallicities (using FMR and mass-metallicity relations)
- Saves each redshift slice as a new catalog (HDF5 and CSV formats)
- Creates diagnostic plots

### 2. `run_volume_complete_analysis.py`

Example script showing how to run the analysis. Can be used as a command-line tool.

## Usage

### Method 1: Command Line

```bash
cd /home/jarmijo/skysim5000_analysis
python analysis/run_volume_complete_analysis.py /path/to/skysim5000.h5
```

### Method 2: Python Script

```python
from analysis.volume_complete_samples import main

# Update the data_file path in the script first
main()
```

### Method 3: Custom Analysis

```python
from data_loader.loader import DataLoader
from analysis.volume_complete_samples import (
    create_volume_complete_sample,
    calculate_derived_quantities,
    save_volume_complete_catalog
)

# Load data
loader = DataLoader('/path/to/skysim5000.h5')
data = loader.load_data()

# Create volume complete sample for z=[0.35, 0.7]
subset_loader, mask, abs_mag_limit = create_volume_complete_sample(
    loader, z_min=0.35, z_max=0.7, band='r', apparent_mag_limit=25.0
)

# Calculate derived quantities
results = calculate_derived_quantities(subset_loader)

# Save catalog
save_volume_complete_catalog(results, 0.35, 0.7, 'output_dir')
```

## Output Files

For each redshift bin, the script creates:

1. **HDF5 file**: `skysim5000_volume_complete_z{z_min}-{z_max}.h5`
   - Contains all galaxy properties in compressed format
   - Includes metadata (z_min, z_max, n_galaxies, etc.)

2. **CSV file**: `skysim5000_volume_complete_z{z_min}-{z_max}.csv`
   - Same data in CSV format for easy access

3. **Diagnostic plot**: `volume_complete_z{z_min}-{z_max}_properties.png`
   - 4-panel plot showing:
     - Stellar mass distribution
     - Color-magnitude diagram
     - Star formation main sequence
     - Mass-metallicity relation

## Output Columns

Each catalog contains the following columns:

### Basic Properties
- `redshift_true`: True redshift
- `stellar_mass`: Stellar mass [M_sun]
- `log_stellar_mass`: log10(stellar mass)
- `Mag_abs_{u,g,r,i,z}`: Absolute magnitudes in each band

### Colors
- `colour_ur`: (u-r) color
- `colour_gr`: (g-r) color
- `colour_ri`: (r-i) color
- `colour_gi`: (g-i) color

### Star Formation
- `sfr_uv_obs`: Observed UV-based SFR [M_sun/yr]
- `log_sfr_uv_obs`: log10(observed SFR)
- `sfr_uv_corr`: Dust-corrected SFR [M_sun/yr]
- `log_sfr_uv_corr`: log10(dust-corrected SFR)
- `A_v`: V-band dust extinction
- `ssfr`: Specific SFR [yr^-1]
- `log_ssfr`: log10(sSFR)
- `is_star_forming`: Boolean flag for star-forming galaxies
- `is_quenched`: Boolean flag for quenched galaxies

### Metallicity
- `metallicity_fmr`: Metallicity from Fundamental Metallicity Relation
- `fmr_mu`: FMR mu parameter
- `metallicity_mzr`: Metallicity from mass-metallicity relation

### Sample Metadata
- `z_min`: Minimum redshift of the bin
- `z_max`: Maximum redshift of the bin
- `abs_mag_limit`: Absolute magnitude limit for volume completeness

## Volume Completeness

The volume complete samples are created by:

1. Calculating the absolute magnitude limit at z_max for each bin:
   ```
   M_lim = m_lim - DM(z_max) - K(z_max)
   ```
   where:
   - m_lim = apparent magnitude limit (25.0 for r-band)
   - DM = distance modulus
   - K = K-correction (approximated as -0.5*z for r-band)

2. Selecting only galaxies with M_r < M_lim

This ensures that we could detect all galaxies above this magnitude limit throughout the entire redshift bin.

## Example Results

For the default configuration (r < 25.0):

- z=[0.00, 0.35]: M_r < -16.8
- z=[0.35, 0.70]: M_r < -18.9  
- z=[0.70, 1.00]: M_r < -20.0

## Notes

- The K-correction is simplified. For more accurate results, use SED-based K-corrections
- The script assumes the SkySim5000 "small" dataset by default
- Metallicities are calculated using empirical relations (FMR and MZR)
- SFR is calculated from UV luminosity with dust correction