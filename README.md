# SkySim5000 Analysis Tools

A Python package for analyzing SkySim5000 galaxy catalog data, including luminosity functions, color distributions, star formation rates, and metallicity calculations.

## Structure

- `data_loader/`: Module for loading and preprocessing SkySim5000 data
- `luminosity_function/`: Luminosity function calculations using 1/Vmax method
- `colour_analysis/`: Color-magnitude and color-color diagram analysis
- `sfr_calculator/`: Star formation rate calculations with dust corrections
- `metallicity/`: Fundamental Metallicity Relation (FMR) calculations
- `utils/`: Utility functions and constants
- `tests/`: Unit tests for all modules

## Installation

```bash
pip install -e .
```

## Usage

```python
from skysim5000_analysis import DataLoader, LuminosityFunction, ColourAnalysis, SFRCalculator, MetallicityCalculator

# Load data
loader = DataLoader('/path/to/skysim5000.hdf5')
data = loader.load_data()

# Calculate luminosity function
lf = LuminosityFunction(data)
phi, mag_centers = lf.calculate_vmax_lf()

# Analyze colors
colors = ColourAnalysis(data)
cmd_data = colors.create_cmd()

# Calculate SFRs
sfr_calc = SFRCalculator(data)
log_sfr = sfr_calc.calculate_sfr_uv()

# Calculate metallicities
met_calc = MetallicityCalculator(data)
metallicity = met_calc.calculate_fmr()
```

## Requirements

- numpy
- scipy
- pandas
- h5py
- matplotlib
- astropy