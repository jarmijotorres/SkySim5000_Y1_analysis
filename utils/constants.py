"""
Physical constants and default parameters
"""

from astropy.cosmology import FlatLambdaCDM

# Default cosmology (matching SkySim5000)
COSMOLOGY = FlatLambdaCDM(H0=71.0, Om0=0.2648, Ob0=0.0448)

# Physical constants
L_SUN_ERG = 3.846e33  # Solar luminosity in erg/s
M_SUN_G = 1.989e33    # Solar mass in grams
PC_CM = 3.086e18      # Parsec in cm
MPC_CM = 3.086e24     # Megaparsec in cm

# Magnitude system
MAG_ZEROPOINT = 48.6  # AB magnitude zero point

# Solar metallicity
Z_SOLAR = 8.69  # 12 + log(O/H)_solar

# Default survey parameters
DEFAULT_AREA_DEG2 = 9.0 * 13.0  # Default survey area in deg^2
DEFAULT_MAG_LIM = 25.0          # Default magnitude limit
DEFAULT_Z_MIN = 0.25            # Default minimum redshift
DEFAULT_Z_MAX = 1.00            # Default maximum redshift

# Dust attenuation parameters (Calzetti+2000)
CALZETTI_RV = 4.05
CALZETTI_RATIOS = {
    1500: 10.33 / 4.05,  # A_1500 / A_V
    1600: 10.33 / 4.05,  # A_1600 / A_V (approximate)
    2800: 4.05 / 4.05    # A_2800 / A_V = 1.0
}

# Star formation rate calibrations
SFR_CALIBRATIONS = {
    'kennicutt2012': {
        'name': 'Kennicutt & Evans (2012)',
        'constant': 43.35,  # log(SFR) = log(L_UV) - constant
        'reference': 'Kennicutt & Evans 2012, ARA&A, 50, 531'
    },
    'madau2014': {
        'name': 'Madau & Dickinson (2014)',
        'factor': 1.4e-28,  # SFR = factor * L_UV
        'reference': 'Madau & Dickinson 2014, ARA&A, 52, 415'
    }
}

# Star-forming main sequence parameters
SFMS_RELATIONS = {
    'elbaz2007': {
        'name': 'Elbaz et al. (2007)',
        'slope': 0.77,
        'intercept': -7.65,
        'reference': 'Elbaz et al. 2007, A&A, 468, 33'
    },
    'whitaker2012': {
        'name': 'Whitaker et al. (2012)',
        'z_bins': [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.5)],
        'slopes': [0.70, 0.75, 0.80, 0.85],
        'intercepts': [-7.0, -6.8, -6.5, -6.2],
        'reference': 'Whitaker et al. 2012, ApJ, 754, L29'
    }
}

# Fundamental Metallicity Relation parameters (Mannucci+2010)
FMR_PARAMS = {
    'alpha': 0.32,
    'coefficients': [8.90, 0.37, -0.14, -0.19],
    'valid_range': (8.7, 11.2),  # Valid mu range
    'reference': 'Mannucci et al. 2010, MNRAS, 408, 2115'
}

# Mass-metallicity relation parameters
MZR_RELATIONS = {
    'tremonti2004': {
        'name': 'Tremonti et al. (2004)',
        'coefficients': [-1.492, 1.847, -0.08026],
        'reference': 'Tremonti et al. 2004, ApJ, 613, 898'
    },
    'andrews2013': {
        'name': 'Andrews & Martini (2013)',
        'coefficients': [8.798, 0.639, -0.137],
        'reference': 'Andrews & Martini 2013, ApJ, 765, 140'
    }
}

# Color cuts for red/blue classification
COLOR_CUTS = {
    'gr': 0.5,   # g-r color cut
    'ur': 1.75,  # u-r color cut
    'gi': 0.8    # g-i color cut
}

# Quenching thresholds
QUENCHING_THRESHOLDS = {
    'log_ssfr': -11.0,  # log10(sSFR) threshold for quenched galaxies
    'sfr_offset': -1.0  # Offset from SFMS for quenched galaxies
}