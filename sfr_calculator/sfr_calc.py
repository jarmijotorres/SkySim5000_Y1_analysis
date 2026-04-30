"""
Star Formation Rate calculator using UV luminosity
"""

import numpy as np
from typing import Tuple, Optional, Dict
import warnings


class SFRCalculator:
    """
    Calculate star formation rates from UV luminosity
    """
    
    # Physical constants
    L_SUN_ERG = 3.846e33  # L_sun in erg/s
    
    # Dust attenuation curve ratios (Calzetti+2000)
    # A_lambda / A_V ratios
    CALZETTI_RATIOS = {
        1500: 10.33 / 4.05,  # A_1500 / A_V
        1600: 10.33 / 4.05,  # Approximate, close to 1500 Å
        2800: 4.05 / 4.05    # A_2800 / A_V = 1.0
    }
    
    def __init__(self):
        """Initialize SFR calculator"""
        pass
    
    def calculate_uv_luminosity(self, sed_flux: np.ndarray, 
                               bin_width: float = 306.0) -> np.ndarray:
        """
        Calculate UV luminosity from SED flux
        
        Parameters
        ----------
        sed_flux : np.ndarray
            SED flux in the UV bin (L_sun/Å)
        bin_width : float
            Width of the wavelength bin in Angstroms
            
        Returns
        -------
        np.ndarray
            UV luminosity in erg/s
        """
        # Convert to float64 to avoid overflow
        sed_flux_64 = sed_flux.astype(np.float64)
        
        # L_UV in L_sun
        L_UV_Lsun = sed_flux_64 * bin_width
        
        # Convert to erg/s
        L_UV_erg = L_UV_Lsun * self.L_SUN_ERG
        
        return L_UV_erg
    
    def calculate_sfr_uv(self, L_UV_erg: np.ndarray,
                        calibration: str = 'kennicutt2012') -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate SFR from UV luminosity
        
        Parameters
        ----------
        L_UV_erg : np.ndarray
            UV luminosity in erg/s
        calibration : str
            SFR calibration to use:
            - 'kennicutt2012': Kennicutt & Evans (2012)
            - 'madau2014': Madau & Dickinson (2014)
            
        Returns
        -------
        log_sfr : np.ndarray
            log10(SFR) in M_sun/yr
        sfr : np.ndarray
            SFR in M_sun/yr
        """
        # Mask for valid luminosities
        valid = L_UV_erg > 0
        
        # Initialize arrays
        log_sfr = np.full(len(L_UV_erg), np.nan)
        
        if calibration == 'kennicutt2012':
            # Kennicutt & Evans (2012) calibration
            # log(SFR) = log(L_UV) - 43.35
            log_sfr[valid] = np.log10(L_UV_erg[valid]) - 43.35
        elif calibration == 'madau2014':
            # Madau & Dickinson (2014) calibration
            # SFR = 1.4e-28 * L_UV
            # log(SFR) = log(1.4e-28) + log(L_UV)
            log_sfr[valid] = np.log10(1.4e-28) + np.log10(L_UV_erg[valid])
        else:
            raise ValueError(f"Unknown calibration: {calibration}")
        
        # Linear SFR
        sfr = np.where(valid, 10**log_sfr, np.nan)
        
        return log_sfr, sfr
    
    def apply_dust_correction(self, log_sfr: np.ndarray, A_V: np.ndarray,
                            wavelength: float = 1600.0,
                            attenuation_law: str = 'calzetti') -> np.ndarray:
        """
        Apply dust correction to SFR
        
        Parameters
        ----------
        log_sfr : np.ndarray
            Observed log10(SFR) without dust correction
        A_V : np.ndarray
            V-band extinction
        wavelength : float
            UV wavelength in Angstroms (default: 1600)
        attenuation_law : str
            Dust attenuation law to use (default: 'calzetti')
            
        Returns
        -------
        np.ndarray
            Dust-corrected log10(SFR)
        """
        if attenuation_law == 'calzetti':
            # Get A_lambda/A_V ratio
            if wavelength in self.CALZETTI_RATIOS:
                ratio = self.CALZETTI_RATIOS[wavelength]
            else:
                # Approximate for nearby wavelengths
                warnings.warn(f"No exact Calzetti ratio for {wavelength}Å, using 1600Å")
                ratio = self.CALZETTI_RATIOS[1600]
            
            # Calculate A_lambda
            A_lambda = np.clip(A_V, 0.0, 10.0) * ratio
            
            # Apply correction: SFR_corr = SFR_obs * 10^(0.4 * A_lambda)
            log_sfr_corr = log_sfr + 0.4 * A_lambda
            
        else:
            raise ValueError(f"Unknown attenuation law: {attenuation_law}")
        
        return log_sfr_corr
    
    def calculate_ssfr(self, sfr: np.ndarray, stellar_mass: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate specific star formation rate (sSFR)
        
        Parameters
        ----------
        sfr : np.ndarray
            Star formation rate in M_sun/yr
        stellar_mass : np.ndarray
            Stellar mass in M_sun
            
        Returns
        -------
        ssfr : np.ndarray
            Specific SFR in yr^-1
        log_ssfr : np.ndarray
            log10(sSFR)
        """
        # Mask for valid values
        valid = (sfr > 0) & (stellar_mass > 0)
        
        # Calculate sSFR
        ssfr = np.full(len(sfr), np.nan)
        ssfr[valid] = sfr[valid] / stellar_mass[valid]
        
        # Log sSFR
        log_ssfr = np.full(len(ssfr), np.nan)
        log_ssfr[valid] = np.log10(ssfr[valid])
        
        return ssfr, log_ssfr
    
    def classify_sfms(self, log_ssfr: np.ndarray, 
                     quenched_threshold: float = -11.0) -> Dict[str, np.ndarray]:
        """
        Classify galaxies based on star-forming main sequence
        
        Parameters
        ----------
        log_ssfr : np.ndarray
            log10(sSFR) values
        quenched_threshold : float
            log10(sSFR) threshold for quenched galaxies
            
        Returns
        -------
        dict
            Dictionary with classification masks
        """
        star_forming = log_ssfr > quenched_threshold
        quenched = log_ssfr <= quenched_threshold
        
        return {
            'star_forming': star_forming,
            'quenched': quenched,
            'sf_fraction': np.nanmean(star_forming),
            'quenched_fraction': np.nanmean(quenched)
        }
    
    def sfms_relation(self, log_mass: np.ndarray, z: float = 0.0,
                     relation: str = 'elbaz2007') -> np.ndarray:
        """
        Calculate expected SFR from star-forming main sequence relations
        
        Parameters
        ----------
        log_mass : np.ndarray
            log10(stellar mass / M_sun)
        z : float
            Redshift
        relation : str
            SFMS relation to use:
            - 'elbaz2007': Elbaz et al. (2007)
            - 'whitaker2012': Whitaker et al. (2012)
            
        Returns
        -------
        np.ndarray
            Expected log10(SFR) from SFMS
        """
        if relation == 'elbaz2007':
            # Elbaz et al. (2007) at z~0
            # log(SFR) = 0.77 * log(M*) - 7.65
            log_sfr_sfms = 0.77 * log_mass - 7.65
            
        elif relation == 'whitaker2012':
            # Whitaker et al. (2012)
            # More complex, redshift-dependent
            if z < 0.5:
                alpha = 0.70
                beta = -7.0
            elif z < 1.0:
                alpha = 0.75
                beta = -6.8
            elif z < 1.5:
                alpha = 0.80
                beta = -6.5
            else:
                alpha = 0.85
                beta = -6.2
            
            log_sfr_sfms = alpha * log_mass + beta
            
        else:
            raise ValueError(f"Unknown SFMS relation: {relation}")
        
        return log_sfr_sfms
    
    def calculate_sfr_offset(self, log_sfr: np.ndarray, log_mass: np.ndarray,
                           z: float = 0.0, relation: str = 'elbaz2007') -> np.ndarray:
        """
        Calculate offset from the star-forming main sequence
        
        Parameters
        ----------
        log_sfr : np.ndarray
            Observed log10(SFR)
        log_mass : np.ndarray
            log10(stellar mass)
        z : float
            Redshift
        relation : str
            SFMS relation to use
            
        Returns
        -------
        np.ndarray
            Δlog(SFR) = log(SFR_obs) - log(SFR_SFMS)
        """
        log_sfr_sfms = self.sfms_relation(log_mass, z, relation)
        return log_sfr - log_sfr_sfms
    
    def burstiness_parameter(self, sfr_history: np.ndarray, 
                           time_bins: np.ndarray) -> float:
        """
        Calculate burstiness parameter from SFR history
        
        Parameters
        ----------
        sfr_history : np.ndarray
            SFR history array
        time_bins : np.ndarray
            Time bins for SFR history
            
        Returns
        -------
        float
            Burstiness parameter (ratio of recent to past SFR)
        """
        if len(sfr_history) < 2:
            return np.nan
        
        # Recent SFR (last bin)
        sfr_recent = sfr_history[-1]
        
        # Average past SFR
        sfr_past = np.mean(sfr_history[:-1])
        
        if sfr_past > 0:
            return sfr_recent / sfr_past
        else:
            return np.nan