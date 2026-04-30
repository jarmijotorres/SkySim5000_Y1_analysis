"""
Metallicity calculator using empirical scaling relations
"""

import numpy as np
from typing import Tuple, Dict, Optional
import warnings


class MetallicityCalculator:
    """
    Calculate gas-phase metallicities using empirical relations
    """
    
    # Solar metallicity
    Z_SOLAR = 8.69  # 12 + log(O/H)_solar
    
    def __init__(self):
        """Initialize metallicity calculator"""
        pass
    
    def fmr_metallicity(self, log_mstar: np.ndarray, log_sfr: np.ndarray,
                       alpha: float = 0.32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate metallicity using the Fundamental Metallicity Relation
        (Mannucci et al. 2010)
        
        Parameters
        ----------
        log_mstar : np.ndarray
            log10(stellar mass / M_sun)
        log_sfr : np.ndarray
            log10(SFR / M_sun/yr), dust-corrected
        alpha : float
            FMR parameter (default 0.32 from Mannucci+2010)
            
        Returns
        -------
        Z : np.ndarray
            12 + log10(O/H)
        mu : np.ndarray
            The combined FMR parameter mu_alpha
        """
        # Calculate mu parameter
        mu = log_mstar - alpha * log_sfr
        
        # Mannucci+2010 polynomial coefficients (Table 3)
        # Valid for mu in [8.7, 11.2] approximately
        Z = (8.90
             + 0.37 * (mu - 10.0)
             - 0.14 * (mu - 10.0)**2
             - 0.19 * (mu - 10.0)**3)
        
        # Warn about extrapolation
        if np.any((mu < 8.7) | (mu > 11.2)):
            n_extrap = np.sum((mu < 8.7) | (mu > 11.2))
            warnings.warn(f"FMR extrapolated for {n_extrap} galaxies outside valid range [8.7, 11.2]")
        
        return Z, mu
    
    def mass_metallicity_relation(self, log_mstar: np.ndarray,
                                 relation: str = 'tremonti2004') -> np.ndarray:
        """
        Calculate metallicity from mass-metallicity relation
        
        Parameters
        ----------
        log_mstar : np.ndarray
            log10(stellar mass / M_sun)
        relation : str
            Which relation to use:
            - 'tremonti2004': Tremonti et al. (2004)
            - 'kewley2008': Kewley & Ellison (2008) PP04 O3N2
            - 'andrews2013': Andrews & Martini (2013)
            
        Returns
        -------
        np.ndarray
            12 + log10(O/H)
        """
        if relation == 'tremonti2004':
            # Tremonti et al. (2004) - polynomial fit
            x = log_mstar
            Z = (-1.492 + 1.847*x - 0.08026*x**2)
            
        elif relation == 'kewley2008':
            # Kewley & Ellison (2008) - PP04 O3N2 calibration
            # Asymptotic form
            Z = 8.90 + 0.57 * (log_mstar - 10.0) - 0.19 * (log_mstar - 10.0)**2
            
        elif relation == 'andrews2013':
            # Andrews & Martini (2013)
            # Using their equation 3
            x = log_mstar - 10.0
            Z = 8.798 + 0.639 * x - 0.137 * x**2
            
        else:
            raise ValueError(f"Unknown mass-metallicity relation: {relation}")
        
        return Z
    
    def metallicity_scatter(self, Z: np.ndarray, sigma: float = 0.1) -> np.ndarray:
        """
        Add intrinsic scatter to metallicity values
        
        Parameters
        ----------
        Z : np.ndarray
            Metallicity values
        sigma : float
            Intrinsic scatter in dex
            
        Returns
        -------
        np.ndarray
            Metallicity with added scatter
        """
        scatter = np.random.normal(0, sigma, size=len(Z))
        return Z + scatter
    
    def metallicity_gradient(self, Z_center: np.ndarray, 
                           radius_kpc: np.ndarray,
                           gradient: float = -0.1) -> np.ndarray:
        """
        Apply radial metallicity gradient
        
        Parameters
        ----------
        Z_center : np.ndarray
            Central metallicity
        radius_kpc : np.ndarray
            Galactocentric radius in kpc
        gradient : float
            Metallicity gradient in dex/kpc (typically negative)
            
        Returns
        -------
        np.ndarray
            Metallicity at given radius
        """
        return Z_center + gradient * radius_kpc
    
    def effective_yield(self, Z: np.ndarray, gas_fraction: np.ndarray) -> np.ndarray:
        """
        Calculate effective yield from closed-box model
        
        Parameters
        ----------
        Z : np.ndarray
            Metallicity (12 + log O/H)
        gas_fraction : np.ndarray
            Gas mass fraction
            
        Returns
        -------
        np.ndarray
            Effective yield
        """
        # Convert from 12 + log(O/H) to linear metallicity
        Z_linear = 10**(Z - 12.0)
        Z_solar_linear = 10**(self.Z_SOLAR - 12.0)
        
        # Normalized metallicity
        Z_norm = Z_linear / Z_solar_linear
        
        # Effective yield from closed-box model
        # y_eff = Z / ln(1/f_gas)
        with np.errstate(divide='ignore', invalid='ignore'):
            y_eff = Z_norm / (-np.log(gas_fraction))
            y_eff[gas_fraction >= 1] = np.nan
            y_eff[gas_fraction <= 0] = np.nan
        
        return y_eff
    
    def alpha_enhancement(self, log_mstar: np.ndarray,
                         log_sfr: np.ndarray) -> np.ndarray:
        """
        Estimate [α/Fe] enhancement based on stellar mass and SFR
        
        Parameters
        ----------
        log_mstar : np.ndarray
            log10(stellar mass / M_sun)
        log_sfr : np.ndarray
            log10(SFR / M_sun/yr)
            
        Returns
        -------
        np.ndarray
            [α/Fe] values
        """
        # Simple prescription: massive, quiescent galaxies are α-enhanced
        log_ssfr = log_sfr - log_mstar
        
        # Base [α/Fe] on sSFR
        alpha_fe = np.zeros_like(log_ssfr)
        
        # Quiescent galaxies (log sSFR < -11)
        quiescent = log_ssfr < -11
        alpha_fe[quiescent] = 0.3
        
        # Transition region
        transition = (log_ssfr >= -11) & (log_ssfr < -10)
        alpha_fe[transition] = 0.3 * (11 + log_ssfr[transition])
        
        # Star-forming galaxies
        # alpha_fe remains 0
        
        return alpha_fe
    
    def nitrogen_to_oxygen(self, Z: np.ndarray) -> np.ndarray:
        """
        Calculate N/O ratio from oxygen abundance
        
        Parameters
        ----------
        Z : np.ndarray
            12 + log(O/H)
            
        Returns
        -------
        np.ndarray
            log(N/O)
        """
        # Empirical relation from Pérez-Montero & Contini (2009)
        # Secondary nitrogen production at high metallicity
        
        # For low metallicity (primary production)
        log_NO_low = -1.5
        
        # For high metallicity (secondary production)
        # log(N/O) increases with 12 + log(O/H)
        log_NO = np.where(Z < 8.0,
                         log_NO_low,
                         log_NO_low + 1.0 * (Z - 8.0))
        
        return log_NO
    
    def metallicity_distribution_stats(self, Z: np.ndarray) -> Dict:
        """
        Calculate statistics of metallicity distribution
        
        Parameters
        ----------
        Z : np.ndarray
            Metallicity values
            
        Returns
        -------
        dict
            Statistics dictionary
        """
        valid = np.isfinite(Z)
        
        if not np.any(valid):
            return {
                'mean': np.nan,
                'median': np.nan,
                'std': np.nan,
                'mad': np.nan,
                'percentiles': [np.nan] * 5,
                'solar_fraction': np.nan,
                'subsolar_fraction': np.nan,
                'supersolar_fraction': np.nan
            }
        
        Z_valid = Z[valid]
        
        # Calculate median absolute deviation
        mad = np.median(np.abs(Z_valid - np.median(Z_valid)))
        
        # Fractions relative to solar
        solar_fraction = np.mean(np.abs(Z_valid - self.Z_SOLAR) < 0.1)
        subsolar_fraction = np.mean(Z_valid < self.Z_SOLAR - 0.1)
        supersolar_fraction = np.mean(Z_valid > self.Z_SOLAR + 0.1)
        
        return {
            'mean': np.mean(Z_valid),
            'median': np.median(Z_valid),
            'std': np.std(Z_valid),
            'mad': mad,
            'percentiles': np.percentile(Z_valid, [10, 25, 50, 75, 90]),
            'solar_fraction': solar_fraction,
            'subsolar_fraction': subsolar_fraction,
            'supersolar_fraction': supersolar_fraction
        }