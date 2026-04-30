"""
Luminosity Function calculator using the 1/Vmax method
"""

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d
from typing import Tuple, Optional, Dict
import warnings


class LuminosityFunction:
    """
    Calculate galaxy luminosity functions using the 1/Vmax method
    """
    
    def __init__(self, cosmology: Optional[FlatLambdaCDM] = None):
        """
        Initialize the luminosity function calculator
        
        Parameters
        ----------
        cosmology : astropy.cosmology.FlatLambdaCDM, optional
            Cosmology to use. If None, uses default (H0=71, Om0=0.2648, Ob0=0.0448)
        """
        if cosmology is None:
            self.cosmo = FlatLambdaCDM(H0=71.0, Om0=0.2648, Ob0=0.0448)
        else:
            self.cosmo = cosmology
            
    def calculate_vmax(self, mag_abs: np.ndarray, redshifts: np.ndarray,
                      mag_lim: float, z_min: float, z_max: float,
                      area_deg2: float) -> np.ndarray:
        """
        Calculate Vmax for each galaxy
        
        Parameters
        ----------
        mag_abs : np.ndarray
            Absolute magnitudes
        redshifts : np.ndarray
            Galaxy redshifts
        mag_lim : float
            Apparent magnitude limit
        z_min : float
            Minimum redshift
        z_max : float
            Maximum redshift
        area_deg2 : float
            Survey area in square degrees
            
        Returns
        -------
        np.ndarray
            Vmax values in Mpc^3
        """
        # Convert area to steradians
        area_sr = area_deg2 * (np.pi / 180.0)**2
        f_sky = area_sr / (4.0 * np.pi)
        
        # Create interpolation functions for efficient computation
        z_interp = np.linspace(z_min, z_max, 10000)
        mu_interp = self.cosmo.distmod(z_interp).value
        
        # Invert: given M_abs, find z_max
        M_lim_interp = mag_lim - mu_interp
        
        # M_lim_interp is monotonically decreasing → need to flip for interp1d
        z_from_Mlim = interp1d(
            M_lim_interp[::-1], z_interp[::-1],
            bounds_error=False,
            fill_value=(z_max, z_min)
        )
        
        # Calculate z_max for each galaxy
        z_max_arr = np.clip(z_from_Mlim(mag_abs), z_min, z_max)
        
        # Calculate comoving volumes
        Vc_interp = self.cosmo.comoving_volume(z_interp).value  # Mpc^3
        Vc_from_z = interp1d(z_interp, Vc_interp, bounds_error=False,
                           fill_value=(Vc_interp[0], Vc_interp[-1]))
        
        # Calculate Vmax
        Vmax = f_sky * (Vc_from_z(z_max_arr) - Vc_from_z(z_min))
        
        return Vmax
    
    def calculate_lf(self, mag_abs: np.ndarray, vmax: np.ndarray,
                    mag_bins: Optional[np.ndarray] = None,
                    min_galaxies: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the luminosity function using 1/Vmax method
        
        Parameters
        ----------
        mag_abs : np.ndarray
            Absolute magnitudes
        vmax : np.ndarray
            Vmax values for each galaxy
        mag_bins : np.ndarray, optional
            Magnitude bin edges. If None, uses default binning
        min_galaxies : int
            Minimum number of galaxies per bin
            
        Returns
        -------
        mag_centers : np.ndarray
            Magnitude bin centers
        phi : np.ndarray
            Luminosity function values [Mpc^-3 mag^-1]
        phi_err : np.ndarray
            Poisson errors on phi
        """
        if mag_bins is None:
            mag_bins = np.arange(-25.5, -15.0, 0.5)
            
        mag_centers = 0.5 * (mag_bins[:-1] + mag_bins[1:])
        bin_width = mag_bins[1] - mag_bins[0]
        
        phi = np.zeros(len(mag_centers))
        phi_err = np.zeros(len(mag_centers))
        n_gal = np.zeros(len(mag_centers), dtype=int)
        
        # Handle potential division by zero
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            
            for k, (mlo, mhi) in enumerate(zip(mag_bins[:-1], mag_bins[1:])):
                in_bin = (mag_abs >= mlo) & (mag_abs < mhi)
                n_gal[k] = in_bin.sum()
                
                if n_gal[k] == 0:
                    phi[k] = np.nan
                    phi_err[k] = np.nan
                    continue
                    
                contributions = 1.0 / vmax[in_bin]
                phi[k] = contributions.sum() / bin_width  # Mpc^-3 mag^-1
                phi_err[k] = np.sqrt(np.sum(contributions**2)) / bin_width
        
        # Mask bins with too few galaxies
        mask = n_gal < min_galaxies
        phi[mask] = np.nan
        phi_err[mask] = np.nan
        
        return mag_centers, phi, phi_err
    
    def fit_schechter(self, mag_centers: np.ndarray, phi: np.ndarray,
                     phi_err: np.ndarray) -> Dict[str, float]:
        """
        Fit a Schechter function to the luminosity function
        
        Parameters
        ----------
        mag_centers : np.ndarray
            Magnitude bin centers
        phi : np.ndarray
            Luminosity function values
        phi_err : np.ndarray
            Errors on phi
            
        Returns
        -------
        dict
            Fitted parameters: phi_star, M_star, alpha
        """
        from scipy.optimize import curve_fit
        
        def schechter_mag(M, phi_star, M_star, alpha):
            """Schechter function in magnitude space"""
            x = 10.0 ** (0.4 * (M_star - M))
            return (np.log(10) / 2.5) * phi_star * x**(alpha + 1) * np.exp(-x)
        
        # Use only valid bins
        valid = np.isfinite(phi) & np.isfinite(phi_err) & (phi > 0)
        
        if valid.sum() < 3:
            raise ValueError("Not enough valid data points for fitting")
        
        # Initial guess
        p0 = [1e-3, -21.0, -1.2]
        bounds = ([0, -24.0, -2.5], [1.0, -18.0, 0.0])
        
        try:
            popt, pcov = curve_fit(
                schechter_mag,
                mag_centers[valid],
                phi[valid],
                p0=p0,
                sigma=phi_err[valid],
                absolute_sigma=True,
                bounds=bounds,
                maxfev=10000
            )
            
            perr = np.sqrt(np.diag(pcov))
            
            return {
                'phi_star': popt[0],
                'phi_star_err': perr[0],
                'M_star': popt[1],
                'M_star_err': perr[1],
                'alpha': popt[2],
                'alpha_err': perr[2]
            }
            
        except Exception as e:
            warnings.warn(f"Schechter fit failed: {e}")
            return None
    
    def fit_double_schechter(self, mag_centers: np.ndarray, phi: np.ndarray,
                           phi_err: np.ndarray) -> Dict[str, float]:
        """
        Fit a double Schechter function to the luminosity function
        
        Parameters
        ----------
        mag_centers : np.ndarray
            Magnitude bin centers
        phi : np.ndarray
            Luminosity function values
        phi_err : np.ndarray
            Errors on phi
            
        Returns
        -------
        dict
            Fitted parameters: phi1_star, phi2_star, M_star, alpha1, alpha2
        """
        from scipy.optimize import curve_fit
        
        def double_schechter_mag(M, phi1_star, phi2_star, M_star, alpha1, alpha2):
            """Double Schechter function"""
            x = 10.0 ** (0.4 * (M_star - M))
            factor = (np.log(10) / 2.5) * np.exp(-x)
            comp1 = phi1_star * x**(alpha1 + 1)
            comp2 = phi2_star * x**(alpha2 + 1)
            return factor * (comp1 + comp2)
        
        # Use only valid bins
        valid = np.isfinite(phi) & np.isfinite(phi_err) & (phi > 0)
        
        if valid.sum() < 5:
            raise ValueError("Not enough valid data points for double Schechter fitting")
        
        # Initial guess
        p0 = [1e-3, 1e-4, -21.0, -0.5, -1.5]
        bounds = ([0, 0, -24.0, -2.5, -2.5], [1.0, 1.0, -18.0, 0.5, 0.0])
        
        try:
            popt, pcov = curve_fit(
                double_schechter_mag,
                mag_centers[valid],
                phi[valid],
                p0=p0,
                sigma=phi_err[valid],
                absolute_sigma=True,
                bounds=bounds,
                maxfev=10000
            )
            
            perr = np.sqrt(np.diag(pcov))
            
            return {
                'phi1_star': popt[0],
                'phi1_star_err': perr[0],
                'phi2_star': popt[1],
                'phi2_star_err': perr[1],
                'M_star': popt[2],
                'M_star_err': perr[2],
                'alpha1': popt[3],
                'alpha1_err': perr[3],
                'alpha2': popt[4],
                'alpha2_err': perr[4]
            }
            
        except Exception as e:
            warnings.warn(f"Double Schechter fit failed: {e}")
            return None