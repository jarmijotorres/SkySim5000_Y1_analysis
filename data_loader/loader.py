"""
Data loader for SkySim5000 galaxy catalog
"""

import numpy as np
import h5py
import pandas as pd
from typing import Dict, Optional, Union, List


class DataLoader:
    """
    Class to load and preprocess SkySim5000 galaxy catalog data
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the data loader
        
        Parameters
        ----------
        file_path : str
            Path to the SkySim5000 HDF5 file
        """
        self.file_path = file_path
        self.data = {}
        self._file = None
        
    def __enter__(self):
        """Context manager entry"""
        self._file = h5py.File(self.file_path, 'r')
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self._file:
            self._file.close()
            
    def load_data(self, columns: Optional[List[str]] = None, 
                  group: str = 'SkySim5000_small') -> Dict[str, np.ndarray]:
        """
        Load data from the HDF5 file
        
        Parameters
        ----------
        columns : list of str, optional
            Specific columns to load. If None, loads all columns
        group : str
            HDF5 group name (default: 'SkySim5000_small')
            
        Returns
        -------
        dict
            Dictionary with column names as keys and numpy arrays as values
        """
        with h5py.File(self.file_path, 'r') as f:
            if group not in f:
                raise ValueError(f"Group '{group}' not found in HDF5 file")
                
            available_columns = list(f[group].keys())
            
            if columns is None:
                columns = available_columns
            else:
                # Validate requested columns
                invalid_cols = set(columns) - set(available_columns)
                if invalid_cols:
                    raise ValueError(f"Columns not found: {invalid_cols}")
            
            # Load data
            for col in columns:
                self.data[col] = f[f'{group}/{col}'][:]
                
        return self.data
    
    def get_magnitudes(self, band: str = 'r', z0: bool = True) -> np.ndarray:
        """
        Get absolute magnitudes for a specific band
        
        Parameters
        ----------
        band : str
            Photometric band ('u', 'g', 'r', 'i', 'z')
        z0 : bool
            If True, returns rest-frame magnitudes at z=0
            
        Returns
        -------
        np.ndarray
            Absolute magnitudes
        """
        if z0:
            key = f'Mag_true_{band}_lsst_z0'
        else:
            key = f'mag_{band}_lsst'
            
        if key not in self.data:
            raise ValueError(f"Magnitude data '{key}' not loaded")
            
        return self.data[key]
    
    def get_colors(self, bands: tuple = ('g', 'r')) -> np.ndarray:
        """
        Calculate galaxy colors
        
        Parameters
        ----------
        bands : tuple of str
            Two bands to calculate color (e.g., ('g', 'r') for g-r)
            
        Returns
        -------
        np.ndarray
            Color array
        """
        mag1 = self.get_magnitudes(bands[0], z0=True)
        mag2 = self.get_magnitudes(bands[1], z0=True)
        return mag1 - mag2
    
    def get_stellar_masses(self) -> np.ndarray:
        """Get stellar masses"""
        if 'stellar_mass' not in self.data:
            raise ValueError("Stellar mass data not loaded")
        return self.data['stellar_mass']
    
    def get_redshifts(self, true: bool = True) -> np.ndarray:
        """
        Get redshifts
        
        Parameters
        ----------
        true : bool
            If True, returns true redshifts; otherwise photometric
            
        Returns
        -------
        np.ndarray
            Redshift array
        """
        key = 'redshift_true' if true else 'redshift'
        if key not in self.data:
            raise ValueError(f"Redshift data '{key}' not loaded")
        return self.data[key]
    
    def get_sed_data(self) -> np.ndarray:
        """Get SED data for UV luminosity"""
        key = 'sed_1246_306_disk_no_host_extinction'
        if key not in self.data:
            raise ValueError("SED data not loaded")
        return self.data[key]
    
    def get_dust_extinction(self) -> np.ndarray:
        """Get dust extinction (A_v)"""
        key = 'A_v_disk'
        if key not in self.data:
            raise ValueError("Dust extinction data not loaded")
        return self.data[key]
    
    def apply_selection(self, mask: np.ndarray) -> 'DataLoader':
        """
        Apply a selection mask to all loaded data
        
        Parameters
        ----------
        mask : np.ndarray
            Boolean mask to apply
            
        Returns
        -------
        DataLoader
            New DataLoader instance with selected data
        """
        new_loader = DataLoader(self.file_path)
        new_loader.data = {key: val[mask] for key, val in self.data.items()}
        return new_loader
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert loaded data to pandas DataFrame"""
        return pd.DataFrame(self.data)
    
    def summary(self) -> None:
        """Print summary of loaded data"""
        print(f"SkySim5000 Data Summary")
        print(f"=======================")
        print(f"File: {self.file_path}")
        print(f"Loaded columns: {len(self.data)}")
        
        if self.data:
            n_galaxies = len(next(iter(self.data.values())))
            print(f"Number of galaxies: {n_galaxies:,}")
            print(f"\nColumns:")
            for key, val in self.data.items():
                print(f"  - {key}: shape {val.shape}, dtype {val.dtype}")