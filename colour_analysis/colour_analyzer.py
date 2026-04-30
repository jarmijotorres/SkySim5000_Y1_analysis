"""
Colour analysis tools for galaxy populations
"""

import numpy as np
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


class ColourAnalysis:
    """
    Analyze galaxy colours and create colour-magnitude diagrams
    """
    
    def __init__(self, data_loader=None):
        """
        Initialize colour analysis
        
        Parameters
        ----------
        data_loader : DataLoader instance, optional
            DataLoader with loaded galaxy data
        """
        self.data_loader = data_loader
        
    def calculate_colour(self, mag1: np.ndarray, mag2: np.ndarray) -> np.ndarray:
        """
        Calculate colour from two magnitude arrays
        
        Parameters
        ----------
        mag1 : np.ndarray
            First band magnitudes (bluer band)
        mag2 : np.ndarray
            Second band magnitudes (redder band)
            
        Returns
        -------
        np.ndarray
            Colour array (mag1 - mag2)
        """
        return mag1 - mag2
    
    def create_cmd_data(self, x_data: np.ndarray, y_data: np.ndarray,
                       x_bins: int = 50, y_bins: int = 50,
                       min_count: int = 2) -> Dict:
        """
        Create data for colour-magnitude diagram
        
        Parameters
        ----------
        x_data : np.ndarray
            X-axis data (e.g., magnitude or log stellar mass)
        y_data : np.ndarray
            Y-axis data (e.g., colour)
        x_bins : int
            Number of bins in x direction
        y_bins : int
            Number of bins in y direction
        min_count : int
            Minimum count per bin
            
        Returns
        -------
        dict
            Dictionary containing binned data and statistics
        """
        # Create 2D histogram
        counts, x_edges, y_edges, _ = binned_statistic_2d(
            x_data, y_data,
            values=None,
            statistic='count',
            bins=[x_bins, y_bins]
        )
        
        # Mask low-count bins
        counts_masked = np.where(counts >= min_count, counts, np.nan)
        
        # Log scale for visualization
        log_counts = np.log10(counts_masked)
        
        # Bin centers
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        
        # Smooth for contours
        counts_smooth = gaussian_filter(
            np.where(counts >= min_count, counts, 0).T,
            sigma=0.75
        )
        
        return {
            'counts': counts,
            'counts_masked': counts_masked,
            'log_counts': log_counts,
            'counts_smooth': counts_smooth,
            'x_edges': x_edges,
            'y_edges': y_edges,
            'x_centers': x_centers,
            'y_centers': y_centers
        }
    
    def classify_red_blue(self, colour: np.ndarray, 
                         colour_cut: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classify galaxies as red or blue based on colour cut
        
        Parameters
        ----------
        colour : np.ndarray
            Galaxy colours
        colour_cut : float
            Colour cut value (galaxies with colour > cut are red)
            
        Returns
        -------
        red_mask : np.ndarray
            Boolean mask for red galaxies
        blue_mask : np.ndarray
            Boolean mask for blue galaxies
        """
        red_mask = colour >= colour_cut
        blue_mask = colour < colour_cut
        return red_mask, blue_mask
    
    def calculate_red_fraction(self, colour: np.ndarray, 
                             colour_cut: float = 0.5) -> float:
        """
        Calculate fraction of red galaxies
        
        Parameters
        ----------
        colour : np.ndarray
            Galaxy colours
        colour_cut : float
            Colour cut value
            
        Returns
        -------
        float
            Fraction of red galaxies
        """
        red_mask, _ = self.classify_red_blue(colour, colour_cut)
        return red_mask.mean()
    
    def bimodality_test(self, colour: np.ndarray, bins: int = 100) -> Dict:
        """
        Test for bimodality in colour distribution
        
        Parameters
        ----------
        colour : np.ndarray
            Galaxy colours
        bins : int
            Number of bins for histogram
            
        Returns
        -------
        dict
            Dictionary with bimodality test results
        """
        from scipy.stats import gaussian_kde
        from scipy.signal import find_peaks
        
        # Create histogram
        hist, bin_edges = np.histogram(colour[np.isfinite(colour)], bins=bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        # Find peaks
        peaks, properties = find_peaks(hist, height=hist.max()*0.1)
        
        # KDE for smoother distribution
        kde = gaussian_kde(colour[np.isfinite(colour)])
        x_kde = np.linspace(colour[np.isfinite(colour)].min(), 
                           colour[np.isfinite(colour)].max(), 200)
        y_kde = kde(x_kde)
        
        # Find peaks in KDE
        kde_peaks, _ = find_peaks(y_kde, height=y_kde.max()*0.1)
        
        return {
            'n_peaks_hist': len(peaks),
            'peak_positions_hist': bin_centers[peaks] if len(peaks) > 0 else [],
            'n_peaks_kde': len(kde_peaks),
            'peak_positions_kde': x_kde[kde_peaks] if len(kde_peaks) > 0 else [],
            'kde_x': x_kde,
            'kde_y': y_kde,
            'is_bimodal': len(kde_peaks) >= 2
        }
    
    def plot_cmd(self, x_data: np.ndarray, y_data: np.ndarray,
                xlabel: str = r'$M_r$', ylabel: str = r'$(g-r)$',
                title: str = 'Colour-Magnitude Diagram',
                colour_cut: Optional[float] = None,
                invert_x: bool = True,
                figsize: Tuple[float, float] = (8, 6),
                cmap: str = 'jet') -> plt.Figure:
        """
        Plot colour-magnitude diagram
        
        Parameters
        ----------
        x_data : np.ndarray
            X-axis data
        y_data : np.ndarray
            Y-axis data (colour)
        xlabel : str
            X-axis label
        ylabel : str
            Y-axis label
        title : str
            Plot title
        colour_cut : float, optional
            If provided, draws horizontal line at this colour
        invert_x : bool
            Whether to invert x-axis (for magnitudes)
        figsize : tuple
            Figure size
        cmap : str
            Colormap name
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object
        """
        cmd_data = self.create_cmd_data(x_data, y_data)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Density plot
        im = ax.pcolormesh(
            cmd_data['x_edges'], 
            cmd_data['y_edges'],
            cmd_data['log_counts'].T,
            cmap=cmap,
            shading='auto',
            rasterized=True
        )
        
        # Contours
        ax.contour(
            cmd_data['x_centers'],
            cmd_data['y_centers'],
            cmd_data['counts_smooth'],
            levels=5,
            colors='white',
            linewidths=0.6,
            alpha=0.7
        )
        
        # Colour cut line
        if colour_cut is not None:
            ax.axhline(colour_cut, color='cyan', ls='--', lw=1.5,
                      label=f'Colour cut = {colour_cut}')
            ax.legend()
        
        # Colorbar
        cb = plt.colorbar(im, ax=ax, pad=0.02)
        cb.set_label(r'$\log_{10}(N_{\rm gal})$')
        
        # Labels and formatting
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        if invert_x:
            ax.invert_xaxis()
            
        ax.tick_params(which='both', direction='in', top=True, right=True)
        
        return fig
    
    def analyze_colour_gradients(self, x_data: np.ndarray, y_data: np.ndarray,
                               x_bins: int = 20) -> Dict:
        """
        Analyze colour gradients as function of x variable
        
        Parameters
        ----------
        x_data : np.ndarray
            X-axis data (e.g., magnitude or mass)
        y_data : np.ndarray
            Colour data
        x_bins : int
            Number of bins in x
            
        Returns
        -------
        dict
            Dictionary with gradient analysis results
        """
        # Bin the data
        bin_edges = np.percentile(x_data, np.linspace(0, 100, x_bins+1))
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        mean_colours = []
        std_colours = []
        median_colours = []
        
        for i in range(len(bin_edges)-1):
            mask = (x_data >= bin_edges[i]) & (x_data < bin_edges[i+1])
            if mask.sum() > 0:
                mean_colours.append(np.mean(y_data[mask]))
                std_colours.append(np.std(y_data[mask]))
                median_colours.append(np.median(y_data[mask]))
            else:
                mean_colours.append(np.nan)
                std_colours.append(np.nan)
                median_colours.append(np.nan)
        
        return {
            'bin_centers': bin_centers,
            'mean_colours': np.array(mean_colours),
            'std_colours': np.array(std_colours),
            'median_colours': np.array(median_colours)
        }