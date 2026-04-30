"""
Plotting utilities and style configurations
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def set_plot_style():
    """Set consistent plotting style for all figures"""
    
    # Font sizes
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 13
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 11
    
    # Figure properties
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.rcParams['figure.dpi'] = 100
    
    # Axes properties
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['axes.grid'] = False
    plt.rcParams['axes.axisbelow'] = True
    
    # Ticks
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['xtick.major.size'] = 5
    plt.rcParams['ytick.major.size'] = 5
    plt.rcParams['xtick.minor.size'] = 3
    plt.rcParams['ytick.minor.size'] = 3
    
    # Lines
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['lines.markersize'] = 6
    
    # Legend
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.8
    plt.rcParams['legend.loc'] = 'best'
    
    # Colors
    plt.rcParams['image.cmap'] = 'viridis'


def get_color_palette(n_colors=10):
    """Get a nice color palette"""
    if n_colors <= 10:
        return plt.cm.tab10(np.linspace(0, 1, n_colors))
    else:
        return plt.cm.viridis(np.linspace(0, 1, n_colors))


def add_colorbar(im, ax, label='', orientation='vertical', pad=0.02):
    """Add a properly sized colorbar to an axis"""
    if orientation == 'vertical':
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=pad)
        cb = plt.colorbar(im, cax=cax, orientation=orientation)
    else:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=pad)
        cb = plt.colorbar(im, cax=cax, orientation=orientation)
    
    cb.set_label(label)
    return cb


def plot_histogram_with_stats(data, ax=None, bins=50, 
                             label='', color='steelblue', alpha=0.7,
                             show_stats=True, stat_loc='upper right'):
    """Plot histogram with statistical annotations"""
    if ax is None:
        fig, ax = plt.subplots()
    
    # Remove NaN values
    data_clean = data[np.isfinite(data)]
    
    # Plot histogram
    n, bins, patches = ax.hist(data_clean, bins=bins, 
                              color=color, alpha=alpha, 
                              edgecolor='white', linewidth=0.5,
                              label=label)
    
    # Add statistics
    if show_stats and len(data_clean) > 0:
        mean_val = np.mean(data_clean)
        median_val = np.median(data_clean)
        std_val = np.std(data_clean)
        
        stats_text = f'Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}'
        
        # Position based on stat_loc
        if stat_loc == 'upper right':
            x, y = 0.95, 0.95
            ha, va = 'right', 'top'
        elif stat_loc == 'upper left':
            x, y = 0.05, 0.95
            ha, va = 'left', 'top'
        else:
            x, y = 0.95, 0.05
            ha, va = 'right', 'bottom'
        
        ax.text(x, y, stats_text, transform=ax.transAxes,
                ha=ha, va=va,
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor='white', alpha=0.8))
    
    return ax


def plot_2d_density(x, y, ax=None, bins=50, cmap='magma',
                   xlabel='', ylabel='', title='',
                   show_contours=True, colorbar=True):
    """Create 2D density plot with optional contours"""
    if ax is None:
        fig, ax = plt.subplots()
    
    # Remove NaN values
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    
    # Create 2D histogram
    h = ax.hist2d(x_clean, y_clean, bins=bins, 
                  cmap=cmap, norm=LogNorm())
    
    # Add contours
    if show_contours:
        from scipy.ndimage import gaussian_filter
        counts, xedges, yedges = np.histogram2d(x_clean, y_clean, bins=bins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        
        # Smooth for contours
        counts_smooth = gaussian_filter(counts.T, sigma=1.0)
        
        # Contour levels
        levels = np.percentile(counts_smooth[counts_smooth > 0], [10, 30, 50, 70, 90])
        
        ax.contour(counts_smooth, extent=extent, 
                  levels=levels, colors='white', 
                  linewidths=0.5, alpha=0.5)
    
    # Add colorbar
    if colorbar:
        add_colorbar(h[3], ax, label='Count')
    
    # Labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    return ax


def plot_comparison(data1, data2, labels=['Data 1', 'Data 2'],
                   bins=50, colors=['steelblue', 'tomato'],
                   xlabel='', ylabel='Normalized Count', title=''):
    """Plot comparison of two distributions"""
    fig, ax = plt.subplots()
    
    # Plot both histograms
    for data, label, color in zip([data1, data2], labels, colors):
        data_clean = data[np.isfinite(data)]
        ax.hist(data_clean, bins=bins, density=True,
                alpha=0.6, color=color, label=label,
                edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    
    return fig, ax


def save_figure(fig, filename, dpi=150, bbox_inches='tight', **kwargs):
    """Save figure with consistent settings"""
    fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
    print(f"Figure saved: {filename}")


def create_subplots(nrows=1, ncols=1, figsize=None, **kwargs):
    """Create subplots with consistent settings"""
    if figsize is None:
        figsize = (6*ncols, 5*nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    
    # Ensure axes is always a list
    if nrows * ncols == 1:
        axes = [axes]
    elif nrows == 1 or ncols == 1:
        axes = axes.flatten()
    
    return fig, axes