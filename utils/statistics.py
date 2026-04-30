"""
Statistical utilities for galaxy population analysis
"""

import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional


def calculate_statistics(data: np.ndarray, name: str = 'Data') -> Dict:
    """
    Calculate comprehensive statistics for a dataset
    
    Parameters
    ----------
    data : np.ndarray
        Input data array
    name : str
        Name for the dataset
        
    Returns
    -------
    dict
        Dictionary with statistical measures
    """
    # Remove NaN values
    data_clean = data[np.isfinite(data)]
    
    if len(data_clean) == 0:
        return {
            'name': name,
            'count': 0,
            'mean': np.nan,
            'median': np.nan,
            'std': np.nan,
            'mad': np.nan,
            'min': np.nan,
            'max': np.nan,
            'percentiles': {p: np.nan for p in [5, 16, 25, 50, 75, 84, 95]}
        }
    
    # Calculate percentiles
    percentiles = [5, 16, 25, 50, 75, 84, 95]
    perc_values = np.percentile(data_clean, percentiles)
    
    # Median absolute deviation
    mad = np.median(np.abs(data_clean - np.median(data_clean)))
    
    return {
        'name': name,
        'count': len(data_clean),
        'mean': np.mean(data_clean),
        'median': np.median(data_clean),
        'std': np.std(data_clean),
        'mad': mad,
        'min': np.min(data_clean),
        'max': np.max(data_clean),
        'percentiles': dict(zip(percentiles, perc_values))
    }


def bootstrap_error(data: np.ndarray, statistic=np.mean, 
                   n_bootstrap: int = 1000, confidence: float = 0.68) -> Tuple[float, float]:
    """
    Calculate bootstrap error on a statistic
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    statistic : callable
        Statistic function (default: np.mean)
    n_bootstrap : int
        Number of bootstrap samples
    confidence : float
        Confidence level (default: 0.68 for 1-sigma)
        
    Returns
    -------
    lower : float
        Lower confidence bound
    upper : float
        Upper confidence bound
    """
    data_clean = data[np.isfinite(data)]
    
    if len(data_clean) < 2:
        return np.nan, np.nan
    
    # Bootstrap resampling
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        resample = np.random.choice(data_clean, size=len(data_clean), replace=True)
        bootstrap_stats.append(statistic(resample))
    
    # Calculate confidence intervals
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_stats, 100 * alpha)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha))
    
    return lower, upper


def weighted_statistics(values: np.ndarray, weights: np.ndarray) -> Dict:
    """
    Calculate weighted statistics
    
    Parameters
    ----------
    values : np.ndarray
        Data values
    weights : np.ndarray
        Weights for each value
        
    Returns
    -------
    dict
        Weighted statistics
    """
    # Clean data
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values_clean = values[mask]
    weights_clean = weights[mask]
    
    if len(values_clean) == 0:
        return {
            'weighted_mean': np.nan,
            'weighted_std': np.nan,
            'weighted_median': np.nan,
            'effective_n': 0
        }
    
    # Normalize weights
    weights_norm = weights_clean / np.sum(weights_clean)
    
    # Weighted mean
    weighted_mean = np.sum(values_clean * weights_norm)
    
    # Weighted variance
    weighted_var = np.sum(weights_norm * (values_clean - weighted_mean)**2)
    weighted_std = np.sqrt(weighted_var)
    
    # Weighted median (approximate)
    sorted_indices = np.argsort(values_clean)
    cumsum_weights = np.cumsum(weights_norm[sorted_indices])
    median_idx = np.searchsorted(cumsum_weights, 0.5)
    weighted_median = values_clean[sorted_indices[median_idx]]
    
    # Effective sample size
    effective_n = 1.0 / np.sum(weights_norm**2)
    
    return {
        'weighted_mean': weighted_mean,
        'weighted_std': weighted_std,
        'weighted_median': weighted_median,
        'effective_n': effective_n
    }


def ks_test_2sample(sample1: np.ndarray, sample2: np.ndarray) -> Dict:
    """
    Perform two-sample Kolmogorov-Smirnov test
    
    Parameters
    ----------
    sample1 : np.ndarray
        First sample
    sample2 : np.ndarray
        Second sample
        
    Returns
    -------
    dict
        KS test results
    """
    # Clean samples
    sample1_clean = sample1[np.isfinite(sample1)]
    sample2_clean = sample2[np.isfinite(sample2)]
    
    if len(sample1_clean) == 0 or len(sample2_clean) == 0:
        return {
            'ks_statistic': np.nan,
            'p_value': np.nan,
            'reject_null': False
        }
    
    # Perform KS test
    ks_stat, p_value = stats.ks_2samp(sample1_clean, sample2_clean)
    
    return {
        'ks_statistic': ks_stat,
        'p_value': p_value,
        'reject_null': p_value < 0.05
    }


def anderson_darling_test(sample: np.ndarray, dist: str = 'norm') -> Dict:
    """
    Perform Anderson-Darling test for normality
    
    Parameters
    ----------
    sample : np.ndarray
        Data sample
    dist : str
        Distribution to test against ('norm', 'expon', 'logistic', 'gumbel')
        
    Returns
    -------
    dict
        AD test results
    """
    sample_clean = sample[np.isfinite(sample)]
    
    if len(sample_clean) < 8:
        return {
            'ad_statistic': np.nan,
            'critical_values': [],
            'significance_levels': [],
            'reject_normality': False
        }
    
    # Perform AD test
    result = stats.anderson(sample_clean, dist=dist)
    
    # Check if we reject normality at 5% level
    reject = result.statistic > result.critical_values[2]  # 5% level
    
    return {
        'ad_statistic': result.statistic,
        'critical_values': result.critical_values,
        'significance_levels': result.significance_level,
        'reject_normality': reject
    }


def bimodality_coefficient(data: np.ndarray) -> float:
    """
    Calculate bimodality coefficient (BC)
    BC = (skewness^2 + 1) / (kurtosis + 3)
    BC > 5/9 suggests bimodality
    
    Parameters
    ----------
    data : np.ndarray
        Input data
        
    Returns
    -------
    float
        Bimodality coefficient
    """
    data_clean = data[np.isfinite(data)]
    
    if len(data_clean) < 4:
        return np.nan
    
    skewness = stats.skew(data_clean)
    kurtosis = stats.kurtosis(data_clean, fisher=False)  # Pearson's definition
    
    bc = (skewness**2 + 1) / (kurtosis + 3)
    
    return bc


def outlier_detection(data: np.ndarray, method: str = 'iqr', 
                     threshold: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect outliers in data
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    method : str
        Method for outlier detection ('iqr', 'zscore', 'mad')
    threshold : float
        Threshold for outlier detection
        
    Returns
    -------
    outliers : np.ndarray
        Boolean mask for outliers
    bounds : tuple
        Lower and upper bounds for inliers
    """
    data_clean = data[np.isfinite(data)]
    
    if len(data_clean) == 0:
        return np.zeros(len(data), dtype=bool), (np.nan, np.nan)
    
    if method == 'iqr':
        # Interquartile range method
        q1 = np.percentile(data_clean, 25)
        q3 = np.percentile(data_clean, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
    elif method == 'zscore':
        # Z-score method
        mean = np.mean(data_clean)
        std = np.std(data_clean)
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        
    elif method == 'mad':
        # Median absolute deviation method
        median = np.median(data_clean)
        mad = np.median(np.abs(data_clean - median))
        mad_scaled = 1.4826 * mad  # Scale to match standard deviation
        lower_bound = median - threshold * mad_scaled
        upper_bound = median + threshold * mad_scaled
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create outlier mask
    outliers = (data < lower_bound) | (data > upper_bound)
    
    return outliers, (lower_bound, upper_bound)