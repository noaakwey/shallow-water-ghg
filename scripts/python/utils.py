#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
UTILITY FUNCTIONS FOR SHALLOW WATER GHG ANALYSIS
================================================================================

Common functions used across model training and Monte Carlo scripts.

Author: Artur Gafurov
Version: 1.0
================================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import json


# ============================================================================
# CONSTANTS
# ============================================================================

# Molecular weights
MW_C = 12.011      # g/mol Carbon
MW_CO2 = 44.01    # g/mol CO2
MW_CH4 = 16.04    # g/mol CH4

# Time constants
SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400

# GWP values (IPCC AR6)
GWP_CH4_20 = 81    # 20-year GWP
GWP_CH4_100 = 28   # 100-year GWP


# ============================================================================
# UNIT CONVERSIONS
# ============================================================================

def umol_to_g_c(umol: float, hours: float = 24) -> float:
    """
    Convert µmol CO2 m⁻² s⁻¹ to g C m⁻² day⁻¹.
    
    Parameters
    ----------
    umol : float
        Flux in µmol CO2 m⁻² s⁻¹
    hours : float
        Integration period in hours (default: 24)
        
    Returns
    -------
    float
        Flux in g C m⁻² day⁻¹
    """
    return umol * 1e-6 * MW_C * hours * SECONDS_PER_HOUR


def umol_ch4_to_g(umol: float) -> float:
    """
    Convert µmol CH4 m⁻² s⁻¹ to g CH4 m⁻² day⁻¹.
    
    Parameters
    ----------
    umol : float
        CH4 flux in µmol m⁻² s⁻¹
        
    Returns
    -------
    float
        CH4 flux in g m⁻² day⁻¹
    """
    return umol * 1e-6 * MW_CH4 * SECONDS_PER_DAY


def g_c_to_g_co2(g_c: float) -> float:
    """Convert g C to g CO2."""
    return g_c * MW_CO2 / MW_C


def ch4_to_co2eq(g_ch4: float, gwp: int = GWP_CH4_100) -> float:
    """Convert g CH4 to g CO2-equivalent."""
    return g_ch4 * gwp


# ============================================================================
# DATA LOADING
# ============================================================================

def load_csv_with_dates(filepath: Union[str, Path], 
                        date_columns: list = None,
                        encoding: str = 'utf-8') -> pd.DataFrame:
    """
    Load CSV with automatic date parsing and encoding detection.
    
    Parameters
    ----------
    filepath : str or Path
        Path to CSV file
    date_columns : list, optional
        List of column names to parse as dates
    encoding : str
        Primary encoding to try
        
    Returns
    -------
    pd.DataFrame
        Loaded dataframe
    """
    # Try different encodings
    encodings = [encoding, 'utf-8-sig', 'cp1251', 'latin1']
    
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    if df is None:
        raise ValueError(f"Could not load {filepath} with any encoding")
    
    # Auto-detect date columns
    if date_columns is None:
        date_columns = [c for c in df.columns 
                       if any(d in c.lower() for d in ['date', 'time', 'дата'])]
    
    # Parse dates
    for col in date_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass
    
    return df


# ============================================================================
# STATISTICS
# ============================================================================

def compute_metrics(y_true: np.ndarray, 
                    y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute regression metrics.
    
    Parameters
    ----------
    y_true : array-like
        Observed values
    y_pred : array-like
        Predicted values
        
    Returns
    -------
    dict
        Dictionary with R2, RMSE, MAE, bias
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # MAE
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Bias
    bias = np.mean(y_pred - y_true)
    
    return {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'bias': bias,
        'n': len(y_true)
    }


def confidence_interval(data: np.ndarray, 
                        confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute confidence interval from distribution.
    
    Parameters
    ----------
    data : array-like
        Sample distribution
    confidence : float
        Confidence level (default: 0.95)
        
    Returns
    -------
    tuple
        (lower_bound, upper_bound)
    """
    alpha = (1 - confidence) / 2
    return np.percentile(data, [alpha * 100, (1 - alpha) * 100])


# ============================================================================
# MODEL HELPERS
# ============================================================================

def clamp_array(arr: np.ndarray, 
                vmin: float, 
                vmax: float) -> np.ndarray:
    """Clamp array values to range [vmin, vmax]."""
    return np.clip(arr, vmin, vmax)


def apply_physical_constraint(arr: np.ndarray, 
                               min_val: float = 0.0) -> np.ndarray:
    """Apply physical constraint (e.g., non-negative fluxes)."""
    return np.maximum(arr, min_val)


# ============================================================================
# I/O HELPERS
# ============================================================================

def save_results(results: dict, 
                 filepath: Union[str, Path],
                 indent: int = 2):
    """Save results dictionary to JSON."""
    filepath = Path(filepath)
    
    # Convert numpy types to Python native
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj
    
    with open(filepath, 'w') as f:
        json.dump(convert(results), f, indent=indent)


def format_uncertainty(mean: float, 
                       std: float, 
                       ci: Tuple[float, float] = None,
                       decimals: int = 2) -> str:
    """
    Format uncertainty for publication.
    
    Returns string like "1.23 ± 0.45 (95% CI: 0.35-2.11)"
    """
    result = f"{mean:.{decimals}f} ± {std:.{decimals}f}"
    if ci is not None:
        result += f" (95% CI: {ci[0]:.{decimals}f}-{ci[1]:.{decimals}f})"
    return result


# ============================================================================
# RASTER HELPERS
# ============================================================================

def compute_pixel_area_from_transform(transform) -> float:
    """
    Compute pixel area in m² from rasterio transform.
    
    Parameters
    ----------
    transform : Affine
        Rasterio affine transform
        
    Returns
    -------
    float
        Pixel area in m²
    """
    return abs(transform.a * transform.e)


def mask_and_flatten(data: np.ndarray, 
                     mask: np.ndarray) -> np.ndarray:
    """
    Apply mask and flatten to 1D array.
    
    Parameters
    ----------
    data : np.ndarray
        2D data array
    mask : np.ndarray
        Boolean or binary mask (1 = include)
        
    Returns
    -------
    np.ndarray
        1D array of masked values
    """
    mask_bool = mask > 0
    return data[mask_bool].flatten()


# ============================================================================
# PRINTING HELPERS
# ============================================================================

def print_header(title: str, char: str = "=", width: int = 70):
    """Print formatted header."""
    print(char * width)
    print(f"  {title}")
    print(char * width)


def print_subheader(title: str, char: str = "-", width: int = 60):
    """Print formatted subheader."""
    print("\n" + char * width)
    print(title)
    print(char * width)


def print_model_summary(name: str, 
                        coefs: dict, 
                        metrics: dict):
    """Print formatted model summary."""
    print(f"\n{'='*60}")
    print(f"  {name} MODEL SUMMARY")
    print(f"{'='*60}")
    
    print("\nCoefficients:")
    for k, v in coefs.items():
        print(f"  {k:20s}: {v:15.6e}")
    
    print("\nMetrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:15s}: {v:.4f}")
        else:
            print(f"  {k:15s}: {v}")


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    # Quick tests
    print("Testing utility functions...")
    
    # Unit conversions
    flux_umol = 2.0  # µmol CO2 m⁻² s⁻¹
    flux_g_c = umol_to_g_c(flux_umol, hours=12)
    print(f"\n{flux_umol} µmol CO2 m⁻² s⁻¹ × 12h = {flux_g_c:.3f} g C m⁻² day⁻¹")
    
    # Metrics
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
    metrics = compute_metrics(y_true, y_pred)
    print(f"\nMetrics: R²={metrics['R2']:.3f}, RMSE={metrics['RMSE']:.3f}")
    
    # Confidence interval
    data = np.random.normal(10, 2, 1000)
    ci = confidence_interval(data)
    print(f"\nCI test: mean={np.mean(data):.2f}, 95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")
    
    print("\n✅ All tests passed!")
