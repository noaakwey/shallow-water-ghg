#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
NEE MONTE CARLO UNCERTAINTY ANALYSIS
================================================================================

Propagate measurement and model uncertainties through the carbon flux 
calculation pipeline using Monte Carlo simulation.

Uncertainty sources:
    1. Model RMSE (GPP, Reco)
    2. Diurnal correction factors (F_GPP, F_Reco)
    3. Ice-free period duration
    4. Daylight hours estimation

Usage:
    python mc_nee_uncertainty.py \
        --fluxes /path/to/Instantaneous_Fluxes_11am_umol_s.vrt \
        --mask /path/to/Shallow_Water_Mask.tif \
        --n-iter 2000 \
        --output-plot Figure5_MC_Results.png

Author: [Your Name]
Version: 1.0
================================================================================
"""

import argparse
import multiprocessing
import warnings

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import seaborn as sns
from joblib import Parallel, delayed
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS
# ============================================================================

MOL_TO_G_C = 12.011    # Molar mass of carbon
MOL_TO_G_CO2 = 44.01   # Molar mass of CO2


# ============================================================================
# DATA LOADING
# ============================================================================

def resolve_band(src, spec, default_index: int) -> int:
    """Resolve band specification to index."""
    if isinstance(spec, int):
        return spec
    if isinstance(spec, str):
        s = spec.strip()
        if s.isdigit():
            return int(s)
        for i, d in enumerate(src.descriptions, start=1):
            if d == spec:
                return i
        return default_index
    return default_index


def load_fluxes(fluxes_path: str, reco_spec, gpp_spec) -> tuple:
    """
    Load flux raster with Reco and GPP bands.
    
    Parameters
    ----------
    fluxes_path : str
        Path to flux raster (GeoTIFF or VRT)
    reco_spec : int or str
        Band index or name for Reco
    gpp_spec : int or str
        Band index or name for GPP
        
    Returns
    -------
    tuple
        (reco_array, gpp_array, profile, transform)
    """
    with rasterio.open(fluxes_path) as src:
        reco_idx = resolve_band(src, reco_spec, default_index=1)
        gpp_idx = resolve_band(src, gpp_spec, default_index=2)
        
        reco = src.read(reco_idx).astype(np.float32)
        gpp = src.read(gpp_idx).astype(np.float32)
        
        return reco, gpp, src.profile, src.transform


def load_mask(mask_path: str, mask_spec=None) -> tuple:
    """Load shallow water mask raster."""
    with rasterio.open(mask_path) as src:
        idx = resolve_band(src, mask_spec, default_index=1) if mask_spec else 1
        mask = src.read(idx).astype(np.float32)
        return mask, src.profile, src.transform


def compute_pixel_area(transform) -> float:
    """Compute pixel area in m² from transform."""
    return abs(transform.a * transform.e)


# ============================================================================
# MONTE CARLO SIMULATION
# ============================================================================

def single_mc_iteration(iter_idx: int,
                        reco_masked: np.ndarray,
                        gpp_masked: np.ndarray,
                        n_pixels: int,
                        pixel_area: float,
                        rmse_gpp: float,
                        rmse_reco: float,
                        fgpp_mean: float,
                        fgpp_sd: float,
                        freco_mean: float,
                        freco_sd: float,
                        ice_days_mean: float,
                        ice_days_sd: float,
                        daylight_hours_mean: float,
                        daylight_hours_sd: float,
                        fullday_hours_mean: float,
                        fullday_hours_sd: float,
                        per_pixel_noise: bool,
                        base_seed: int) -> tuple:
    """
    Single Monte Carlo iteration.
    
    Returns
    -------
    tuple
        (gpp_total, reco_total, nee_total) in mol C
    """
    rng = np.random.default_rng(base_seed + iter_idx)
    umol_to_mol = 1e-6

    # Sample diurnal correction factors
    fgpp_i = np.clip(rng.normal(fgpp_mean, fgpp_sd), 
                     0.5 * fgpp_mean, 1.5 * fgpp_mean)
    freco_i = np.clip(rng.normal(freco_mean, freco_sd), 
                      0.5 * freco_mean, 1.5 * freco_mean)
    
    # Sample ice-free period and daylight hours
    ice_days_i = np.clip(rng.normal(ice_days_mean, ice_days_sd), 150, 250)
    daylight_i = np.clip(rng.normal(daylight_hours_mean, daylight_hours_sd), 8, 16)
    full_i = 24.0

    # Conversion factors
    gpp_factor = umol_to_mol * fgpp_i * daylight_i * 3600.0 * ice_days_i
    reco_factor = umol_to_mol * freco_i * full_i * 3600.0 * ice_days_i

    # Add model noise
    if per_pixel_noise:
        gpp_noisy = np.maximum(
            gpp_masked + rng.normal(0, rmse_gpp, n_pixels), 0)
        reco_noisy = np.maximum(
            reco_masked + rng.normal(0, rmse_reco, n_pixels), 0)
    else:
        gpp_noisy = np.maximum(gpp_masked + rng.normal(0, rmse_gpp), 0)
        reco_noisy = np.maximum(reco_masked + rng.normal(0, rmse_reco), 0)

    # Calculate totals
    gpp_total = np.sum(gpp_noisy) * gpp_factor * pixel_area
    reco_total = np.sum(reco_noisy) * reco_factor * pixel_area
    
    return gpp_total, reco_total, (reco_total - gpp_total)


def monte_carlo_nee(reco_11am: np.ndarray,
                    gpp_11am: np.ndarray,
                    mask: np.ndarray,
                    pixel_area: float,
                    n_iter: int = 1000,
                    **kwargs) -> tuple:
    """
    Run Monte Carlo simulation for NEE uncertainty.
    
    Parameters
    ----------
    reco_11am : np.ndarray
        Reco flux at 11:00 AM (µmol m⁻² s⁻¹)
    gpp_11am : np.ndarray
        GPP at 11:00 AM (µmol m⁻² s⁻¹)
    mask : np.ndarray
        Shallow water mask (1 = include)
    pixel_area : float
        Pixel area in m²
    n_iter : int
        Number of MC iterations
    **kwargs
        Model and uncertainty parameters
        
    Returns
    -------
    tuple
        (gpp_array, reco_array, nee_array, total_area)
    """
    mask_bool = mask > 0
    if not np.any(mask_bool):
        raise ValueError("Empty mask - no valid pixels")
    
    total_area = np.count_nonzero(mask_bool) * pixel_area
    
    # Extract masked pixels (clamp to non-negative)
    reco_masked = np.maximum(reco_11am[mask_bool], 0.0).astype(np.float32)
    gpp_masked = np.maximum(gpp_11am[mask_bool], 0.0).astype(np.float32)
    
    # Clean up memory
    del reco_11am, gpp_11am, mask
    
    # Configure parallel execution
    n_jobs = kwargs.pop('n_jobs', -1)
    if n_jobs == -1:
        n_jobs = min(multiprocessing.cpu_count(), 64)
    
    print(f"Monte Carlo simulation ({n_iter} iterations, {n_jobs} threads)...")
    print(f"Valid pixels: {len(reco_masked):,}")
    print(f"Total area: {total_area/1e6:.1f} km²")
    
    # Run parallel MC
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(single_mc_iteration)(
            i, reco_masked, gpp_masked, len(reco_masked), pixel_area, **kwargs
        )
        for i in tqdm(range(n_iter), desc="MC iterations")
    )
    
    res_arr = np.array(results)
    return res_arr[:, 0], res_arr[:, 1], res_arr[:, 2], total_area


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_distributions(gpp: np.ndarray,
                       reco: np.ndarray,
                       nee: np.ndarray,
                       total_area_m2: float,
                       output_path: str):
    """
    Generate publication-quality distribution plots (Figure 5).
    
    Parameters
    ----------
    gpp, reco, nee : np.ndarray
        MC simulation results (mol C)
    total_area_m2 : float
        Total shallow water area (m²)
    output_path : str
        Output file path
    """
    # Convert to g C m⁻² yr⁻¹
    conv = MOL_TO_G_C / total_area_m2
    gpp_vals = gpp * conv
    reco_vals = reco * conv
    nee_vals = nee * conv
    
    # Configure style
    sns.set_style("ticks")
    sns.set_context("paper", font_scale=1.4)
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    colors = ['#2ca02c', '#d62728', '#1f77b4']
    labels = ['(a) GPP', '(b) Reco', '(c) NEE']
    data = [gpp_vals, reco_vals, nee_vals]
    
    for ax, d, col, lab in zip(axes, data, colors, labels):
        # Histogram with KDE
        sns.histplot(d, kde=True, color=col, ax=ax, 
                     stat="density", alpha=0.4, edgecolor=None)
        
        # Mean and CI
        mean = np.mean(d)
        ci = np.percentile(d, [2.5, 97.5])
        
        ax.axvline(mean, color=col, linestyle='--', linewidth=2)
        ax.axvline(ci[0], color='gray', linestyle=':', linewidth=1)
        ax.axvline(ci[1], color='gray', linestyle=':', linewidth=1)
        
        ax.set_title(lab, fontweight='bold', loc='left')
        ax.set_xlabel(r'Flux (g C m$^{-2}$ yr$^{-1}$)')
        ax.set_ylabel('Density' if ax == axes[0] else '')
        
        # Statistics box
        stats_text = f"Mean: {mean:.1f}\n95% CI: [{ci[0]:.1f}, {ci[1]:.1f}]"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", 
                          ec="gray", alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: {output_path}")


def print_summary(gpp: np.ndarray,
                  reco: np.ndarray,
                  nee: np.ndarray,
                  total_area_m2: float):
    """Print summary statistics."""
    
    def format_stats(name: str, arr: np.ndarray):
        vals = arr * MOL_TO_G_C / total_area_m2
        mean = np.mean(vals)
        std = np.std(vals)
        ci = np.percentile(vals, [2.5, 97.5])
        print(f"{name}: {mean:.1f} ± {std:.1f} g C m⁻² yr⁻¹, "
              f"95% CI: [{ci[0]:.1f}, {ci[1]:.1f}]")
    
    print("\n" + "=" * 70)
    print("MONTE CARLO RESULTS")
    print("=" * 70)
    
    format_stats("GPP ", gpp)
    format_stats("Reco", reco)
    format_stats("NEE ", nee)
    
    # Total emissions
    area_km2 = total_area_m2 / 1e6
    nee_total_mean = np.mean(nee) * MOL_TO_G_C / 1e9  # Gg C
    nee_total_std = np.std(nee) * MOL_TO_G_C / 1e9
    
    print(f"\nTotal area: {area_km2:.1f} km²")
    print(f"Total NEE: {nee_total_mean:.2f} ± {nee_total_std:.2f} Gg C yr⁻¹")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='NEE Monte Carlo Uncertainty Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python mc_nee_uncertainty.py \\
        --fluxes Instantaneous_Fluxes_11am_umol_s.vrt \\
        --mask Shallow_Water_Mask.tif \\
        --n-iter 2000 \\
        --output-plot Figure5_MC_Results.png
        """
    )
    
    # Input files
    parser.add_argument("--fluxes", required=True,
                        help="Path to flux raster (Reco + GPP bands)")
    parser.add_argument("--mask", required=True,
                        help="Path to shallow water mask")
    parser.add_argument("--output-plot", default="mc_nee_results.png",
                        help="Output plot filename")
    
    # Band specification
    parser.add_argument("--reco-band", default=1,
                        help="Reco band index or name (default: 1)")
    parser.add_argument("--gpp-band", default=2,
                        help="GPP band index or name (default: 2)")
    parser.add_argument("--mask-band", default=1,
                        help="Mask band index (default: 1)")
    
    # MC parameters
    parser.add_argument("--n-iter", type=int, default=2000,
                        help="Number of MC iterations (default: 2000)")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Parallel jobs, -1 for all cores (default: -1)")
    
    # Model uncertainties
    parser.add_argument("--rmse-gpp", type=float, default=0.25,
                        help="GPP model RMSE (µmol m⁻² s⁻¹, default: 0.25)")
    parser.add_argument("--rmse-reco", type=float, default=0.28,
                        help="Reco model RMSE (µmol m⁻² s⁻¹, default: 0.28)")
    
    # Diurnal correction factors
    parser.add_argument("--fgpp-mean", type=float, default=0.904,
                        help="GPP diurnal factor mean (default: 0.904)")
    parser.add_argument("--fgpp-sd", type=float, default=0.03,
                        help="GPP diurnal factor std (default: 0.03)")
    parser.add_argument("--freco-mean", type=float, default=0.978,
                        help="Reco diurnal factor mean (default: 0.978)")
    parser.add_argument("--freco-sd", type=float, default=0.03,
                        help="Reco diurnal factor std (default: 0.03)")
    
    # Temporal parameters
    parser.add_argument("--ice-days-mean", type=float, default=200,
                        help="Ice-free days mean (default: 200)")
    parser.add_argument("--ice-days-sd", type=float, default=10,
                        help="Ice-free days std (default: 10)")
    parser.add_argument("--daylight-hours-mean", type=float, default=12,
                        help="Daylight hours mean (default: 12)")
    parser.add_argument("--daylight-hours-sd", type=float, default=1,
                        help="Daylight hours std (default: 1)")
    parser.add_argument("--fullday-hours-mean", type=float, default=24,
                        help="Full day hours (default: 24)")
    parser.add_argument("--fullday-hours-sd", type=float, default=0.5,
                        help="Full day hours std (default: 0.5)")
    
    args, _ = parser.parse_known_args()

    print("=" * 70)
    print("  NEE MONTE CARLO UNCERTAINTY ANALYSIS")
    print("=" * 70)
    
    # Load data
    print("\nLoading flux raster...")
    reco, gpp, _, transform = load_fluxes(
        args.fluxes, args.reco_band, args.gpp_band)
    
    print("Loading mask...")
    mask, _, _ = load_mask(args.mask, args.mask_band)
    
    pixel_area = compute_pixel_area(transform)
    print(f"Pixel area: {pixel_area:.1f} m²")
    
    # MC parameters
    params = {
        'rmse_gpp': args.rmse_gpp,
        'rmse_reco': args.rmse_reco,
        'fgpp_mean': args.fgpp_mean,
        'fgpp_sd': args.fgpp_sd,
        'freco_mean': args.freco_mean,
        'freco_sd': args.freco_sd,
        'ice_days_mean': args.ice_days_mean,
        'ice_days_sd': args.ice_days_sd,
        'daylight_hours_mean': args.daylight_hours_mean,
        'daylight_hours_sd': args.daylight_hours_sd,
        'fullday_hours_mean': args.fullday_hours_mean,
        'fullday_hours_sd': args.fullday_hours_sd,
        'per_pixel_noise': True,
        'base_seed': 42,
        'n_jobs': args.n_jobs
    }

    # Run MC simulation
    gpp_vals, reco_vals, nee_vals, area = monte_carlo_nee(
        reco, gpp, mask, pixel_area, args.n_iter, **params)
    
    # Output results
    plot_distributions(gpp_vals, reco_vals, nee_vals, area, args.output_plot)
    print_summary(gpp_vals, reco_vals, nee_vals, area)
    
    print("\n" + "=" * 70)
    print("  ✅ ANALYSIS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
