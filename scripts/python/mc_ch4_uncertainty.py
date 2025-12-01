#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
CH₄ MONTE CARLO UNCERTAINTY ANALYSIS
================================================================================

Propagate uncertainties through the CH₄ flux calculation using Monte Carlo.

Uncertainty sources:
    1. Model RMSE = 0.026 µmol m⁻² s⁻¹ (per-pixel noise)
    2. F_CH4 = 0.876 ± 0.05 (diurnal correction)
    3. Ice-free days = 200 ± 10

Workflow:
    CH4_daily = CH4_moment × F_CH4
    CH4_annual = CH4_daily × UMOL_TO_G_DAY × ice_free_days
    Total = Σ(CH4_annual × pixel_area)

Usage:
    python mc_ch4_uncertainty.py \
        --ch4-raster CH4_Moment_Mean_umol_s.tif \
        --mask Shallow_Water_Mask.tif \
        --n-iter 2000

Author: [Your Name]
Version: 1.0
================================================================================
"""

import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from joblib import Parallel, delayed
from tqdm import tqdm
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS
# ============================================================================

MW_CH4 = 16.04  # Molar mass of CH₄ (g/mol)
SECONDS_PER_DAY = 86400
UMOL_TO_G_DAY = 1e-6 * MW_CH4 * SECONDS_PER_DAY  # = 1.3858

GWP_CH4_100 = 28  # Global Warming Potential (100 year)


# ============================================================================
# SINGLE MC ITERATION
# ============================================================================

def mc_iteration(iter_idx: int,
                 ch4_moment_flat: np.ndarray,
                 pixel_area_m2: float,
                 rmse: float,
                 f_ch4_mean: float,
                 f_ch4_sd: float,
                 ice_days_mean: float,
                 ice_days_sd: float,
                 seed: int) -> tuple:
    """
    Single Monte Carlo iteration for CH₄.
    
    Returns
    -------
    tuple
        (total_t, mean_annual_g_m2, f_ch4_used, ice_days_used)
    """
    rng = np.random.default_rng(seed + iter_idx)
    
    # Sample F_CH4
    f_ch4 = np.clip(rng.normal(f_ch4_mean, f_ch4_sd), 0.5, 1.2)
    
    # Sample ice-free days
    ice_days = int(np.clip(rng.normal(ice_days_mean, ice_days_sd), 170, 230))
    
    # Add per-pixel noise from model RMSE
    noise = rng.normal(0, rmse, len(ch4_moment_flat))
    ch4_noisy = np.maximum(ch4_moment_flat + noise, 0)
    
    # Calculate fluxes
    # CH4_daily = CH4_moment × F_CH4
    ch4_daily = ch4_noisy * f_ch4  # µmol m⁻² s⁻¹
    
    # CH4_annual = CH4_daily × conversion × days
    ch4_annual_g = ch4_daily * UMOL_TO_G_DAY * ice_days  # g m⁻² yr⁻¹
    
    # Total emission
    total_g = np.sum(ch4_annual_g) * pixel_area_m2
    total_t = total_g / 1e6  # g → tonnes
    
    mean_annual = np.mean(ch4_annual_g)
    
    return total_t, mean_annual, f_ch4, ice_days


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CH₄ Monte Carlo Uncertainty Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python mc_ch4_uncertainty.py \\
        --ch4-raster CH4_Moment_Mean_umol_s.tif \\
        --mask Shallow_Water_Mask.tif \\
        --n-iter 2000
        """
    )
    
    # Input files
    parser.add_argument('--ch4-raster', required=True,
                        help='Path to CH₄ moment raster (µmol m⁻² s⁻¹)')
    parser.add_argument('--mask', required=True,
                        help='Path to shallow water mask (1=shallow, 0=other)')
    
    # Output
    parser.add_argument('--output-prefix', default='ch4_mc',
                        help='Prefix for output files (default: ch4_mc)')
    
    # Model parameters
    parser.add_argument('--rmse', type=float, default=0.026,
                        help='Model RMSE (µmol m⁻² s⁻¹, default: 0.026)')
    parser.add_argument('--f-ch4-mean', type=float, default=0.876,
                        help='F_CH4 mean (moment→daily, default: 0.876)')
    parser.add_argument('--f-ch4-sd', type=float, default=0.05,
                        help='F_CH4 std dev (default: 0.05)')
    parser.add_argument('--ice-days-mean', type=float, default=200,
                        help='Ice-free days mean (default: 200)')
    parser.add_argument('--ice-days-sd', type=float, default=10,
                        help='Ice-free days std dev (default: 10)')
    
    # MC parameters
    parser.add_argument('--n-iter', type=int, default=2000,
                        help='Number of MC iterations (default: 2000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Parallel jobs, -1 for all cores (default: -1)')
    
    args = parser.parse_args()
    
    # =========================================================================
    # HEADER
    # =========================================================================
    
    print("=" * 70)
    print("  CH₄ MONTE CARLO UNCERTAINTY ANALYSIS")
    print("=" * 70)
    
    print(f"\nInput files:")
    print(f"  CH₄ raster: {args.ch4_raster}")
    print(f"  Mask:       {args.mask}")
    
    print(f"\nModel parameters:")
    print(f"  RMSE:       {args.rmse} µmol m⁻² s⁻¹")
    print(f"  F_CH4:      {args.f_ch4_mean} ± {args.f_ch4_sd}")
    print(f"  Ice days:   {args.ice_days_mean} ± {args.ice_days_sd}")
    
    print(f"\nMC parameters:")
    print(f"  Iterations: {args.n_iter}")
    print(f"  Seed:       {args.seed}")
    
    # =========================================================================
    # LOAD RASTERS
    # =========================================================================
    
    print("\n" + "-" * 70)
    print("Loading rasters...")
    
    with rasterio.open(args.ch4_raster) as src:
        ch4_data = src.read(1)
        ch4_transform = src.transform
        ch4_crs = src.crs
        ch4_shape = ch4_data.shape
        pixel_res = abs(ch4_transform[0])
        pixel_area_m2 = pixel_res ** 2
        
    print(f"  CH₄ shape: {ch4_data.shape}")
    print(f"  Pixel size: {pixel_res:.1f} m")
    print(f"  Pixel area: {pixel_area_m2:.0f} m²")
    
    with rasterio.open(args.mask) as src_mask:
        mask_native = src_mask.read(1)
        mask_transform = src_mask.transform
        mask_res = abs(mask_transform[0])
        
    print(f"  Mask shape: {mask_native.shape}")
    print(f"  Mask pixel: {mask_res:.1f} m")
    
    # Resample mask if needed
    if mask_native.shape != ch4_shape:
        print(f"\n  ⚠ Shape mismatch! Resampling mask to CH₄ grid...")
        
        mask_data = np.zeros(ch4_shape, dtype=mask_native.dtype)
        
        with rasterio.open(args.mask) as src_mask:
            reproject(
                source=rasterio.band(src_mask, 1),
                destination=mask_data,
                src_transform=mask_transform,
                src_crs=src_mask.crs,
                dst_transform=ch4_transform,
                dst_crs=ch4_crs,
                resampling=Resampling.nearest
            )
        
        print(f"  Resampled mask shape: {mask_data.shape}")
    else:
        mask_data = mask_native
    
    # =========================================================================
    # APPLY MASK
    # =========================================================================
    
    valid = (mask_data == 1) & np.isfinite(ch4_data) & (ch4_data > 0)
    ch4_flat = ch4_data[valid]
    n_pixels = len(ch4_flat)
    
    total_area_km2 = n_pixels * pixel_area_m2 / 1e6
    
    print(f"\nShallow water statistics:")
    print(f"  Valid pixels:    {n_pixels:,}")
    print(f"  Total area:      {total_area_km2:.1f} km²")
    print(f"  CH₄ moment mean: {ch4_flat.mean():.5f} µmol m⁻² s⁻¹")
    print(f"  CH₄ moment std:  {ch4_flat.std():.5f}")
    print(f"  CH₄ moment range: [{ch4_flat.min():.5f}, {ch4_flat.max():.5f}]")
    
    # =========================================================================
    # DETERMINISTIC CALCULATION
    # =========================================================================
    
    print("\n" + "-" * 70)
    print("Deterministic calculation (no uncertainty):")
    
    ch4_daily_det = ch4_flat * args.f_ch4_mean
    ch4_annual_det = ch4_daily_det * UMOL_TO_G_DAY * args.ice_days_mean
    total_det_t = np.sum(ch4_annual_det) * pixel_area_m2 / 1e6
    
    print(f"  CH₄ daily mean:  {ch4_daily_det.mean():.5f} µmol m⁻² s⁻¹")
    print(f"  CH₄ annual mean: {ch4_annual_det.mean():.2f} g m⁻² yr⁻¹")
    print(f"  Total:           {total_det_t:.0f} t CH₄ yr⁻¹")
    print(f"  CO₂-eq:          {total_det_t * GWP_CH4_100 / 1000:.1f} kt yr⁻¹")
    
    # =========================================================================
    # MONTE CARLO
    # =========================================================================
    
    print("\n" + "-" * 70)
    print(f"Running Monte Carlo ({args.n_iter} iterations)...")
    
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(mc_iteration)(
            i, ch4_flat, pixel_area_m2,
            args.rmse, args.f_ch4_mean, args.f_ch4_sd,
            args.ice_days_mean, args.ice_days_sd, args.seed
        )
        for i in tqdm(range(args.n_iter), desc="MC iterations")
    )
    
    results = np.array(results)
    total_t = results[:, 0]
    annual_g = results[:, 1]
    f_ch4_used = results[:, 2]
    ice_days_used = results[:, 3]
    
    # CO₂-equivalent
    co2eq_kt = total_t * GWP_CH4_100 / 1000
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("MONTE CARLO RESULTS")
    print("=" * 70)
    
    def print_stats(name, data, unit):
        mean = np.mean(data)
        std = np.std(data)
        ci = np.percentile(data, [2.5, 97.5])
        print(f"\n{name}:")
        print(f"  Mean ± SD:  {mean:.2f} ± {std:.2f} {unit}")
        print(f"  95% CI:     [{ci[0]:.2f}, {ci[1]:.2f}] {unit}")
        return mean, std, ci
    
    ann_mean, ann_std, ann_ci = print_stats(
        "Annual flux (per unit area)", annual_g, "g CH₄ m⁻² yr⁻¹")
    
    tot_mean, tot_std, tot_ci = print_stats(
        f"Total emission ({total_area_km2:.0f} km²)", total_t, "t CH₄ yr⁻¹")
    
    co2_mean, co2_std, co2_ci = print_stats(
        f"CO₂-equivalent (GWP₁₀₀={GWP_CH4_100})", co2eq_kt, "kt CO₂-eq yr⁻¹")
    
    print(f"\nParameter distributions (actual used):")
    print(f"  F_CH4:     {f_ch4_used.mean():.4f} ± {f_ch4_used.std():.4f}")
    print(f"  Ice days:  {ice_days_used.mean():.1f} ± {ice_days_used.std():.1f}")
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    
    # Full MC results
    mc_df = pd.DataFrame({
        'iteration': range(args.n_iter),
        'total_t_CH4': total_t,
        'annual_g_m2': annual_g,
        'co2eq_kt': co2eq_kt,
        'f_ch4': f_ch4_used,
        'ice_days': ice_days_used
    })
    mc_csv = f"{args.output_prefix}_iterations.csv"
    mc_df.to_csv(mc_csv, index=False)
    print(f"\nSaved: {mc_csv}")
    
    # Summary statistics
    summary = {
        'parameter': [
            'annual_mean_g_m2_yr', 'annual_sd', 'annual_ci_lo', 'annual_ci_hi',
            'total_mean_t_yr', 'total_sd', 'total_ci_lo', 'total_ci_hi',
            'co2eq_mean_kt_yr', 'co2eq_sd', 'co2eq_ci_lo', 'co2eq_ci_hi',
            'area_km2', 'n_pixels', 'n_iterations',
            'rmse', 'f_ch4_mean', 'f_ch4_sd', 'ice_days_mean', 'ice_days_sd'
        ],
        'value': [
            ann_mean, ann_std, ann_ci[0], ann_ci[1],
            tot_mean, tot_std, tot_ci[0], tot_ci[1],
            co2_mean, co2_std, co2_ci[0], co2_ci[1],
            total_area_km2, n_pixels, args.n_iter,
            args.rmse, args.f_ch4_mean, args.f_ch4_sd,
            args.ice_days_mean, args.ice_days_sd
        ]
    }
    summary_df = pd.DataFrame(summary)
    summary_csv = f"{args.output_prefix}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")
    
    # =========================================================================
    # PLOT
    # =========================================================================
    
    sns.set_style("ticks")
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    datasets = [
        (annual_g, r'Annual flux (g CH$_4$ m$^{-2}$ yr$^{-1}$)',
         '(a) Annual Flux', '#2ca02c', ann_mean, ann_ci),
        (total_t, r'Total emission (t CH$_4$ yr$^{-1}$)',
         '(b) Total Emission', '#ff7f0e', tot_mean, tot_ci),
        (co2eq_kt, r'CO$_2$-eq (kt CO$_2$-eq yr$^{-1}$)',
         '(c) Climate Impact', '#d62728', co2_mean, co2_ci)
    ]
    
    for ax, (data, xlabel, title, color, mean, ci) in zip(axes, datasets):
        sns.histplot(data, kde=True, color=color, ax=ax, stat="density",
                     alpha=0.4, edgecolor=None, bins=50)
        
        ax.axvline(mean, color=color, linestyle='--', linewidth=2, label='Mean')
        ax.axvline(ci[0], color='gray', linestyle=':', linewidth=1.5)
        ax.axvline(ci[1], color='gray', linestyle=':', linewidth=1.5)
        
        ylim = ax.get_ylim()
        ax.fill_betweenx(ylim, ci[0], ci[1], alpha=0.1, color='gray')
        ax.set_ylim(ylim)
        
        ax.set_title(title, fontweight='bold', loc='left', fontsize=12)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('Density' if ax == axes[0] else '', fontsize=11)
        
        stats_text = (f"Mean: {mean:.1f}\nSD: {np.std(data):.1f}\n"
                      f"95% CI: [{ci[0]:.1f}, {ci[1]:.1f}]")
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                fontsize=10, bbox=dict(boxstyle="round,pad=0.4",
                                       fc="white", ec="gray", alpha=0.9))
    
    plt.suptitle(
        f'CH$_4$ Monte Carlo Analysis (n={args.n_iter}, area={total_area_km2:.0f} km²)',
        fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plot_file = f"{args.output_prefix}_distributions.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_file}")
    plt.close()
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    
    print("\n" + "=" * 70)
    print(f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║              CH₄ ANNUAL BUDGET - MONTE CARLO RESULTS                       ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  ANNUAL FLUX:                                                              ║
║    {ann_mean:.2f} ± {ann_std:.2f} g CH₄ m⁻² yr⁻¹                                       ║
║    95% CI: [{ann_ci[0]:.2f}, {ann_ci[1]:.2f}]                                             ║
║                                                                            ║
║  TOTAL EMISSION ({total_area_km2:.0f} km²):                                            ║
║    {tot_mean:.0f} ± {tot_std:.0f} t CH₄ yr⁻¹                                            ║
║    95% CI: [{tot_ci[0]:.0f}, {tot_ci[1]:.0f}]                                               ║
║                                                                            ║
║  CO₂-EQUIVALENT (GWP₁₀₀ = {GWP_CH4_100}):                                            ║
║    {co2_mean:.1f} ± {co2_std:.1f} kt CO₂-eq yr⁻¹                                       ║
║    95% CI: [{co2_ci[0]:.1f}, {co2_ci[1]:.1f}]                                             ║
║                                                                            ║
║  Parameters:                                                               ║
║    RMSE = {args.rmse} µmol m⁻² s⁻¹                                              ║
║    F_CH4 = {args.f_ch4_mean} ± {args.f_ch4_sd}                                             ║
║    Ice-free = {args.ice_days_mean:.0f} ± {args.ice_days_sd:.0f} days                                        ║
║                                                                            ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")
    print("=" * 70)


if __name__ == '__main__':
    main()
