#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
CH₄ FLUX MODEL TRAINING PIPELINE
================================================================================

Train a quadratic regression model for methane (CH₄) emissions using 
Landsat Land Surface Temperature (LST).

Model:
    CH4_moment = γ₀ + γ₁×LST + γ₂×LST²

Physical basis:
    - Methanogenesis in anaerobic sediments is strongly temperature-dependent
    - Ebullition (bubble release) increases with sediment temperature
    - Quadratic form captures the accelerating response at high temperatures

Usage:
    python train_ch4_model.py \
        --ec-data /path/to/ch4_halfhourly.csv \
        --satellite-data /path/to/s2_landsat_predictors.csv \
        --output-dir ./output

Author: [Your Name]
Version: 1.0
================================================================================
"""

import argparse
import json
from pathlib import Path
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import LeaveOneOut

# ============================================================================
# CONFIGURATION
# ============================================================================

# Time window for matching EC and satellite observations
SATELLITE_TIME_WINDOW_MINUTES = 30

# Landsat overpass time (local, approximate)
LANDSAT_OVERPASS_HOUR = 10  # ~10:30 AM local
LANDSAT_OVERPASS_MINUTE = 30

# Quality control thresholds
USTAR_THRESHOLD = 0.05  # m/s minimum friction velocity


# ============================================================================
# DATA LOADING
# ============================================================================

def load_ch4_data(filepath: str, encoding: str = 'utf-8') -> pd.DataFrame:
    """
    Load half-hourly CH₄ eddy covariance data.
    
    Expected columns:
        - DateTime: timestamp
        - CH4 or CH4_f: gap-filled methane flux (µmol m⁻² s⁻¹)
        - Tair: air temperature (°C)
        - qc_CH4: quality flag (0=best, 1=good, 2=filled)
    """
    print("=" * 60)
    print("LOADING CH₄ EDDY COVARIANCE DATA")
    print("=" * 60)
    
    # Try different encodings
    for enc in [encoding, 'cp1251', 'utf-8-sig', 'latin1']:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            print(f"Loaded with encoding: {enc}")
            break
        except:
            continue
    else:
        raise ValueError(f"Could not load file: {filepath}")
    
    print(f"Records: {len(df)}, Columns: {len(df.columns)}")
    
    # Parse datetime
    date_col = None
    for col in ['DateTime', 'datetime', 'TIMESTAMP', 'Date']:
        if col in df.columns:
            date_col = col
            break
    
    if date_col:
        df['datetime'] = pd.to_datetime(df[date_col])
        print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # Identify CH4 column
    ch4_col = None
    for col in ['CH4', 'CH4_f', 'FCH4', 'ch4_flux']:
        if col in df.columns:
            ch4_col = col
            break
    
    if ch4_col:
        valid_ch4 = df[ch4_col].notna().sum()
        print(f"CH₄ column: {ch4_col}, valid values: {valid_ch4}")
        print(f"CH₄ range: [{df[ch4_col].min():.4f}, {df[ch4_col].max():.4f}] µmol m⁻² s⁻¹")
    
    return df


def load_satellite_predictors(filepath: str) -> pd.DataFrame:
    """
    Load satellite predictor data (S2 + Landsat LST).
    """
    print("\n" + "=" * 60)
    print("LOADING SATELLITE PREDICTORS")
    print("=" * 60)
    
    df = pd.read_csv(filepath)
    print(f"Scenes: {len(df)}, Columns: {len(df.columns)}")
    
    # Parse dates
    for col in ['date', 'datetime_local']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    # Check LST availability
    if 'has_landsat' in df.columns:
        landsat_count = df['has_landsat'].sum()
        print(f"Scenes with Landsat LST: {landsat_count} / {len(df)}")
    
    lst_cols = [c for c in df.columns if 'LST' in c]
    print(f"LST columns: {lst_cols}")
    
    return df


# ============================================================================
# TRAINING DATA PREPARATION
# ============================================================================

def create_ch4_training_data(ch4_df: pd.DataFrame, 
                              sat_df: pd.DataFrame,
                              time_window_min: int = 30) -> pd.DataFrame:
    """
    Create CH₄ training dataset by matching EC and satellite observations.
    
    For each Landsat scene, extract the corresponding CH₄ flux measurement
    from the eddy covariance data at the satellite overpass time.
    
    Parameters
    ----------
    ch4_df : pd.DataFrame
        Half-hourly CH₄ flux data
    sat_df : pd.DataFrame
        Satellite predictor data with LST
    time_window_min : int
        Time window for matching (minutes)
        
    Returns
    -------
    pd.DataFrame
        Training data with CH₄ and LST paired observations
    """
    print("\n" + "=" * 60)
    print("CREATING CH₄ TRAINING DATASET")
    print("=" * 60)
    
    # Filter satellite data with valid Landsat LST
    if 'has_landsat' in sat_df.columns:
        sat_with_lst = sat_df[sat_df['has_landsat'] == 1].copy()
    else:
        sat_with_lst = sat_df[sat_df['LST_mean'].notna()].copy()
    
    print(f"Satellite scenes with LST: {len(sat_with_lst)}")
    
    # Prepare datetime column in CH4 data
    if 'datetime' not in ch4_df.columns:
        for col in ['DateTime', 'TIMESTAMP']:
            if col in ch4_df.columns:
                ch4_df['datetime'] = pd.to_datetime(ch4_df[col])
                break
    
    # Identify CH4 column
    ch4_col = None
    for col in ['CH4', 'CH4_f', 'FCH4']:
        if col in ch4_df.columns:
            ch4_col = col
            break
    
    if ch4_col is None:
        raise ValueError("CH4 column not found in EC data")
    
    # Match observations
    training_records = []
    
    for _, sat_row in sat_with_lst.iterrows():
        # Get satellite observation date
        if 'datetime_local' in sat_row and pd.notna(sat_row['datetime_local']):
            sat_time = pd.to_datetime(sat_row['datetime_local'])
        elif 'date' in sat_row:
            sat_date = pd.to_datetime(sat_row['date'])
            # Landsat overpass ~10:30 local time
            sat_time = sat_date.replace(hour=LANDSAT_OVERPASS_HOUR, 
                                        minute=LANDSAT_OVERPASS_MINUTE)
        else:
            continue
        
        # Find corresponding EC observation
        time_diff = abs(ch4_df['datetime'] - sat_time)
        min_idx = time_diff.idxmin()
        min_diff = time_diff[min_idx]
        
        # Check if within time window
        if min_diff > pd.Timedelta(minutes=time_window_min):
            continue
        
        ec_row = ch4_df.loc[min_idx]
        
        # Check CH4 quality
        ch4_value = ec_row[ch4_col]
        if pd.isna(ch4_value):
            continue
        
        # Quality flag check (if available)
        if 'qc_CH4' in ec_row and ec_row['qc_CH4'] > 1:
            continue  # Skip gap-filled data for training
        
        # Extract LST
        lst_mean = sat_row.get('LST_mean', np.nan)
        lst_max = sat_row.get('LST_max', np.nan)
        
        if pd.isna(lst_mean):
            continue
        
        training_records.append({
            'datetime': sat_time,
            'date': sat_time.date(),
            'doy': sat_time.timetuple().tm_yday,
            'CH4': ch4_value,
            'LST_mean': lst_mean,
            'LST_max': lst_max,
            'Tair': ec_row.get('Tair', np.nan),
            'time_diff_min': min_diff.total_seconds() / 60,
            'landsat_date': sat_row.get('landsat_date', None)
        })
    
    training_df = pd.DataFrame(training_records)
    
    print(f"\nMatched observations: {len(training_df)}")
    
    if len(training_df) > 0:
        print(f"\nCH₄ statistics:")
        print(f"  Mean: {training_df['CH4'].mean():.4f} µmol m⁻² s⁻¹")
        print(f"  Std:  {training_df['CH4'].std():.4f}")
        print(f"  Range: [{training_df['CH4'].min():.4f}, {training_df['CH4'].max():.4f}]")
        
        print(f"\nLST statistics:")
        print(f"  Mean: {training_df['LST_mean'].mean():.1f} °C")
        print(f"  Range: [{training_df['LST_mean'].min():.1f}, {training_df['LST_mean'].max():.1f}] °C")
    
    return training_df


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_ch4_model(df: pd.DataFrame, 
                    model_type: str = 'quadratic') -> dict:
    """
    Train CH₄ flux model using LST.
    
    Model types:
        - linear: CH4 = γ₀ + γ₁×LST
        - quadratic: CH4 = γ₀ + γ₁×LST + γ₂×LST²
    
    Parameters
    ----------
    df : pd.DataFrame
        Training data with CH4 and LST_mean columns
    model_type : str
        'linear' or 'quadratic'
        
    Returns
    -------
    dict
        Model results with coefficients, metrics, predictions
    """
    print("\n" + "=" * 60)
    print(f"TRAINING CH₄ MODEL ({model_type.upper()})")
    print("=" * 60)
    
    # Check required columns
    required = ['CH4', 'LST_mean']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"❌ Missing columns: {missing}")
        return None
    
    # Prepare data
    df_clean = df.dropna(subset=required)
    print(f"Using {len(df_clean)} observations")
    
    y = df_clean['CH4'].values
    LST = df_clean['LST_mean'].values
    
    # Build design matrix
    if model_type == 'linear':
        X = np.column_stack([
            np.ones(len(y)),
            LST
        ])
        feature_names = ['Intercept', 'LST']
    else:  # quadratic
        X = np.column_stack([
            np.ones(len(y)),
            LST,
            LST ** 2
        ])
        feature_names = ['Intercept', 'LST', 'LST²']
    
    # Fit model
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    coefs = model.coef_
    
    # Predictions
    y_pred = X @ coefs
    
    # LOOCV
    loo = LeaveOneOut()
    y_pred_cv = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X):
        model_cv = LinearRegression(fit_intercept=False)
        model_cv.fit(X[train_idx], y[train_idx])
        y_pred_cv[test_idx] = X[test_idx] @ model_cv.coef_
    
    # Metrics
    r2_train = r2_score(y, y_pred)
    r2_cv = r2_score(y, y_pred_cv)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    rmse_cv = np.sqrt(mean_squared_error(y, y_pred_cv))
    mae = mean_absolute_error(y, y_pred)
    
    print("\n--- Coefficients ---")
    for name, coef in zip(feature_names, coefs):
        print(f"  {name:15s}: {coef:15.10f}")
    
    print("\n--- Metrics ---")
    print(f"  R²_train:   {r2_train:.4f}")
    print(f"  R²_LOOCV:   {r2_cv:.4f}")
    print(f"  RMSE:       {rmse:.4f} µmol m⁻² s⁻¹")
    print(f"  RMSE_LOOCV: {rmse_cv:.4f} µmol m⁻² s⁻¹")
    print(f"  MAE:        {mae:.4f} µmol m⁻² s⁻¹")
    
    # Temperature response analysis
    print("\n--- Temperature Response ---")
    lst_range = np.linspace(LST.min(), LST.max(), 5)
    
    if model_type == 'quadratic':
        print("  LST (°C)  |  CH₄ (µmol m⁻² s⁻¹)")
        print("  " + "-" * 35)
        for lst_val in lst_range:
            ch4_pred = coefs[0] + coefs[1] * lst_val + coefs[2] * lst_val**2
            print(f"  {lst_val:8.1f}  |  {ch4_pred:12.4f}")
        
        # Optimum temperature (vertex of parabola)
        if coefs[2] != 0:
            lst_optimum = -coefs[1] / (2 * coefs[2])
            ch4_optimum = coefs[0] + coefs[1] * lst_optimum + coefs[2] * lst_optimum**2
            print(f"\n  Vertex at LST = {lst_optimum:.1f} °C (CH₄ = {ch4_optimum:.4f})")
    
    # LST range for clamping
    lst_min, lst_max = LST.min(), LST.max()
    print(f"\n  Training LST range: [{lst_min:.2f}, {lst_max:.2f}] °C")
    
    return {
        'model': model,
        'coefs': dict(zip(feature_names, coefs)),
        'metrics': {
            'R2_train': r2_train, 
            'R2_cv': r2_cv, 
            'RMSE': rmse,
            'RMSE_cv': rmse_cv,
            'MAE': mae
        },
        'y_true': y,
        'y_pred': y_pred,
        'y_pred_cv': y_pred_cv,
        'LST': LST,
        'feature_names': feature_names,
        'model_type': model_type,
        'lst_range': {'min': lst_min, 'max': lst_max}
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_ch4_diagnostics(result: dict, output_path: str = None):
    """
    Generate diagnostic plots for CH₄ model.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    y_true = result['y_true']
    y_pred = result['y_pred']
    y_pred_cv = result['y_pred_cv']
    LST = result['LST']
    coefs = list(result['coefs'].values())
    
    # 1. CH4 vs LST with fitted curve
    ax = axes[0, 0]
    ax.scatter(LST, y_true, alpha=0.7, s=80, c='darkorange', 
               edgecolor='black', linewidth=0.5, label='Observations')
    
    # Fitted curve
    lst_smooth = np.linspace(LST.min(), LST.max(), 100)
    if len(coefs) == 3:  # quadratic
        ch4_smooth = coefs[0] + coefs[1] * lst_smooth + coefs[2] * lst_smooth**2
    else:  # linear
        ch4_smooth = coefs[0] + coefs[1] * lst_smooth
    
    ax.plot(lst_smooth, ch4_smooth, 'b-', lw=2, label='Fitted model')
    ax.set_xlabel('LST (°C)')
    ax.set_ylabel('CH₄ flux (µmol m⁻² s⁻¹)')
    ax.set_title(f"CH₄ vs LST (R² = {result['metrics']['R2_train']:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Observed vs Predicted
    ax = axes[0, 1]
    ax.scatter(y_true, y_pred, alpha=0.7, s=80, c='darkorange',
               edgecolor='black', linewidth=0.5)
    
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'k--', lw=1.5, label='1:1 line')
    ax.set_xlabel('CH₄ observed (µmol m⁻² s⁻¹)')
    ax.set_ylabel('CH₄ predicted (µmol m⁻² s⁻¹)')
    ax.set_title(f"Training: R² = {result['metrics']['R2_train']:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. LOOCV validation
    ax = axes[1, 0]
    ax.scatter(y_true, y_pred_cv, alpha=0.7, s=80, c='darkorange',
               edgecolor='black', linewidth=0.5)
    
    lims = [min(y_true.min(), y_pred_cv.min()), max(y_true.max(), y_pred_cv.max())]
    ax.plot(lims, lims, 'k--', lw=1.5, label='1:1 line')
    ax.set_xlabel('CH₄ observed (µmol m⁻² s⁻¹)')
    ax.set_ylabel('CH₄ predicted LOOCV (µmol m⁻² s⁻¹)')
    ax.set_title(f"LOOCV: R² = {result['metrics']['R2_cv']:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Residuals
    ax = axes[1, 1]
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, alpha=0.7, s=80, c='darkorange',
               edgecolor='black', linewidth=0.5)
    ax.axhline(0, color='k', linestyle='--', lw=1.5)
    ax.set_xlabel('CH₄ predicted (µmol m⁻² s⁻¹)')
    ax.set_ylabel('Residuals (µmol m⁻² s⁻¹)')
    ax.set_title('Residuals vs Predicted')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved: {output_path}")
    
    plt.show()


# ============================================================================
# GEE CODE GENERATION
# ============================================================================

def generate_ch4_gee_code(result: dict, output_path: str = None) -> str:
    """
    Generate JavaScript code for GEE with CH₄ model coefficients.
    """
    coefs = result['coefs']
    metrics = result['metrics']
    lst_range = result['lst_range']
    
    code = f'''
//======================================================================
// CH₄ FLUX MODEL COEFFICIENTS
// Auto-generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}
//======================================================================

// Model: CH4_moment = γ₀ + γ₁×LST + γ₂×LST²
// R²_train = {metrics['R2_train']:.3f}, R²_LOOCV = {metrics['R2_cv']:.3f}
// RMSE = {metrics['RMSE']:.4f} µmol m⁻² s⁻¹
// n = {len(result['y_true'])} scenes

var CH4_INTERCEPT = {coefs['Intercept']:.10f};
var CH4_LST       = {coefs['LST']:.10f};
var CH4_LST2      = {coefs['LST²']:.10f};

// LST clamp range (from training data)
var LST_MIN = {lst_range['min']:.2f};
var LST_MAX = {lst_range['max']:.2f};

// Diurnal correction factor (moment → daily)
var F_CH4 = 0.876;  // ± 0.05

// Conversion: µmol m⁻² s⁻¹ → g CH₄ m⁻² day⁻¹
var UMOL_TO_G_DAY = 1.3858;  // 1e-6 × 16.04 × 86400

// Ice-free period
var ICE_FREE_DAYS = 200;

// GWP for CO₂-equivalent
var GWP_CH4 = 28;

//======================================================================
// CALCULATE CH₄ FLUX
//======================================================================

function calculateCH4(lstImage) {{
  // Clamp LST to training range
  var lstClamped = lstImage.clamp(LST_MIN, LST_MAX);
  
  // Quadratic model: CH4 = γ₀ + γ₁×LST + γ₂×LST²
  var ch4Moment = ee.Image.constant(CH4_INTERCEPT)
    .add(lstClamped.multiply(CH4_LST))
    .add(lstClamped.pow(2).multiply(CH4_LST2))
    .max(0)  // Physical constraint: non-negative
    .rename('CH4_moment');
  
  // Daily average
  var ch4Daily = ch4Moment.multiply(F_CH4).rename('CH4_daily');
  
  return ee.Image.cat([ch4Moment, ch4Daily]);
}}
'''
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(code)
        print(f"\nGEE code saved: {output_path}")
    
    print(code)
    return code


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train CH₄ flux model using LST'
    )
    
    parser.add_argument('--ec-data', 
                        help='Path to CH₄ eddy covariance data (CSV)')
    parser.add_argument('--satellite-data',
                        help='Path to satellite predictor data (CSV)')
    parser.add_argument('--training-data',
                        help='Path to pre-prepared training data (CSV)')
    parser.add_argument('--output-dir', default='./output',
                        help='Output directory')
    parser.add_argument('--model-type', choices=['linear', 'quadratic'], 
                        default='quadratic',
                        help='Model type (default: quadratic)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip diagnostic plots')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("  CH₄ FLUX MODEL TRAINING PIPELINE")
    print("=" * 70)
    
    # Load or create training data
    if args.training_data:
        # Use pre-prepared training data
        training_df = pd.read_csv(args.training_data)
        print(f"Loaded training data: {len(training_df)} observations")
    elif args.ec_data and args.satellite_data:
        # Create training data from EC and satellite
        ch4_df = load_ch4_data(args.ec_data)
        sat_df = load_satellite_predictors(args.satellite_data)
        training_df = create_ch4_training_data(ch4_df, sat_df)
        
        # Save training data
        training_path = output_dir / 'ch4_training_data.csv'
        training_df.to_csv(training_path, index=False)
        print(f"\nTraining data saved: {training_path}")
    else:
        print("❌ Provide either --training-data or both --ec-data and --satellite-data")
        return
    
    if len(training_df) < 5:
        print(f"❌ Insufficient training data: {len(training_df)} observations")
        return
    
    # Train model
    result = train_ch4_model(training_df, model_type=args.model_type)
    
    if result is None:
        print("\n❌ Model training failed")
        return
    
    # Save results
    results_export = {
        'model_type': result['model_type'],
        'coefficients': result['coefs'],
        'metrics': result['metrics'],
        'lst_range': result['lst_range'],
        'n_observations': len(result['y_true'])
    }
    
    results_path = output_dir / 'ch4_model_results.json'
    with open(results_path, 'w') as f:
        json.dump(results_export, f, indent=2)
    print(f"\nResults saved: {results_path}")
    
    # Generate plots
    if not args.no_plots:
        plot_ch4_diagnostics(result, output_dir / 'ch4_model_diagnostics.png')
    
    # Generate GEE code
    generate_ch4_gee_code(result, output_dir / 'ch4_gee_coefficients.js')
    
    print("\n" + "=" * 70)
    print("  ✅ CH₄ MODEL TRAINING COMPLETED")
    print("=" * 70)
    
    # Summary
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  CH₄ MODEL SUMMARY                                                    ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  Model: CH4_moment = γ₀ + γ₁×LST + γ₂×LST²                           ║
║                                                                       ║
║  Coefficients:                                                        ║
║    γ₀ (Intercept) = {result['coefs']['Intercept']:12.6f}                           ║
║    γ₁ (LST)       = {result['coefs']['LST']:12.6f}                           ║
║    γ₂ (LST²)      = {result['coefs']['LST²']:12.6f}                           ║
║                                                                       ║
║  Performance:                                                         ║
║    R² (train)     = {result['metrics']['R2_train']:.4f}                                  ║
║    R² (LOOCV)     = {result['metrics']['R2_cv']:.4f}                                  ║
║    RMSE           = {result['metrics']['RMSE']:.4f} µmol m⁻² s⁻¹                       ║
║                                                                       ║
║  Training data:                                                       ║
║    n              = {len(result['y_true'])}                                         ║
║    LST range      = [{result['lst_range']['min']:.1f}, {result['lst_range']['max']:.1f}] °C                          ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
