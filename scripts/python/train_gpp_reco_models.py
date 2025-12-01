#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
GPP AND RECO MODEL TRAINING PIPELINE
================================================================================

Train regression models for Gross Primary Production (GPP) and Ecosystem 
Respiration (Reco) using Sentinel-2 optical indices and Landsat LST.

Models:
    GPP = α₀ + α₁×LST_mean + α₂×MTCI + α₃×AWEInsh + α₄×LST_max² + α₅×(AWEInsh×LST_mean)
    Reco = β₀ + β₁×LST_max + β₂×AWEInsh² + β₃×(NDWI×LST_mean) + β₄×(AWEInsh×LST_mean)

Usage:
    python train_gpp_reco_models.py \
        --training-data /path/to/training_data.csv \
        --output-dir ./output

Author: [Your Name]
Version: 1.0
================================================================================
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut

# ============================================================================
# CONFIGURATION
# ============================================================================

# Required columns for each model
RECO_REQUIRED = ['Reco', 'LST_max', 'LST_mean', 'AWEInsh', 'NDWI']
GPP_REQUIRED = ['GPP_f', 'LST_mean', 'LST_max', 'AWEInsh', 'MTCI']

# Alternative chlorophyll index if MTCI unavailable
ALT_CHL_INDEX = 'CIG'


# ============================================================================
# DATA LOADING
# ============================================================================

def load_training_data(filepath: str) -> pd.DataFrame:
    """
    Load and validate training dataset.
    
    Parameters
    ----------
    filepath : str
        Path to CSV file with training data
        
    Returns
    -------
    pd.DataFrame
        Loaded and validated training data
    """
    print("=" * 60)
    print("LOADING TRAINING DATA")
    print("=" * 60)
    
    df = pd.read_csv(filepath)
    print(f"Loaded: {len(df)} scenes, {len(df.columns)} columns")
    
    # Parse dates
    if 'datetime_local' in df.columns:
        df['datetime_local'] = pd.to_datetime(df['datetime_local'])
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Print available columns
    print(f"\nColumns: {list(df.columns)}")
    
    # Target variable statistics
    if 'Reco' in df.columns and 'GPP_f' in df.columns:
        print(f"\nReco: mean={df['Reco'].mean():.3f}, std={df['Reco'].std():.3f}, "
              f"range=[{df['Reco'].min():.3f}, {df['Reco'].max():.3f}]")
        print(f"GPP_f: mean={df['GPP_f'].mean():.3f}, std={df['GPP_f'].std():.3f}, "
              f"range=[{df['GPP_f'].min():.3f}, {df['GPP_f'].max():.3f}]")
    
    return df


# ============================================================================
# PREDICTOR ANALYSIS
# ============================================================================

def analyze_predictors(df: pd.DataFrame, 
                       target_cols: list = ['Reco', 'GPP_f']) -> tuple:
    """
    Analyze predictor correlations and ranges.
    
    Parameters
    ----------
    df : pd.DataFrame
        Training data
    target_cols : list
        Target variable column names
        
    Returns
    -------
    tuple
        (correlations dict, ranges dict)
    """
    print("\n" + "=" * 60)
    print("PREDICTOR ANALYSIS")
    print("=" * 60)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    predictor_cols = [c for c in numeric_cols 
                      if c not in target_cols + ['doy', 'has_landsat']]
    
    print(f"Numeric predictors: {len(predictor_cols)}")
    
    # Correlations with targets
    correlations = {}
    for target in target_cols:
        if target in df.columns:
            corr = df[predictor_cols].corrwith(
                df[target]).sort_values(key=abs, ascending=False)
            correlations[target] = corr
            print(f"\nTop-10 correlations with {target}:")
            for pred, r in corr.head(10).items():
                print(f"  {pred:20s}: r = {r:+.3f}")
    
    # Key predictor ranges
    key_preds = ['LST_mean', 'LST_max', 'MTCI', 'NDWI', 'AWEInsh', 'CIG', 'NDVI']
    available_preds = [p for p in key_preds if p in df.columns]
    
    print("\n" + "-" * 60)
    print("PREDICTOR RANGES (for GEE clamp)")
    print("-" * 60)
    
    ranges = {}
    for pred in available_preds:
        pmin, pmax = df[pred].min(), df[pred].max()
        pmean, pstd = df[pred].mean(), df[pred].std()
        ranges[pred] = {'min': pmin, 'max': pmax, 'mean': pmean, 'std': pstd}
        print(f"{pred:15s}: [{pmin:10.4f}, {pmax:10.4f}] | "
              f"mean={pmean:8.4f} ± {pstd:.4f}")
    
    return correlations, ranges


def check_mtci_stability(df: pd.DataFrame) -> bool:
    """
    Check MTCI stability in training data.
    
    Returns True if MTCI is stable (range < 20).
    """
    print("\n" + "=" * 60)
    print("⚠️  MTCI STABILITY ANALYSIS")
    print("=" * 60)
    
    if 'MTCI' not in df.columns:
        print("MTCI not found in data")
        return False
    
    mtci = df['MTCI']
    
    print(f"MTCI range in training: [{mtci.min():.2f}, {mtci.max():.2f}]")
    print(f"Range span: {mtci.max() - mtci.min():.2f}")
    
    # Check for outliers
    q01, q99 = mtci.quantile([0.01, 0.99])
    outliers = ((mtci < q01) | (mtci > q99)).sum()
    print(f"Outliers beyond 1-99%: {outliers} ({100*outliers/len(mtci):.1f}%)")
    
    is_stable = (mtci.max() - mtci.min()) < 20
    
    if is_stable:
        print(f"\n✅ MTCI is stable in training data")
        print(f"   Recommended clamp: [{mtci.min():.2f}, {mtci.max():.2f}]")
    else:
        print(f"\n⚠️  MTCI has wide range!")
        print("   Consider using CIG or other chlorophyll index")
    
    return is_stable


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_reco_model(df: pd.DataFrame, use_robust: bool = False) -> dict:
    """
    Train Reco model using LST and water indices.
    
    Model specification:
        Reco = β₀ + β₁×LST_max + β₂×AWEInsh² + β₃×(NDWI×LST_mean) + β₄×(AWEInsh×LST_mean)
    
    Parameters
    ----------
    df : pd.DataFrame
        Training data with required columns
    use_robust : bool
        Use Huber robust regression instead of OLS
        
    Returns
    -------
    dict
        Model results with coefficients, metrics, and predictions
    """
    print("\n" + "=" * 60)
    print("TRAINING RECO MODEL")
    print("=" * 60)
    
    # Check required columns
    missing = [c for c in RECO_REQUIRED if c not in df.columns]
    if missing:
        print(f"❌ Missing columns: {missing}")
        return None
    
    # Prepare data
    df_clean = df.dropna(subset=RECO_REQUIRED)
    print(f"Using {len(df_clean)} observations")
    
    y = df_clean['Reco'].values
    
    # Extract predictors
    LST_max = df_clean['LST_max'].values
    LST_mean = df_clean['LST_mean'].values
    AWEInsh = df_clean['AWEInsh'].values
    NDWI = df_clean['NDWI'].values
    
    # Derived features
    AWEInsh2 = AWEInsh ** 2
    NDWI_LST_mean = NDWI * LST_mean
    AWEInsh_LST_mean = AWEInsh * LST_mean
    
    # Design matrix
    X = np.column_stack([
        np.ones(len(y)),  # intercept
        LST_max,
        AWEInsh2,
        NDWI_LST_mean,
        AWEInsh_LST_mean
    ])
    
    feature_names = ['Intercept', 'LST_max', 'AWEInsh²', 
                     'NDWI×LST_mean', 'AWEInsh×LST_mean']
    
    # Select model
    if use_robust:
        model = HuberRegressor(epsilon=1.35, max_iter=1000)
        X_fit = X[:, 1:]  # HuberRegressor has built-in intercept
    else:
        model = LinearRegression(fit_intercept=False)
        X_fit = X
    
    # Fit model
    model.fit(X_fit, y)
    
    # Extract coefficients
    if use_robust:
        coefs = np.concatenate([[model.intercept_], model.coef_])
    else:
        coefs = model.coef_
    
    # Predictions
    y_pred = X @ coefs
    
    # Leave-One-Out Cross-Validation
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
    mae = mean_absolute_error(y, y_pred)
    
    print("\n--- Coefficients ---")
    for name, coef in zip(feature_names, coefs):
        print(f"  {name:20s}: {coef:15.10f}")
    
    print("\n--- Metrics ---")
    print(f"  R²_train:  {r2_train:.4f}")
    print(f"  R²_LOOCV:  {r2_cv:.4f}")
    print(f"  RMSE:      {rmse:.4f} µmol m⁻² s⁻¹")
    print(f"  MAE:       {mae:.4f} µmol m⁻² s⁻¹")
    
    return {
        'model': model,
        'coefs': dict(zip(feature_names, coefs)),
        'metrics': {'R2_train': r2_train, 'R2_cv': r2_cv, 
                    'RMSE': rmse, 'MAE': mae},
        'y_true': y,
        'y_pred': y_pred,
        'y_pred_cv': y_pred_cv,
        'feature_names': feature_names
    }


def train_gpp_model(df: pd.DataFrame, 
                    use_robust: bool = False,
                    use_alt_chl: bool = False) -> dict:
    """
    Train GPP model using LST and chlorophyll indices.
    
    Model specification:
        GPP = α₀ + α₁×LST_mean + α₂×MTCI + α₃×AWEInsh + α₄×LST_max² + α₅×(AWEInsh×LST_mean)
    
    Parameters
    ----------
    df : pd.DataFrame
        Training data
    use_robust : bool
        Use Huber robust regression
    use_alt_chl : bool
        Use CIG instead of MTCI
        
    Returns
    -------
    dict
        Model results
    """
    print("\n" + "=" * 60)
    chl_index = ALT_CHL_INDEX if use_alt_chl else 'MTCI'
    print(f"TRAINING GPP MODEL (chlorophyll index: {chl_index})")
    print("=" * 60)
    
    # Check required columns
    required = ['GPP_f', 'LST_mean', 'LST_max', 'AWEInsh', chl_index]
    missing = [c for c in required if c not in df.columns]
    
    if missing:
        if 'MTCI' in missing and 'CIG' in df.columns:
            print("   Trying CIG instead of MTCI...")
            chl_index = 'CIG'
            required = ['GPP_f', 'LST_mean', 'LST_max', 'AWEInsh', 'CIG']
            missing = [c for c in required if c not in df.columns]
    
    if missing:
        print(f"❌ Missing columns: {missing}")
        return None
    
    # Prepare data
    df_clean = df.dropna(subset=required)
    print(f"Using {len(df_clean)} observations")
    print(f"Chlorophyll index: {chl_index}")
    
    y = df_clean['GPP_f'].values
    
    # Extract predictors
    LST_mean = df_clean['LST_mean'].values
    LST_max = df_clean['LST_max'].values
    AWEInsh = df_clean['AWEInsh'].values
    CHL = df_clean[chl_index].values
    
    # Derived features
    LST_max2 = LST_max ** 2
    AWEInsh_LST_mean = AWEInsh * LST_mean
    
    # Design matrix
    X = np.column_stack([
        np.ones(len(y)),
        LST_mean,
        CHL,
        AWEInsh,
        LST_max2,
        AWEInsh_LST_mean
    ])
    
    feature_names = ['Intercept', 'LST_mean', chl_index, 
                     'AWEInsh', 'LST_max²', 'AWEInsh×LST_mean']
    
    # Select model
    if use_robust:
        model = HuberRegressor(epsilon=1.35, max_iter=1000)
        X_fit = X[:, 1:]
    else:
        model = LinearRegression(fit_intercept=False)
        X_fit = X
    
    # Fit model
    model.fit(X_fit, y)
    
    # Extract coefficients
    if use_robust:
        coefs = np.concatenate([[model.intercept_], model.coef_])
    else:
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
    mae = mean_absolute_error(y, y_pred)
    
    print("\n--- Coefficients ---")
    for name, coef in zip(feature_names, coefs):
        print(f"  {name:20s}: {coef:15.10f}")
    
    print("\n--- Metrics ---")
    print(f"  R²_train:  {r2_train:.4f}")
    print(f"  R²_LOOCV:  {r2_cv:.4f}")
    print(f"  RMSE:      {rmse:.4f} µmol m⁻² s⁻¹")
    print(f"  MAE:       {mae:.4f} µmol m⁻² s⁻¹")
    
    # Contribution analysis
    print("\n--- Predictor Contributions ---")
    for i, (name, coef) in enumerate(zip(feature_names, coefs)):
        if i == 0:
            continue
        x_mean = X[:, i].mean()
        x_std = X[:, i].std()
        contribution = coef * x_mean
        sensitivity = coef * x_std
        print(f"  {name:20s}: mean_contrib={contribution:+8.3f}, "
              f"sensitivity={sensitivity:+8.3f}")
    
    return {
        'model': model,
        'coefs': dict(zip(feature_names, coefs)),
        'metrics': {'R2_train': r2_train, 'R2_cv': r2_cv, 
                    'RMSE': rmse, 'MAE': mae},
        'y_true': y,
        'y_pred': y_pred,
        'y_pred_cv': y_pred_cv,
        'chl_index': chl_index,
        'feature_names': feature_names
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_model_diagnostics(reco_result: dict, 
                           gpp_result: dict, 
                           output_path: str = None):
    """
    Generate diagnostic plots for both models.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # --- RECO ---
    ax = axes[0, 0]
    ax.scatter(reco_result['y_true'], reco_result['y_pred'], 
               alpha=0.7, s=60, c='coral')
    ax.plot([0, 2.5], [0, 2.5], 'k--', lw=1.5, label='1:1')
    ax.set_xlabel('Reco observed (µmol m⁻² s⁻¹)')
    ax.set_ylabel('Reco predicted (µmol m⁻² s⁻¹)')
    ax.set_title(f"Reco: R²={reco_result['metrics']['R2_train']:.3f}")
    ax.legend()
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 2.5)
    
    ax = axes[0, 1]
    ax.scatter(reco_result['y_true'], reco_result['y_pred_cv'], 
               alpha=0.7, s=60, c='coral')
    ax.plot([0, 2.5], [0, 2.5], 'k--', lw=1.5)
    ax.set_xlabel('Reco observed (µmol m⁻² s⁻¹)')
    ax.set_ylabel('Reco predicted LOOCV (µmol m⁻² s⁻¹)')
    ax.set_title(f"Reco LOOCV: R²={reco_result['metrics']['R2_cv']:.3f}")
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 2.5)
    
    ax = axes[0, 2]
    residuals_reco = reco_result['y_true'] - reco_result['y_pred']
    ax.scatter(reco_result['y_pred'], residuals_reco, alpha=0.7, s=60, c='coral')
    ax.axhline(0, color='k', linestyle='--', lw=1.5)
    ax.set_xlabel('Reco predicted (µmol m⁻² s⁻¹)')
    ax.set_ylabel('Residuals')
    ax.set_title('Reco: Residuals vs Predicted')
    
    # --- GPP ---
    ax = axes[1, 0]
    ax.scatter(gpp_result['y_true'], gpp_result['y_pred'], 
               alpha=0.7, s=60, c='forestgreen')
    ax.plot([0, 3], [0, 3], 'k--', lw=1.5, label='1:1')
    ax.set_xlabel('GPP observed (µmol m⁻² s⁻¹)')
    ax.set_ylabel('GPP predicted (µmol m⁻² s⁻¹)')
    chl_idx = gpp_result.get('chl_index', 'MTCI')
    ax.set_title(f"GPP ({chl_idx}): R²={gpp_result['metrics']['R2_train']:.3f}")
    ax.legend()
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    
    ax = axes[1, 1]
    ax.scatter(gpp_result['y_true'], gpp_result['y_pred_cv'], 
               alpha=0.7, s=60, c='forestgreen')
    ax.plot([0, 3], [0, 3], 'k--', lw=1.5)
    ax.set_xlabel('GPP observed (µmol m⁻² s⁻¹)')
    ax.set_ylabel('GPP predicted LOOCV (µmol m⁻² s⁻¹)')
    ax.set_title(f"GPP LOOCV: R²={gpp_result['metrics']['R2_cv']:.3f}")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    
    ax = axes[1, 2]
    residuals_gpp = gpp_result['y_true'] - gpp_result['y_pred']
    ax.scatter(gpp_result['y_pred'], residuals_gpp, 
               alpha=0.7, s=60, c='forestgreen')
    ax.axhline(0, color='k', linestyle='--', lw=1.5)
    ax.set_xlabel('GPP predicted (µmol m⁻² s⁻¹)')
    ax.set_ylabel('Residuals')
    ax.set_title('GPP: Residuals vs Predicted')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved: {output_path}")
    
    plt.show()


# ============================================================================
# GEE CODE GENERATION
# ============================================================================

def generate_gee_code(reco_result: dict, 
                      gpp_result: dict, 
                      predictor_ranges: dict, 
                      output_path: str = None) -> str:
    """
    Generate JavaScript code for Google Earth Engine with model coefficients.
    """
    reco_coefs = reco_result['coefs']
    gpp_coefs = gpp_result['coefs']
    chl_index = gpp_result.get('chl_index', 'MTCI')
    
    code = f'''
//======================================================================
// GPP AND RECO MODEL COEFFICIENTS
// Auto-generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}
//======================================================================

//--------------- RECO (instantaneous at 11:00 AM) --------------
// Reco = β₀ + β₁×LST_max + β₂×AWEInsh² + β₃×(NDWI×LST_mean) + β₄×(AWEInsh×LST_mean)
// R²_train = {reco_result['metrics']['R2_train']:.3f}, R²_LOOCV = {reco_result['metrics']['R2_cv']:.3f}
// RMSE = {reco_result['metrics']['RMSE']:.4f} µmol m⁻² s⁻¹

var RECO_INTERCEPT        = {reco_coefs['Intercept']:.10f};
var RECO_LST_MAX          = {reco_coefs['LST_max']:.10f};
var RECO_AWEINSH2         = {reco_coefs['AWEInsh²']:.15e};
var RECO_NDWI_LST_MEAN    = {reco_coefs['NDWI×LST_mean']:.10f};
var RECO_AWEINSH_LST_MEAN = {reco_coefs['AWEInsh×LST_mean']:.15e};

//--------------- GPP (instantaneous at 11:00 AM) ---------------
// GPP = α₀ + α₁×LST_mean + α₂×{chl_index} + α₃×AWEInsh + α₄×LST_max² + α₅×(AWEInsh×LST_mean)
// R²_train = {gpp_result['metrics']['R2_train']:.3f}, R²_LOOCV = {gpp_result['metrics']['R2_cv']:.3f}
// RMSE = {gpp_result['metrics']['RMSE']:.4f} µmol m⁻² s⁻¹

var GPP_INTERCEPT        = {gpp_coefs['Intercept']:.10f};
var GPP_LST_MEAN         = {gpp_coefs['LST_mean']:.10f};
var GPP_{chl_index}      = {gpp_coefs[chl_index]:.10f};
var GPP_AWEINSH          = {gpp_coefs['AWEInsh']:.10f};
var GPP_LST_MAX2         = {gpp_coefs['LST_max²']:.10f};
var GPP_AWEINSH_LST_MEAN = {gpp_coefs['AWEInsh×LST_mean']:.15e};

//======================================================================
// PREDICTOR CLAMP RANGES
// (from training data, {len(reco_result['y_true'])} scenes)
//======================================================================
'''
    
    for pred, ranges in predictor_ranges.items():
        code += f'''
var {pred.upper().replace("_", "_")}_MIN = {ranges['min']:.6f};
var {pred.upper().replace("_", "_")}_MAX = {ranges['max']:.6f};'''
    
    code += '''

//======================================================================
// CLAMP FUNCTION
//======================================================================

function clampPredictors(img) {
  var MTCI_clamped    = img.select('MTCI').clamp(MTCI_MIN, MTCI_MAX).rename('MTCI');
  var AWEInsh_clamped = img.select('AWEInsh').clamp(AWEINSH_MIN, AWEINSH_MAX).rename('AWEInsh');
  var NDWI_clamped    = img.select('NDWI').clamp(NDWI_MIN, NDWI_MAX).rename('NDWI');
  
  return img
    .addBands(MTCI_clamped, null, true)
    .addBands(AWEInsh_clamped, null, true)
    .addBands(NDWI_clamped, null, true);
}
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
        description='Train GPP and Reco models for shallow water flux estimation'
    )
    
    parser.add_argument('--training-data', required=True,
                        help='Path to training data CSV')
    parser.add_argument('--output-dir', default='./output',
                        help='Output directory for results')
    parser.add_argument('--model-type', choices=['ols', 'robust'], default='ols',
                        help='Regression model type (default: ols)')
    parser.add_argument('--use-cig', action='store_true',
                        help='Use CIG instead of MTCI for GPP model')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip diagnostic plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("  GPP AND RECO MODEL TRAINING PIPELINE")
    print("=" * 70)
    
    # Load data
    try:
        training_df = load_training_data(args.training_data)
    except FileNotFoundError:
        print(f"\n❌ File not found: {args.training_data}")
        return
    
    # Analyze predictors
    correlations, predictor_ranges = analyze_predictors(training_df)
    check_mtci_stability(training_df)
    
    # Train models
    use_robust = (args.model_type == 'robust')
    
    reco_result = train_reco_model(training_df, use_robust=use_robust)
    gpp_result = train_gpp_model(training_df, use_robust=use_robust, 
                                  use_alt_chl=args.use_cig)
    
    if reco_result is None or gpp_result is None:
        print("\n❌ Model training failed. Check your data.")
        return
    
    # Save results
    results = {
        'reco_coefs': reco_result['coefs'],
        'reco_metrics': reco_result['metrics'],
        'gpp_coefs': gpp_result['coefs'],
        'gpp_metrics': gpp_result['metrics'],
        'predictor_ranges': predictor_ranges
    }
    
    results_path = output_dir / 'model_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")
    
    # Generate plots
    if not args.no_plots:
        plot_model_diagnostics(
            reco_result, gpp_result,
            output_path=output_dir / 'model_diagnostics.png'
        )
    
    # Generate GEE code
    generate_gee_code(
        reco_result, gpp_result, predictor_ranges,
        output_path=output_dir / 'gee_coefficients.js'
    )
    
    print("\n" + "=" * 70)
    print("  ✅ PIPELINE COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
