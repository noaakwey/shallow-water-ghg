# Methodology

## Overview

This document describes the methodology for estimating greenhouse gas (CO₂ and CH₄) emissions from shallow water zones using satellite remote sensing and eddy covariance measurements.

## Study Area

The Kuibyshev Reservoir is the largest reservoir in Europe by surface area (6,450 km²). Shallow water zones (<2 m depth) constitute approximately 440 km² and are characterized by:

- Extensive macrophyte beds
- High organic matter accumulation
- Seasonal flooding dynamics
- Active carbon cycling

## Data Sources

### Eddy Covariance Tower

A flux tower was installed over the shallow water zone to measure:
- CO₂ fluxes (open-path IRGA)
- CH₄ fluxes (closed-path analyzer)
- Meteorological variables

Data processing followed standard protocols using REddyProc for gap-filling and flux partitioning.

### Satellite Remote Sensing

| Platform | Product | Key Variables | Resolution |
|----------|---------|---------------|------------|
| Sentinel-2 | Surface Reflectance | Spectral indices | 10-20 m |
| Landsat 8/9 | Level-2 ST | Land Surface Temperature | 30 m |
| Sentinel-1 | GRD | Radar backscatter | 10 m |

## Model Development

### Temporal Matching

Satellite observations were matched with eddy covariance measurements using:

1. **Sentinel-2**: 11:00 AM local time (±30 min window)
2. **Landsat**: 10:30 AM local time (±5 day window for LST)

### GPP Model

The GPP model uses chlorophyll-sensitive indices and temperature:

```
GPP_11am = α₀ + α₁×LST_mean + α₂×MTCI + α₃×AWEInsh + α₄×LST_max² + α₅×(AWEInsh×LST_mean)
```

**Rationale**:
- MTCI (Meris Terrestrial Chlorophyll Index): Proxy for phytoplankton chlorophyll
- AWEInsh: Water-specific index that captures submerged vegetation
- LST: Temperature control on photosynthesis

### Reco Model

Ecosystem respiration is modeled as:

```
Reco_11am = β₀ + β₁×LST_max + β₂×AWEInsh² + β₃×(NDWI×LST_mean) + β₄×(AWEInsh×LST_mean)
```

**Rationale**:
- Strong temperature dependence (Q10 relationship)
- Water indices capture organic matter availability
- Interaction terms account for coupled effects

### CH₄ Model

Methane emissions follow a quadratic temperature response:

```
CH4_moment = γ₀ + γ₁×LST + γ₂×LST²
```

**Rationale**:
- Methanogenesis is strongly temperature-dependent
- Quadratic form captures accelerating response at high temperatures
- Ebullition increases non-linearly with sediment warming

## Diurnal Correction

Satellite observations capture instantaneous fluxes at overpass time. Diurnal correction factors convert these to daily averages:

| Flux | Factor (F) | Derivation |
|------|------------|------------|
| GPP | 0.904 ± 0.03 | Ratio of 11:00 to daily mean |
| Reco | 0.978 ± 0.03 | Ratio of 11:00 to daily mean |
| CH₄ | 0.876 ± 0.05 | Ratio of 10:30 to daily mean |

Factors were empirically derived from the eddy covariance time series.

## Predictor Clamping

To prevent extrapolation beyond the training domain, predictors are clamped to their observed ranges:

| Predictor | Min | Max | Units |
|-----------|-----|-----|-------|
| LST_mean | 5.11 | 25.35 | °C |
| LST_max | 5.47 | 27.38 | °C |
| MTCI | -4.15 | 4.83 | - |
| NDWI | -0.067 | 0.097 | - |
| AWEInsh | 2470 | 4749 | - |

## Shallow Water Mask

A phenological approach identifies shallow water zones:

1. **Permanent water**: MNDWI > 0 in summer (June-August)
2. **Spring flooding**: MNDWI > -0.1 in spring (April-May)
3. **Autumn exposure**: MNDWI > -0.1 in autumn (October)
4. **Shallow water** = Spring flooded AND NOT Autumn flooded

This captures zones that are inundated during spring flooding but become exposed as water levels drop.

## Annual Budget Calculation

Annual fluxes are calculated as:

```
GPP_annual = GPP_daily × F_GPP × daylight_hours × 3600 × ice_free_days
Reco_annual = Reco_daily × F_Reco × 24 × 3600 × ice_free_days
NEE_annual = Reco_annual - GPP_annual
CH4_annual = CH4_daily × F_CH4 × UMOL_TO_G_DAY × ice_free_days
```

Where:
- Ice-free days = 200 (April 20 - November 5)
- Daylight hours = 12 (average)
- UMOL_TO_G_DAY = 1.3858 (conversion factor)

## Uncertainty Analysis

Monte Carlo simulation propagates uncertainties through the calculation:

### Uncertainty Sources

1. **Model RMSE**: Per-pixel Gaussian noise
2. **Diurnal factors**: Normal distribution around mean
3. **Ice-free period**: ±10 days uncertainty
4. **Daylight hours**: ±1 hour uncertainty

### Monte Carlo Protocol

1. Sample parameter values from distributions
2. Add per-pixel noise to flux maps
3. Calculate annual totals
4. Repeat 2000 times
5. Extract mean, SD, and 95% CI from distribution

## Validation

Models were validated using Leave-One-Out Cross-Validation (LOOCV):

| Model | R² (train) | R² (LOOCV) | RMSE |
|-------|------------|------------|------|
| GPP | 0.52 | 0.49 | 0.25 µmol m⁻² s⁻¹ |
| Reco | 0.80 | 0.77 | 0.28 µmol m⁻² s⁻¹ |
| CH₄ | 0.50 | 0.47 | 0.026 µmol m⁻² s⁻¹ |

## Limitations

1. **Temporal sampling**: Satellite observations limited to cloud-free conditions
2. **Spatial extrapolation**: Models calibrated at single tower location
3. **MTCI instability**: Index can be numerically unstable over water
4. **Diurnal patterns**: Assumed constant correction factors

## References

1. Baldocchi, D. (2003). Assessing the eddy covariance technique for evaluating carbon dioxide exchange rates of ecosystems. Global Change Biology, 9(4), 479-492.

2. Wutzler, T., et al. (2018). Basic and extensible post-processing of eddy covariance flux data with REddyProc. Biogeosciences, 15, 5015-5030.

3. Gorelick, N., et al. (2017). Google Earth Engine: Planetary-scale geospatial analysis for everyone. Remote Sensing of Environment, 202, 18-27.
