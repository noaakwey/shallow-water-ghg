# Shallow Water Greenhouse Gas Emissions: Remote Sensing Approach

[![DOI](https://img.shields.io/badge/DOI-10.xxxx/xxxxx-blue.svg)](https://doi.org/10.xxxx/xxxxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GEE](https://img.shields.io/badge/Google%20Earth%20Engine-Ready-green.svg)](https://earthengine.google.com/)

## Overview

This repository contains the complete codebase for estimating greenhouse gas emissions (CO₂ and CH₄) from shallow water zones of the Kuibyshev Reservoir using satellite remote sensing and eddy covariance measurements.

**Key Features:**
- Integration of Sentinel-2, Landsat LST data
- Machine learning models for GPP, Reco, and CH₄ flux estimation
- Monte Carlo uncertainty analysis
- Google Earth Engine implementation for regional mapping

## Repository Structure

```
├── scripts/
│   ├── python/
│   │   ├── train_gpp_reco_models.py    # GPP and Reco model training
│   │   ├── train_ch4_model.py          # CH₄ model training
│   │   ├── mc_nee_uncertainty.py       # NEE Monte Carlo analysis
│   │   ├── mc_ch4_uncertainty.py       # CH₄ Monte Carlo analysis
│   │   └── utils.py                    # Utility functions
│   └── gee/
│       ├── s2_landsat_predictors.js    # S2 + Landsat LST extraction
│       ├── gpp_reco_mapping.js         # GPP/Reco flux mapping
│       └── ch4_mapping.js              # CH₄ flux mapping
├── data/     # in work
│   ├── sample/
│   │   └── training_data_sample.csv    # Sample training data structure
│   └── README.md                       # Data availability statement
├── docs/    # in work
│   └── methodology.md                  # Detailed methodology
├── figures/ # in work
│   └── .gitkeep
└── requirements.txt
```

## Installation

### Python Environment

```bash
# Clone repository
git clone https://github.com/noaakwey/shallow-water-ghg.git
cd shallow-water-ghg

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Google Earth Engine

1. Sign up for [Google Earth Engine](https://earthengine.google.com/)
2. Install the [GEE Python API](https://developers.google.com/earth-engine/guides/python_install) (optional)
3. Import scripts to your GEE Code Editor

## Usage

### 1. Model Training

#### GPP and Reco Models

```bash
python scripts/python/train_gpp_reco_models.py \
    --training-data /path/to/training_data.csv \
    --output-dir ./output \
    --model-type ols \
    --cross-validation loocv
```

#### CH₄ Model

```bash
python scripts/python/train_ch4_model.py \
    --training-data /path/to/ch4_training_data.csv \
    --output-dir ./output \
    --model-type quadratic
```

### 2. Monte Carlo Uncertainty Analysis

#### NEE Uncertainty

```bash
python scripts/python/mc_nee_uncertainty.py \
    --fluxes /path/to/Instantaneous_Fluxes_11am_umol_s.vrt \ # vrt or tiff
    --mask /path/to/Shallow_Water_Mask.tif \
    --reco-band 1 \
    --gpp-band 2 \
    --rmse-gpp 0.25 \
    --rmse-reco 0.28 \
    --fgpp-mean 0.904 --fgpp-sd 0.03 \
    --freco-mean 0.978 --freco-sd 0.03 \
    --ice-days-mean 200 --ice-days-sd 10 \
    --daylight-hours-mean 12 --daylight-hours-sd 1 \
    --n-iter 2000 \
    --n-jobs -1 \
    --output-plot Figure5_MC_Results.png
```

#### CH₄ Uncertainty

```bash
python scripts/python/mc_ch4_uncertainty.py \
    --ch4-raster /path/to/CH4_Moment_Mean_umol_s.tif \ # vrt or tiff
    --mask /path/to/Shallow_Water_Mask.tif \
    --rmse 0.026 \
    --f-ch4-mean 0.876 --f-ch4-sd 0.05 \
    --ice-days-mean 200 --ice-days-sd 10 \
    --n-iter 2000 \
    --output-prefix ch4_mc
```

### 3. Google Earth Engine Scripts

Access the GEE scripts directly:

| Script | Description | GEE Link |
|--------|-------------|----------|
| `s2_landsat_predictors.js` | Extract S2 + Landsat predictors | [Open in GEE](https://code.earthengine.google.com/175976d82aef53b421ac6681ebe029aa) |
| `gpp_reco_mapping.js` | Map GPP/Reco fluxes | [Open in GEE](https://code.earthengine.google.com/370b1f16125ba492533d47eb5f1f5498) |
| `ch4_mapping.js` | Map CH₄ fluxes | [Open in GEE](https://code.earthengine.google.com/72db3490539d593dcfab9a188aa30085) |

## Model Specifications

### GPP Model

```
GPP_11am = α₀ + α₁×LST_mean + α₂×MTCI + α₃×AWEInsh + α₄×LST_max² + α₅×(AWEInsh×LST_mean)
```

| Coefficient | Value | Description |
|-------------|-------|-------------|
| α₀ | 22.530 | Intercept |
| α₁ | -0.793 | LST mean sensitivity |
| α₂ | -0.111 | MTCI (chlorophyll) |
| α₃ | -0.007 | AWEInsh (water index) |
| α₄ | -0.003 | LST max squared |
| α₅ | 2.93e-4 | Interaction term |

**Performance:** R²(LOOCV) = 0.487, RMSE = 0.25 µmol m⁻² s⁻¹

### Reco Model

```
Reco_11am = β₀ + β₁×LST_max + β₂×AWEInsh² + β₃×(NDWI×LST_mean) + β₄×(AWEInsh×LST_mean)
```

| Coefficient | Value | Description |
|-------------|-------|-------------|
| β₀ | 2.461 | Intercept |
| β₁ | -0.062 | LST max sensitivity |
| β₂ | -2.58e-7 | AWEInsh squared |
| β₃ | 0.395 | NDWI×LST interaction |
| β₄ | 4.45e-5 | AWEInsh×LST interaction |

**Performance:** R²(LOOCV) = 0.765, RMSE = 0.28 µmol m⁻² s⁻¹

### CH₄ Model

```
CH4_moment = γ₀ + γ₁×LST + γ₂×LST²
```

| Coefficient | Value | Description |
|-------------|-------|-------------|
| γ₀ | 0.00534 | Intercept |
| γ₁ | -0.00218 | LST linear term |
| γ₂ | 0.000212 | LST quadratic term |

**Performance:** R²(LOOCV) = 0.466, RMSE = 0.026 µmol m⁻² s⁻¹

## Diurnal Correction Factors

| Flux | Factor (F) | Description |
|------|------------|-------------|
| GPP | 0.904 ± 0.03 | 11:00 AM → Daily average |
| Reco | 0.978 ± 0.03 | 11:00 AM → Daily average |
| CH₄ | 0.876 ± 0.05 | 10:30 AM → Daily average |

## Data Availability

### Eddy Covariance Data

Half-hourly eddy covariance measurements (CO₂ and CH₄ fluxes) from the Kuibyshev Reservoir tower site are available upon request. Contact: [Project Superviser](mailto:MVKozhevnikova@kpfu.ru)

### Satellite Data

All satellite data used in this study are freely available:
- **Sentinel-2 SR**: `COPERNICUS/S2_SR` (Google Earth Engine)
- **Landsat 8/9 L2**: `LANDSAT/LC08/C02/T1_L2`, `LANDSAT/LC09/C02/T1_L2`

### Derived Products

Raster products (flux maps, masks) are available upon request or can be generated using the provided GEE scripts.

## Predictor Ranges (Clamp Values)

For stable model application, predictors should be clamped to training data ranges:

| Predictor | Min | Max |
|-----------|-----|-----|
| LST_mean | 5.11 °C | 25.35 °C |
| LST_max | 5.47 °C | 27.38 °C |
| MTCI | -4.15 | 4.83 |
| NDWI | -0.067 | 0.097 |
| AWEInsh | 2470 | 4749 |

## Citation

If you use this code or methodology, please cite:

```bibtex
@article{articleID2025,
  title={Satellite-based estimation of CO₂ and CH₄ fluxes from shallow waters of a large temperate reservoir: Integrating eddy covariance with Sentinel-2 and Landsat thermal imagery},
  author={Artur Gafurov et al},
  journal={Remote Sensing of Environment},
  year={2025(6)},
  doi={10.xxxx/xxxxx}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Eddy covariance data processing: REddyProc
- Satellite processing library: [satellite-processing](https://github.com/noaakwey/satellite-processing)
- Monte Carlo analysis: NumPy, Joblib

## Contact

For questions or collaboration inquiries, please contact:
- **Author**: [Artur Gafurov]
- **Email**: [e-mail](amgafurov@kpfu.ru)
- **Institution**: [Kazan Federal University](https://en.kpfu.ru)
