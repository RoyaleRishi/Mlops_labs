# C2W4 Lab 1 — Weather Data Feature Engineering (Modified)

Modified version of the [C2W4 Lab 1](https://github.com/https-deeplearning-ai/machine-learning-engineering-for-production-public/tree/main/course2/week4-ungraded-labs) feature engineering lab using `tf.Transform` + Apache Beam on the Jena Climate Dataset.

## Changes from Base Lab

| | Base Lab | This Submission |
|---|---|---|
| **Prediction target** | `T (degC)` — temperature | `wv (m/s)` — wind speed |
| **Features dropped** | 5 features correlated with temperature | 7 features (different subset) based on wind speed correlation analysis |
| **Normalization** | `tft.scale_to_0_1` (min-max) | `tft.scale_to_z_score` (standardization) |
| **History window** | 120 steps (5 days) | 72 steps (3 days) |
| **Forecast horizon** | 12 hours | 6 hours |
| **Batch size** | 72 | 48 |
| **Output tensor shape** | `(72, 120, 13)` | `(48, 72, 12)` |

## Rationale

**Target change:** Wind speed prediction is physically driven by pressure gradients and air density rather than thermal features, requiring a different feature selection strategy. The vapor pressure cluster (`VPmax`, `VPact`, `sh`, `H2OC`) is strongly correlated with temperature but has near-zero correlation with wind speed and is dropped here. `p (mbar)` and `rho (g/m**3)` become more central as they govern atmospheric dynamics.

**Normalization change:** Wind speed is right-skewed (~1.5 skewness) and bounded at zero. Min-max scaling would compress the useful range due to occasional high-wind outliers. Z-score normalization is more robust here and is consistent with how the transform graph would be applied at serving time.

**Window change:** Temperature has strong multi-day thermal inertia — a 5-day lookback is well-motivated. Wind speed is more locally variable, so a 3-day window is sufficient and reduces sequence length by 40%.

## Usage

```bash
pip install -r requirements.txt

# Download dataset
# https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip
mkdir -p data && mv jena_climate_2009_2016.csv data/

jupyter notebook C2_W4_Lab_1_WeatherData_WindSpeed.ipynb
```
