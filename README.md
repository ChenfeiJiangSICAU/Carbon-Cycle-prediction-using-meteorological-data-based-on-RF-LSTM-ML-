# Carbon-Cycle-prediction-using-meteorological-data-based-on-RF-LSTM-ML
This project models relationships between atmospheric variables and carbon dynamics using machine learning. NetCDF weather data are processed to extract city-level features across pressure levels and converted into tabular form. Random Forest and XGBoost are used for nonlinear regression, evaluated with R², RMSE, and MAE.

## Overview

This project explores the relationship between atmospheric variables and carbon cycle dynamics using machine learning methods. Multi-dimensional meteorological data in NetCDF format are processed to extract city-level features for Ya'an, China, across multiple pressure levels. The resulting dataset is transformed into a structured tabular format suitable for regression modeling.


## Data Processing

The workflow includes:

- Loading multiple NetCDF files (with automatic skipping of corrupted files)
- Combining datasets along the time dimension
- Selecting a specific geographic location using nearest-neighbor matching
- Extracting variables at selected pressure levels (1000, 850, 500 hPa)
- Converting multi-dimensional data into a tabular DataFrame
- Reshaping data using pivot operations
- Flattening multi-level column indices

---

## Features

The dataset includes:

- Temperature (t)
- Relative humidity (r)
- Wind components (u, v)

All variables are extracted at multiple pressure levels and represented as independent features (e.g., `t_850`, `r_500`).

Additional time-based features:

- Hour of day
- Day of year

---

## Models

Two machine learning models are implemented:

### Random Forest
- Ensemble-based method
- Handles nonlinear relationships effectively
- Provides feature importance for interpretability

### XGBoost
- Gradient boosting framework
- Efficient and high-performing on tabular data
- Captures complex feature interactions

---

## Evaluation

Model performance is evaluated using:

- R² (coefficient of determination)
- RMSE (root mean squared error)
- MAE (mean absolute error)

---


## Requirements

```bash
pip install xarray pandas numpy scikit-learn xgboost pyarrow netcdf4
```

**Notes**：
- NetCDF files are processed using xarray
- Corrupted files are automatically detected and skipped
- Data are aligned along the time dimension before model training


**Future Work**:
- Integration of real carbon flux datasets (e.g., FLUXNET)
- Expansion to multi-location analysis
- Incorporation of temporal modeling approaches (e.g., LSTM, Transformer)
- Advanced feature engineering and uncertainty analysis
