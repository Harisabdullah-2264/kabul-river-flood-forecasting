# Explainable Machine Learning for Flood-Flow Prediction of the Kabul River at Nowshera

Code and data for the MSc thesis *"Explainable Machine Learning for Flood-Flow Prediction of the Kabul River at Nowshera"* (GIK Institute). The project predicts daily discharge of the Kabul River at Nowshera 1, 2, and 3 days ahead and uses SHAP to explain the predictions.

> **Before you publish this repo:** replace every occurrence of `HARIS-USERNAME` in the thesis and in this README with your actual GitHub username, and rename the repository if you wish. Then upload the three scripts and (optionally) the data into the folder structure below.

## What the project does

- **Target:** daily discharge `Q(t+1)`, `Q(t+2)`, `Q(t+3)` at Nowshera (m³/s).
- **Models:** Random Forest, XGBoost, LSTM, and a simple MEPX empirical benchmark.
- **Baseline:** persistence (tomorrow = today).
- **Metrics:** NSE, KGE, RMSE, MAE, reported at each lead time.
- **Explainability:** global, seasonal (snowmelt / monsoon / dry), and single-event SHAP.

## Data

- **Forcing:** ERA5-Land basin-mean daily precipitation, 2 m air temperature, and snow water equivalent over the Kabul basin upstream of Nowshera (transboundary, ~92,570 km²).
- **Discharge:** WAPDA daily record at Nowshera.
- **Period:** 1990–August 2022.
- **Analysis table:** `data/feature_matrix_full.csv` (one row per day; current-day discharge, lagged discharge, backward-looking rainfall accumulations, temperature and snow features, and the 1/2/3-day-ahead targets).
- **Split (leakage-free, temporal):** train 1990–2015; validation 2016–2019 (used by the LSTM only); test 2020–2022.

> If the WAPDA discharge data cannot be redistributed, keep `data/` out of the public repo and add a short note here explaining how to obtain it; the ERA5-Land forcing can be regenerated with the Google Earth Engine script.

## Repository structure

```
.
├── README.md
├── gee/
│   └── era5land_extraction.js      # Google Earth Engine: ERA5-Land basin-mean forcing extraction
├── src/
│   ├── feature_engineering.py      # builds data/feature_matrix_full.csv from raw inputs
│   └── train_and_shap.py           # train/val/test split, 4 models, metrics, SHAP analyses
├── data/
│   └── feature_matrix_full.csv     # analysis-ready table (or a note on how to obtain it)
└── requirements.txt
```

## Environment

Python 3.10 or newer. Install dependencies:

```bash
pip install -r requirements.txt
```

Suggested `requirements.txt`:

```
numpy
pandas
scikit-learn
xgboost
tensorflow
shap
matplotlib
```

(The LSTM uses TensorFlow/Keras; if you trained it with PyTorch instead, edit this list accordingly.)

## How to reproduce the results

1. **Forcing:** run `gee/era5land_extraction.js` in the Google Earth Engine Code Editor and export the basin-mean daily series.
2. **Features:** run `python src/feature_engineering.py` to merge the discharge record with the forcing and write `data/feature_matrix_full.csv`.
3. **Models, metrics, and SHAP:** run `python src/train_and_shap.py` to reproduce the train/validation/test split, train the four models, compute NSE/KGE/RMSE/MAE at the three lead times, and generate the SHAP figures reported in Chapter 5.

## Citing

If you use this code or data, please cite the thesis (add the full citation once it is finalised).
