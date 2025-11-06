# ⚡ Elmy Electricity Price Delta Forecasting

<p align="center">
  <img src="docs/media/windfarm.png" alt="European renewable generation assets" width="560"/>
</p>

Forecast the day-ahead gap between Intraday and SPOT electricity prices using the dataset released for the [Challenge Data – Prédiction de prix de l'électricité par Elmy](https://challengedata.ens.fr/participants/challenges/140/) competition. This repository refresh brings clean utilities, tests, and documentation for candidates who want to present a tidy, production-aware solution.

---

## 🔍 What’s Included
- 📦 **Reusable data tools** for column pruning, imputation, scaling, and lagged feature creation.
- 🎯 **Challenge-specific scoring** via the Weighted Accuracy metric required by Elmy.
- 🧪 **Pytest coverage** validating preprocessing, feature engineering, metrics, and plotting helpers.
- 🗂 **Curated structure** separating utilities (`src/utils`), experiments (`notebooks`), and visuals (`docs/media`).

## 🗺 Repository Map
```
.
├── src/utils/            # Core scripts for ETL, lag features, scoring, and plotting
├── tests/                # pytest modules covering critical behaviours
├── notebooks/            # Challenge exploration & modelling notebooks (archived)
├── docs/media/           # Visual assets (hero image, charts)
├── requirements.txt      # Version-pinned dependencies
└── README.md             # Project overview for reviewers
```

## 🚀 Quickstart
1. **Create and activate a virtual environment**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: .\env\Scripts\activate
   ```
2. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```
3. **Fetch the Challenge Data files** and place the raw CSVs in `data/raw/` (gitignored).
4. **Clean and scale the dataset**
   ```bash
   python src/utils/process_data.py \
     --input data/raw/elmy_train.csv \
     --output data/interim/train_clean.csv \
     --scaler standard
   ```
5. **Create lagged features for supervised models**
   ```bash
   python src/utils/lag_data.py \
     --input data/interim/train_clean.csv \
     --target-column spot_id_delta \
     --n-lags 24 \
     --output-features data/interim/train_lagged.csv \
     --output-target data/interim/train_target.csv
   ```
6. **Evaluate model learning curves** once a model is trained and saved with `joblib.dump`
   ```bash
   python src/utils/plot_learning_curves.py \
     --model models/gradient_boost.joblib \
     --train data/interim/train_lagged.csv \
     --validation data/interim/val_lagged.csv \
     --target-column spot_id_delta \
     --save figures/learning_curve.png
   ```
7. **Run the test suite**
   ```bash
   pytest
   ```

## 🛠 Utility Summary
| Module | Purpose | Example Use |
| --- | --- | --- |
| `process_data.py` | Configurable preprocessing (row/column drops, imputation, scaling) with CLI entry point. | `ProcessConfig(scaler="standard")` |
| `lag_data.py` | Generate aligned lagged feature matrices for time-series models. | `LagConfig(n_lags=24)` |
| `weighted_accuracy.py` | Weighted accuracy scorer enforcing challenge evaluation rules. | `weighted_accuracy_scorer` |
| `plot_learning_curves.py` | Headless Matplotlib helper for RMSE learning curves. | CLI export to `figures/learning_curve.png` |

## 🧪 Quality Guardrails
- Tests run headless with Matplotlib’s `Agg` backend (see `tests/conftest.py`).
- Utility functions raise descriptive errors on shape mismatches and unsupported options.
- Requirements are version-pinned to keep reproductions consistent across machines.

## 🗣 Talking Points for Interviews
- Walk through how the CLI suite standardises preprocessing and lag generation for Intraday vs SPOT price modelling.
- Highlight the weighted accuracy implementation to show metric alignment with the official leaderboard.
- Share notebook insights (e.g., `analyse_models.ipynb`, `enrich_and_visualize_data.ipynb`) to demonstrate exploratory depth and feature reasoning.
