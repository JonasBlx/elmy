# ☀️ Elmy Predictive Solar Maintenance

<p align="center">
  <img src="docs/media/windfarm.png" alt="Operational solar and wind farm at sunset" width="620"/>
</p>

> Forecast solar farm health and anticipate price swings using a refreshed, production-ready toolkit built on the ENS Challenge Data – Elmy Electricity Price Forecast dataset.

---

## ✨ Highlights
- 🔌 **Predictive maintenance ready:** engineered for solar inverter and tariff signals.
- 🧰 **Reusable toolkit:** dataclass-powered utilities for cleaning, lag features, scoring, and learning-curve inspection.
- 🧪 **Confidence built-in:** pytest coverage across preprocessing, feature engineering, metrics, and plotting.
- 📊 **Recruiter-friendly visuals:** ready-made hooks to showcase notebooks and CLI exports in interviews.

## 🗂 Repo Tour
```
.
├── src/utils/            # ETL, feature lags, scoring, plotting CLIs
├── tests/                # pytest suite validating the toolkit
├── notebooks/            # archived experiments & visual diagnostics
├── docs/media/           # drop presentation-ready images here
├── requirements.txt      # reproducible dependency set
└── README.md             # you are here 😊
```

## ⚙️ Quickstart
1. 🧪 Create an environment  
   `python -m venv env && source env/bin/activate`
2. 📦 Install dependencies  
   `pip install -r requirements.txt`
3. 📥 Fetch the [ENS Challenge Data – Elmy Electricity Price Forecast](https://challengedata.ens.fr/participants/challenges/140/) dataset → store raw files under `data/raw/`.
4. 🧼 Clean & scale  
   `python src/utils/process_data.py --input data/raw/elmy.csv --output data/interim/clean.csv --scaler standard`
5. ⏱ Build lags  
   `python src/utils/lag_data.py --input data/interim/clean.csv --target-column price_delta --n-lags 24 --output-features data/interim/lagged.csv --output-target data/interim/target.csv`
6. 📈 Inspect learning curves  
   `python src/utils/plot_learning_curves.py --model models/xgb.joblib --train data/interim/lagged_train.csv --validation data/interim/lagged_val.csv --target-column price_delta --save figures/learning_curve.png`
7. ✅ Run tests  
   `pytest`

## 🧠 Core Utilities
| Emoji | Module | What it delivers |
| --- | --- | --- |
| 🧽 | `process_data.py` | Drop columns/rows, impute, scale, and export via CLI |
| ⏳ | `lag_data.py` | Structured lag matrices with optional CLI batching |
| 🎯 | `weighted_accuracy.py` | Challenge-compliant weighted accuracy scorer |
| 📉 | `plot_learning_curves.py` | Publication-ready RMSE diagnostics (CLI + Matplotlib) |

## 💬 Talking Points
- Demonstrate the preprocessing, lag engineering, and evaluation steps with the refreshed CLIs.
- Use notebooks such as `analyse_models.ipynb` to showcase experimentation and benchmarking workflow.
- Emphasise the weighted accuracy metric alignment with the ENS challenge scoring.
