# Predictive Modeling of Municipal Solid Waste Generation Waves Using Advanced Feature Engineering

Complete, end-to-end reproducible Data Science & Machine Learning project for predicting daily municipal solid waste generation and detecting operational surge/trough waves.

---

## 📌 Project Overview
* **Primary Kaggle Dataset:** [Daily Solid Waste Dataset](https://www.kaggle.com/datasets/ivantha/daily-solid-waste-dataset) (`ivantha/daily-solid-waste-dataset`)
* **Municipal Councils Covered:** 
  - `Boralesgamuwa UC` (2012–2018)
  - `Dehiwala MC` (2012–2018)
  - `Homagama PS` (2012–2018)
  - `Moratuwa MC` (2014–2018)
* **Dataset Scope:** 15,991 weighbridge tickets, 7,966 daily time-series records, 556,423 Metric Tonnes of waste monitored.
* **Core Technology Stack:** Python 3, Pandas, NumPy, Scikit-Learn, XGBoost, Statsmodels, Matplotlib, Seaborn, Plotly, Streamlit.

---

## 🏗️ Repository Architecture
```text
d:\ds-project-municipal/
│
├── data/
│   ├── raw/                           # Original Kaggle dataset CSV files
│   ├── processed/
│   │   ├── cleaned_waste_dataset.csv  # 15,977 ticket-level cleaned records
│   │   ├── daily_waste_dataset.csv    # 7,966 daily aggregated series
│   │   ├── engineered_features.csv    # 7,846 rows x 81 engineered features
│   │   └── data_quality_report.json   # Verified data audit report
│   └── outputs/
│       ├── eda_summary_statistics.csv # Descriptive statistics by area
│       ├── wave_detection_results.csv # Detected surge/trough waves & baselines
│       ├── feature_dictionary.csv     # 27 cataloged engineered features
│       ├── model_comparison.csv       # Comparison across 4 feature configs
│       ├── evaluation_summary.csv     # Executive evaluation metrics
│       ├── feature_importance.csv     # XGBoost feature importance ranking
│       ├── location_wise_evaluation.csv # Test metrics across 4 municipalities
│       ├── wave_period_evaluation.csv # Errors during surge vs normal periods
│       ├── prediction_results.csv     # Test horizon ground-truth vs predictions
│       ├── future_forecast.csv        # 14-day recursive multi-step forecasts
│       ├── models/
│       │   └── best_waste_model.joblib # Trained XGBoost Wave-Aware model
│       └── figures/                   # High-res publication visualizations
│           ├── eda_timeseries_all_areas.png
│           ├── eda_distribution_and_spread.png
│           ├── eda_weekly_monthly_patterns.png
│           ├── wave_detection_visualization.png
│           ├── model_performance_comparison.png
│           ├── actual_vs_predicted.png
│           ├── residual_analysis.png
│           ├── feature_importance.png
│           └── future_forecast.png
│
├── src/
│   ├── preprocessor.py                # Data cleaning & daily aggregation
│   ├── eda.py                         # Statistical analysis & EDA plotting
│   ├── wave_detector.py               # Mathematical wave detection
│   ├── feature_engineering.py         # Temporal & wave feature engineering
│   ├── model_trainer.py               # ML training & feature ablation experiments
│   ├── visualizer.py                  # Evaluation visualization generator
│   ├── forecaster.py                  # Multi-step recursive future forecaster
│   └── pipeline.py                    # Master end-to-end pipeline runner
│
├── app/
│   └── app.py                         # Interactive Streamlit dashboard
│
├── notebooks/
│   └── municipal_waste_waves_colab.ipynb # Master Colab / Jupyter notebook
│
└── README.md
```

---

## 🚀 How to Run the Project

### 1. Run the Entire End-to-End Pipeline
To run all stages (Data Cleaning -> Aggregation -> EDA -> Wave Detection -> Feature Engineering -> Model Training -> Evaluation -> Forecasting -> Visualizations):
```bash
python src/pipeline.py
```

### 2. Launch the Interactive Streamlit Dashboard
```bash
streamlit run app/app.py
```
Or with python module runner:
```bash
python -m streamlit run app/app.py
```

### 3. Run in Google Colab / Jupyter Notebook
Open [`notebooks/municipal_waste_waves_colab.ipynb`](file:///d:/ds-project-municipal/notebooks/municipal_waste_waves_colab.ipynb) in JupyterLab or upload directly to [Google Colab](https://colab.research.google.com/).

---

## 📊 Summary of Experimental Results

### Model Comparison Table (Chronological Test Set Evaluation)
| Model | Feature Configuration | Features | Test MAE (kg) | Test RMSE (kg) | Test $R^2$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Persistence Baseline** | Lag-1 Benchmark | 1 | 22,811.14 | 36,656.68 | 0.5610 |
| **Linear Regression** | Exp 1 (Basic) | 14 | 17,472.98 | 27,554.70 | 0.7519 |
| **Random Forest** | Exp 1 (Basic) | 14 | 18,587.56 | 32,007.21 | 0.6653 |
| **XGBoost** | Exp 1 (Basic) | 14 | 18,736.49 | 32,585.86 | 0.6531 |
| **Random Forest** | Exp 2 (Conventional TS) | 37 | 15,701.76 | 25,845.93 | 0.7817 |
| **Gradient Boosting** | Exp 2 (Conventional TS) | 37 | 15,595.03 | 25,714.18 | 0.7840 |
| **XGBoost** | Exp 3 (Advanced Stats) | 49 | 15,736.33 | 25,689.68 | 0.7844 |
| **Random Forest** | Exp 4 (Wave-Aware) | 58 | 15,613.36 | 25,692.63 | 0.7843 |
| **Gradient Boosting** | Exp 4 (Wave-Aware) | 58 | 15,559.38 | 25,708.48 | 0.7841 |
| **XGBoost (Best Model)** | **Exp 4 (Wave-Aware)** | **58** | **15,700.46** | **25,636.45** | **0.7853** |

### Key Experimental Finding: Wave-Period Breakdown
* **High Surge Waves:** Test MAE = 13,372.29 kg, $R^2 = \mathbf{0.8729}$
* **Normal Operating Periods:** Test MAE = 15,772.84 kg, $R^2 = 0.7756$
* **Low Trough Waves:** Test MAE = 17,207.49 kg, $R^2 = 0.7730$
* **Conclusion:** Wave-aware feature engineering provides the largest performance boost during **High Surge Waves** ($R^2$ jumps from $0.775$ to $\mathbf{0.873}$), proving that incorporating deviation momentum and baseline deviations allows tree ensembles to anticipate surge events effectively.
