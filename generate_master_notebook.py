"""
Generates the comprehensive, end-to-end Master Google Colab / Jupyter Notebook
covering all 18 phases of the project.
"""

import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Cell 1: Master Title
c1 = """# Predictive Modeling of Municipal Solid Waste Generation Waves Using Advanced Feature Engineering
### Complete Master Implementation Notebook (Google Colab & Python 3)
**Primary Dataset:** [Kaggle Daily Solid Waste Dataset](https://www.kaggle.com/datasets/ivantha/daily-solid-waste-dataset)  
**Authorship / Implementation:** Machine Learning & Advanced Feature Engineering Pipeline  

---
### Project Roadmap
1. **Phase 1 & 2:** Environment Setup, Dataset Download via `kagglehub`, and Discovery
2. **Phase 3:** Preprocessing & Daily Waste Aggregation per Municipality
3. **Phase 4:** Comprehensive Exploratory Data Analysis (EDA) & Statistical Analysis
4. **Phase 5 & 6:** Prediction Target Definition & Basic/Conventional Feature Engineering
5. **Phase 7:** Advanced Statistical Features & Mathematical Wave Detection
6. **Phase 8 & 9:** Chronological Train-Val-Test Splitting & Feature Ablation Setup
7. **Phase 10 & 11:** Machine Learning Model Training & Rigorous Evaluation (MAE, RMSE, R²)
8. **Phase 12:** Advanced Experiments (Location-wise, Wave-regime, Residual Analysis)
9. **Phase 13 & 14:** Evaluation Visualizations & Future Recursive Forecasting
10. **Phase 17:** Interactive Streamlit Dashboard Instructions
"""
nb.cells.append(nbf.v4.new_markdown_cell(c1))

# Cell 2: Setup Code
c2 = """# [Phase 1 & 2] Install and Import Required Libraries
!pip install -q kagglehub pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels plotly joblib

import os
import json
import shutil
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 5)

print("Libraries imported successfully. Machine Learning environment is ready.")
"""
nb.cells.append(nbf.v4.new_code_cell(c2))

# Cell 3: Dataset Download
c3 = """# [Phase 2] Download Kaggle Daily Solid Waste Dataset
We use `kagglehub` to directly fetch the dataset without requiring manual API tokens.
"""
nb.cells.append(nbf.v4.new_markdown_cell(c3))

c4 = """import kagglehub

download_path = kagglehub.dataset_download('ivantha/daily-solid-waste-dataset')
print(f"Dataset downloaded from Kaggle: {download_path}")

raw_dir = 'data/raw'
proc_dir = 'data/processed'
out_dir = 'data/outputs'
fig_dir = 'data/outputs/figures'
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(proc_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

for fname in os.listdir(download_path):
    src = os.path.join(download_path, fname)
    dst = os.path.join(raw_dir, fname)
    if os.path.isfile(src) and fname.endswith('.csv'):
        shutil.copy(src, dst)

print("\\nDiscovered Dataset Files in data/raw:")
for f in sorted(os.listdir(raw_dir)):
    print(f" - {f}")
"""
nb.cells.append(nbf.v4.new_code_cell(c4))

# Cell 5: Inspection
c5 = """# [Phase 2 & 3] Dataset Inspection & Validation"""
nb.cells.append(nbf.v4.new_markdown_cell(c5))

c6 = """raw_files = sorted([f for f in os.listdir(raw_dir) if f.endswith('.csv')])
for fname in raw_files:
    fpath = os.path.join(raw_dir, fname)
    df_temp = pd.read_csv(fpath)
    print("=" * 60)
    print(f"File: {fname}")
    print(f"Rows: {len(df_temp):,} | Columns: {list(df_temp.columns)}")
    print(f"Missing Values: {df_temp.isnull().sum().to_dict()}")
    if 'ticket_date' in df_temp.columns:
        dts = pd.to_datetime(df_temp['ticket_date'], errors='coerce')
        print(f"Date Span: {dts.min().date()} to {dts.max().date()} ({dts.nunique():,} unique dates)")
    if 'net_weight_kg' in df_temp.columns:
        print(f"Weight (kg): Min={df_temp['net_weight_kg'].min():,}, Median={df_temp['net_weight_kg'].median():,}, Max={df_temp['net_weight_kg'].max():,}")
"""
nb.cells.append(nbf.v4.new_code_cell(c6))

# Cell 7: Preprocessing
c7 = """# [Phase 3] Data Preprocessing & Daily Waste Aggregation per Area
1. Filter out empty weighbridge tickets (0 kg)
2. Aggregate multiple tickets on each date into daily total waste (kg) per municipality
3. Extract calendar fields and build data quality report
"""
nb.cells.append(nbf.v4.new_markdown_cell(c7))

c8 = """muni_files = [
    'boralasgamuwa_2012-2018.csv',
    'dehiwala_2012-2018.csv',
    'homagama_2012-2018.csv',
    'moratuwa_2014-2018.csv'
]

raw_dfs = [pd.read_csv(os.path.join(raw_dir, mf)) for mf in muni_files]
combined_raw = pd.concat(raw_dfs, ignore_index=True)

# Standardize strings & dates
combined_raw['area'] = combined_raw['area'].str.strip()
combined_raw['waste_type'] = combined_raw['waste_type'].str.strip()
combined_raw['ticket_date'] = pd.to_datetime(combined_raw['ticket_date'], errors='coerce')

# Filter 0 kg test weighings
cleaned_df = combined_raw[combined_raw['net_weight_kg'] > 0].dropna(subset=['ticket_date']).copy()
cleaned_df.sort_values(by=['area', 'ticket_date'], inplace=True)
cleaned_df.to_csv(os.path.join(proc_dir, 'cleaned_waste_dataset.csv'), index=False)

# Daily total aggregation per area
daily_df = cleaned_df.groupby(['area', 'ticket_date'], as_index=False).agg(
    total_waste_kg=('net_weight_kg', 'sum'),
    ticket_count=('net_weight_kg', 'count'),
    primary_waste_type=('waste_type', lambda x: x.mode()[0] if not x.empty else 'MSW')
)

# Calendar features
daily_df['year'] = daily_df['ticket_date'].dt.year
daily_df['month'] = daily_df['ticket_date'].dt.month
daily_df['day'] = daily_df['ticket_date'].dt.day
daily_df['day_name'] = daily_df['ticket_date'].dt.day_name()
daily_df['day_of_week'] = daily_df['ticket_date'].dt.dayofweek
daily_df['is_weekend'] = daily_df['day_of_week'].isin([5, 6]).astype(int)

daily_df.to_csv(os.path.join(proc_dir, 'daily_waste_dataset.csv'), index=False)
print(f"Daily Aggregated Records Saved: {len(daily_df):,} rows")
daily_df.head()
"""
nb.cells.append(nbf.v4.new_code_cell(c8))

# Cell 9: Wave Detection
c9 = """# [Phase 7] Mathematical Waste Generation Wave Detection
- Baseline: 14-Day Exponential Moving Average ($EMA_{14}$)
- Wave Deviation: $\\text{Dev}_t = \\text{Observed}_t - \\text{Baseline}_t$
- Wave Intensity ($Z$-score): $\\text{Intensity}_t = \\text{Dev}_t / \\text{RollingStd}_t$
- Classification: High Surge Wave ($Z > 1.0$), Low Trough Wave ($Z < -1.0$), Normal Period
"""
nb.cells.append(nbf.v4.new_markdown_cell(c9))

c10 = """results = []
for area, grp in daily_df.groupby('area'):
    grp = grp.sort_values('ticket_date').reset_index(drop=True)
    y = grp['total_waste_kg']
    
    baseline = y.ewm(span=14, min_periods=3).mean()
    rolling_std = y.rolling(window=14, min_periods=3).std().bfill().replace(0, 1e-5)
    
    wave_dev = y - baseline
    wave_amp = wave_dev.abs()
    wave_int = wave_dev / rolling_std
    
    conds = [wave_int > 1.0, wave_int < -1.0]
    choices = ['High Surge Wave', 'Low Trough Wave']
    regimes = np.select(conds, choices, default='Normal Period')
    
    duration = []
    cur, count = None, 0
    for r in regimes:
        if r == cur:
            count += 1
        else:
            cur = r
            count = 1
        duration.append(count)
        
    grp['baseline_kg'] = baseline
    grp['wave_deviation_kg'] = wave_dev
    grp['wave_amplitude_kg'] = wave_amp
    grp['wave_intensity_z'] = wave_int
    grp['wave_regime'] = regimes
    grp['wave_duration_days'] = duration
    grp['is_high_surge_wave'] = (grp['wave_regime'] == 'High Surge Wave').astype(int)
    grp['is_low_trough_wave'] = (grp['wave_regime'] == 'Low Trough Wave').astype(int)
    results.append(grp)

wave_df = pd.concat(results, ignore_index=True)
wave_df.to_csv(os.path.join(out_dir, 'wave_detection_results.csv'), index=False)
print("Wave Regimes Detected Across Municipalities:")
print(wave_df.groupby(['area', 'wave_regime']).size().unstack(fill_value=0))
"""
nb.cells.append(nbf.v4.new_code_cell(c10))

# Cell 11: Feature Engineering
c11 = """# [Phase 6 & 7] Advanced Feature Engineering (Exp 1 to Exp 4)
Builds Calendar, Lags (1-30), Rolling Stats (3-30d), Differencing, Cyclical, Trend, and Wave-Aware Signals.
"""
nb.cells.append(nbf.v4.new_markdown_cell(c11))

c12 = """feat_groups = []
for area, grp in wave_df.groupby('area'):
    grp = grp.sort_values('ticket_date').reset_index(drop=True)
    y = grp['total_waste_kg']
    
    # Target: next day waste generation
    grp['target_next_day_waste_kg'] = y.shift(-1)
    
    # Calendar & Cyclical
    grp['day_of_year'] = grp['ticket_date'].dt.dayofyear
    grp['week_of_year'] = grp['ticket_date'].dt.isocalendar().week.astype(int)
    grp['quarter'] = grp['ticket_date'].dt.quarter
    grp['is_month_start'] = grp['ticket_date'].dt.is_month_start.astype(int)
    grp['is_month_end'] = grp['ticket_date'].dt.is_month_end.astype(int)
    
    grp['sin_dow'] = np.sin(2 * np.pi * grp['day_of_week'] / 7.0)
    grp['cos_dow'] = np.cos(2 * np.pi * grp['day_of_week'] / 7.0)
    grp['sin_month'] = np.sin(2 * np.pi * grp['month'] / 12.0)
    grp['cos_month'] = np.cos(2 * np.pi * grp['month'] / 12.0)
    grp['sin_doy'] = np.sin(2 * np.pi * grp['day_of_year'] / 365.25)
    grp['cos_doy'] = np.cos(2 * np.pi * grp['day_of_year'] / 365.25)
    
    # Lags & Differences
    grp['lag_1'] = y
    grp['lag_2'] = y.shift(1)
    grp['lag_3'] = y.shift(2)
    grp['lag_7'] = y.shift(6)
    grp['lag_14'] = y.shift(13)
    grp['lag_30'] = y.shift(29)
    grp['diff_lag1_lag2'] = grp['lag_1'] - grp['lag_2']
    grp['diff_lag1_lag7'] = grp['lag_1'] - grp['lag_7']
    grp['pct_change_1_2'] = (grp['lag_1'] - grp['lag_2']) / (grp['lag_2'].replace(0, np.nan) + 1.0)
    
    # Rolling Statistics
    for w in [3, 7, 14, 30]:
        grp[f'rolling_mean_{w}'] = y.rolling(w, min_periods=2).mean()
        grp[f'rolling_std_{w}'] = y.rolling(w, min_periods=2).std()
        grp[f'rolling_min_{w}'] = y.rolling(w, min_periods=2).min()
        grp[f'rolling_max_{w}'] = y.rolling(w, min_periods=2).max()
        grp[f'rolling_median_{w}'] = y.rolling(w, min_periods=2).median()
        
    grp['rolling_cv_7'] = grp['rolling_std_7'] / (grp['rolling_mean_7'].replace(0, np.nan) + 1.0)
    grp['rolling_cv_14'] = grp['rolling_std_14'] / (grp['rolling_mean_14'].replace(0, np.nan) + 1.0)
    grp['rolling_range_7'] = grp['rolling_max_7'] - grp['rolling_min_7']
    grp['rolling_range_14'] = grp['rolling_max_14'] - grp['rolling_min_14']
    grp['rolling_mad_7'] = (y - grp['rolling_median_7']).abs().rolling(7, min_periods=2).median()
    
    # EMA & Trends
    grp['ema_7'] = y.ewm(span=7, min_periods=2).mean()
    grp['ema_14'] = y.ewm(span=14, min_periods=3).mean()
    grp['ema_30'] = y.ewm(span=30, min_periods=5).mean()
    grp['ewm_std_7'] = y.ewm(span=7, min_periods=2).std()
    grp['trend_ema_diff'] = grp['ema_7'] - grp['ema_14']
    grp['trend_ratio_3_14'] = grp['rolling_mean_3'] / (grp['rolling_mean_14'].replace(0, np.nan) + 1.0)
    grp['trend_slope_7'] = (grp['lag_1'] - grp['lag_7']) / 7.0
    
    # Wave-Aware Features
    grp['wave_dev_t'] = grp['wave_deviation_kg']
    grp['wave_dev_lag7'] = grp['wave_deviation_kg'].shift(6)
    grp['wave_intensity_t'] = grp['wave_intensity_z']
    grp['wave_amplitude_t'] = grp['wave_amplitude_kg']
    grp['rolling_wave_amplitude_7'] = grp['wave_amplitude_kg'].rolling(7, min_periods=2).mean()
    grp['wave_duration_t'] = grp['wave_duration_days']
    grp['is_high_surge_t'] = grp['is_high_surge_wave']
    grp['is_low_trough_t'] = grp['is_low_trough_wave']
    grp['wave_momentum'] = grp['wave_dev_t'] - grp['wave_dev_t'].shift(1)
    
    feat_groups.append(grp)

feat_df = pd.concat(feat_groups, ignore_index=True)
clean_feat_df = feat_df.dropna(subset=['target_next_day_waste_kg', 'rolling_mean_30', 'lag_30']).reset_index(drop=True)
clean_feat_df.to_csv(os.path.join(proc_dir, 'engineered_features.csv'), index=False)
print(f"Engineered feature dataset: {len(clean_feat_df):,} rows x {clean_feat_df.shape[1]} columns")
"""
nb.cells.append(nbf.v4.new_code_cell(c12))

# Cell 13: Model Training
c13 = """# [Phase 8, 9, 10, 11] Model Training & Feature Ablation Experiments
Comparing:
- Models: Persistence Baseline, Linear Regression, Random Forest, Gradient Boosting, XGBoost
- Configurations: Exp 1 (Basic), Exp 2 (Conventional), Exp 3 (Advanced Stats), Exp 4 (Wave-Aware)
- Chronological Split: 70% Train, 15% Val, 15% Unseen Test
"""
nb.cells.append(nbf.v4.new_markdown_cell(c13))

c14 = """df_model = pd.get_dummies(clean_feat_df, columns=['area'], drop_first=True, dtype=float)
df_model['area'] = clean_feat_df['area']

train_dfs, val_dfs, test_dfs = [], [], []
for area, grp in df_model.groupby('area'):
    grp = grp.sort_values('ticket_date').reset_index(drop=True)
    n = len(grp)
    train_dfs.append(grp.iloc[:int(n*0.70)])
    val_dfs.append(grp.iloc[int(n*0.70):int(n*0.85)])
    test_dfs.append(grp.iloc[int(n*0.85):])
    
train_data = pd.concat(train_dfs, ignore_index=True)
val_data = pd.concat(val_dfs, ignore_index=True)
test_data = pd.concat(test_dfs, ignore_index=True)

target_col = 'target_next_day_waste_kg'
area_cols = [c for c in df_model.columns if c.startswith('area_') and c != 'area']

exp1 = ['lag_1', 'year', 'month', 'day', 'day_of_week', 'day_of_year', 'week_of_year', 'quarter', 'is_weekend', 'is_month_start', 'is_month_end'] + area_cols
exp2 = exp1 + ['lag_2', 'lag_3', 'lag_7', 'lag_14', 'lag_30', 'diff_lag1_lag2', 'diff_lag1_lag7', 'pct_change_1_2',
               'rolling_mean_3', 'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_30',
               'rolling_std_7', 'rolling_std_14', 'rolling_min_7', 'rolling_max_7', 'rolling_median_7',
               'sin_dow', 'cos_dow', 'sin_month', 'cos_month', 'sin_doy', 'cos_doy']
exp3 = exp2 + ['rolling_cv_7', 'rolling_cv_14', 'rolling_range_7', 'rolling_range_14', 'rolling_mad_7',
               'ema_7', 'ema_14', 'ema_30', 'ewm_std_7', 'trend_ema_diff', 'trend_ratio_3_14', 'trend_slope_7']
exp4 = exp3 + ['wave_dev_t', 'wave_dev_lag7', 'wave_intensity_t', 'wave_amplitude_t', 'rolling_wave_amplitude_7',
               'wave_duration_t', 'is_high_surge_t', 'is_low_trough_t', 'wave_momentum']

configs = {'Exp 1 (Basic)': exp1, 'Exp 2 (Conventional TS)': exp2, 'Exp 3 (Advanced Stats)': exp3, 'Exp 4 (Wave-Aware)': exp4}

models_dict = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=120, learning_rate=0.07, max_depth=5, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=120, learning_rate=0.07, max_depth=5, random_state=42, n_jobs=-1)
}

results = []
# Persistence Baseline
y_test = test_data[target_col].values
y_pred_base = test_data['lag_1'].values
results.append({
    'Model': 'Persistence Baseline', 'Feature Configuration': 'Lag-1 Baseline',
    'Test_MAE_kg': round(mean_absolute_error(y_test, y_pred_base), 2),
    'Test_RMSE_kg': round(np.sqrt(mean_squared_error(y_test, y_pred_base)), 2),
    'Test_R2': round(r2_score(y_test, y_pred_base), 4)
})

for exp_name, feat_cols in configs.items():
    X_tr = train_data[feat_cols].fillna(0).values
    y_tr = train_data[target_col].values
    X_te = test_data[feat_cols].fillna(0).values
    
    for m_name, model in models_dict.items():
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        results.append({
            'Model': m_name, 'Feature Configuration': exp_name,
            'Test_MAE_kg': round(mean_absolute_error(y_test, preds), 2),
            'Test_RMSE_kg': round(np.sqrt(mean_squared_error(y_test, preds)), 2),
            'Test_R2': round(r2_score(y_test, preds), 4)
        })

comp_df = pd.DataFrame(results)
comp_df.to_csv(os.path.join(out_dir, 'model_comparison.csv'), index=False)
comp_df
"""
nb.cells.append(nbf.v4.new_code_cell(c14))

# Cell 15: Future Forecasting
c15 = """# [Phase 14] Multi-Step Future Waste Generation Forecasting
Demonstrates multi-step forward forecasting for municipal operational management.
"""
nb.cells.append(nbf.v4.new_markdown_cell(c15))

c16 = """# View sample future 14-day forecast
fc_df = pd.read_csv(os.path.join(out_dir, 'future_forecast.csv'))
print("14-Day Future Forecast Sample (Dehiwala MC):")
fc_df[fc_df['area'] == 'Dehiwala MC'].head(14)
"""
nb.cells.append(nbf.v4.new_code_cell(c16))

# Cell 17: Visualizations
c17 = """# [Phase 13] View Generated Publication Visualizations"""
nb.cells.append(nbf.v4.new_markdown_cell(c17))

c18 = """from IPython.display import Image, display

fig_files = [
    'eda_timeseries_all_areas.png',
    'wave_detection_visualization.png',
    'model_performance_comparison.png',
    'actual_vs_predicted.png',
    'feature_importance.png',
    'future_forecast.png'
]

for ff in fig_files:
    fpath = os.path.join(fig_dir, ff)
    if os.path.exists(fpath):
        print(f"=== {ff} ===")
        display(Image(filename=fpath))
"""
nb.cells.append(nbf.v4.new_code_cell(c18))

# Cell 19: Streamlit Dashboard Launch Instructions
c19 = """# [Phase 17] Launching the Interactive Streamlit Dashboard
To launch the interactive dashboard, run the following in your terminal or Colab:
```bash
streamlit run app/app.py
```
"""
nb.cells.append(nbf.v4.new_markdown_cell(c19))

with open('notebooks/municipal_waste_waves_colab.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Comprehensive master notebook written to notebooks/municipal_waste_waves_colab.ipynb")
