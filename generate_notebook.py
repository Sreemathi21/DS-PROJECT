import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Cell 1: Intro Markdown
c1 = """# Predictive Modeling of Municipal Solid Waste Generation Waves Using Advanced Feature Engineering
### Master Implementation Notebook (Google Colab & Local Python)
**Dataset:** [Kaggle Daily Solid Waste Dataset](https://www.kaggle.com/datasets/ivantha/daily-solid-waste-dataset)

This notebook guides you step-by-step through:
1. Environment Setup & Dependency Installation
2. Downloading & Loading the primary Kaggle Dataset
3. Systematic Dataset Inspection
4. Data Preprocessing & Daily Waste Aggregation per Municipality
5. Comprehensive Exploratory Data Analysis (EDA)
6. Feature Engineering (Lags, Rolling Stats, Cyclical, Wave Features)
7. Waste Generation Wave Detection
8. Chronological Time-Series Splitting
9. Machine Learning Model Training (Persistence, Linear Regression, Random Forest, Gradient Boosting, XGBoost)
10. Model Evaluation & Comparison
11. Future Waste Generation Forecasting
"""
nb.cells.append(nbf.v4.new_markdown_cell(c1))

# Cell 2: Setup Code
c2 = """# Step 1: Install & Import Required Libraries
!pip install -q kagglehub pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels plotly

import os
import json
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 5)

print("Environment setup complete. Libraries imported successfully.")
"""
nb.cells.append(nbf.v4.new_code_cell(c2))

# Cell 3: Download Markdown
c3 = """## Step 2: Download the Kaggle Daily Solid Waste Dataset
We use `kagglehub.dataset_download` which directly accesses public Kaggle datasets without requiring an API key.
"""
nb.cells.append(nbf.v4.new_markdown_cell(c3))

# Cell 4: Download Code
c4 = """import kagglehub

# Download dataset
download_path = kagglehub.dataset_download('ivantha/daily-solid-waste-dataset')
print(f"Dataset downloaded to: {download_path}")

# Create directories
raw_dir = 'data/raw'
proc_dir = 'data/processed'
output_dir = 'data/outputs'
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(proc_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Copy CSV files to local project directory
for fname in os.listdir(download_path):
    src = os.path.join(download_path, fname)
    dst = os.path.join(raw_dir, fname)
    if os.path.isfile(src) and fname.endswith('.csv'):
        shutil.copy(src, dst)

print("\\nCopied CSV files:")
for f in sorted(os.listdir(raw_dir)):
    print(f" - {f}")
"""
nb.cells.append(nbf.v4.new_code_cell(c4))

# Cell 5: Inspect Markdown
c5 = """## Step 3: Inspect the Actual Dataset Files
We examine row counts, columns, data types, date ranges, and waste quantities for all 6 dataset files.
"""
nb.cells.append(nbf.v4.new_markdown_cell(c5))

# Cell 6: Inspect Code
c6 = """raw_files = sorted([f for f in os.listdir(raw_dir) if f.endswith('.csv')])

for fname in raw_files:
    fpath = os.path.join(raw_dir, fname)
    df_temp = pd.read_csv(fpath)
    print(f"==================================================")
    print(f"File: {fname}")
    print(f"Shape: {df_temp.shape[0]:,} rows x {df_temp.shape[1]} columns")
    print(f"Columns: {list(df_temp.columns)}")
    print(f"Nulls: {df_temp.isnull().sum().to_dict()}")
    if 'ticket_date' in df_temp.columns:
        dts = pd.to_datetime(df_temp['ticket_date'], errors='coerce')
        print(f"Date Span: {dts.min().date()} to {dts.max().date()} (Unique dates: {dts.nunique():,})")
    if 'net_weight_kg' in df_temp.columns:
        print(f"Net Weight (kg) -> Min: {df_temp['net_weight_kg'].min():,}, Median: {df_temp['net_weight_kg'].median():,}, Mean: {df_temp['net_weight_kg'].mean():,.1f}, Max: {df_temp['net_weight_kg'].max():,}")
    if 'area' in df_temp.columns:
        print(f"Areas: {list(df_temp['area'].unique())}")
    if 'waste_type' in df_temp.columns:
        print(f"Waste Categories ({df_temp['waste_type'].nunique()} types): {list(df_temp['waste_type'].unique()[:5])}...")
    print()
"""
nb.cells.append(nbf.v4.new_code_cell(c6))

# Cell 7: Preprocessing Markdown
c7 = """## Step 4: Data Preprocessing & Daily Waste Aggregation
In municipal operations, multiple truckloads occur daily with weighbridge tickets.
We clean the tickets and aggregate them into daily total waste (kg) per municipality.
"""
nb.cells.append(nbf.v4.new_markdown_cell(c7))

# Cell 8: Preprocessing Code
c8 = """muni_files = [
    'boralasgamuwa_2012-2018.csv',
    'dehiwala_2012-2018.csv',
    'homagama_2012-2018.csv',
    'moratuwa_2014-2018.csv'
]

raw_dfs = []
for mf in muni_files:
    fpath = os.path.join(raw_dir, mf)
    df_temp = pd.read_csv(fpath)
    df_temp['source_file'] = mf
    raw_dfs.append(df_temp)

combined_raw = pd.concat(raw_dfs, ignore_index=True)
print(f"Combined raw weighbridge records: {len(combined_raw):,}")

# Standardize strings
combined_raw['area'] = combined_raw['area'].str.strip()
combined_raw['waste_type'] = combined_raw['waste_type'].str.strip()
combined_raw['ticket_date'] = pd.to_datetime(combined_raw['ticket_date'], errors='coerce')

# Filter out empty weighbridge tickets (0 kg has no waste)
cleaned_df = combined_raw[combined_raw['net_weight_kg'] > 0].copy()
cleaned_df = cleaned_df.dropna(subset=['ticket_date']).copy()
cleaned_df.sort_values(by=['area', 'ticket_date'], inplace=True)
cleaned_df.reset_index(drop=True, inplace=True)

# Save cleaned ticket-level data
cleaned_df.to_csv(os.path.join(proc_dir, 'cleaned_waste_dataset.csv'), index=False)
print(f"Cleaned ticket-level records saved: {len(cleaned_df):,} rows")

# Daily aggregation per area
daily_df = cleaned_df.groupby(['area', 'ticket_date'], as_index=False).agg(
    total_waste_kg=('net_weight_kg', 'sum'),
    ticket_count=('net_weight_kg', 'count'),
    primary_waste_type=('waste_type', lambda x: x.mode()[0] if not x.empty else 'MSW')
)

# Extract calendar features
daily_df['year'] = daily_df['ticket_date'].dt.year
daily_df['month'] = daily_df['ticket_date'].dt.month
daily_df['day'] = daily_df['ticket_date'].dt.day
daily_df['day_name'] = daily_df['ticket_date'].dt.day_name()
daily_df['day_of_week'] = daily_df['ticket_date'].dt.dayofweek
daily_df['is_weekend'] = daily_df['day_of_week'].isin([5, 6]).astype(int)

# Save daily aggregated dataset
daily_df.to_csv(os.path.join(proc_dir, 'daily_waste_dataset.csv'), index=False)
print(f"Daily aggregated records saved: {len(daily_df):,} rows")
print("\\nFirst 5 rows of daily aggregated dataset:")
daily_df.head()
"""
nb.cells.append(nbf.v4.new_code_cell(c8))

os.makedirs('notebooks', exist_ok=True)
nb_path = 'notebooks/municipal_waste_waves_colab.ipynb'
with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook written to {nb_path}")
