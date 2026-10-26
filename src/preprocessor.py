"""
Data Preprocessing Module for Municipal Solid Waste Dataset
Dataset: ivantha/daily-solid-waste-dataset
"""

import os
import json
import pandas as pd
import numpy as np

def load_and_inspect_raw(raw_dir: str):
    """
    Scans the raw data folder and inspects every CSV file without prior assumptions.
    Returns a summary dictionary for every file.
    """
    csv_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
    summary = {}
    
    for fname in sorted(csv_files):
        fpath = os.path.join(raw_dir, fname)
        df = pd.read_csv(fpath)
        
        # Check date column if present
        date_col = next((c for c in df.columns if 'date' in c.lower()), None)
        date_min, date_max = None, None
        if date_col:
            dt_series = pd.to_datetime(df[date_col], errors='coerce')
            date_min = str(dt_series.min().date()) if dt_series.notnull().any() else None
            date_max = str(dt_series.max().date()) if dt_series.notnull().any() else None
            
        summary[fname] = {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "column_count": len(df.columns),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": {col: int(val) for col, val in df.isnull().sum().items()},
            "duplicate_records": int(df.duplicated().sum()),
            "date_range": [date_min, date_max] if date_min else None,
            "areas": list(df['area'].unique()) if 'area' in df.columns else None,
            "waste_types": list(df['waste_type'].unique()) if 'waste_type' in df.columns else None,
            "weight_min_kg": float(df['net_weight_kg'].min()) if 'net_weight_kg' in df.columns else None,
            "weight_max_kg": float(df['net_weight_kg'].max()) if 'net_weight_kg' in df.columns else None,
            "weight_mean_kg": float(df['net_weight_kg'].mean()) if 'net_weight_kg' in df.columns else None
        }
    return summary


def preprocess_municipal_data(raw_dir: str, proc_dir: str):
    """
    Cleans and consolidates the four municipal council records:
    1. boralasgamuwa_2012-2018.csv
    2. dehiwala_2012-2018.csv
    3. homagama_2012-2018.csv
    4. moratuwa_2014-2018.csv
    
    Outputs:
    - cleaned_waste_dataset.csv (ticket-level cleaned records)
    - daily_waste_dataset.csv (daily aggregated time-series per area)
    - data_quality_report.json
    """
    os.makedirs(proc_dir, exist_ok=True)
    muni_files = [
        'boralasgamuwa_2012-2018.csv',
        'dehiwala_2012-2018.csv',
        'homagama_2012-2018.csv',
        'moratuwa_2014-2018.csv'
    ]
    
    raw_dfs = []
    for mf in muni_files:
        fpath = os.path.join(raw_dir, mf)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Raw file {mf} not found in {raw_dir}")
        df_temp = pd.read_csv(fpath)
        df_temp['source_file'] = mf
        raw_dfs.append(df_temp)
        
    combined_raw = pd.concat(raw_dfs, ignore_index=True)
    initial_row_count = len(combined_raw)
    
    # 1. Standardize string formatting
    combined_raw['area'] = combined_raw['area'].str.strip()
    combined_raw['waste_type'] = combined_raw['waste_type'].str.strip()
    
    # 2. Convert ticket_date to standard datetime (YYYY-MM-DD)
    combined_raw['ticket_date'] = pd.to_datetime(combined_raw['ticket_date'], errors='coerce')
    invalid_dates = int(combined_raw['ticket_date'].isnull().sum())
    
    # 3. Handle zero and negative weights
    zero_weights_count = int((combined_raw['net_weight_kg'] == 0).sum())
    negative_weights_count = int((combined_raw['net_weight_kg'] < 0).sum())
    
    # Filter out weighbridge zero test tickets (0 kg has no waste)
    cleaned_df = combined_raw[combined_raw['net_weight_kg'] > 0].copy()
    cleaned_df = cleaned_df.dropna(subset=['ticket_date']).copy()
    cleaned_df.sort_values(by=['area', 'ticket_date'], inplace=True)
    cleaned_df.reset_index(drop=True, inplace=True)
    
    # Save cleaned ticket-level data
    cleaned_csv_path = os.path.join(proc_dir, 'cleaned_waste_dataset.csv')
    cleaned_df.to_csv(cleaned_csv_path, index=False)
    
    # 4. Daily Waste Aggregation per Area
    # In municipal operations, multiple truckloads occur daily. We aggregate to total daily waste (kg) per area.
    daily_df = cleaned_df.groupby(['area', 'ticket_date'], as_index=False).agg(
        total_waste_kg=('net_weight_kg', 'sum'),
        ticket_count=('net_weight_kg', 'count'),
        primary_waste_type=('waste_type', lambda x: x.mode()[0] if not x.empty else 'MSW')
    )
    
    # Add calendar fields
    daily_df['year'] = daily_df['ticket_date'].dt.year
    daily_df['month'] = daily_df['ticket_date'].dt.month
    daily_df['day'] = daily_df['ticket_date'].dt.day
    daily_df['day_name'] = daily_df['ticket_date'].dt.day_name()
    daily_df['day_of_week'] = daily_df['ticket_date'].dt.dayofweek
    daily_df['is_weekend'] = daily_df['day_of_week'].isin([5, 6]).astype(int)
    
    daily_csv_path = os.path.join(proc_dir, 'daily_waste_dataset.csv')
    daily_df.to_csv(daily_csv_path, index=False)
    
    # 5. Build comprehensive Data Quality Report
    report = {
        "dataset_name": "Kaggle Daily Solid Waste Dataset (Municipal Councils)",
        "source_files": muni_files,
        "initial_raw_rows": initial_row_count,
        "cleaned_ticket_rows": len(cleaned_df),
        "zero_weight_tickets_removed": zero_weights_count,
        "negative_weights_found": negative_weights_count,
        "invalid_dates_found": invalid_dates,
        "exact_duplicates_found": int(combined_raw.duplicated().sum()),
        "aggregated_daily_rows": len(daily_df),
        "municipalities": list(daily_df['area'].unique()),
        "municipality_summary": {},
        "total_waste_recorded_kg": float(daily_df['total_waste_kg'].sum()),
        "total_waste_metric_tonnes": float(round(daily_df['total_waste_kg'].sum() / 1000.0, 2))
    }
    
    for area in daily_df['area'].unique():
        sub = daily_df[daily_df['area'] == area]
        report["municipality_summary"][area] = {
            "daily_records_count": len(sub),
            "date_start": str(sub['ticket_date'].min().date()),
            "date_end": str(sub['ticket_date'].max().date()),
            "mean_daily_kg": float(round(sub['total_waste_kg'].mean(), 2)),
            "median_daily_kg": float(round(sub['total_waste_kg'].median(), 2)),
            "std_daily_kg": float(round(sub['total_waste_kg'].std(), 2)),
            "min_daily_kg": float(round(sub['total_waste_kg'].min(), 2)),
            "max_daily_kg": float(round(sub['total_waste_kg'].max(), 2))
        }
        
    report_path = os.path.join(proc_dir, 'data_quality_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"Data preprocessing successfully finished.")
    print(f"Cleaned dataset: {cleaned_csv_path} ({len(cleaned_df):,} rows)")
    print(f"Daily aggregated dataset: {daily_csv_path} ({len(daily_df):,} rows)")
    print(f"Data quality report: {report_path}")
    return report

if __name__ == "__main__":
    raw_path = r"d:\ds-project-municipal\data\raw"
    proc_path = r"d:\ds-project-municipal\data\processed"
    preprocess_municipal_data(raw_path, proc_path)
