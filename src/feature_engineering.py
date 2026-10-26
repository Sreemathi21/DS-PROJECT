"""
Phase 6 & Phase 7: Unified Feature Engineering Module
Builds Conventional, Advanced Statistical, and Wave-Aware Features.
Strictly prevents data leakage by calculating all rolling, difference,
and wave metrics using only historical information available at time t.
"""

import os
import pandas as pd
import numpy as np

def create_all_features(wave_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes daily waste dataframe with wave detection results and builds:
    - Target: target_next_day_waste_kg (t+1)
    - Exp 1: Basic features (Calendar + Lag 1)
    - Exp 2: Conventional Time-Series (Lags 1-30, Rolling 3-30d, Diffs, Cyclical)
    - Exp 3: Advanced Statistical (EMA 7/14/30, Rolling CV, Range, MAD, Trend)
    - Exp 4: Wave-Aware Features (Historical Deviation, Amplitude, Intensity, Duration, Wave Momentum)
    """
    df = wave_df.copy().sort_values(['area', 'ticket_date']).reset_index(drop=True)
    df['ticket_date'] = pd.to_datetime(df['ticket_date'])
    
    engineered_groups = []
    
    for area, grp in df.groupby('area'):
        grp = grp.copy().sort_values('ticket_date').reset_index(drop=True)
        y = grp['total_waste_kg']
        
        # Target variable: Next day waste generation Y(t+1)
        grp['target_next_day_waste_kg'] = y.shift(-1)
        
        # ----------------------------------------------------
        # 1. Calendar Features (Exp 1)
        # ----------------------------------------------------
        grp['year'] = grp['ticket_date'].dt.year
        grp['month'] = grp['ticket_date'].dt.month
        grp['day'] = grp['ticket_date'].dt.day
        grp['day_of_week'] = grp['ticket_date'].dt.dayofweek
        grp['day_of_year'] = grp['ticket_date'].dt.dayofyear
        grp['week_of_year'] = grp['ticket_date'].dt.isocalendar().week.astype(int)
        grp['quarter'] = grp['ticket_date'].dt.quarter
        grp['is_weekend'] = grp['day_of_week'].isin([5, 6]).astype(int)
        grp['is_month_start'] = grp['ticket_date'].dt.is_month_start.astype(int)
        grp['is_month_end'] = grp['ticket_date'].dt.is_month_end.astype(int)
        
        # Cyclical Calendar Encoding (Exp 2)
        grp['sin_dow'] = np.sin(2 * np.pi * grp['day_of_week'] / 7.0)
        grp['cos_dow'] = np.cos(2 * np.pi * grp['day_of_week'] / 7.0)
        grp['sin_month'] = np.sin(2 * np.pi * grp['month'] / 12.0)
        grp['cos_month'] = np.cos(2 * np.pi * grp['month'] / 12.0)
        grp['sin_doy'] = np.sin(2 * np.pi * grp['day_of_year'] / 365.25)
        grp['cos_doy'] = np.cos(2 * np.pi * grp['day_of_year'] / 365.25)
        
        # ----------------------------------------------------
        # 2. Lag Features (Exp 2)
        # Using historical observations at t, t-1, etc.
        # Note: at prediction time t, y_t is lag_1 for target y_{t+1}
        # ----------------------------------------------------
        grp['lag_1'] = y                    # Observed waste today (t)
        grp['lag_2'] = y.shift(1)          # t-1
        grp['lag_3'] = y.shift(2)          # t-2
        grp['lag_7'] = y.shift(6)          # t-6 (same day last week)
        grp['lag_14'] = y.shift(13)        # 2 weeks ago
        grp['lag_30'] = y.shift(29)        # ~1 month ago
        
        # Differences
        grp['diff_lag1_lag2'] = grp['lag_1'] - grp['lag_2']
        grp['diff_lag1_lag7'] = grp['lag_1'] - grp['lag_7']
        grp['pct_change_1_2'] = (grp['lag_1'] - grp['lag_2']) / (grp['lag_2'].replace(0, np.nan) + 1.0)
        
        # Rolling Statistics on y (using only up to time t)
        for w in [3, 7, 14, 30]:
            grp[f'rolling_mean_{w}'] = y.rolling(w, min_periods=2).mean()
            grp[f'rolling_std_{w}'] = y.rolling(w, min_periods=2).std()
            grp[f'rolling_min_{w}'] = y.rolling(w, min_periods=2).min()
            grp[f'rolling_max_{w}'] = y.rolling(w, min_periods=2).max()
            grp[f'rolling_median_{w}'] = y.rolling(w, min_periods=2).median()
            
        # ----------------------------------------------------
        # 3. Advanced Statistical & Trend Features (Exp 3)
        # ----------------------------------------------------
        # Rolling Coefficient of Variation (Std / Mean)
        grp['rolling_cv_7'] = grp['rolling_std_7'] / (grp['rolling_mean_7'].replace(0, np.nan) + 1.0)
        grp['rolling_cv_14'] = grp['rolling_std_14'] / (grp['rolling_mean_14'].replace(0, np.nan) + 1.0)
        
        # Rolling Range (Max - Min)
        grp['rolling_range_7'] = grp['rolling_max_7'] - grp['rolling_min_7']
        grp['rolling_range_14'] = grp['rolling_max_14'] - grp['rolling_min_14']
        
        # Rolling Median Absolute Deviation (MAD proxy)
        med_7 = grp['rolling_median_7']
        grp['rolling_mad_7'] = (y - med_7).abs().rolling(7, min_periods=2).median()
        
        # Exponential Moving Averages (EMA)
        grp['ema_7'] = y.ewm(span=7, min_periods=2).mean()
        grp['ema_14'] = y.ewm(span=14, min_periods=3).mean()
        grp['ema_30'] = y.ewm(span=30, min_periods=5).mean()
        grp['ewm_std_7'] = y.ewm(span=7, min_periods=2).std()
        
        # Trend indicators
        grp['trend_ema_diff'] = grp['ema_7'] - grp['ema_14']   # Short-term vs medium-term momentum
        grp['trend_ratio_3_14'] = grp['rolling_mean_3'] / (grp['rolling_mean_14'].replace(0, np.nan) + 1.0)
        grp['trend_slope_7'] = (grp['lag_1'] - grp['lag_7']) / 7.0
        
        # ----------------------------------------------------
        # 4. Wave-Aware Predictive Features (Exp 4)
        # ----------------------------------------------------
        # Wave metrics computed at time t (using information up to t)
        grp['wave_dev_t'] = grp['wave_deviation_kg']
        grp['wave_dev_lag7'] = grp['wave_deviation_kg'].shift(6)
        grp['wave_intensity_t'] = grp['wave_intensity_z']
        grp['wave_amplitude_t'] = grp['wave_amplitude_kg']
        grp['rolling_wave_amplitude_7'] = grp['wave_amplitude_kg'].rolling(7, min_periods=2).mean()
        grp['wave_duration_t'] = grp['wave_duration_days']
        grp['is_high_surge_t'] = grp['is_high_surge_wave']
        grp['is_low_trough_t'] = grp['is_low_trough_wave']
        
        # Wave momentum (Rate of change of deviation)
        grp['wave_momentum'] = grp['wave_dev_t'] - grp['wave_dev_t'].shift(1)
        
        engineered_groups.append(grp)
        
    full_df = pd.concat(engineered_groups, ignore_index=True)
    
    # Drop rows that have NaN in the target (the very last day of each municipality has no next day)
    # Also drop initial 30 days warmup where rolling_mean_30 has NaNs
    clean_feature_df = full_df.dropna(subset=['target_next_day_waste_kg', 'rolling_mean_30', 'lag_30']).copy()
    clean_feature_df.reset_index(drop=True, inplace=True)
    
    return clean_feature_df

def create_feature_dictionary():
    """
    Generates a structured feature catalog explaining Feature Name, Category, Definition,
    and Prediction-Time Availability.
    """
    catalog = [
        # Target
        {"Feature Name": "target_next_day_waste_kg", "Category": "Target Variable", 
         "Definition": "Total waste generated at day t+1 in kg", "Calculation Method": "Y(t+1)", "Prediction Availability": "Ground Truth Target"},
        
        # Exp 1
        {"Feature Name": "lag_1", "Category": "Exp 1: Basic", 
         "Definition": "Observed waste generation at current day t", "Calculation Method": "Y(t)", "Prediction Availability": "Available at time t"},
        {"Feature Name": "day_of_week", "Category": "Exp 1: Basic", 
         "Definition": "Day of week index (0=Mon, 6=Sun)", "Calculation Method": "ticket_date.dt.dayofweek", "Prediction Availability": "Known calendar"},
        {"Feature Name": "is_weekend", "Category": "Exp 1: Basic", 
         "Definition": "Binary flag for Saturday or Sunday", "Calculation Method": "day_of_week in [5,6]", "Prediction Availability": "Known calendar"},
        {"Feature Name": "month", "Category": "Exp 1: Basic", 
         "Definition": "Month of year (1-12)", "Calculation Method": "ticket_date.dt.month", "Prediction Availability": "Known calendar"},
        {"Feature Name": "quarter", "Category": "Exp 1: Basic", 
         "Definition": "Fiscal/calendar quarter (1-4)", "Calculation Method": "ticket_date.dt.quarter", "Prediction Availability": "Known calendar"},
        
        # Exp 2
        {"Feature Name": "lag_2, lag_3, lag_7, lag_14, lag_30", "Category": "Exp 2: Conventional TS", 
         "Definition": "Historical waste generation at days t-1, t-2, t-6, t-13, t-29", "Calculation Method": "Y.shift(k)", "Prediction Availability": "Historical only"},
        {"Feature Name": "diff_lag1_lag2", "Category": "Exp 2: Conventional TS", 
         "Definition": "Day-to-day rate of change in waste generation", "Calculation Method": "lag_1 - lag_2", "Prediction Availability": "Available at t"},
        {"Feature Name": "diff_lag1_lag7", "Category": "Exp 2: Conventional TS", 
         "Definition": "Week-over-week difference for same weekday", "Calculation Method": "lag_1 - lag_7", "Prediction Availability": "Available at t"},
        {"Feature Name": "rolling_mean_3, 7, 14, 30", "Category": "Exp 2: Conventional TS", 
         "Definition": "Short to long rolling averages over historical windows", "Calculation Method": "Rolling mean over window w", "Prediction Availability": "Available at t"},
        {"Feature Name": "rolling_std_7, 14", "Category": "Exp 2: Conventional TS", 
         "Definition": "Rolling volatility of waste generation", "Calculation Method": "Rolling std over window w", "Prediction Availability": "Available at t"},
        {"Feature Name": "sin_dow, cos_dow", "Category": "Exp 2: Conventional TS", 
         "Definition": "Cyclical sine & cosine encoding of day of week", "Calculation Method": "sin/cos(2*pi*dow/7)", "Prediction Availability": "Deterministic"},
        {"Feature Name": "sin_month, cos_month", "Category": "Exp 2: Conventional TS", 
         "Definition": "Cyclical sine & cosine encoding of month", "Calculation Method": "sin/cos(2*pi*month/12)", "Prediction Availability": "Deterministic"},
         
        # Exp 3
        {"Feature Name": "rolling_cv_7, 14", "Category": "Exp 3: Advanced Statistical", 
         "Definition": "Coefficient of Variation (relative volatility)", "Calculation Method": "rolling_std / rolling_mean", "Prediction Availability": "Available at t"},
        {"Feature Name": "rolling_range_7", "Category": "Exp 3: Advanced Statistical", 
         "Definition": "Spread between peak and trough within 7 days", "Calculation Method": "rolling_max_7 - rolling_min_7", "Prediction Availability": "Available at t"},
        {"Feature Name": "rolling_mad_7", "Category": "Exp 3: Advanced Statistical", 
         "Definition": "Median Absolute Deviation (robust dispersion)", "Calculation Method": "median(|Y - med|)", "Prediction Availability": "Available at t"},
        {"Feature Name": "ema_7, ema_14, ema_30", "Category": "Exp 3: Advanced Statistical", 
         "Definition": "Exponentially weighted moving averages", "Calculation Method": "Y.ewm(span=s).mean()", "Prediction Availability": "Available at t"},
        {"Feature Name": "trend_ema_diff", "Category": "Exp 3: Advanced Statistical", 
         "Definition": "Short vs medium EMA convergence momentum", "Calculation Method": "ema_7 - ema_14", "Prediction Availability": "Available at t"},
        {"Feature Name": "trend_slope_7", "Category": "Exp 3: Advanced Statistical", 
         "Definition": "Local linear trend slope over past 7 days", "Calculation Method": "(lag_1 - lag_7) / 7.0", "Prediction Availability": "Available at t"},
         
        # Exp 4
        {"Feature Name": "wave_dev_t", "Category": "Exp 4: Wave-Aware", 
         "Definition": "Current deviation from 14-day baseline at day t", "Calculation Method": "Observed(t) - Baseline(t)", "Prediction Availability": "Available at t"},
        {"Feature Name": "wave_dev_lag7", "Category": "Exp 4: Wave-Aware", 
         "Definition": "Baseline deviation observed 7 days prior", "Calculation Method": "wave_dev.shift(6)", "Prediction Availability": "Available at t"},
        {"Feature Name": "wave_intensity_t", "Category": "Exp 4: Wave-Aware", 
         "Definition": "Standardized wave deviation (Z-score)", "Calculation Method": "wave_dev / rolling_std", "Prediction Availability": "Available at t"},
        {"Feature Name": "rolling_wave_amplitude_7", "Category": "Exp 4: Wave-Aware", 
         "Definition": "Rolling 7-day average wave amplitude magnitude", "Calculation Method": "mean(|wave_dev|)", "Prediction Availability": "Available at t"},
        {"Feature Name": "wave_duration_t", "Category": "Exp 4: Wave-Aware", 
         "Definition": "Consecutive days the municipal series is in current wave regime", "Calculation Method": "Count of consecutive regime days", "Prediction Availability": "Available at t"},
        {"Feature Name": "is_high_surge_t", "Category": "Exp 4: Wave-Aware", 
         "Definition": "Flag indicating if current day is in a high surge surge wave (Z > 1.0)", "Calculation Method": "Binary indicator", "Prediction Availability": "Available at t"},
        {"Feature Name": "is_low_trough_t", "Category": "Exp 4: Wave-Aware", 
         "Definition": "Flag indicating if current day is in a low trough wave (Z < -1.0)", "Calculation Method": "Binary indicator", "Prediction Availability": "Available at t"},
        {"Feature Name": "wave_momentum", "Category": "Exp 4: Wave-Aware", 
         "Definition": "Rate of acceleration/deceleration of the wave deviation", "Calculation Method": "wave_dev(t) - wave_dev(t-1)", "Prediction Availability": "Available at t"}
    ]
    return pd.DataFrame(catalog)

if __name__ == '__main__':
    wave_path = 'data/outputs/wave_detection_results.csv'
    wave_df = pd.read_csv(wave_path)
    
    print("Generating engineered features...")
    feat_df = create_all_features(wave_df)
    
    out_feat_path = 'data/processed/engineered_features.csv'
    feat_df.to_csv(out_feat_path, index=False)
    print(f"Engineered feature dataset saved to: {out_feat_path} ({len(feat_df):,} rows x {feat_df.shape[1]} columns)")
    
    cat_df = create_feature_dictionary()
    dict_path = 'data/outputs/feature_dictionary.csv'
    cat_df.to_csv(dict_path, index=False)
    print(f"Feature catalog saved to: {dict_path} ({len(cat_df)} features cataloged)")
