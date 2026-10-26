"""
Phase 14: Future Waste Generation Forecasting Module
Implements multi-step recursive forecasting forward in time (e.g. 7, 14, 30 days)
for municipal planning, fleet scheduling, and disposal management.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_future_forecast(area_name: str,
                             horizon_days: int = 14,
                             feat_df_path: str = 'data/processed/engineered_features.csv',
                             model_path: str = 'data/outputs/models/best_waste_model.joblib',
                             output_dir: str = 'data/outputs') -> pd.DataFrame:
    """
    Generates a recursive multi-step forecast for the specified municipality.
    """
    model_bundle = joblib.load(model_path)
    model = model_bundle['model']
    features = model_bundle['features']
    
    # Load historical daily series
    raw_feat = pd.read_csv(feat_df_path)
    raw_feat['ticket_date'] = pd.to_datetime(raw_feat['ticket_date'])
    area_history = raw_feat[raw_feat['area'] == area_name].sort_values('ticket_date').copy()
    
    if area_history.empty:
        raise ValueError(f"Area {area_name} not found in historical data.")
        
    last_date = area_history['ticket_date'].max()
    print(f"Generating {horizon_days}-day forecast for '{area_name}' starting after {last_date.date()}...")
    
    # We maintain a running history series of waste values
    history_weights = list(area_history['total_waste_kg'].values)
    history_dates = list(area_history['ticket_date'].values)
    
    forecast_records = []
    
    for step in range(1, horizon_days + 1):
        future_date = pd.to_datetime(last_date) + pd.Timedelta(days=step)
        
        # 1. Deterministic Calendar Features
        dow = future_date.dayofweek
        month = future_date.month
        doy = future_date.dayofyear
        
        feat_dict = {
            'year': future_date.year,
            'month': month,
            'day': future_date.day,
            'day_of_week': dow,
            'day_of_year': doy,
            'week_of_year': future_date.isocalendar().week,
            'quarter': future_date.quarter,
            'is_weekend': 1 if dow in [5, 6] else 0,
            'is_month_start': 1 if future_date.is_month_start else 0,
            'is_month_end': 1 if future_date.is_month_end else 0,
            
            # Cyclical
            'sin_dow': np.sin(2 * np.pi * dow / 7.0),
            'cos_dow': np.cos(2 * np.pi * dow / 7.0),
            'sin_month': np.sin(2 * np.pi * month / 12.0),
            'cos_month': np.cos(2 * np.pi * month / 12.0),
            'sin_doy': np.sin(2 * np.pi * doy / 365.25),
            'cos_doy': np.cos(2 * np.pi * doy / 365.25),
            
            # Area Dummies
            'area_Dehiwala MC': 1.0 if area_name == 'Dehiwala MC' else 0.0,
            'area_Homagama PS': 1.0 if area_name == 'Homagama PS' else 0.0,
            'area_Moratuwa MC': 1.0 if area_name == 'Moratuwa MC' else 0.0
        }
        
        # 2. Lag Features from running history
        feat_dict['lag_1'] = history_weights[-1]
        feat_dict['lag_2'] = history_weights[-2] if len(history_weights) >= 2 else history_weights[-1]
        feat_dict['lag_3'] = history_weights[-3] if len(history_weights) >= 3 else history_weights[-1]
        feat_dict['lag_7'] = history_weights[-7] if len(history_weights) >= 7 else history_weights[-1]
        feat_dict['lag_14'] = history_weights[-14] if len(history_weights) >= 14 else history_weights[-1]
        feat_dict['lag_30'] = history_weights[-30] if len(history_weights) >= 30 else history_weights[-1]
        
        feat_dict['diff_lag1_lag2'] = feat_dict['lag_1'] - feat_dict['lag_2']
        feat_dict['diff_lag1_lag7'] = feat_dict['lag_1'] - feat_dict['lag_7']
        feat_dict['pct_change_1_2'] = (feat_dict['lag_1'] - feat_dict['lag_2']) / (feat_dict['lag_2'] + 1.0)
        
        # 3. Rolling Statistics
        recent_s = pd.Series(history_weights)
        for w in [3, 7, 14, 30]:
            sub_w = recent_s.tail(w)
            feat_dict[f'rolling_mean_{w}'] = sub_w.mean()
            feat_dict[f'rolling_std_{w}'] = sub_w.std() if len(sub_w) > 1 else 0.0
            feat_dict[f'rolling_min_{w}'] = sub_w.min()
            feat_dict[f'rolling_max_{w}'] = sub_w.max()
            feat_dict[f'rolling_median_{w}'] = sub_w.median()
            
        feat_dict['rolling_cv_7'] = feat_dict['rolling_std_7'] / (feat_dict['rolling_mean_7'] + 1.0)
        feat_dict['rolling_cv_14'] = feat_dict['rolling_std_14'] / (feat_dict['rolling_mean_14'] + 1.0)
        feat_dict['rolling_range_7'] = feat_dict['rolling_max_7'] - feat_dict['rolling_min_7']
        feat_dict['rolling_range_14'] = feat_dict['rolling_max_14'] - feat_dict['rolling_min_14']
        feat_dict['rolling_mad_7'] = (sub_w - feat_dict['rolling_median_7']).abs().median()
        
        # EMA
        feat_dict['ema_7'] = recent_s.ewm(span=7).mean().iloc[-1]
        feat_dict['ema_14'] = recent_s.ewm(span=14).mean().iloc[-1]
        feat_dict['ema_30'] = recent_s.ewm(span=30).mean().iloc[-1]
        feat_dict['ewm_std_7'] = recent_s.ewm(span=7).std().iloc[-1]
        
        feat_dict['trend_ema_diff'] = feat_dict['ema_7'] - feat_dict['ema_14']
        feat_dict['trend_ratio_3_14'] = feat_dict['rolling_mean_3'] / (feat_dict['rolling_mean_14'] + 1.0)
        feat_dict['trend_slope_7'] = (feat_dict['lag_1'] - feat_dict['lag_7']) / 7.0
        
        # 4. Wave Features
        wave_dev = feat_dict['lag_1'] - feat_dict['ema_14']
        wave_int = wave_dev / (feat_dict['rolling_std_14'] + 1e-5)
        feat_dict['wave_dev_t'] = wave_dev
        feat_dict['wave_dev_lag7'] = recent_s.iloc[-7] - feat_dict['ema_14'] if len(recent_s) >= 7 else 0.0
        feat_dict['wave_intensity_t'] = wave_int
        feat_dict['wave_amplitude_t'] = abs(wave_dev)
        feat_dict['rolling_wave_amplitude_7'] = abs(recent_s.tail(7) - feat_dict['ema_14']).mean()
        feat_dict['wave_duration_t'] = 1
        feat_dict['is_high_surge_t'] = 1 if wave_int > 1.0 else 0
        feat_dict['is_low_trough_t'] = 1 if wave_int < -1.0 else 0
        feat_dict['wave_momentum'] = wave_dev - (feat_dict['lag_2'] - feat_dict['ema_14'])
        
        # Vectorize for model
        input_row = pd.DataFrame([feat_dict])[features].fillna(0).values
        pred_val = float(model.predict(input_row)[0])
        # Municipal waste cannot be negative
        pred_val = max(pred_val, 100.0)
        
        history_weights.append(pred_val)
        history_dates.append(future_date)
        
        forecast_records.append({
            'area': area_name,
            'forecast_step': step,
            'forecast_date': str(future_date.date()),
            'predicted_waste_kg': round(pred_val, 2),
            'predicted_waste_tonnes': round(pred_val / 1000.0, 2),
            'baseline_estimate_tonnes': round(feat_dict['ema_14'] / 1000.0, 2)
        })
        
    fc_df = pd.DataFrame(forecast_records)
    return fc_df, area_history

def forecast_all_municipalities(horizon_days: int = 14, output_dir: str = 'data/outputs'):
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    feat_df_path = 'data/processed/engineered_features.csv'
    raw_feat = pd.read_csv(feat_df_path)
    areas = raw_feat['area'].unique()
    
    all_forecasts = []
    
    fig, axes = plt.subplots(len(areas), 1, figsize=(15, 12), sharex=False)
    
    for idx, area in enumerate(areas):
        fc_df, hist_df = generate_future_forecast(area, horizon_days=horizon_days)
        all_forecasts.append(fc_df)
        
        ax = axes[idx]
        recent_hist = hist_df.tail(60).copy()
        recent_hist['ticket_date'] = pd.to_datetime(recent_hist['ticket_date'])
        
        fc_dates = pd.to_datetime(fc_df['forecast_date'])
        
        # Plot history
        ax.plot(recent_hist['ticket_date'], recent_hist['total_waste_kg'] / 1000.0, 
                color='#34495e', label='Historical Observed Waste (60 Days)', linewidth=1.5)
        
        # Plot forecast
        ax.plot(fc_dates, fc_df['predicted_waste_tonnes'], 
                color='#e74c3c', marker='o', linestyle='--', linewidth=1.8, 
                label=f'{horizon_days}-Day Future Forecast')
        
        # Connecting line
        ax.plot([recent_hist['ticket_date'].iloc[-1], fc_dates.iloc[0]], 
                [recent_hist['total_waste_kg'].iloc[-1] / 1000.0, fc_df['predicted_waste_tonnes'].iloc[0]], 
                color='#e74c3c', linestyle=':')
        
        ax.set_title(f"Future Municipal Solid Waste Forecast - {area} ({horizon_days} Days Ahead)", fontsize=11, fontweight='bold')
        ax.set_ylabel("Waste (Tonnes)", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        if idx == 0:
            ax.legend(loc='upper right', frameon=True)
            
    axes[-1].set_xlabel("Date", fontsize=11)
    plt.tight_layout()
    fc_fig_path = os.path.join(fig_dir, 'future_forecast.png')
    plt.savefig(fc_fig_path, dpi=200)
    plt.close()
    
    combined_fc = pd.concat(all_forecasts, ignore_index=True)
    fc_csv = os.path.join(output_dir, 'future_forecast.csv')
    combined_fc.to_csv(fc_csv, index=False)
    print(f"\nAll municipal future forecasts saved to: {fc_csv}")
    print(f"Forecast plot saved to: {fc_fig_path}")
    print("\nForecast Sample (First 7 days for Dehiwala MC):")
    print(combined_fc[combined_fc['area'] == 'Dehiwala MC'].head(7).to_string(index=False))
    return combined_fc

if __name__ == '__main__':
    forecast_all_municipalities(horizon_days=14)
