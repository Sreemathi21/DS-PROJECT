"""
Phase 8, 9, 10, 11, 12: Model Training, Evaluation, and Advanced Experiments
Implements:
1. Chronological Train-Val-Test Splitting (70% - 15% - 15%)
2. Feature Configurations: Exp 1 (Basic), Exp 2 (Conventional), Exp 3 (Advanced Stats), Exp 4 (Wave-Aware)
3. Models: Persistence, Linear Regression, Random Forest, Gradient Boosting, XGBoost
4. Evaluation: MAE, RMSE, R²
5. Location-wise & Wave-period error analysis
6. Feature Importance extraction
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

def prepare_feature_subsets():
    """
    Defines the 4 experimental feature sets.
    """
    area_dummies = ['area_Dehiwala MC', 'area_Homagama PS', 'area_Moratuwa MC'] # Boralesgamuwa is reference
    
    # Exp 1: Basic
    exp1 = ['lag_1', 'year', 'month', 'day', 'day_of_week', 'day_of_year', 
            'week_of_year', 'quarter', 'is_weekend', 'is_month_start', 'is_month_end'] + area_dummies
            
    # Exp 2: Conventional Time-Series
    exp2 = exp1 + [
        'lag_2', 'lag_3', 'lag_7', 'lag_14', 'lag_30',
        'diff_lag1_lag2', 'diff_lag1_lag7', 'pct_change_1_2',
        'rolling_mean_3', 'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_30',
        'rolling_std_7', 'rolling_std_14', 'rolling_min_7', 'rolling_max_7', 'rolling_median_7',
        'sin_dow', 'cos_dow', 'sin_month', 'cos_month', 'sin_doy', 'cos_doy'
    ]
    
    # Exp 3: Advanced Statistical & Trend
    exp3 = exp2 + [
        'rolling_cv_7', 'rolling_cv_14', 'rolling_range_7', 'rolling_range_14',
        'rolling_mad_7', 'ema_7', 'ema_14', 'ema_30', 'ewm_std_7',
        'trend_ema_diff', 'trend_ratio_3_14', 'trend_slope_7'
    ]
    
    # Exp 4: Wave-Aware Features
    exp4 = exp3 + [
        'wave_dev_t', 'wave_dev_lag7', 'wave_intensity_t', 'wave_amplitude_t',
        'rolling_wave_amplitude_7', 'wave_duration_t', 'is_high_surge_t', 
        'is_low_trough_t', 'wave_momentum'
    ]
    
    return {
        'Exp 1 (Basic)': exp1,
        'Exp 2 (Conventional TS)': exp2,
        'Exp 3 (Advanced Stats)': exp3,
        'Exp 4 (Wave-Aware)': exp4
    }

def split_chronological(df: pd.DataFrame, train_ratio=0.70, val_ratio=0.15):
    """
    Performs chronological splitting for each municipality independently,
    preserving strict temporal order.
    """
    train_dfs, val_dfs, test_dfs = [], [], []
    
    for area, grp in df.groupby('area'):
        grp = grp.sort_values('ticket_date').reset_index(drop=True)
        n = len(grp)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_dfs.append(grp.iloc[:n_train])
        val_dfs.append(grp.iloc[n_train:n_train + n_val])
        test_dfs.append(grp.iloc[n_train + n_val:])
        
    train_data = pd.concat(train_dfs, ignore_index=True)
    val_data = pd.concat(val_dfs, ignore_index=True)
    test_data = pd.concat(test_dfs, ignore_index=True)
    
    print(f"Chronological Split:")
    print(f"  Training Set:   {len(train_data):,} rows ({len(train_data)/len(df)*100:.1f}%)")
    print(f"  Validation Set: {len(val_data):,} rows ({len(val_data)/len(df)*100:.1f}%)")
    print(f"  Testing Set:    {len(test_data):,} rows ({len(test_data)/len(df)*100:.1f}%)")
    
    return train_data, val_data, test_data

def evaluate_predictions(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return round(mae, 2), round(rmse, 2), round(r2, 4)

def run_model_experiments(feat_df_path: str = 'data/processed/engineered_features.csv',
                          output_dir: str = 'data/outputs'):
    os.makedirs(output_dir, exist_ok=True)
    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    raw_df = pd.read_csv(feat_df_path)
    raw_df['ticket_date'] = pd.to_datetime(raw_df['ticket_date'])
    
    # One-hot encode area for modeling
    df = pd.get_dummies(raw_df, columns=['area'], drop_first=True, dtype=float)
    # Restore area text column for grouping and analysis
    df['area'] = raw_df['area']
    
    target_col = 'target_next_day_waste_kg'
    train_df, val_df, test_df = split_chronological(df)
    
    exp_configs = prepare_feature_subsets()
    
    # Store all experiment results
    results_records = []
    trained_models = {}
    test_predictions_dict = {}
    
    # ----------------------------------------------------
    # Baseline Model: Persistence (Next day = Today's lag_1)
    # ----------------------------------------------------
    y_test = test_df[target_col].values
    y_pred_persist = test_df['lag_1'].values
    p_mae, p_rmse, p_r2 = evaluate_predictions(y_test, y_pred_persist)
    results_records.append({
        'Model': 'Persistence Baseline',
        'Feature Configuration': 'Lag-1 Baseline',
        'Features_Count': 1,
        'Val_MAE_kg': round(mean_absolute_error(val_df[target_col], val_df['lag_1']), 2),
        'Test_MAE_kg': p_mae,
        'Test_RMSE_kg': p_rmse,
        'Test_R2': p_r2
    })
    test_predictions_dict['Persistence Baseline'] = y_pred_persist
    
    # Define candidate ML algorithms
    algo_constructors = {
        'Linear Regression': lambda: LinearRegression(),
        'Random Forest': lambda: RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_split=4, random_state=42, n_jobs=-1),
        'Gradient Boosting': lambda: GradientBoostingRegressor(n_estimators=120, learning_rate=0.07, max_depth=5, random_state=42),
        'XGBoost': lambda: xgb.XGBRegressor(n_estimators=120, learning_rate=0.07, max_depth=5, random_state=42, n_jobs=-1)
    }
    
    print("\n=== Training Machine Learning Models across 4 Feature Configurations ===")
    
    best_overall_score = -float('inf')
    best_model_name = None
    best_model_obj = None
    best_feature_list = None
    best_exp_name = None
    
    for exp_name, feat_cols in exp_configs.items():
        print(f"\n--- Running: {exp_name} ({len(feat_cols)} features) ---")
        
        # Prepare arrays
        X_train = train_df[feat_cols].fillna(0).values
        y_train = train_df[target_col].values
        X_val = val_df[feat_cols].fillna(0).values
        y_val = val_df[target_col].values
        X_test = test_df[feat_cols].fillna(0).values
        
        for algo_name, algo_init in algo_constructors.items():
            model = algo_init()
            model.fit(X_train, y_train)
            
            # Predict val and test
            pred_val = model.predict(X_val)
            pred_test = model.predict(X_test)
            
            val_mae, val_rmse, val_r2 = evaluate_predictions(y_val, pred_val)
            test_mae, test_rmse, test_r2 = evaluate_predictions(y_test, pred_test)
            
            key_name = f"{algo_name} | {exp_name}"
            test_predictions_dict[key_name] = pred_test
            
            results_records.append({
                'Model': algo_name,
                'Feature Configuration': exp_name,
                'Features_Count': len(feat_cols),
                'Val_MAE_kg': val_mae,
                'Test_MAE_kg': test_mae,
                'Test_RMSE_kg': test_rmse,
                'Test_R2': test_r2
            })
            
            print(f"  {algo_name:<20} | Val MAE: {val_mae:>10,.1f} kg | Test MAE: {test_mae:>10,.1f} kg | Test R²: {test_r2:.4f}")
            
            # Track best model on validation R2
            if val_r2 > best_overall_score:
                best_overall_score = val_r2
                best_model_name = algo_name
                best_exp_name = exp_name
                best_model_obj = model
                best_feature_list = feat_cols
                
    results_df = pd.DataFrame(results_records)
    results_csv = os.path.join(output_dir, 'model_comparison.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"\nModel comparison table saved to: {results_csv}")
    
    # Save the best model
    best_model_path = os.path.join(models_dir, 'best_waste_model.joblib')
    joblib.dump({
        'model': best_model_obj,
        'model_name': best_model_name,
        'feature_config': best_exp_name,
        'features': best_feature_list
    }, best_model_path)
    print(f"Best Model Selected: {best_model_name} with {best_exp_name}")
    print(f"Saved to: {best_model_path}")
    
    # ----------------------------------------------------
    # Phase 7 & 12: Feature Importance Extraction
    # ----------------------------------------------------
    if hasattr(best_model_obj, 'feature_importances_'):
        imp_df = pd.DataFrame({
            'Feature': best_feature_list,
            'Importance': best_model_obj.feature_importances_
        }).sort_values('Importance', ascending=False)
        imp_csv = os.path.join(output_dir, 'feature_importance.csv')
        imp_df.to_csv(imp_csv, index=False)
        print(f"Feature importance saved to: {imp_csv}")
        print("\nTop 10 Most Influential Features:")
        print(imp_df.head(10).to_string(index=False))
        
    # ----------------------------------------------------
    # Phase 12: Advanced Experiments
    # A. Location-wise evaluation
    # B. Wave-period evaluation (Surge vs Trough vs Normal)
    # C. Residual analysis
    # ----------------------------------------------------
    best_test_pred = best_model_obj.predict(test_df[best_feature_list].fillna(0).values)
    
    pred_results_df = test_df[['ticket_date', 'area', 'wave_regime', target_col]].copy()
    pred_results_df['predicted_waste_kg'] = best_test_pred
    pred_results_df['residual_kg'] = pred_results_df[target_col] - pred_results_df['predicted_waste_kg']
    pred_results_df['absolute_error_kg'] = pred_results_df['residual_kg'].abs()
    
    pred_csv = os.path.join(output_dir, 'prediction_results.csv')
    pred_results_df.to_csv(pred_csv, index=False)
    print(f"Test prediction results saved to: {pred_csv}")
    
    # Location-wise breakdown
    loc_breakdown = []
    for area, grp in pred_results_df.groupby('area'):
        loc_mae, loc_rmse, loc_r2 = evaluate_predictions(grp[target_col], grp['predicted_waste_kg'])
        loc_breakdown.append({
            'Municipality': area,
            'Test_Records': len(grp),
            'Mean_Actual_kg': round(grp[target_col].mean(), 2),
            'MAE_kg': loc_mae,
            'RMSE_kg': loc_rmse,
            'R2_Score': loc_r2
        })
    loc_df = pd.DataFrame(loc_breakdown)
    loc_csv = os.path.join(output_dir, 'location_wise_evaluation.csv')
    loc_df.to_csv(loc_csv, index=False)
    print(f"\nLocation-wise evaluation saved to: {loc_csv}")
    print(loc_df.to_string(index=False))
    
    # Wave-period evaluation (Normal vs High Surge vs Low Trough)
    wave_eval = []
    for regime, grp in pred_results_df.groupby('wave_regime'):
        w_mae, w_rmse, w_r2 = evaluate_predictions(grp[target_col], grp['predicted_waste_kg'])
        wave_eval.append({
            'Wave_Regime': regime,
            'Observations': len(grp),
            'Mean_Actual_kg': round(grp[target_col].mean(), 2),
            'MAE_kg': w_mae,
            'RMSE_kg': w_rmse,
            'R2_Score': w_r2
        })
    wave_eval_df = pd.DataFrame(wave_eval)
    wave_eval_csv = os.path.join(output_dir, 'wave_period_evaluation.csv')
    wave_eval_df.to_csv(wave_eval_csv, index=False)
    print(f"\nWave-period evaluation saved to: {wave_eval_csv}")
    print(wave_eval_df.to_string(index=False))
    
    return results_df, loc_df, wave_eval_df

if __name__ == '__main__':
    run_model_experiments()
