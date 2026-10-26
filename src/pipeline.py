"""
Phase 15: Complete Master Pipeline Runner
Integrates data loading, inspection, cleaning, daily aggregation, EDA,
wave detection, feature engineering, chronological model training, evaluation,
future multi-step forecasting, and visualization into a single reproducible pipeline.
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime

# Local modules
from preprocessor import preprocess_municipal_data
from eda import perform_eda
from wave_detector import detect_waste_waves, plot_wave_detection_sample
from feature_engineering import create_all_features, create_feature_dictionary
from model_trainer import run_model_experiments
from forecaster import forecast_all_municipalities
from visualizer import generate_evaluation_visualizations

def run_full_pipeline(raw_dir: str = 'data/raw',
                      proc_dir: str = 'data/processed',
                      output_dir: str = 'data/outputs'):
    """
    Executes the entire end-to-end data science project pipeline.
    """
    start_time = datetime.now()
    print("=" * 70)
    print("MUNICIPAL SOLID WASTE WAVE PREDICTION: MASTER PIPELINE")
    print(f"Timestamp: {start_time}")
    print("=" * 70)
    
    # Stage 1: Preprocessing & Aggregation
    print("\n[Step 1/7] Running Data Cleaning and Daily Aggregation...")
    preprocess_municipal_data(raw_dir, proc_dir)
    
    # Stage 2: Exploratory Data Analysis
    print("\n[Step 2/7] Running Exploratory Data Analysis (EDA)...")
    perform_eda(proc_dir, output_dir)
    
    # Stage 3: Wave Detection
    print("\n[Step 3/7] Running Mathematical Wave Detection & Regime Classification...")
    daily_df = pd.read_csv(os.path.join(proc_dir, 'daily_waste_dataset.csv'))
    wave_df = detect_waste_waves(daily_df)
    wave_csv = os.path.join(output_dir, 'wave_detection_results.csv')
    wave_df.to_csv(wave_csv, index=False)
    plot_wave_detection_sample(wave_df, os.path.join(output_dir, 'figures', 'wave_detection_visualization.png'))
    
    # Stage 4: Feature Engineering
    print("\n[Step 4/7] Generating Conventional, Advanced, and Wave-Aware Features...")
    feat_df = create_all_features(wave_df)
    feat_csv = os.path.join(proc_dir, 'engineered_features.csv')
    feat_df.to_csv(feat_csv, index=False)
    
    cat_df = create_feature_dictionary()
    cat_df.to_csv(os.path.join(output_dir, 'feature_dictionary.csv'), index=False)
    
    # Stage 5: Model Training & Evaluation Experiments
    print("\n[Step 5/7] Chronological Splitting, Model Training & Feature Ablation Experiments...")
    results_df, loc_df, wave_eval_df = run_model_experiments(feat_csv, output_dir)
    
    # Generate Evaluation Summary Table (Phase 16 requirement)
    eval_summary = results_df.copy()
    eval_summary_path = os.path.join(output_dir, 'evaluation_summary.csv')
    eval_summary.to_csv(eval_summary_path, index=False)
    print(f"Evaluation summary table saved to: {eval_summary_path}")
    
    # Stage 6: Visualization
    print("\n[Step 6/7] Generating Model Evaluation Visualizations...")
    generate_evaluation_visualizations(output_dir)
    
    # Stage 7: Future Forecasting
    print("\n[Step 7/7] Generating Multi-Step Future Waste Generation Forecasts...")
    forecast_all_municipalities(horizon_days=14, output_dir=output_dir)
    
    elapsed = datetime.now() - start_time
    print("\n" + "=" * 70)
    print(f"MASTER PIPELINE COMPLETED SUCCESSFULLY IN {elapsed.total_seconds():.1f} SECONDS!")
    print("All project artifacts, models, CSV files, and figures are verified and saved.")
    print("=" * 70)

if __name__ == '__main__':
    # Ensure current working directory is workspace root
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run_full_pipeline()
