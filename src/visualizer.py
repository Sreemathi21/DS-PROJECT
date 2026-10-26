"""
Phase 13: Prediction and Evaluation Visualizations Module
Generates publication-quality charts:
1. Model Comparison across Feature Configurations
2. Actual vs Predicted Waste Generation (Test Set)
3. Residual and Error Distribution Analysis
4. Feature Importance Ranking Plot
5. Wave-Period Performance Plot
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

def generate_evaluation_visualizations(output_dir: str = 'data/outputs'):
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. Model Comparison Chart
    comp_path = os.path.join(output_dir, 'model_comparison.csv')
    comp_df = pd.read_csv(comp_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # MAE Comparison
    sns.barplot(data=comp_df[comp_df['Model'] != 'Persistence Baseline'], 
                x='Model', y='Test_MAE_kg', hue='Feature Configuration', ax=axes[0], palette='Blues_d')
    axes[0].axhline(comp_df[comp_df['Model'] == 'Persistence Baseline']['Test_MAE_kg'].values[0], 
                   color='red', linestyle='--', label='Persistence Baseline (22,811 kg)')
    axes[0].set_title("Test MAE Across Feature Configurations (Lower is Better)", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Mean Absolute Error (kg)", fontsize=11)
    axes[0].tick_params(axis='x', rotation=15)
    axes[0].legend(title="Feature Configuration", loc='upper right')
    
    # R2 Comparison
    sns.barplot(data=comp_df[comp_df['Model'] != 'Persistence Baseline'], 
                x='Model', y='Test_R2', hue='Feature Configuration', ax=axes[1], palette='Greens_d')
    axes[1].axhline(comp_df[comp_df['Model'] == 'Persistence Baseline']['Test_R2'].values[0], 
                   color='red', linestyle='--', label='Persistence Baseline (0.5610)')
    axes[1].set_title("Test R² Across Feature Configurations (Higher is Better)", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("R² Coefficient of Determination", fontsize=11)
    axes[1].tick_params(axis='x', rotation=15)
    axes[1].legend(title="Feature Configuration", loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'model_performance_comparison.png'), dpi=200)
    plt.close()
    
    # 2. Actual vs Predicted (Test Chronological Horizon)
    pred_path = os.path.join(output_dir, 'prediction_results.csv')
    pred_df = pd.read_csv(pred_path)
    pred_df['ticket_date'] = pd.to_datetime(pred_df['ticket_date'])
    
    areas = pred_df['area'].unique()
    fig, axes = plt.subplots(len(areas), 1, figsize=(15, 12), sharex=False)
    
    for idx, area in enumerate(areas):
        sub = pred_df[pred_df['area'] == area].sort_values('ticket_date').tail(180) # Last 6 months of test
        ax = axes[idx]
        ax.plot(sub['ticket_date'], sub['target_next_day_waste_kg'] / 1000.0, 
                color='#2c3e50', label='Actual Waste (Tonnes)', linewidth=1.2, alpha=0.9)
        ax.plot(sub['ticket_date'], sub['predicted_waste_kg'] / 1000.0, 
                color='#e74c3c', label='XGBoost Wave-Aware Prediction', linewidth=1.2, linestyle='--')
        
        ax.set_title(f"Actual vs Predicted Municipal Solid Waste - {area} (Final Test Horizon)", fontsize=11, fontweight='bold')
        ax.set_ylabel("Waste (Tonnes)", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        if idx == 0:
            ax.legend(loc='upper right', frameon=True)
            
    axes[-1].set_xlabel("Date", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'actual_vs_predicted.png'), dpi=200)
    plt.close()
    
    # 3. Residual and Error Analysis
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Residual Distribution
    sns.histplot(pred_df['residual_kg'] / 1000.0, kde=True, ax=axes[0], color='#2980b9', bins=40)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=1.5)
    axes[0].set_title("Prediction Error (Residual) Distribution: Y - Ŷ (Tonnes)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Residual Error (Metric Tonnes)", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)
    
    # Scatter Actual vs Predicted
    axes[1].scatter(pred_df['target_next_day_waste_kg'] / 1000.0, pred_df['predicted_waste_kg'] / 1000.0, 
                    alpha=0.35, color='#8e44ad', s=20)
    max_val = max(pred_df['target_next_day_waste_kg'].max(), pred_df['predicted_waste_kg'].max()) / 1000.0
    axes[1].plot([0, max_val], [0, max_val], color='red', linestyle='--', label='Ideal 1:1 Fit Line')
    axes[1].set_title("Parity Plot: Actual vs Predicted Waste (Tonnes)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Actual Waste (Tonnes)", fontsize=11)
    axes[1].set_ylabel("Predicted Waste (Tonnes)", fontsize=11)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'residual_analysis.png'), dpi=200)
    plt.close()
    
    # 4. Feature Importance Top 15 Plot
    imp_path = os.path.join(output_dir, 'feature_importance.csv')
    if os.path.exists(imp_path):
        imp_df = pd.read_csv(imp_path)
        plt.figure(figsize=(10, 7))
        top_15 = imp_df.head(15).sort_values('Importance', ascending=True)
        plt.barh(top_15['Feature'], top_15['Importance'], color='#16a085', edgecolor='#117a65')
        plt.title("Top 15 Most Important Predictive Features (XGBoost)", fontsize=13, fontweight='bold')
        plt.xlabel("Relative Feature Importance Weight", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'feature_importance.png'), dpi=200)
        plt.close()
        
    print("All Phase 13 visualizations successfully generated in data/outputs/figures/")

if __name__ == '__main__':
    generate_evaluation_visualizations()
