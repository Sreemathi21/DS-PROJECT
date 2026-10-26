"""
Phase 4: Exploratory Data Analysis (EDA) Module
Calculates rigorous statistical summaries and generates high-resolution visualizations
saved to data/outputs/figures/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def perform_eda(proc_dir: str = 'data/processed', output_dir: str = 'data/outputs'):
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    daily_path = os.path.join(proc_dir, 'daily_waste_dataset.csv')
    df = pd.read_csv(daily_path)
    df['ticket_date'] = pd.to_datetime(df['ticket_date'])
    
    print("=== Phase 4: Statistical Analysis ===")
    
    # 1. Overall & Area-wise Descriptive Statistics
    stats_list = []
    
    # Combined overall
    tot_waste = df['total_waste_kg'].sum()
    mean_w = df['total_waste_kg'].mean()
    med_w = df['total_waste_kg'].median()
    std_w = df['total_waste_kg'].std()
    min_w = df['total_waste_kg'].min()
    max_w = df['total_waste_kg'].max()
    q25 = df['total_waste_kg'].quantile(0.25)
    q75 = df['total_waste_kg'].quantile(0.75)
    q95 = df['total_waste_kg'].quantile(0.95)
    
    stats_list.append({
        'Entity': 'All Municipalities Combined',
        'Record_Count': len(df),
        'Total_Waste_Tonnes': round(tot_waste / 1000.0, 2),
        'Mean_kg': round(mean_w, 2),
        'Median_kg': round(med_w, 2),
        'Std_kg': round(std_w, 2),
        'Min_kg': round(min_w, 2),
        'Q25_kg': round(q25, 2),
        'Q75_kg': round(q75, 2),
        'Q95_kg': round(q95, 2),
        'Max_kg': round(max_w, 2)
    })
    
    # Per Area statistics
    for area, grp in df.groupby('area'):
        stats_list.append({
            'Entity': area,
            'Record_Count': len(grp),
            'Total_Waste_Tonnes': round(grp['total_waste_kg'].sum() / 1000.0, 2),
            'Mean_kg': round(grp['total_waste_kg'].mean(), 2),
            'Median_kg': round(grp['total_waste_kg'].median(), 2),
            'Std_kg': round(grp['total_waste_kg'].std(), 2),
            'Min_kg': round(grp['total_waste_kg'].min(), 2),
            'Q25_kg': round(grp['total_waste_kg'].quantile(0.25), 2),
            'Q75_kg': round(grp['total_waste_kg'].quantile(0.75), 2),
            'Q95_kg': round(grp['total_waste_kg'].quantile(0.95), 2),
            'Max_kg': round(grp['total_waste_kg'].max(), 2)
        })
        
    stats_df = pd.DataFrame(stats_list)
    stats_csv_path = os.path.join(output_dir, 'eda_summary_statistics.csv')
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"Summary statistics saved to: {stats_csv_path}")
    print(stats_df.to_string())
    
    # 2. Time-Series Trajectory Plot (All Municipalities)
    print("\nGenerating Figure 1: Time-Series Trajectories by Municipality...")
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=False)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (area, grp) in enumerate(df.groupby('area')):
        grp_sorted = grp.sort_values('ticket_date')
        axes[idx].plot(grp_sorted['ticket_date'], grp_sorted['total_waste_kg'] / 1000.0, 
                       color=colors[idx], label=f'{area} (Tonnes/day)', linewidth=0.9, alpha=0.85)
        # 30-day moving average overlay
        ma30 = grp_sorted.set_index('ticket_date')['total_waste_kg'].rolling(30, min_periods=7).mean() / 1000.0
        axes[idx].plot(ma30.index, ma30.values, color='black', linewidth=1.5, label='30-Day Moving Average')
        
        axes[idx].set_title(f"Historical Daily Municipal Waste Generation - {area}", fontsize=12, fontweight='bold')
        axes[idx].set_ylabel("Waste (Metric Tonnes)", fontsize=10)
        axes[idx].grid(True, linestyle='--', alpha=0.5)
        axes[idx].legend(loc='upper right', frameon=True)
        
    axes[-1].set_xlabel("Date (Year-Month)", fontsize=11)
    plt.tight_layout()
    fig1_path = os.path.join(fig_dir, 'eda_timeseries_all_areas.png')
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    
    # 3. Waste Distribution and Boxplots by Area
    print("Generating Figure 2: Distributions and Outlier Analysis...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Boxplot
    sns.boxplot(data=df, x='area', y='total_waste_kg', ax=axes[0], palette='Set2')
    axes[0].set_title("Distribution and Spread of Daily Waste by Municipality", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Municipal Council", fontsize=11)
    axes[0].set_ylabel("Daily Waste Generation (kg)", fontsize=11)
    axes[0].tick_params(axis='x', rotation=15)
    
    # KDE distribution
    for area, grp in df.groupby('area'):
        sns.kdeplot(grp['total_waste_kg'], ax=axes[1], label=area, fill=True, alpha=0.3)
    axes[1].set_title("Density Estimation (KDE) of Daily Waste (kg)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Daily Waste Generation (kg)", fontsize=11)
    axes[1].set_ylabel("Density", fontsize=11)
    axes[1].legend(title="Municipality")
    
    plt.tight_layout()
    fig2_path = os.path.join(fig_dir, 'eda_distribution_and_spread.png')
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    
    # 4. Weekly and Monthly Variation Analysis
    print("Generating Figure 3: Weekly and Monthly Periodic Waves...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Day of Week boxplot
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    sns.boxplot(data=df, x='day_name', y='total_waste_kg', order=day_order, ax=axes[0, 0], palette='Blues_r')
    axes[0, 0].set_title("Day of Week Variation (All Municipalities)", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Day of Week", fontsize=10)
    axes[0, 0].set_ylabel("Waste Generation (kg)", fontsize=10)
    axes[0, 0].tick_params(axis='x', rotation=30)
    
    # Monthly boxplot
    sns.boxplot(data=df, x='month', y='total_waste_kg', ax=axes[0, 1], palette='Purples')
    axes[0, 1].set_title("Monthly Variation (Annual Cycle)", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("Month of Year (1=Jan, 12=Dec)", fontsize=10)
    axes[0, 1].set_ylabel("Waste Generation (kg)", fontsize=10)
    
    # Yearly trend by area
    yearly_df = df.groupby(['year', 'area'])['total_waste_kg'].sum().reset_index()
    sns.barplot(data=yearly_df, x='year', y='total_waste_kg', hue='area', ax=axes[1, 0], palette='tab10')
    axes[1, 0].set_title("Annual Total Waste Generation (kg) by Municipality", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Year", fontsize=10)
    axes[1, 0].set_ylabel("Total Annual Waste (kg)", fontsize=10)
    axes[1, 0].legend(title="Area", loc='upper left')
    
    # Heatmap of Month vs Day of Week
    pivot_heat = df.pivot_table(index='day_name', columns='month', values='total_waste_kg', aggfunc='mean')
    pivot_heat = pivot_heat.reindex(day_order)
    sns.heatmap(pivot_heat / 1000.0, cmap='YlOrRd', annot=True, fmt=".1f", ax=axes[1, 1], cbar_kws={'label': 'Mean Waste (Tonnes)'})
    axes[1, 1].set_title("Average Waste Heatmap: Day of Week vs Month (Tonnes)", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("Month of Year", fontsize=10)
    axes[1, 1].set_ylabel("Day of Week", fontsize=10)
    
    plt.tight_layout()
    fig3_path = os.path.join(fig_dir, 'eda_weekly_monthly_patterns.png')
    plt.savefig(fig3_path, dpi=200)
    plt.close()
    
    print("EDA Visualizations successfully generated in data/outputs/figures/")
    return stats_df

if __name__ == '__main__':
    perform_eda()
