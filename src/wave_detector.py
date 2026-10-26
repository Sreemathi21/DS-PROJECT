"""
Phase 7: Waste Generation Wave Detection Module
Mathematically detects waste generation waves (surges and troughs), calculates baselines,
wave deviations, amplitudes, intensities, and wave regimes.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def detect_waste_waves(daily_df: pd.DataFrame, 
                       baseline_span: int = 14, 
                       threshold_z: float = 1.0) -> pd.DataFrame:
    """
    Detects waves per municipality:
    - Baseline: 14-day Exponential Moving Average (EMA)
    - Wave Deviation: Observed - Baseline
    - Wave Intensity (Z-score): Deviation / Rolling 14-day Std
    - Wave Classification: 'High Surge Wave', 'Low Trough Wave', 'Normal Period'
    - Wave Duration: consecutive days in the same regime
    """
    df = daily_df.copy().sort_values(['area', 'ticket_date']).reset_index(drop=True)
    
    # Containers for results per municipality
    results = []
    
    for area, grp in df.groupby('area'):
        grp = grp.copy().sort_values('ticket_date').reset_index(drop=True)
        y = grp['total_waste_kg']
        
        # 1. Baseline estimation using 14-day EMA
        # min_periods=3 ensures robust baseline even at start
        baseline = y.ewm(span=baseline_span, min_periods=3).mean()
        rolling_std = y.rolling(window=baseline_span, min_periods=3).std().bfill()
        rolling_std = rolling_std.replace(0, 1e-5)
        
        # 2. Wave Deviation and Amplitude
        wave_deviation = y - baseline
        wave_amplitude = wave_deviation.abs()
        wave_intensity = wave_deviation / rolling_std
        
        # 3. Wave Classification
        # Statistical threshold: |Z| > threshold_z
        conditions = [
            wave_intensity > threshold_z,
            wave_intensity < -threshold_z
        ]
        choices = ['High Surge Wave', 'Low Trough Wave']
        wave_class = np.select(conditions, choices, default='Normal Period')
        
        # 4. Wave Duration (consecutive days in current regime)
        duration = []
        cur_regime = None
        cur_count = 0
        for wc in wave_class:
            if wc == cur_regime:
                cur_count += 1
            else:
                cur_regime = wc
                cur_count = 1
            duration.append(cur_count)
            
        grp['baseline_kg'] = baseline
        grp['wave_deviation_kg'] = wave_deviation
        grp['wave_amplitude_kg'] = wave_amplitude
        grp['wave_intensity_z'] = wave_intensity
        grp['wave_regime'] = wave_class
        grp['wave_duration_days'] = duration
        
        # Binary flags for easy ML feature ingestion
        grp['is_high_surge_wave'] = (grp['wave_regime'] == 'High Surge Wave').astype(int)
        grp['is_low_trough_wave'] = (grp['wave_regime'] == 'Low Trough Wave').astype(int)
        
        results.append(grp)
        
    out_df = pd.concat(results, ignore_index=True)
    return out_df

def plot_wave_detection_sample(wave_df: pd.DataFrame, 
                               output_fig_path: str = 'data/outputs/figures/wave_detection_visualization.png'):
    """
    Plots detected waves for each municipality over a 1-year window to clearly visualize
    the actual waste, estimated baseline, and color-coded surge/trough waves.
    """
    os.makedirs(os.path.dirname(output_fig_path), exist_ok=True)
    areas = wave_df['area'].unique()
    fig, axes = plt.subplots(len(areas), 1, figsize=(15, 12), sharex=False)
    
    for idx, area in enumerate(areas):
        sub = wave_df[wave_df['area'] == area].sort_values('ticket_date').copy()
        sub['ticket_date'] = pd.to_datetime(sub['ticket_date'])
        
        # Select 2016 (a complete year with good continuity)
        sample = sub[(sub['ticket_date'] >= '2016-01-01') & (sub['ticket_date'] <= '2016-12-31')].copy()
        if len(sample) < 50:
            sample = sub.tail(300).copy()
            
        ax = axes[idx]
        dates = sample['ticket_date']
        waste_t = sample['total_waste_kg'] / 1000.0
        base_t = sample['baseline_kg'] / 1000.0
        
        # Plot actual and baseline
        ax.plot(dates, waste_t, color='#2c3e50', linewidth=1.1, label='Observed Waste (Tonnes/day)', alpha=0.9)
        ax.plot(dates, base_t, color='#e67e22', linewidth=2.0, linestyle='--', label='14-Day Baseline (EMA)')
        
        # Highlight High Surge Waves
        surge = sample[sample['wave_regime'] == 'High Surge Wave']
        if not surge.empty:
            ax.scatter(surge['ticket_date'], surge['total_waste_kg'] / 1000.0, 
                       color='#e74c3c', s=25, label='High Surge Wave (Peak)', zorder=5)
            
        # Highlight Low Trough Waves
        trough = sample[sample['wave_regime'] == 'Low Trough Wave']
        if not trough.empty:
            ax.scatter(trough['ticket_date'], trough['total_waste_kg'] / 1000.0, 
                       color='#3498db', s=25, label='Low Trough Wave (Dip)', zorder=5)
            
        ax.set_title(f"Mathematical Wave Detection & Baseline Tracking - {area} (2016)", fontsize=11, fontweight='bold')
        ax.set_ylabel("Waste (Tonnes)", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        if idx == 0:
            ax.legend(loc='upper right', frameon=True, ncol=4)
            
    axes[-1].set_xlabel("Date", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_fig_path, dpi=200)
    plt.close()
    print(f"Wave detection plot saved to: {output_fig_path}")

if __name__ == '__main__':
    daily_path = 'data/processed/daily_waste_dataset.csv'
    df = pd.read_csv(daily_path)
    wave_results = detect_waste_waves(df)
    
    out_csv = 'data/outputs/wave_detection_results.csv'
    wave_results.to_csv(out_csv, index=False)
    print(f"Wave detection results saved to: {out_csv} ({len(wave_results):,} rows)")
    
    regime_counts = wave_results.groupby(['area', 'wave_regime']).size().unstack(fill_value=0)
    print("\nDetected Wave Regimes Distribution:")
    print(regime_counts.to_string())
    
    plot_wave_detection_sample(wave_results)
