# Master Results Catalog: Municipal Solid Waste Wave Prediction

This directory (`d:/ds-project-municipal/results/`) serves as the official, standalone repository for all project results. It contains **10 detailed data tables** and **15 high-resolution (300 DPI) publication-quality figures** generated from the actual Kaggle dataset ([`ivantha/daily-solid-waste-dataset`](https://www.kaggle.com/datasets/ivantha/daily-solid-waste-dataset)).

---

## 📋 Index of 10 Data Tables (`results/tables/`)

| Table File | Title | Description | Key Metric |
| :--- | :--- | :--- | :--- |
| **[`table_01_dataset_inventory_audit.csv`](file:///D:/ds-project-municipal/results/tables/table_01_dataset_inventory_audit.csv)** | Raw Dataset Inventory & Audit | Complete row, column, date span, and missing value breakdown of all CSV files. | 15,991 raw collection records |
| **[`table_02_preprocessing_data_cleaning.csv`](file:///D:/ds-project-municipal/results/tables/table_02_preprocessing_data_cleaning.csv)** | Data Preprocessing Audit | Step-by-step record transformation from raw weighings to daily aggregated series. | 14 test tickets removed; 7,966 daily rows |
| **[`table_03_descriptive_statistics_by_area.csv`](file:///D:/ds-project-municipal/results/tables/table_03_descriptive_statistics_by_area.csv)** | Descriptive Statistics by Area | Record count, Mean, Median, Std, Min, Quantiles (25%, 75%, 95%), and Max by area. | Total 556,423 Metric Tonnes recorded |
| **[`table_04_day_of_week_and_monthly_patterns.csv`](file:///D:/ds-project-municipal/results/tables/table_04_day_of_week_and_monthly_patterns.csv)** | Day of Week & Monthly Cycles | Breakdown of average daily waste generation across Monday–Sunday and Jan–Dec. | Identifies weekly surge and lull periods |
| **[`table_05_waste_composition_by_type.csv`](file:///D:/ds-project-municipal/results/tables/table_05_waste_composition_by_type.csv)** | Waste Category Composition | Ticket counts, total tonnages, and percentage contributions by waste category. | MSW & Organic comprise >85% of volume |
| **[`table_06_wave_detection_regime_summary.csv`](file:///D:/ds-project-municipal/results/tables/table_06_wave_detection_regime_summary.csv)** | Wave Detection & Regime Summary | Frequency of High Surge Waves, Low Trough Waves, and Normal Operating Days. | 956 surge wave days detected across councils |
| **[`table_07_feature_engineering_catalog.csv`](file:///D:/ds-project-municipal/results/tables/table_07_feature_engineering_catalog.csv)** | Feature Engineering Dictionary | Complete catalog of 27 core engineered features, math definitions, and tiers. | 81 total columns in engineered matrix |
| **[`table_08_machine_learning_model_comparison.csv`](file:///D:/ds-project-municipal/results/tables/table_08_machine_learning_model_comparison.csv)** | Machine Learning Model Benchmark | Performance of 5 model families across Exp 1 (Basic) to Exp 4 (Wave-Aware). | XGBoost Exp 4 achieves Test $R^2 = 0.7853$ |
| **[`table_09_wave_period_evaluation.csv`](file:///D:/ds-project-municipal/results/tables/table_09_wave_period_evaluation.csv)** | Wave-Period Regime Evaluation | Model prediction error breakdown during surge waves vs normal operating periods. | High Surge Wave Test $R^2 = \mathbf{0.8729}$ |
| **[`table_10_future_forecast_schedule.csv`](file:///D:/ds-project-municipal/results/tables/table_10_future_forecast_schedule.csv)** | 14-Day Future Forecast Schedule | Recursive multi-step forward waste generation forecast for all 4 municipal councils. | Operational planning schedule in Tonnes |

---

## 📈 Index of 15 Publication Graphs (`results/graphs/` at 300 DPI)

| Graph File | Visual Title | Core Analytical Insight |
| :--- | :--- | :--- |
| **[`graph_01_multi_area_historical_timeseries.png`](file:///D:/ds-project-municipal/results/graphs/graph_01_multi_area_historical_timeseries.png)** | Multi-Area Historical Trajectories | 4-panel time series showing multi-year waste generation with 30-day moving average trendlines. |
| **[`graph_02_waste_distributions_and_spread.png`](file:///D:/ds-project-municipal/results/graphs/graph_02_waste_distributions_and_spread.png)** | Area Distributions & Density | Boxplots and KDE curves illustrating the vastly different population scales of the councils. |
| **[`graph_03_day_of_week_weekly_waves.png`](file:///D:/ds-project-municipal/results/graphs/graph_03_day_of_week_weekly_waves.png)** | Weekly Periodic Generation Waves | Shows weekly cyclical dynamics with weekend surges and midweek dips. |
| **[`graph_04_monthly_seasonal_cycles.png`](file:///D:/ds-project-municipal/results/graphs/graph_04_monthly_seasonal_cycles.png)** | Annual Monthly Seasonality | Monthly boxplots revealing seasonal shifts corresponding to regional monsoon and festive cycles. |
| **[`graph_05_day_vs_month_heatmap.png`](file:///D:/ds-project-municipal/results/graphs/graph_05_day_vs_month_heatmap.png)** | Weekday vs Month Generation Heatmap | Matrix of average waste generation in tonnes, identifying peak collection calendar cells. |
| **[`graph_06_waste_composition_pie_and_bar.png`](file:///D:/ds-project-municipal/results/graphs/graph_06_waste_composition_pie_and_bar.png)** | Waste Composition Breakdown | Category share pie chart alongside area-wise stacked bar charts of waste streams. |
| **[`graph_07_mathematical_wave_detection.png`](file:///D:/ds-project-municipal/results/graphs/graph_07_mathematical_wave_detection.png)** | Mathematical Wave Detection | Visualizes observed waste, 14-day EMA dynamic baseline, red surge peaks, and blue trough dips. |
| **[`graph_08_wave_deviation_intensity_distribution.png`](file:///D:/ds-project-municipal/results/graphs/graph_08_wave_deviation_intensity_distribution.png)** | Wave Deviation & Z-Score Intensity | Demonstrates the Gaussian-like distribution of capacity deviations and operational thresholds. |
| **[`graph_09_model_mae_rmse_comparison.png`](file:///D:/ds-project-municipal/results/graphs/graph_09_model_mae_rmse_comparison.png)** | Test MAE & RMSE Model Benchmark | Bar chart comparing errors across Exp 1, 2, 3, and 4 against the Persistence Baseline. |
| **[`graph_10_model_r2_explanatory_power.png`](file:///D:/ds-project-municipal/results/graphs/graph_10_model_r2_explanatory_power.png)** | Explanatory Power ($R^2$) Progression | Quantifies performance gains from $R^2 = 0.561$ (Baseline) to $0.7853$ (Wave-Aware XGBoost). |
| **[`graph_11_feature_importance_ranking.png`](file:///D:/ds-project-municipal/results/graphs/graph_11_feature_importance_ranking.png)** | Top 20 Predictive Features | Highlights dominance of `ema_14`, `rolling_mean_7`, and wave momentum features. |
| **[`graph_12_test_horizon_actual_vs_predicted.png`](file:///D:/ds-project-municipal/results/graphs/graph_12_test_horizon_actual_vs_predicted.png)** | Test Horizon Actual vs Predicted | Unseen test set overlay proving accurate alignment between ground truth and predictions. |
| **[`graph_13_residual_and_parity_analysis.png`](file:///D:/ds-project-municipal/results/graphs/graph_13_residual_and_parity_analysis.png)** | Residual Distribution & Parity Fit | Error histogram centered at zero and 1:1 parity scatter plot confirming unbiased predictions. |
| **[`graph_14_wave_period_performance_contrast.png`](file:///D:/ds-project-municipal/results/graphs/graph_14_wave_period_performance_contrast.png)** | Wave-Period Performance Contrast | Direct empirical proof: model achieves highest accuracy ($R^2 = 0.8729$) during surge waves. |
| **[`graph_15_future_14_day_forecast_trajectory.png`](file:///D:/ds-project-municipal/results/graphs/graph_15_future_14_day_forecast_trajectory.png)** | 14-Day Forward Forecast Horizon | Multi-panel operational forecast trajectories for fleet scheduling across all 4 councils. |
