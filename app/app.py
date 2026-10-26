"""
Phase 17: Interactive Streamlit Dashboard
Municipal Solid Waste Generation Waves & Predictive Modeling System
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Municipal Waste Wave Prediction & Analytics",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_all_project_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc_dir = os.path.join(base_dir, 'data', 'processed')
    out_dir = os.path.join(base_dir, 'data', 'outputs')
    
    with open(os.path.join(proc_dir, 'data_quality_report.json'), 'r') as f:
        quality_rep = json.load(f)
        
    daily_df = pd.read_csv(os.path.join(proc_dir, 'daily_waste_dataset.csv'))
    daily_df['ticket_date'] = pd.to_datetime(daily_df['ticket_date'])
    
    wave_df = pd.read_csv(os.path.join(out_dir, 'wave_detection_results.csv'))
    wave_df['ticket_date'] = pd.to_datetime(wave_df['ticket_date'])
    
    model_comp = pd.read_csv(os.path.join(out_dir, 'model_comparison.csv'))
    feat_imp = pd.read_csv(os.path.join(out_dir, 'feature_importance.csv'))
    pred_res = pd.read_csv(os.path.join(out_dir, 'prediction_results.csv'))
    pred_res['ticket_date'] = pd.to_datetime(pred_res['ticket_date'])
    future_fc = pd.read_csv(os.path.join(out_dir, 'future_forecast.csv'))
    
    return quality_rep, daily_df, wave_df, model_comp, feat_imp, pred_res, future_fc

try:
    quality_rep, daily_df, wave_df, model_comp, feat_imp, pred_res, future_fc = load_all_project_data()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading project data: {e}. Please ensure the pipeline has been run.")
    data_loaded = False

if data_loaded:
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    section = st.sidebar.radio(
        "Select Section:",
        ["1. Executive Overview", 
         "2. Historical Analysis & EDA", 
         "3. Waste Wave Detection", 
         "4. Future Forecasting", 
         "5. Model Performance & Evaluation",
         "6. Results Gallery (10 Tables & 15 Graphs)"]
    )
    
    # ----------------------------------------------------
    # SECTION 1: EXECUTIVE OVERVIEW
    # ----------------------------------------------------
    if section == "1. Executive Overview":
        st.markdown('<div class="main-header">Predictive Modeling of Municipal Solid Waste Generation Waves</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Advanced Feature Engineering & Machine Learning Forecasting System</div>', unsafe_allow_html=True)
        
        # High level KPIs
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Total Waste Recorded", f"{quality_rep['total_waste_metric_tonnes']:,.1f} Tonnes")
        with kpi2:
            st.metric("Total Historical Records", f"{quality_rep['cleaned_ticket_rows']:,} Tickets")
        with kpi3:
            st.metric("Daily Time-Series Points", f"{quality_rep['aggregated_daily_rows']:,} Days")
        with kpi4:
            st.metric("Municipalities Monitored", f"{len(quality_rep['municipalities'])} Regions")
            
        st.write("---")
        
        col_l, col_r = st.columns([1, 1])
        with col_l:
            st.subheader("Municipal Summary Table")
            summary_rows = []
            for muni, stats in quality_rep['municipality_summary'].items():
                summary_rows.append({
                    "Municipality": muni,
                    "Date Range": f"{stats['date_start']} to {stats['date_end']}",
                    "Total Days": stats['daily_records_count'],
                    "Mean Daily (Tonnes)": round(stats['mean_daily_kg'] / 1000.0, 1),
                    "Max Peak (Tonnes)": round(stats['max_daily_kg'] / 1000.0, 1)
                })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
            
        with col_r:
            st.subheader("Total Waste Generation Share")
            area_shares = daily_df.groupby('area')['total_waste_kg'].sum().reset_index()
            fig_pie = px.pie(area_shares, values='total_waste_kg', names='area', 
                             hole=0.4, color_discrete_sequence=px.colors.qualitative.Bold)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

    # ----------------------------------------------------
    # SECTION 2: HISTORICAL ANALYSIS & EDA
    # ----------------------------------------------------
    elif section == "2. Historical Analysis & EDA":
        st.header("Historical Waste Generation Analysis & Temporal Dynamics")
        
        muni_choice = st.selectbox("Select Municipality for Detailed Analysis:", quality_rep['municipalities'])
        sub_muni = daily_df[daily_df['area'] == muni_choice].sort_values('ticket_date')
        
        # Interactive Time Series
        st.subheader(f"Daily Waste Trajectory ({muni_choice})")
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=sub_muni['ticket_date'], y=sub_muni['total_waste_kg'] / 1000.0,
                                    mode='lines', name='Daily Waste (Tonnes)', line=dict(color='#2563EB', width=1)))
        # 30-day rolling
        ma30 = sub_muni.set_index('ticket_date')['total_waste_kg'].rolling(30, min_periods=7).mean() / 1000.0
        fig_ts.add_trace(go.Scatter(x=ma30.index, y=ma30.values,
                                    mode='lines', name='30-Day Moving Average', line=dict(color='#DC2626', width=2.5)))
        fig_ts.update_layout(xaxis_title="Date", yaxis_title="Waste (Metric Tonnes)", height=400)
        st.plotly_chart(fig_ts, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Day of Week Variation")
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            fig_dow = px.box(daily_df, x='day_name', y='total_waste_kg', category_orders={'day_name': day_order},
                             color='day_name', labels={'total_waste_kg': 'Waste (kg)', 'day_name': 'Day of Week'})
            fig_dow.update_layout(showlegend=False, height=380)
            st.plotly_chart(fig_dow, use_container_width=True)
            
        with col2:
            st.subheader("Monthly Seasonal Cycles")
            fig_month = px.box(daily_df, x='month', y='total_waste_kg', color='month',
                               labels={'total_waste_kg': 'Waste (kg)', 'month': 'Month (1-12)'})
            fig_month.update_layout(showlegend=False, height=380)
            st.plotly_chart(fig_month, use_container_width=True)

    # ----------------------------------------------------
    # SECTION 3: WASTE GENERATION WAVE DETECTION
    # ----------------------------------------------------
    elif section == "3. Waste Wave Detection":
        st.header("Mathematical Waste Generation Wave Detection")
        st.markdown("""
        Waste waves are mathematically identified using an Exponential Moving Average ($EMA_{14}$) baseline 
        and rolling standard deviations:
        * **High Surge Waves:** Waste generation deviates $> +1.0\\,\\sigma$ above normal capacity baseline.
        * **Low Trough Waves:** Waste generation deviates $< -1.0\\,\\sigma$ below normal capacity baseline.
        * **Normal Period:** Typical operating range within $\\pm 1.0\\,\\sigma$.
        """)
        
        muni_wave = st.selectbox("Select Municipality for Wave Breakdown:", quality_rep['municipalities'], key='wave_muni')
        wave_sub = wave_df[wave_df['area'] == muni_wave].sort_values('ticket_date').copy()
        
        # Summary counts
        reg_counts = wave_sub['wave_regime'].value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("High Surge Waves (Surges)", f"{reg_counts.get('High Surge Wave', 0)} Days")
        c2.metric("Low Trough Waves (Dips)", f"{reg_counts.get('Low Trough Wave', 0)} Days")
        c3.metric("Normal Operating Days", f"{reg_counts.get('Normal Period', 0)} Days")
        
        # Interactive Wave Plot (Last 365 Days)
        st.subheader("Wave Regimes & Baseline Deviation Plot")
        recent_waves = wave_sub.tail(365)
        
        fig_w = go.Figure()
        # Observed
        fig_w.add_trace(go.Scatter(x=recent_waves['ticket_date'], y=recent_waves['total_waste_kg'] / 1000.0,
                                   mode='lines', name='Observed (Tonnes)', line=dict(color='#475569', width=1.2)))
        # Baseline
        fig_w.add_trace(go.Scatter(x=recent_waves['ticket_date'], y=recent_waves['baseline_kg'] / 1000.0,
                                   mode='lines', name='14-Day Baseline (EMA)', line=dict(color='#D97706', width=2, dash='dash')))
        
        # Surges
        surges = recent_waves[recent_waves['wave_regime'] == 'High Surge Wave']
        if not surges.empty:
            fig_w.add_trace(go.Scatter(x=surges['ticket_date'], y=surges['total_waste_kg'] / 1000.0,
                                       mode='markers', name='High Surge Wave', marker=dict(color='#DC2626', size=8, symbol='circle')))
        # Troughs
        troughs = recent_waves[recent_waves['wave_regime'] == 'Low Trough Wave']
        if not troughs.empty:
            fig_w.add_trace(go.Scatter(x=troughs['ticket_date'], y=troughs['total_waste_kg'] / 1000.0,
                                       mode='markers', name='Low Trough Wave', marker=dict(color='#2563EB', size=8, symbol='circle')))
            
        fig_w.update_layout(xaxis_title="Date", yaxis_title="Waste (Metric Tonnes)", height=450)
        st.plotly_chart(fig_w, use_container_width=True)

    # ----------------------------------------------------
    # SECTION 4: FUTURE FORECASTING
    # ----------------------------------------------------
    elif section == "4. Future Forecasting":
        st.header("Multi-Step Future Waste Generation Forecasting")
        st.markdown("Forecast future municipal waste generation to plan truck dispatching and landfill allocation.")
        
        col_fc1, col_fc2 = st.columns([1, 2])
        with col_fc1:
            fc_area = st.selectbox("Select Municipality:", quality_rep['municipalities'], key='fc_area')
            horizon = st.slider("Forecast Horizon (Days):", min_value=7, max_value=30, value=14, step=7)
            
            sub_fc = future_fc[future_fc['area'] == fc_area].head(horizon)
            st.dataframe(sub_fc[['forecast_date', 'predicted_waste_tonnes', 'baseline_estimate_tonnes']], use_container_width=True)
            
        with col_fc2:
            st.subheader(f"{horizon}-Day Forecast Horizon ({fc_area})")
            
            # Historical tail
            hist_sub = daily_df[daily_df['area'] == fc_area].sort_values('ticket_date').tail(60)
            
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(x=hist_sub['ticket_date'], y=hist_sub['total_waste_kg'] / 1000.0,
                                        mode='lines', name='Historical (Last 60 Days)', line=dict(color='#334155', width=1.5)))
            
            fig_fc.add_trace(go.Scatter(x=pd.to_datetime(sub_fc['forecast_date']), y=sub_fc['predicted_waste_tonnes'],
                                        mode='lines+markers', name=f'{horizon}-Day AI Forecast', 
                                        line=dict(color='#DC2626', width=2.5, dash='dash'),
                                        marker=dict(size=6, color='#DC2626')))
            
            fig_fc.update_layout(xaxis_title="Date", yaxis_title="Predicted Waste (Metric Tonnes)", height=450)
            st.plotly_chart(fig_fc, use_container_width=True)

    # ----------------------------------------------------
    # SECTION 5: MODEL PERFORMANCE & EVALUATION
    # ----------------------------------------------------
    elif section == "5. Model Performance & Evaluation":
        st.header("Model Performance & Feature Ablation Experiments")
        
        st.subheader("Empirical Model Comparison")
        st.dataframe(model_comp.style.highlight_min(subset=['Test_MAE_kg', 'Test_RMSE_kg'], color='#D1FAE5')
                                      .highlight_max(subset=['Test_R2'], color='#D1FAE5'), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Model R² Comparison Across Feature Sets")
            fig_r2 = px.bar(model_comp[model_comp['Model'] != 'Persistence Baseline'], 
                            x='Model', y='Test_R2', color='Feature Configuration', barmode='group',
                            labels={'Test_R2': 'Test R² (Explanatory Power)'})
            st.plotly_chart(fig_r2, use_container_width=True)
            
        with col2:
            st.subheader("Top Predictive Features (XGBoost)")
            top_10 = feat_imp.head(10).sort_values('Importance', ascending=True)
            fig_imp = px.bar(top_10, x='Importance', y='Feature', orientation='h', 
                             color='Importance', color_continuous_scale='Teal')
            st.plotly_chart(fig_imp, use_container_width=True)

    # ----------------------------------------------------
    # SECTION 6: RESULTS GALLERY (10 TABLES & 15 GRAPHS)
    # ----------------------------------------------------
    elif section == "6. Results Gallery (10 Tables & 15 Graphs)":
        st.header("Results Gallery: Standalone Tables and Visualizations")
        st.markdown("Inspect all 10 curated data tables and 15 publication-quality (300 DPI) graphs.")
        
        tab_tables, tab_graphs = st.tabs(["📊 10 Data Tables", "📈 15 High-Resolution Graphs"])
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        res_tables_dir = os.path.join(base_dir, 'results', 'tables')
        res_graphs_dir = os.path.join(base_dir, 'results', 'graphs')
        
        with tab_tables:
            st.subheader("Curated Research Tables")
            table_files = sorted([f for f in os.listdir(res_tables_dir) if f.endswith('.csv')])
            chosen_table = st.selectbox("Select Table to View:", table_files)
            if chosen_table:
                t_df = pd.read_csv(os.path.join(res_tables_dir, chosen_table))
                st.dataframe(t_df, use_container_width=True)
                st.caption(f"Rows: {len(t_df)} | Columns: {len(t_df.columns)}")
                
        with tab_graphs:
            st.subheader("Publication-Quality Visualizations (300 DPI)")
            graph_files = sorted([f for f in os.listdir(res_graphs_dir) if f.endswith('.png')])
            chosen_graph = st.selectbox("Select Graph to View:", graph_files)
            if chosen_graph:
                st.image(os.path.join(res_graphs_dir, chosen_graph), caption=chosen_graph, use_container_width=True)

