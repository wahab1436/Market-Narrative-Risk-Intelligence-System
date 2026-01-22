"""
Market Narrative Risk Intelligence Dashboard - Professional Version
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta
import sys
import os

# Page configuration MUST be the first Streamlit command
st.set_page_config(
    page_title="Market Risk Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .warning-text {
        color: #DC2626;
        font-weight: bold;
    }
    .success-text {
        color: #059669;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">Market Narrative Risk Intelligence Dashboard</h1>', unsafe_allow_html=True)
st.markdown("""
Advanced analytics for market stress detection and risk regime identification based on real-time news analysis.
""")

# Sidebar
with st.sidebar:
    st.header("Dashboard Controls")
    
    # Data status
    st.subheader("Data Status")
    data_dir = Path("data/gold")
    if data_dir.exists():
        pred_files = list(data_dir.glob("predictions_*.parquet"))
        feature_files = list(data_dir.glob("features_*.parquet"))
        
        if pred_files:
            latest_pred = max(pred_files, key=lambda x: x.stat().st_mtime)
            st.markdown(f'<p class="success-text">Latest predictions loaded: {latest_pred.name}</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="warning-text">No prediction files found</p>', unsafe_allow_html=True)
            
        if feature_files:
            latest_feature = max(feature_files, key=lambda x: x.stat().st_mtime)
            st.markdown(f'Feature data: {latest_feature.name}')
    else:
        st.error("Data directory not found")
    
    st.divider()
    
    # Date range filter
    st.subheader("Date Range Filter")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now().date() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", datetime.now().date())
    
    # Display metrics
    st.divider()
    st.subheader("System Metrics")
    
    if 'df' in st.session_state and not st.session_state.df.empty:
        df = st.session_state.df
        st.metric("Total Articles", f"{len(df):,}")
        
        if 'timestamp' in df.columns:
            min_date = df['timestamp'].min().date()
            max_date = df['timestamp'].max().date()
            st.metric("Date Range", f"{min_date} to {max_date}")
        
        if 'is_anomaly' in df.columns:
            anomalies = df['is_anomaly'].sum()
            st.metric("Anomalies Detected", f"{anomalies:,}", f"{(anomalies/len(df)*100):.1f}%")

# Main content
@st.cache_data
def load_latest_predictions():
    """Load the latest predictions file."""
    try:
        data_dir = Path("data/gold")
        pred_files = list(data_dir.glob("predictions_*.parquet"))
        
        if not pred_files:
            # Fallback to features file
            feature_files = list(data_dir.glob("features_*.parquet"))
            if feature_files:
                latest_file = max(feature_files, key=lambda x: x.stat().st_mtime)
                st.info(f"Using features file: {latest_file.name}")
                return pd.read_parquet(latest_file)
            else:
                st.error("No data files found. Please run the pipeline first.")
                return pd.DataFrame()
        
        # Load latest predictions
        latest_file = max(pred_files, key=lambda x: x.stat().st_mtime)
        return pd.read_parquet(latest_file)
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load data
if 'df' not in st.session_state:
    with st.spinner("Loading latest data..."):
        df = load_latest_predictions()
        if not df.empty:
            st.session_state.df = df
            st.session_state.data_loaded = True
        else:
            st.session_state.data_loaded = False
else:
    df = st.session_state.df
    st.session_state.data_loaded = True

# Display data if loaded
if st.session_state.data_loaded and not df.empty:
    # Filter by date if available
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
        filtered_df = df[mask].copy()
    else:
        filtered_df = df.copy()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Predictions", "Anomalies", "Raw Data"])
    
    with tab1:
        st.header("System Overview")
        
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'weighted_stress_score' in filtered_df.columns:
                current = filtered_df['weighted_stress_score'].iloc[-1] if len(filtered_df) > 0 else 0
                avg = filtered_df['weighted_stress_score'].mean()
                st.metric("Current Stress", f"{current:.2f}", f"{current-avg:+.2f}")
        
        with col2:
            if 'xgboost_risk_regime' in filtered_df.columns:
                regime_counts = filtered_df['xgboost_risk_regime'].value_counts()
                if not regime_counts.empty:
                    current_regime = regime_counts.index[0]
                    st.metric("Dominant Risk Regime", current_regime.upper())
        
        with col3:
            if 'is_anomaly' in filtered_df.columns:
                anomalies = filtered_df['is_anomaly'].sum()
                rate = (anomalies/len(filtered_df)*100) if len(filtered_df) > 0 else 0
                st.metric("Anomalies", f"{anomalies:,}", f"{rate:.1f}%")
        
        with col4:
            total_articles = len(filtered_df)
            st.metric("Articles Analyzed", f"{total_articles:,}")
        
        st.divider()
        
        # Chart 1: Stress Score Timeline
        if 'timestamp' in filtered_df.columns and 'weighted_stress_score' in filtered_df.columns:
            st.subheader("Market Stress Timeline")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=filtered_df['timestamp'], 
                y=filtered_df['weighted_stress_score'],
                mode='lines',
                name='Stress Score',
                line=dict(color='#3B82F6', width=2)
            ))
            
            # Add model predictions if available
            model_predictions = [col for col in filtered_df.columns if 'prediction' in col]
            for i, pred_col in enumerate(model_predictions[:3]):  # Show up to 3 models
                fig.add_trace(go.Scatter(
                    x=filtered_df['timestamp'],
                    y=filtered_df[pred_col],
                    mode='lines',
                    name=pred_col.replace('_', ' ').title(),
                    line=dict(dash='dash', width=1.5),
                    opacity=0.6
                ))
            
            # Add anomaly markers if available
            if 'is_anomaly' in filtered_df.columns:
                anomalies_df = filtered_df[filtered_df['is_anomaly'] == 1]
                if not anomalies_df.empty:
                    fig.add_trace(go.Scatter(
                        x=anomalies_df['timestamp'],
                        y=anomalies_df['weighted_stress_score'],
                        mode='markers',
                        name='Anomalies',
                        marker=dict(color='#DC2626', size=8, symbol='x')
                    ))
            
            fig.update_layout(
                title="Stress Score and Model Predictions",
                height=400,
                xaxis_title="Date",
                yaxis_title="Stress Score",
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Chart 2: Risk Regime Analysis
        if 'xgboost_risk_regime' in filtered_df.columns:
            st.subheader("Risk Regime Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                regime_counts = filtered_df['xgboost_risk_regime'].value_counts()
                fig = px.pie(
                    values=regime_counts.values,
                    names=regime_counts.index,
                    title="Risk Regime Distribution",
                    color_discrete_sequence=['#10B981', '#F59E0B', '#DC2626']  # Green, Orange, Red
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Show as stacked bar chart over time
                if 'timestamp' in filtered_df.columns:
                    filtered_df['date'] = filtered_df['timestamp'].dt.date
                    daily_regimes = filtered_df.groupby(
                        ['date', 'xgboost_risk_regime']
                    ).size().unstack(fill_value=0)
                    
                    fig = go.Figure()
                    colors = {'low': '#10B981', 'medium': '#F59E0B', 'high': '#DC2626'}
                    
                    for regime in daily_regimes.columns:
                        fig.add_trace(go.Bar(
                            x=daily_regimes.index,
                            y=daily_regimes[regime],
                            name=regime.upper(),
                            marker_color=colors.get(regime, '#6B7280'),
                            opacity=0.8
                        ))
                    
                    fig.update_layout(
                        title="Risk Regimes Over Time",
                        barmode='stack',
                        height=350,
                        xaxis_title="Date",
                        yaxis_title="Number of Articles",
                        legend_title="Risk Regime"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Model Predictions Analysis")
        
        # Find all prediction columns
        pred_cols = [col for col in filtered_df.columns if any(x in col for x in [
            'prediction', 'regime', 'forecast', 'similarity', 'probability'
        ])]
        
        if pred_cols:
            st.subheader("Available Model Outputs")
            
            # Create a grid of model predictions
            num_cols = 3
            rows = [pred_cols[i:i + num_cols] for i in range(0, len(pred_cols), num_cols)]
            
            for row in rows:
                cols = st.columns(num_cols)
                for idx, pred_col in enumerate(row):
                    if idx < len(cols):
                        with cols[idx]:
                            if pred_col in filtered_df.columns:
                                with st.container():
                                    st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                                    st.markdown(f"**{pred_col.replace('_', ' ').title()}**")
                                    
                                    if filtered_df[pred_col].dtype in ['float64', 'int64']:
                                        mean_val = filtered_df[pred_col].mean()
                                        std_val = filtered_df[pred_col].std()
                                        st.metric("Mean", f"{mean_val:.3f}", f"Std: {std_val:.3f}")
                                    
                                    elif filtered_df[pred_col].dtype == 'object':
                                        unique_vals = filtered_df[pred_col].nunique()
                                        st.metric("Unique Values", f"{unique_vals}")
                                    
                                    st.markdown(f'</div>', unsafe_allow_html=True)
            
            # Detailed view for selected prediction
            st.divider()
            st.subheader("Detailed Analysis")
            
            selected_pred = st.selectbox(
                "Select prediction for detailed analysis:",
                options=pred_cols
            )
            
            if selected_pred:
                col1, col2 = st.columns(2)
                
                with col1:
                    if filtered_df[selected_pred].dtype in ['float64', 'int64']:
                        fig = px.histogram(
                            filtered_df, 
                            x=selected_pred,
                            title=f"Distribution of {selected_pred}",
                            nbins=30,
                            color_discrete_sequence=['#3B82F6']
                        )
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif filtered_df[selected_pred].dtype == 'object':
                        value_counts = filtered_df[selected_pred].value_counts()
                        fig = px.bar(
                            x=value_counts.index,
                            y=value_counts.values,
                            title=f"Value Counts for {selected_pred}",
                            color_discrete_sequence=['#3B82F6']
                        )
                        fig.update_layout(height=300, xaxis_title=selected_pred, yaxis_title="Count")
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if 'timestamp' in filtered_df.columns and selected_pred in filtered_df.columns:
                        fig = go.Figure()
                        
                        if filtered_df[selected_pred].dtype in ['float64', 'int64']:
                            fig.add_trace(go.Scatter(
                                x=filtered_df['timestamp'],
                                y=filtered_df[selected_pred],
                                mode='lines',
                                name=selected_pred,
                                line=dict(color='#3B82F6', width=2)
                            ))
                        elif filtered_df[selected_pred].dtype == 'object':
                            # For categorical data, show as scatter
                            unique_vals = filtered_df[selected_pred].unique()
                            colors = px.colors.qualitative.Set3
                            
                            for i, val in enumerate(unique_vals[:5]):  # Limit to 5 categories
                                mask = filtered_df[selected_pred] == val
                                fig.add_trace(go.Scatter(
                                    x=filtered_df.loc[mask, 'timestamp'],
                                    y=[i] * mask.sum(),
                                    mode='markers',
                                    name=str(val),
                                    marker=dict(color=colors[i % len(colors)], size=8)
                                ))
                        
                        fig.update_layout(
                            title=f"{selected_pred} Over Time",
                            height=300,
                            xaxis_title="Date",
                            yaxis_title=selected_pred.replace('_', ' ').title()
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("No prediction columns found in the data.")
    
    with tab3:
        st.header("Anomaly Detection Analysis")
        
        if 'is_anomaly' in filtered_df.columns:
            anomalies = filtered_df[filtered_df['is_anomaly'] == 1]
            
            if not anomalies.empty:
                st.subheader(f"Detected Anomalies: {len(anomalies)}")
                
                # Anomaly statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    if 'anomaly_score' in anomalies.columns:
                        st.metric("Avg Anomaly Score", f"{anomalies['anomaly_score'].mean():.3f}")
                with col2:
                    if 'timestamp' in anomalies.columns:
                        days_with_anomalies = anomalies['timestamp'].dt.date.nunique()
                        st.metric("Days with Anomalies", f"{days_with_anomalies}")
                with col3:
                    anomaly_rate = (len(anomalies) / len(filtered_df)) * 100
                    st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
                
                # Anomaly details table
                st.subheader("Anomaly Details")
                anomaly_cols = ['timestamp']
                if 'headline' in filtered_df.columns:
                    anomaly_cols.append('headline')
                if 'weighted_stress_score' in filtered_df.columns:
                    anomaly_cols.append('weighted_stress_score')
                if 'anomaly_score' in filtered_df.columns:
                    anomaly_cols.append('anomaly_score')
                if 'xgboost_risk_regime' in filtered_df.columns:
                    anomaly_cols.append('xgboost_risk_regime')
                
                display_df = anomalies[anomaly_cols].copy()
                if 'timestamp' in display_df.columns:
                    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'headline': st.column_config.TextColumn(
                            'Headline',
                            width='large'
                        ),
                        'weighted_stress_score': st.column_config.NumberColumn(
                            'Stress Score',
                            format='%.3f'
                        ),
                        'anomaly_score': st.column_config.NumberColumn(
                            'Anomaly Score',
                            format='%.3f'
                        ),
                        'xgboost_risk_regime': st.column_config.TextColumn(
                            'Risk Regime'
                        )
                    }
                )
                
                # Anomaly timeline chart
                if 'timestamp' in filtered_df.columns and 'anomaly_score' in filtered_df.columns:
                    st.subheader("Anomaly Score Timeline")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=filtered_df['timestamp'],
                        y=filtered_df['anomaly_score'],
                        mode='lines',
                        name='Anomaly Score',
                        line=dict(color='#F59E0B', width=2)
                    ))
                    
                    # Highlight anomalies
                    fig.add_trace(go.Scatter(
                        x=anomalies['timestamp'],
                        y=anomalies['anomaly_score'],
                        mode='markers',
                        name='Detected Anomalies',
                        marker=dict(color='#DC2626', size=8, symbol='x')
                    ))
                    
                    # Add threshold line (95th percentile)
                    threshold = filtered_df['anomaly_score'].quantile(0.95)
                    fig.add_hline(
                        y=threshold,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"95th Percentile: {threshold:.3f}",
                        annotation_position="top right"
                    )
                    
                    fig.update_layout(
                        height=400,
                        xaxis_title="Date",
                        yaxis_title="Anomaly Score",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No anomalies detected in the selected period.")
        else:
            st.info("Anomaly detection data not available.")
    
    with tab4:
        st.header("Raw Data Explorer")
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=500
        )
        
        # Data statistics
        with st.expander("Data Statistics", expanded=False):
            numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                st.subheader("Numeric Column Statistics")
                stats_df = filtered_df[numeric_cols].describe().T
                st.dataframe(stats_df, use_container_width=True)
            
            categorical_cols = filtered_df.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                st.subheader("Categorical Column Summary")
                for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
                    unique_count = filtered_df[col].nunique()
                    st.metric(f"{col} Unique Values", f"{unique_count}")
        
        # Column explorer
        with st.expander("Column Information", expanded=False):
            st.subheader("Column Details")
            col_info = []
            for col in filtered_df.columns:
                col_info.append({
                    'Column': col,
                    'Type': str(filtered_df[col].dtype),
                    'Non-Null': filtered_df[col].count(),
                    'Null': filtered_df[col].isnull().sum(),
                    'Unique': filtered_df[col].nunique()
                })
            col_df = pd.DataFrame(col_info)
            st.dataframe(col_df, use_container_width=True, hide_index=True)
        
        # Download option
        st.divider()
        st.subheader("Data Export")
        
        col1, col2 = st.columns(2)
        with col1:
            export_format = st.selectbox(
                "Export Format",
                ["CSV", "Excel", "JSON"],
                index=0
            )
        
        with col2:
            export_scope = st.radio(
                "Export Scope",
                ["Current View (Filtered)", "All Available Data"],
                horizontal=True
            )
        
        if export_scope == "Current View (Filtered)":
            export_data = filtered_df
        else:
            export_data = df
        
        if export_format == "CSV":
            csv_data = export_data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"market_risk_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        elif export_format == "Excel":
            # Note: This requires openpyxl
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                export_data.to_excel(writer, index=False, sheet_name='MarketRiskData')
            st.download_button(
                label="Download Excel",
                data=buffer.getvalue(),
                file_name=f"market_risk_data_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        elif export_format == "JSON":
            json_data = export_data.to_json(orient='records', indent=2)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"market_risk_data_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #6B7280; font-size: 0.9em;'>
    Market Narrative Risk Intelligence System • Version 1.0.0 • Data updated: {}
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)

else:
    # If no data is loaded
    st.error("No data available. Please run the pipeline first.")
    
    with st.expander("Troubleshooting Guide", expanded=True):
        st.markdown("""
        ### Steps to Load Data:
        
        1. **Run the pipeline**:
        ```
        python main.py
        ```
        
        2. **Check data directory structure**:
        Ensure the following directory exists with prediction files:
        ```
        data/gold/predictions_*.parquet
        ```
        
        3. **Verify pipeline execution**:
        Check the pipeline logs for any errors during execution.
        
        4. **Refresh dashboard**:
        After running the pipeline, refresh this page to load new data.
        
        ### Expected Data Structure:
        - Bronze: Raw scraped data
        - Silver: Cleaned and validated data  
        - Gold: Feature-engineered data with predictions
        
        ### Common Issues:
        - Pipeline not completing all steps
        - Permission issues with data directories
        - Insufficient data for model training
        - Memory limitations during processing
        """)
        
        # Show directory structure if data directory exists
        if Path("data").exists():
            st.subheader("Current Data Directory Structure")
            tree_output = []
            for root, dirs, files in os.walk("data"):
                level = root.replace("data", "").count(os.sep)
                indent = "    " * level
                tree_output.append(f"{indent}{os.path.basename(root)}/")
                subindent = "    " * (level + 1)
                for file in files[:5]:  # Show first 5 files per directory
                    tree_output.append(f"{subindent}{file}")
            
            st.code("\n".join(tree_output))
        else:
            st.error("Data directory does not exist. Please run the pipeline first.")

# Add auto-refresh option
st.sidebar.divider()
if st.sidebar.button("Refresh Data", type="secondary"):
    st.cache_data.clear()
    if 'df' in st.session_state:
        del st.session_state.df
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("""
**System Status:** Active  
**Last Refresh:** {}
""".format(datetime.now().strftime("%H:%M:%S")))
