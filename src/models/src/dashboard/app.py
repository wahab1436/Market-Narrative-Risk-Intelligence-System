"""
Streamlit dashboard for market narrative risk intelligence.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import List, Optional

from src.utils.logger import dashboard_logger
from src.utils.config_loader import config_loader


class RiskIntelligenceDashboard:
    """
    Interactive dashboard for market narrative risk analysis.
    """
    
    def __init__(self):
        """Initialize dashboard."""
        st.set_page_config(
            page_title="Market Narrative Risk Intelligence",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        self.config = config_loader.get_config("config")
        self.dashboard_config = self.config.get("dashboard", {})
        self.load_data()
        dashboard_logger.info("Dashboard initialized")
    
    def load_data(self):
        """Load prediction data from gold layer."""
        gold_dir = Path("data/gold")
        
        # Look for files with predictions
        prediction_files = list(gold_dir.glob("*_predictions.parquet"))
        
        if prediction_files:
            # Load the most recent predictions
            latest_file = max(prediction_files, key=lambda x: x.stat().st_mtime)
            self.df = pd.read_parquet(latest_file)
            
            # Ensure timestamp is datetime
            if 'timestamp' in self.df.columns:
                self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            
            dashboard_logger.info(f"Loaded {len(self.df)} records from {latest_file}")
        else:
            # Create sample data if no predictions exist
            self.create_sample_data()
            dashboard_logger.warning("No prediction files found, using sample data")
    
    def create_sample_data(self):
        """Create sample data for demonstration."""
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        
        self.df = pd.DataFrame({
            'timestamp': dates,
            'headline': [f"Market news {i}" for i in range(len(dates))],
            'sentiment_polarity': np.random.uniform(-1, 1, len(dates)),
            'keyword_stress_score': np.random.exponential(0.5, len(dates)),
            'weighted_stress_score': np.random.normal(0, 1, len(dates)),
            'linear_regression_prediction': np.random.normal(0, 0.8, len(dates)),
            'neural_network_prediction': np.random.normal(0, 0.9, len(dates)),
            'xgboost_risk_regime': np.random.choice(['low', 'medium', 'high'], len(dates)),
            'prob_low': np.random.uniform(0, 1, len(dates)),
            'prob_medium': np.random.uniform(0, 1, len(dates)),
            'prob_high': np.random.uniform(0, 1, len(dates)),
            'is_anomaly': np.random.choice([0, 1], len(dates), p=[0.9, 0.1])
        })
    
    def create_sidebar(self):
        """Create dashboard sidebar with filters."""
        with st.sidebar:
            st.title("🎯 Risk Intelligence Dashboard")
            st.markdown("---")
            
            # Date range filter
            st.subheader("Date Range")
            if 'timestamp' in self.df.columns:
                min_date = self.df['timestamp'].min().date()
                max_date = self.df['timestamp'].max().date()
                
                date_range = st.date_input(
                    "Select date range",
                    value=(max_date - timedelta(days=7), max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                
                if len(date_range) == 2:
                    self.start_date, self.end_date = date_range
                else:
                    self.start_date = self.end_date = date_range
            else:
                self.start_date = self.end_date = datetime.now().date()
            
            # Risk regime filter
            st.subheader("Risk Regime")
            risk_regimes = ['All', 'Low', 'Medium', 'High']
            selected_regime = st.selectbox(
                "Filter by risk regime",
                risk_regimes
            )
            self.selected_regime = selected_regime.lower() if selected_regime != 'All' else None
            
            # Anomaly filter
            st.subheader("Anomaly Detection")
            show_anomalies = st.checkbox("Show only anomalies", value=False)
            self.show_anomalies = show_anomalies
            
            # Asset category filter
            st.subheader("Asset Categories")
            asset_columns = [col for col in self.df.columns if col.startswith('mentions_')]
            if asset_columns:
                asset_categories = [col.replace('mentions_', '') for col in asset_columns]
                selected_assets = st.multiselect(
                    "Filter by asset mentions",
                    asset_categories
                )
                self.selected_assets = selected_assets
            else:
                self.selected_assets = []
            
            st.markdown("---")
            st.caption(f"Data points: {len(self.df)}")
            st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    def filter_data(self) -> pd.DataFrame:
        """Apply filters to data."""
        filtered_df = self.df.copy()
        
        # Date filter
        if 'timestamp' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['timestamp'].dt.date >= self.start_date) &
                (filtered_df['timestamp'].dt.date <= self.end_date)
            ]
        
        # Risk regime filter
        if self.selected_regime and 'xgboost_risk_regime' in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df['xgboost_risk_regime'] == self.selected_regime
            ]
        
        # Anomaly filter
        if self.show_anomalies and 'is_anomaly' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['is_anomaly'] == 1]
        
        # Asset category filter
        if self.selected_assets and 'timestamp' in filtered_df.columns:
            # Aggregate by date for asset mentions
            daily_data = filtered_df.copy()
            daily_data['date'] = daily_data['timestamp'].dt.date
            
            # Create asset mention indicators
            asset_mentions = []
            for asset in self.selected_assets:
                col_name = f'mentions_{asset}'
                if col_name in daily_data.columns:
                    asset_mentions.append(col_name)
            
            if asset_mentions:
                # Filter days with mentions of selected assets
                days_with_mentions = daily_data[daily_data[asset_mentions].sum(axis=1) > 0]['date']
                filtered_df = filtered_df[filtered_df['timestamp'].dt.date.isin(days_with_mentions)]
        
        return filtered_df
    
    def create_stress_score_chart(self, df: pd.DataFrame):
        """Create stress score timeline chart."""
        if 'timestamp' not in df.columns:
            return
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Stress Score Timeline', 'Model Predictions Comparison'),
            vertical_spacing=0.15,
            row_heights=[0.6, 0.4]
        )
        
        # Add actual stress score
        if 'weighted_stress_score' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['weighted_stress_score'],
                    mode='lines+markers',
                    name='Actual Stress Score',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=6)
                ),
                row=1, col=1
            )
        
        # Add model predictions if available
        prediction_cols = [
            ('linear_regression_prediction', 'Linear Regression', '#ff7f0e'),
            ('neural_network_prediction', 'Neural Network', '#2ca02c')
        ]
        
        for col, name, color in prediction_cols:
            if col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df[col],
                        mode='lines',
                        name=name,
                        line=dict(color=color, width=1.5, dash='dash'),
                        opacity=0.7
                    ),
                    row=1, col=1
                )
        
        # Add residuals if available
        residual_cols = [
            ('linear_regression_residual', 'Linear Regression Residual'),
            ('neural_network_residual', 'Neural Network Residual')
        ]
        
        for col, name in residual_cols:
            if col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df[col],
                        mode='lines',
                        name=name,
                        line=dict(width=1),
                        opacity=0.5,
                        showlegend=True
                    ),
                    row=2, col=1
                )
        
        # Update layout
        fig.update_layout(
            height=self.dashboard_config.get('chart_height', 500),
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Stress Score", row=1, col=1)
        fig.update_yaxes(title_text="Residuals", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_risk_regime_chart(self, df: pd.DataFrame):
        """Create risk regime evolution chart."""
        if 'timestamp' not in df.columns or 'xgboost_risk_regime' not in df.columns:
            return
        
        # Aggregate by date
        daily_df = df.copy()
        daily_df['date'] = daily_df['timestamp'].dt.date
        
        regime_counts = daily_df.groupby(['date', 'xgboost_risk_regime']).size().unstack(fill_value=0)
        
        fig = go.Figure()
        
        # Define colors for regimes
        regime_colors = {
            'low': '#2ca02c',
            'medium': '#ff7f0e',
            'high': '#d62728'
        }
        
        for regime in regime_counts.columns:
            fig.add_trace(go.Bar(
                x=regime_counts.index,
                y=regime_counts[regime],
                name=regime.capitalize(),
                marker_color=regime_colors.get(regime, '#777'),
                opacity=0.8
            ))
        
        # Update layout
        fig.update_layout(
            title='Risk Regime Distribution Over Time',
            barmode='stack',
            height=400,
            xaxis_title="Date",
            yaxis_title="Number of Articles",
            legend_title="Risk Regime",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add probability heatmap if available
        prob_cols = [col for col in df.columns if col.startswith('prob_')]
        if prob_cols and 'timestamp' in df.columns:
            # Prepare probability data
            prob_df = df[['timestamp'] + prob_cols].copy()
            prob_df['date'] = prob_df['timestamp'].dt.date
            
            # Average probabilities by date
            avg_probs = prob_df.groupby('date')[prob_cols].mean()
            
            # Create heatmap
            fig2 = go.Figure(data=go.Heatmap(
                z=avg_probs.T.values,
                x=avg_probs.index,
                y=[col.replace('prob_', '').capitalize() for col in prob_cols],
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="Probability")
            ))
            
            fig2.update_layout(
                title='Risk Regime Probability Heatmap',
                height=300,
                xaxis_title="Date",
                yaxis_title="Risk Regime"
            )
            
            st.plotly_chart(fig2, use_container_width=True)
    
    def create_anomaly_chart(self, df: pd.DataFrame):
        """Create anomaly detection chart."""
        if 'timestamp' not in df.columns or 'is_anomaly' not in df.columns:
            return
        
        fig = go.Figure()
        
        # Add stress score line
        if 'weighted_stress_score' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['weighted_stress_score'],
                mode='lines',
                name='Stress Score',
                line=dict(color='#1f77b4', width=2),
                opacity=0.7
            ))
        
        # Highlight anomalies
        anomalies = df[df['is_anomaly'] == 1]
        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies['timestamp'],
                y=anomalies.get('weighted_stress_score', 0),
                mode='markers',
                name='Anomalies',
                marker=dict(
                    color='#d62728',
                    size=12,
                    symbol='circle',
                    line=dict(width=2, color='white')
                ),
                hovertext=anomalies.get('headline', 'Anomaly detected')
            ))
        
        # Add anomaly scores if available
        if 'anomaly_score' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['anomaly_score'],
                mode='lines',
                name='Anomaly Score',
                line=dict(color='#ff7f0e', width=1, dash='dash'),
                opacity=0.5,
                yaxis='y2'
            ))
            
            # Add secondary y-axis
            fig.update_layout(
                yaxis2=dict(
                    title="Anomaly Score",
                    overlaying='y',
                    side='right'
                )
            )
        
        fig.update_layout(
            title='Anomaly Detection Timeline',
            height=400,
            xaxis_title="Date",
            yaxis_title="Stress Score",
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_feature_importance_chart(self):
        """Create feature importance visualization."""
        # Load SHAP results if available
        shap_dir = Path("logs")
        shap_files = list(shap_dir.glob("shap_summary_*.png"))
        
        if shap_files:
            # Display SHAP summary plots
            cols = st.columns(min(3, len(shap_files)))
            
            for idx, shap_file in enumerate(shap_files[:3]):
                with cols[idx]:
                    model_name = shap_file.stem.replace('shap_summary_', '')
                    st.subheader(f"{model_name.replace('_', ' ').title()}")
                    st.image(str(shap_file), use_column_width=True)
        else:
            # Fallback to sample feature importance
            st.info("SHAP analysis not available. Run the pipeline to generate feature importance plots.")
            
            # Create sample feature importance
            features = [
                'keyword_stress_score', 'sentiment_polarity',
                'daily_article_count', 'rolling_7d_volatility',
                'market_breadth', 'headline_length'
            ]
            
            importance = np.random.uniform(0.1, 1, len(features))
            
            fig = go.Figure(data=go.Bar(
                x=importance,
                y=features,
                orientation='h',
                marker_color='#1f77b4'
            ))
            
            fig.update_layout(
                title='Sample Feature Importance',
                height=400,
                xaxis_title="Importance",
                yaxis_title="Feature",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def create_metrics_summary(self, df: pd.DataFrame):
        """Create metrics summary cards."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'weighted_stress_score' in df.columns:
                avg_stress = df['weighted_stress_score'].mean()
                delta = avg_stress - self.df['weighted_stress_score'].mean() if 'weighted_stress_score' in self.df.columns else 0
                st.metric(
                    "Avg Stress Score",
                    f"{avg_stress:.2f}",
                    f"{delta:+.2f}"
                )
        
        with col2:
            if 'xgboost_risk_regime' in df.columns:
                high_risk_pct = (df['xgboost_risk_regime'] == 'high').mean() * 100
                st.metric(
                    "High Risk %",
                    f"{high_risk_pct:.1f}%"
                )
        
        with col3:
            if 'is_anomaly' in df.columns:
                anomaly_count = df['is_anomaly'].sum()
                st.metric(
                    "Anomalies Detected",
                    str(anomaly_count)
                )
        
        with col4:
            total_articles = len(df)
            st.metric(
                "Total Articles",
                str(total_articles)
            )
    
    def run(self):
        """Run the dashboard."""
        # Header
        st.title("Market Narrative Risk Intelligence System")
        st.markdown("---")
        
        # Sidebar
        self.create_sidebar()
        
        # Filter data
        filtered_df = self.filter_data()
        
        if filtered_df.empty:
            st.warning("No data available for selected filters.")
            return
        
        # Metrics summary
        self.create_metrics_summary(filtered_df)
        st.markdown("---")
        
        # Stress score timeline
        st.header("Stress Score Analysis")
        self.create_stress_score_chart(filtered_df)
        st.markdown("---")
        
        # Risk regime analysis
        st.header("Risk Regime Analysis")
        self.create_risk_regime_chart(filtered_df)
        st.markdown("---")
        
        # Anomaly detection
        st.header("Anomaly Detection")
        self.create_anomaly_chart(filtered_df)
        st.markdown("---")
        
        # Feature importance
        st.header("Model Explainability")
        self.create_feature_importance_chart()
        
        # Data download
        st.markdown("---")
        st.header("Data Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Download Predictions as CSV"):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Click to download",
                    data=csv,
                    file_name="risk_predictions.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("Generate Report"):
                report_content = self.generate_report(filtered_df)
                st.download_button(
                    label="Download Report",
                    data=report_content,
                    file_name="risk_report.md",
                    mime="text/markdown"
                )
    
    def generate_report(self, df: pd.DataFrame) -> str:
        """Generate markdown report."""
        report = f"""# Market Narrative Risk Intelligence Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Period: {self.start_date} to {self.end_date}

## Summary Statistics
- Total Articles: {len(df)}
- Average Stress Score: {df.get('weighted_stress_score', 0).mean():.2f}
- High Risk Articles: {(df.get('xgboost_risk_regime', '') == 'high').sum() if 'xgboost_risk_regime' in df.columns else 0}
- Anomalies Detected: {df.get('is_anomaly', 0).sum() if 'is_anomaly' in df.columns else 0}

## Key Insights

### 1. Stress Score Trends
The weighted stress score shows the overall market stress level based on news narratives.

### 2. Risk Regime Distribution
Analysis of risk regimes (low, medium, high) based on XGBoost classification.

### 3. Anomaly Detection
Periods identified as anomalous by the Isolation Forest model.

## Recommendations
1. Monitor periods with high stress scores and anomaly flags
2. Review articles classified as high risk for specific narratives
3. Consider correlation with market movements for validation

## Data Sample
{df.head().to_markdown()}
"""
        return report


def main():
    """Main dashboard function."""
    dashboard = RiskIntelligenceDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
