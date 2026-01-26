"""
Professional Market Narrative Risk Intelligence Dashboard.
Modern, clean interface with comprehensive data visualization.
Complete version with proper pipeline integration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys
import traceback
from typing import List, Dict, Optional
import logging

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Market Narrative Risk Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Market Narrative Risk Intelligence System v1.0.0"
    }
)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import scraper with error handling
try:
    from src.scraper import scrape_and_save
    SCRAPER_AVAILABLE = True
    logger.info("Scraper module imported successfully")
except ImportError as e:
    logger.warning(f"Scraper not available: {e}")
    SCRAPER_AVAILABLE = False
    scrape_and_save = None

class MarketRiskDashboard:
    """
    Professional dashboard for market narrative risk intelligence.
    """
    
    def __init__(self):
        """Initialize dashboard with professional settings."""
        self.logger = logger
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        self._init_session_state()
        self.load_data()
        self.logger.info("MarketRiskDashboard initialized successfully")
    
    def _init_session_state(self):
        """Initialize session state variables."""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'current_view' not in st.session_state:
            st.session_state.current_view = 'overview'
        if 'pipeline_running' not in st.session_state:
            st.session_state.pipeline_running = False
    
    def load_data(self):
        """Load the latest valid parquet file, ignoring corrupted/tiny files."""
        try:
            data_dir = Path("data/bronze")
            
            if not data_dir.exists():
                data_dir.mkdir(parents=True, exist_ok=True)
            
            # Find valid parquet files (at least 1KB)
            valid_files = []
            for file in data_dir.glob("*.parquet"):
                if file.stat().st_size > 1024:  # At least 1KB
                    valid_files.append((file.stat().st_mtime, file))
            
            if valid_files:
                # Sort by modification time (newest first)
                valid_files.sort(reverse=True)
                latest_file = valid_files[0][1]
                
                try:
                    self.df = pd.read_parquet(latest_file)
                    
                    if 'timestamp' in self.df.columns:
                        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
                        self.min_date = self.df['timestamp'].min().date()
                        self.max_date = self.df['timestamp'].max().date()
                    else:
                        # Create sample timestamps if none exist
                        self.df['timestamp'] = pd.date_range(
                            end=datetime.now(), 
                            periods=len(self.df), 
                            freq='H'
                        )
                        self.min_date = self.df['timestamp'].min().date()
                        self.max_date = self.df['timestamp'].max().date()
                    
                    self.logger.info(f"Loaded {len(self.df)} records from {latest_file.name}")
                    st.session_state.data_loaded = True
                    return
                    
                except Exception as e:
                    self.logger.error(f"Error reading parquet {latest_file}: {e}")
            
            # Fallback to sample data if no valid files
            self._create_sample_data()
            self.logger.info("Using sample data - no valid parquet files found")
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}", exc_info=True)
            self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample data for demonstration."""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='H')
        
        np.random.seed(42)
        base_trend = np.linspace(0, 2, len(dates))
        seasonal = 0.5 * np.sin(2 * np.pi * np.arange(len(dates)) / 24)
        noise = np.random.normal(0, 0.3, len(dates))
        
        stress_scores = base_trend + seasonal + noise
        sentiment_scores = np.random.uniform(-0.8, 0.8, len(dates))
        
        self.df = pd.DataFrame({
            'timestamp': dates,
            'headline': [f"Market update {i}" for i in range(len(dates))],
            'snippet': [f"Market analysis for period {i}" for i in range(len(dates))],
            'sentiment_polarity': sentiment_scores,
            'market_stress_score': stress_scores,
            'change_percent': np.random.uniform(-5, 5, len(dates)),
            'price': np.random.uniform(100, 5000, len(dates)),
            'asset_tags': [['S&P 500', 'NASDAQ'][i % 2] for i in range(len(dates))],
            'asset_type': ['index' if i % 2 == 0 else 'crypto' for i in range(len(dates))],
            'source': 'yahoo_finance',
            'is_anomaly': np.random.choice([0, 1], len(dates), p=[0.95, 0.05]),
        })
        
        self.min_date = self.df['timestamp'].min().date()
        self.max_date = self.df['timestamp'].max().date()
        
        st.session_state.data_loaded = True
        self.logger.info("Created sample data for demonstration")
    
    def _run_full_pipeline(self):
        """Run the complete pipeline: scraping -> saving."""
        if not SCRAPER_AVAILABLE or scrape_and_save is None:
            st.error("Scraper module not available. Cannot run pipeline.")
            return False
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Collecting market data...")
            progress_bar.progress(30)
            self.logger.info("Starting data collection phase")
            
            result_path = scrape_and_save()
            
            if not result_path:
                st.error("Data collection failed. No data collected.")
                return False
            
            self.logger.info(f"Data collection completed: {result_path}")
            progress_bar.progress(70)
            
            status_text.text("Loading new data...")
            self.load_data()
            
            progress_bar.progress(100)
            status_text.text("Pipeline completed successfully!")
            
            return True
                
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            st.error(f"Pipeline error: {e}")
            return False
            
        finally:
            progress_bar.empty()
            status_text.empty()
    
    def render_header(self):
        """Render professional header."""
        st.markdown("""
        <div style='text-align: center; padding: 20px 0;'>
            <h1 style='margin-bottom: 5px; color: #2c3e50;'>Market Narrative Risk Intelligence System</h1>
            <p style='color: #7f8c8d; font-size: 1.1em; margin-top: 0;'>
            Advanced analytics for market stress detection and risk regime identification
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    def render_sidebar(self):
        """Render professional sidebar with filters and navigation."""
        with st.sidebar:
            # System status
            data_status = "Available" if st.session_state.data_loaded else "No Data"
            data_time = self.max_date.strftime("%Y-%m-%d") if hasattr(self, 'max_date') else "Never"
            
            st.markdown(f"""
            <div style='padding: 10px; background-color: #f8f9fa; border-radius: 5px; margin-bottom: 20px;'>
                <p style='font-size: 0.9em; color: #6c757d; margin: 0;'>
                <strong>System Status:</strong> Operational<br>
                <strong>Data Status:</strong> {data_status}<br>
                <strong>Last Updated:</strong> {data_time}<br>
                <strong>Records Loaded:</strong> {len(self.df) if hasattr(self, 'df') else 0:,}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### Data Collection")
            
            # Show scraper status
            if SCRAPER_AVAILABLE:
                st.success("Data Collection: Available")
            else:
                st.warning("Data Collection: Not Available")
            
            if st.button("Update Market Data", type="primary", use_container_width=True, 
                       disabled=not SCRAPER_AVAILABLE,
                       help="Collect latest market data from Yahoo Finance" if SCRAPER_AVAILABLE else "Scraper not available"):
                with st.spinner("Collecting market data... This may take a minute."):
                    success = self._run_full_pipeline()
                    if success:
                        st.success("Data collection completed! Refreshing dashboard...")
                        import time
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Data collection failed.")
            
            if st.button("Refresh Dashboard", use_container_width=True):
                st.rerun()
            
            st.markdown("---")
            st.markdown("### Navigation")
            
            view_options = {
                'overview': 'System Overview',
                'market_analysis': 'Market Analysis',
                'risk_assessment': 'Risk Assessment',
                'data_insights': 'Data Insights'
            }
            
            selected_view = st.selectbox(
                "Select Dashboard View",
                options=list(view_options.keys()),
                format_func=lambda x: view_options[x],
                key='view_selector'
            )
            st.session_state.current_view = selected_view
            
            st.markdown("---")
            st.markdown("### Data Filters")
            
            if hasattr(self, 'min_date') and hasattr(self, 'max_date'):
                date_range = st.date_input(
                    "Analysis Period",
                    value=(self.min_date, self.max_date),
                    min_value=self.min_date,
                    max_value=self.max_date
                )
                
                if len(date_range) == 2:
                    self.start_date, self.end_date = date_range
                else:
                    self.start_date = self.end_date = date_range[0] if date_range else self.max_date
            else:
                self.start_date = self.end_date = datetime.now().date()
            
            # Asset type filter
            if 'asset_type' in self.df.columns:
                asset_types = ['All'] + sorted(self.df['asset_type'].dropna().unique().tolist())
                selected_asset_type = st.selectbox("Asset Type Filter", asset_types, index=0)
                self.selected_asset_type = selected_asset_type if selected_asset_type != 'All' else None
            else:
                self.selected_asset_type = None
            
            st.markdown("---")
            
            with st.expander("Data Statistics", expanded=False):
                if st.session_state.data_loaded and hasattr(self, 'df'):
                    total_records = len(self.df)
                    
                    # Apply date filters for statistics
                    mask = (self.df['timestamp'].dt.date >= self.start_date) & \
                           (self.df['timestamp'].dt.date <= self.end_date)
                    if self.selected_asset_type and 'asset_type' in self.df.columns:
                        mask = mask & (self.df['asset_type'] == self.selected_asset_type)
                    
                    filtered_df = self.df[mask]
                    filtered_records = len(filtered_df)
                    
                    st.metric("Total Records", f"{total_records:,}")
                    st.metric("Filtered Records", f"{filtered_records:,}")
                    
                    if 'asset_type' in filtered_df.columns:
                        type_counts = filtered_df['asset_type'].value_counts()
                        for asset_type, count in type_counts.items():
                            st.metric(f"{asset_type.title()}", f"{count:,}")
    
    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all sidebar filters to data."""
        filtered_df = df.copy()
        
        if 'timestamp' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['timestamp'].dt.date >= self.start_date) &
                (filtered_df['timestamp'].dt.date <= self.end_date)
            ]
        
        if self.selected_asset_type and 'asset_type' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['asset_type'] == self.selected_asset_type]
        
        return filtered_df
    
    def render_overview(self):
        """Render system overview dashboard."""
        st.markdown("## System Overview")
        
        filtered_df = self._apply_filters(self.df)
        
        if filtered_df.empty:
            st.warning("No data matches current filters. Please adjust your filter settings.")
            return
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'change_percent' in filtered_df.columns:
                avg_change = filtered_df['change_percent'].mean()
                st.metric("Avg Change", f"{avg_change:+.2f}%", f"{avg_change:+.2f}%")
            else:
                st.info("Change data not available")
        
        with col2:
            if 'market_stress_score' in filtered_df.columns:
                current_stress = filtered_df['market_stress_score'].iloc[-1] if len(filtered_df) > 0 else 0
                avg_stress = filtered_df['market_stress_score'].mean()
                st.metric("Current Stress", f"{current_stress:.2f}", f"{current_stress - avg_stress:+.2f}")
            else:
                st.info("Stress data not available")
        
        with col3:
            if 'is_anomaly' in filtered_df.columns:
                anomaly_count = int(filtered_df['is_anomaly'].sum())
                anomaly_rate = (anomaly_count / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
                st.metric("Anomalies", f"{anomaly_count}", f"{anomaly_rate:.1f}% rate")
            else:
                st.info("Anomaly data not available")
        
        with col4:
            total_articles = len(filtered_df)
            st.metric("Articles", f"{total_articles:,}")
        
        st.markdown("---")
        
        st.markdown("### Recent Market Updates")
        display_cols = ['timestamp', 'headline', 'change_percent']
        display_cols = [col for col in display_cols if col in filtered_df.columns]
        
        if display_cols and not filtered_df.empty:
            recent_data = filtered_df[display_cols].tail(10).copy()
            if 'timestamp' in recent_data.columns:
                recent_data['timestamp'] = recent_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            if 'change_percent' in recent_data.columns:
                recent_data['change_percent'] = recent_data['change_percent'].apply(lambda x: f"{x:+.2f}%")
            st.dataframe(recent_data, width=st.session_state.get('table_width', 'stretch'), hide_index=True)
        else:
            st.info("No displayable data available")
    
    def render_market_analysis(self):
        """Render detailed market analysis view."""
        st.markdown("## Market Analysis")
        
        filtered_df = self._apply_filters(self.df)
        
        if filtered_df.empty:
            st.warning("No data available for market analysis")
            return
        
        # Price trend chart
        if 'price' in filtered_df.columns and 'timestamp' in filtered_df.columns:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=filtered_df['timestamp'],
                y=filtered_df['price'],
                mode='lines+markers',
                name='Price',
                line=dict(color=self.colors[0], width=2),
                marker=dict(size=4)
            ))
            
            if len(filtered_df) > 2:
                z = np.polyfit(range(len(filtered_df)), filtered_df['price'], 1)
                p = np.poly1d(z)
                fig.add_trace(go.Scatter(
                    x=filtered_df['timestamp'],
                    y=p(range(len(filtered_df))),
                    mode='lines',
                    name='Trend',
                    line=dict(color='red', width=2, dash='dash')
                ))
            
            fig.update_layout(
                title='Price Trend Over Time',
                height=400,
                xaxis_title="Date",
                yaxis_title="Price",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Change distribution
        if 'change_percent' in filtered_df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_hist = px.histogram(
                    filtered_df,
                    x='change_percent',
                    nbins=30,
                    title='Distribution of Price Changes'
                )
                fig_hist.update_layout(height=400)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                positive_changes = len(filtered_df[filtered_df['change_percent'] > 0])
                negative_changes = len(filtered_df[filtered_df['change_percent'] < 0])
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=['Positive', 'Negative'],
                    values=[positive_changes, negative_changes],
                    marker=dict(colors=['#2ca02c', '#d62728'])
                )])
                fig_pie.update_layout(title='Change Direction', height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
    
    def render_risk_assessment(self):
        """Render risk assessment view."""
        st.markdown("## Risk Assessment")
        
        filtered_df = self._apply_filters(self.df)
        
        if filtered_df.empty:
            st.warning("No data available")
            return
        
        # Stress score analysis
        if 'market_stress_score' in filtered_df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_stress = go.Figure()
                
                fig_stress.add_trace(go.Scatter(
                    x=filtered_df['timestamp'],
                    y=filtered_df['market_stress_score'],
                    mode='lines+markers',
                    name='Stress Score',
                    line=dict(color=self.colors[1], width=2),
                    marker=dict(size=4)
                ))
                
                # Add thresholds
                fig_stress.add_hline(y=3, line_dash="dot", line_color="green", annotation_text="Low Risk")
                fig_stress.add_hline(y=7, line_dash="dot", line_color="red", annotation_text="High Risk")
                
                fig_stress.update_layout(
                    title='Market Stress Score Over Time',
                    height=400,
                    xaxis_title="Date",
                    yaxis_title="Stress Score",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_stress, use_container_width=True)
            
            with col2:
                stress_stats = filtered_df['market_stress_score'].describe()
                st.markdown("### Stress Score Statistics")
                
                metrics = [
                    ("Current", f"{filtered_df['market_stress_score'].iloc[-1]:.2f}"),
                    ("Average", f"{stress_stats['mean']:.2f}"),
                    ("Maximum", f"{stress_stats['max']:.2f}"),
                    ("Minimum", f"{stress_stats['min']:.2f}"),
                    ("Std Dev", f"{stress_stats['std']:.2f}")
                ]
                
                for label, value in metrics:
                    st.metric(label, value)
        
        # Anomaly detection
        if 'is_anomaly' in filtered_df.columns:
            st.markdown("### Anomaly Detection")
            
            fig_anomaly = go.Figure()
            
            normal = filtered_df[filtered_df['is_anomaly'] == 0]
            anomalies = filtered_df[filtered_df['is_anomaly'] == 1]
            
            if 'market_stress_score' in filtered_df.columns:
                fig_anomaly.add_trace(go.Scatter(
                    x=normal['timestamp'],
                    y=normal['market_stress_score'],
                    mode='markers',
                    name='Normal',
                    marker=dict(color='lightblue', size=6)
                ))
                
                fig_anomaly.add_trace(go.Scatter(
                    x=anomalies['timestamp'],
                    y=anomalies['market_stress_score'],
                    mode='markers',
                    name='Anomaly',
                    marker=dict(color='red', size=10, symbol='diamond')
                ))
            
            fig_anomaly.update_layout(
                title='Anomaly Detection Timeline',
                height=400,
                xaxis_title="Date",
                yaxis_title="Stress Score"
            )
            
            st.plotly_chart(fig_anomaly, use_container_width=True)
    
    def render_data_insights(self):
        """Render data insights view."""
        st.markdown("## Data Insights")
        
        filtered_df = self._apply_filters(self.df)
        
        if filtered_df.empty:
            st.warning("No data available")
            return
        
        # Correlation analysis
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['is_anomaly']]
        
        if len(numeric_cols) > 1:
            st.markdown("### Feature Correlations")
            
            corr_cols = numeric_cols[:8]  # Limit to first 8 for readability
            corr_matrix = filtered_df[corr_cols].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                color_continuous_scale='RdBu',
                zmin=-1,
                zmax=1,
                title='Feature Correlation Matrix'
            )
            
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Data export section
        st.markdown("### Data Export")
        
        if not filtered_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Export Format")
                export_format = st.radio(
                    "Select format",
                    ['CSV', 'JSON'],
                    horizontal=True
                )
            
            with col2:
                st.markdown("#### Data Preview")
                st.dataframe(filtered_df.head(5), width='stretch')
            
            st.markdown(f"**Total Records:** {len(filtered_df):,}")
            
            if export_format == 'CSV':
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"market_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            elif export_format == 'JSON':
                json_str = filtered_df.to_json(orient='records', indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"market_data_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
        else:
            st.info("No data to export")
    
    def render_footer(self):
        """Render professional footer."""
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #6c757d; font-size: 0.9em; padding: 20px;'>
        Market Narrative Risk Intelligence System v1.0.0<br>
        Real-time market analysis powered by machine learning
        </div>
        """, unsafe_allow_html=True)
    
    def run(self):
        """Run the dashboard application."""
        try:
            self.render_header()
            self.render_sidebar()
            
            view_handlers = {
                'overview': self.render_overview,
                'market_analysis': self.render_market_analysis,
                'risk_assessment': self.render_risk_assessment,
                'data_insights': self.render_data_insights
            }
            
            current_view = st.session_state.get('current_view', 'overview')
            handler = view_handlers.get(current_view, self.render_overview)
            
            handler()
            
            self.render_footer()
            
            self.logger.info(f"Dashboard view '{current_view}' rendered successfully")
            
        except Exception as e:
            self.logger.error(f"Dashboard error: {e}", exc_info=True)
            st.error(f"An error occurred: {str(e)}")
            with st.expander("Error Details"):
                st.code(traceback.format_exc())


def main():
    """Main entry point for the dashboard."""
    try:
        dashboard = MarketRiskDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Failed to initialize dashboard: {e}")
        st.code(traceback.format_exc())
        st.stop()


if __name__ == "__main__":
    main()
