"""
Professional Market Risk Intelligence Dashboard
Clean, data-driven interface with comprehensive pipeline integration
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
import sys
import traceback
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Market Risk Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/issues',
        'Report a bug': None,
        'About': "Market Risk Intelligence Dashboard v2.0.0 - Professional Financial Risk Analytics Platform"
    }
)

# Custom CSS for professional appearance
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a237e;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #5d6cc0;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3f51b5;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a237e;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Status indicators */
    .status-success {
        color: #4caf50;
        font-weight: 600;
    }
    
    .status-warning {
        color: #ff9800;
        font-weight: 600;
    }
    
    .status-error {
        color: #f44336;
        font-weight: 600;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #3f51b5, #5c6bc0);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #303f9f, #3949ab);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(63, 81, 181, 0.3);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #f5f7ff;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 5px 5px 0 0;
        padding: 0px 20px;
        font-weight: 600;
        color: #5d6cc0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3f51b5;
        color: white !important;
    }
    
    /* Data table styling */
    .dataframe {
        width: 100%;
        border-collapse: collapse;
    }
    
    .dataframe th {
        background-color: #3f51b5;
        color: white;
        font-weight: 600;
        padding: 12px;
        text-align: left;
    }
    
    .dataframe td {
        padding: 10px;
        border-bottom: 1px solid #ddd;
    }
    
    .dataframe tr:hover {
        background-color: #f5f7ff;
    }
</style>
""", unsafe_allow_html=True)

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import pipeline modules
try:
    from scraper import scrape_and_save
    from preprocessing.clean_data import clean_and_save
    from preprocessing.feature_engineering import engineer_and_save
    PIPELINE_AVAILABLE = True
except ImportError as e:
    PIPELINE_AVAILABLE = False

try:
    from models.regression.linear_regression import LinearRegressionModel
    from models.regression.ridge_regression import RidgeRegressionModel
    from models.regression.lasso_regression import LassoRegressionModel
    from models.neural_network import NeuralNetworkModel
    from models.xgboost_model import XGBoostModel
    from models.isolation_forest import IsolationForestModel
    MODELS_AVAILABLE = True
except ImportError as e:
    MODELS_AVAILABLE = False


class MarketRiskDashboard:
    """Professional dashboard for market risk intelligence."""
    
    def __init__(self):
        """Initialize dashboard with professional settings."""
        self.data_loaded = False
        self.scraped_data = None
        self._init_session_state()
        self.load_data()
    
    def _init_session_state(self):
        """Initialize session state variables."""
        if 'scraped_data' not in st.session_state:
            st.session_state.scraped_data = None
        if 'pipeline_status' not in st.session_state:
            st.session_state.pipeline_status = 'idle'
        if 'current_tab' not in st.session_state:
            st.session_state.current_tab = 'Dashboard'
        if 'selected_instrument' not in st.session_state:
            st.session_state.selected_instrument = 'All'
    
    def load_data(self):
        """Load data from all pipeline layers."""
        try:
            # Try to load from gold layer first
            gold_dir = Path("data/gold")
            if gold_dir.exists():
                gold_files = list(gold_dir.glob("*.parquet"))
                if gold_files:
                    latest_gold = max(gold_files, key=lambda x: x.stat().st_mtime)
                    self.df = pd.read_parquet(latest_gold)
                    self.data_source = f"Gold Layer: {latest_gold.name}"
                    self.data_loaded = True
                    return
            
            # Try silver layer
            silver_dir = Path("data/silver")
            if silver_dir.exists():
                silver_files = list(silver_dir.glob("*.parquet"))
                if silver_files:
                    latest_silver = max(silver_files, key=lambda x: x.stat().st_mtime)
                    self.df = pd.read_parquet(latest_silver)
                    self.data_source = f"Silver Layer: {latest_silver.name}"
                    self.data_loaded = True
                    return
            
            # Try bronze layer
            bronze_dir = Path("data/bronze")
            if bronze_dir.exists():
                bronze_files = list(bronze_dir.glob("*.parquet"))
                if bronze_files:
                    latest_bronze = max(bronze_files, key=lambda x: x.stat().st_mtime)
                    self.df = pd.read_parquet(latest_bronze)
                    self.data_source = f"Bronze Layer: {latest_bronze.name}"
                    self.data_loaded = True
                    return
            
            # Create sample data if nothing exists
            self._create_sample_data()
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample market data for demonstration."""
        # This matches the structure from your logs
        instruments = [
            "S&P 500", "Dow Jones", "NASDAQ", "VIX", 
            "Russell 2000", "Gold", "Silver", "US 10Y Bond"
        ]
        
        data = []
        current_time = datetime.now()
        
        # Add successful scrapes
        successful_data = {
            "S&P 500": {"price": 6988.19, "change": 0.00, "change_percent": 0.00},
            "Dow Jones": {"price": 49150.03, "change": 0.00, "change_percent": 0.00},
            "NASDAQ": {"price": 23837.27, "change": 0.00, "change_percent": 0.00},
            "VIX": {"price": 15.81, "change": 0.00, "change_percent": 0.00},
            "Russell 2000": {"price": 2662.87, "change": 0.00, "change_percent": 0.00},
            "Gold": {"price": 5093.30, "change": 0.00, "change_percent": 0.00}
        }
        
        # Add failed scrapes
        failed_instruments = ["Silver", "US 10Y Bond"]
        
        for instrument in instruments:
            record = {
                'timestamp': current_time,
                'instrument': instrument,
                'asset_type': self._get_asset_type(instrument),
                'scraped_at': current_time,
                'status': 'success' if instrument not in failed_instruments else 'failed'
            }
            
            if instrument in successful_data:
                record.update(successful_data[instrument])
                record['headline'] = f"{instrument} Market Update"
                record['market_stress_score'] = np.random.uniform(0.1, 0.9)
            else:
                record.update({
                    'price': None,
                    'change': None,
                    'change_percent': None,
                    'headline': f"Failed to fetch {instrument} data",
                    'market_stress_score': None
                })
            
            data.append(record)
        
        self.df = pd.DataFrame(data)
        self.data_source = "Sample Data (Demo Mode)"
        self.data_loaded = True
    
    def _get_asset_type(self, instrument):
        """Map instrument to asset type."""
        if any(x in instrument for x in ['S&P', 'Dow', 'NASDAQ', 'Russell']):
            return 'index'
        elif 'VIX' in instrument:
            return 'volatility'
        elif 'Gold' in instrument or 'Silver' in instrument:
            return 'commodity'
        elif 'Bond' in instrument:
            return 'bond'
        return 'other'
    
    def render_header(self):
        """Render professional header with status."""
        col1, col2, col3 = st.columns([2, 3, 1])
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if self.data_loaded:
                st.markdown(f'<div class="metric-label">DATA SOURCE</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{self.data_source}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<h1 class="main-header">Market Risk Intelligence Dashboard</h1>', unsafe_allow_html=True)
            st.markdown('<p class="sub-header">Real-time financial market monitoring and risk analytics</p>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            status = "Operational" if self.data_loaded else "No Data"
            color_class = "status-success" if self.data_loaded else "status-warning"
            st.markdown(f'<div class="metric-label">SYSTEM STATUS</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value {color_class}">{status}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
    
    def render_sidebar(self):
        """Render professional sidebar."""
        with st.sidebar:
            st.markdown("### Pipeline Control")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Run Pipeline", type="primary", use_container_width=True):
                    if PIPELINE_AVAILABLE:
                        self.run_pipeline()
                    else:
                        st.error("Pipeline components not available")
            
            with col2:
                if st.button("Refresh Data", use_container_width=True):
                    self.load_data()
                    st.rerun()
            
            st.markdown("---")
            st.markdown("### Data Filters")
            
            # Date range filter
            if 'timestamp' in self.df.columns:
                min_date = self.df['timestamp'].min().date()
                max_date = self.df['timestamp'].max().date()
                date_range = st.date_input(
                    "Analysis Period",
                    value=(max(min_date, max_date - timedelta(days=7)), max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                if len(date_range) == 2:
                    self.start_date, self.end_date = date_range
                else:
                    self.start_date = self.end_date = date_range[0] if date_range else max_date
            
            # Instrument filter
            if 'instrument' in self.df.columns:
                instruments = ['All'] + sorted(self.df['instrument'].unique().tolist())
                selected = st.selectbox("Select Instrument", instruments)
                st.session_state.selected_instrument = selected
            
            # Status filter
            if 'status' in self.df.columns:
                status_options = ['All', 'success', 'failed']
                selected_status = st.selectbox("Data Status", status_options)
                self.selected_status = selected_status if selected_status != 'All' else None
            
            st.markdown("---")
            st.markdown("### System Information")
            
            # Display stats
            if self.data_loaded:
                st.metric("Total Instruments", len(self.df))
                if 'status' in self.df.columns:
                    success_count = (self.df['status'] == 'success').sum()
                    st.metric("Successful Scrapes", success_count)
                if 'price' in self.df.columns:
                    valid_prices = self.df['price'].notna().sum()
                    st.metric("Valid Prices", valid_prices)
    
    def run_pipeline(self):
        """Execute the full pipeline."""
        with st.spinner("Running pipeline..."):
            try:
                # Step 1: Scraping
                st.info("Step 1: Scraping market data...")
                bronze_path = scrape_and_save()
                
                if bronze_path:
                    st.success(f"Scraping completed: {bronze_path}")
                    
                    # Step 2: Cleaning
                    st.info("Step 2: Cleaning data...")
                    silver_path = clean_and_save(bronze_path)
                    
                    if silver_path:
                        st.success(f"Cleaning completed: {silver_path}")
                        
                        # Step 3: Feature engineering
                        st.info("Step 3: Engineering features...")
                        gold_path = engineer_and_save(silver_path)
                        
                        if gold_path:
                            st.success(f"Feature engineering completed: {gold_path}")
                            
                            # Step 4: Model training
                            if MODELS_AVAILABLE:
                                st.info("Step 4: Training models...")
                                self.train_models(gold_path)
                                st.success("Model training completed")
                        
                        # Reload data
                        self.load_data()
                        st.rerun()
                
            except Exception as e:
                st.error(f"Pipeline failed: {str(e)}")
                st.code(traceback.format_exc())
    
    def train_models(self, data_path):
        """Train all available models."""
        df = pd.read_parquet(data_path)
        
        models = [
            ('Linear Regression', LinearRegressionModel()),
            ('Ridge Regression', RidgeRegressionModel()),
            ('Lasso Regression', LassoRegressionModel()),
            ('Neural Network', NeuralNetworkModel()),
            ('XGBoost', XGBoostModel()),
            ('Isolation Forest', IsolationForestModel())
        ]
        
        progress_bar = st.progress(0)
        
        for idx, (name, model) in enumerate(models):
            try:
                progress = (idx + 1) / len(models)
                progress_bar.progress(progress)
                
                with st.spinner(f"Training {name}..."):
                    model.train(df)
                
                st.success(f"{name} trained successfully")
            except Exception as e:
                st.warning(f"{name} failed: {str(e)}")
        
        progress_bar.empty()
    
    def render_dashboard_tab(self):
        """Render main dashboard with key metrics."""
        st.markdown("## Market Overview")
        
        # Filter data based on sidebar selections
        filtered_df = self.df.copy()
        
        if 'timestamp' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['timestamp'].dt.date >= self.start_date) &
                (filtered_df['timestamp'].dt.date <= self.end_date)
            ]
        
        if st.session_state.selected_instrument != 'All':
            filtered_df = filtered_df[filtered_df['instrument'] == st.session_state.selected_instrument]
        
        if hasattr(self, 'selected_status') and self.selected_status:
            filtered_df = filtered_df[filtered_df['status'] == self.selected_status]
        
        # Display metrics in cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if 'price' in filtered_df.columns:
                avg_price = filtered_df['price'].mean()
                st.markdown('<div class="metric-label">AVERAGE PRICE</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">${avg_price:,.2f}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if 'change_percent' in filtered_df.columns:
                avg_change = filtered_df['change_percent'].mean()
                color = "status-success" if avg_change >= 0 else "status-error"
                st.markdown('<div class="metric-label">AVG DAILY CHANGE</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value {color}">{avg_change:+.2f}%</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if 'market_stress_score' in filtered_df.columns:
                avg_stress = filtered_df['market_stress_score'].mean()
                stress_color = "status-error" if avg_stress > 0.7 else "status-warning" if avg_stress > 0.4 else "status-success"
                st.markdown('<div class="metric-label">MARKET STRESS</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value {stress_color}">{avg_stress:.2f}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            success_rate = (filtered_df['status'] == 'success').mean() if 'status' in filtered_df.columns else 0
            rate_color = "status-success" if success_rate > 0.8 else "status-warning" if success_rate > 0.6 else "status-error"
            st.markdown('<div class="metric-label">DATA QUALITY</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value {rate_color}">{success_rate:.1%}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Price visualization
        if 'price' in filtered_df.columns and 'instrument' in filtered_df.columns:
            st.markdown("### Instrument Prices")
            
            # Create price chart
            fig = go.Figure()
            
            for instrument in filtered_df['instrument'].unique():
                instr_data = filtered_df[filtered_df['instrument'] == instrument]
                if instr_data['price'].notna().any():
                    fig.add_trace(go.Scatter(
                        x=instr_data['timestamp'],
                        y=instr_data['price'],
                        mode='lines+markers',
                        name=instrument,
                        hovertemplate=f"{instrument}<br>Price: $%{{y:,.2f}}<br>Date: %{{x}}<extra></extra>"
                    ))
            
            fig.update_layout(
                title="Instrument Price Trends",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                hovermode='x unified',
                height=500,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent data table
        st.markdown("### Recent Market Data")
        display_cols = ['timestamp', 'instrument', 'price', 'change_percent', 'status', 'market_stress_score']
        display_cols = [col for col in display_cols if col in filtered_df.columns]
        
        if display_cols:
            recent_data = filtered_df[display_cols].sort_values('timestamp', ascending=False).head(20)
            if 'timestamp' in recent_data.columns:
                recent_data['timestamp'] = recent_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            
            # Format numeric columns
            if 'price' in recent_data.columns:
                recent_data['price'] = recent_data['price'].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "N/A")
            if 'change_percent' in recent_data.columns:
                recent_data['change_percent'] = recent_data['change_percent'].apply(
                    lambda x: f"{x:+.2f}%" if pd.notnull(x) else "N/A"
                )
            if 'market_stress_score' in recent_data.columns:
                recent_data['market_stress_score'] = recent_data['market_stress_score'].apply(
                    lambda x: f"{x:.3f}" if pd.notnull(x) else "N/A"
                )
            
            st.dataframe(recent_data, use_container_width=True, hide_index=True)
    
    def render_scraped_data_tab(self):
        """Display raw scraped data from the pipeline."""
        st.markdown("## Scraped Market Data")
        
        if not self.data_loaded:
            st.warning("No data available. Please run the pipeline first.")
            return
        
        # Show data statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Records", len(self.df))
            if 'instrument' in self.df.columns:
                st.metric("Unique Instruments", self.df['instrument'].nunique())
        
        with col2:
            if 'status' in self.df.columns:
                success_count = (self.df['status'] == 'success').sum()
                st.metric("Successful Scrapes", success_count)
                st.metric("Success Rate", f"{(success_count/len(self.df)*100):.1f}%")
        
        st.markdown("---")
        
        # Data quality indicators
        st.markdown("### Data Quality Analysis")
        
        if 'price' in self.df.columns:
            missing_prices = self.df['price'].isna().sum()
            valid_prices = self.df['price'].notna().sum()
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=['Valid Prices', 'Missing Prices'],
                    values=[valid_prices, missing_prices],
                    hole=0.4,
                    marker=dict(colors=['#4caf50', '#f44336'])
                )
            ])
            
            fig.update_layout(
                title="Price Data Completeness",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Raw data table
        st.markdown("### Raw Data Table")
        
        # Select columns to display
        available_cols = self.df.columns.tolist()
        default_cols = ['timestamp', 'instrument', 'asset_type', 'price', 'change', 'change_percent', 'status']
        display_cols = st.multiselect(
            "Select columns to display:",
            available_cols,
            default=[col for col in default_cols if col in available_cols]
        )
        
        if display_cols:
            display_df = self.df[display_cols].copy()
            
            # Format timestamp if present
            if 'timestamp' in display_df.columns:
                display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Format numeric columns
            for col in ['price', 'change', 'change_percent', 'market_stress_score']:
                if col in display_df.columns:
                    if col == 'price':
                        display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "N/A")
                    elif col == 'change_percent':
                        display_df[col] = display_df[col].apply(lambda x: f"{x:+.2f}%" if pd.notnull(x) else "N/A")
                    else:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "N/A")
            
            st.dataframe(display_df, use_container_width=True, height=600)
            
            # Download option
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name=f"market_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    def render_analytics_tab(self):
        """Render analytics and insights."""
        st.markdown("## Market Analytics")
        
        if not self.data_loaded:
            st.warning("No data available for analytics.")
            return
        
        # Correlation analysis
        st.markdown("### Correlation Analysis")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            corr_matrix = self.df[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu',
                title="Correlation Matrix"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Stress score distribution
        if 'market_stress_score' in self.df.columns:
            st.markdown("### Stress Score Distribution")
            
            fig = px.histogram(
                self.df,
                x='market_stress_score',
                nbins=30,
                title="Distribution of Market Stress Scores",
                color_discrete_sequence=['#3f51b5']
            )
            
            fig.update_layout(
                xaxis_title="Stress Score",
                yaxis_title="Frequency",
                bargap=0.1
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Stress score by instrument
            if 'instrument' in self.df.columns:
                st.markdown("### Stress Score by Instrument")
                
                stress_by_instrument = self.df.groupby('instrument')['market_stress_score'].mean().sort_values()
                
                fig = px.bar(
                    x=stress_by_instrument.values,
                    y=stress_by_instrument.index,
                    orientation='h',
                    title="Average Stress Score by Instrument",
                    color=stress_by_instrument.values,
                    color_continuous_scale='RdYlGn_r'
                )
                
                fig.update_layout(
                    xaxis_title="Average Stress Score",
                    yaxis_title="Instrument",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def render_settings_tab(self):
        """Render settings and configuration."""
        st.markdown("## System Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Pipeline Settings")
            
            update_frequency = st.selectbox(
                "Data Update Frequency",
                ["Manual", "Hourly", "Daily", "Weekly"],
                index=0
            )
            
            st.checkbox("Enable automatic anomaly detection", value=True)
            st.checkbox("Send email alerts for high stress", value=False)
            st.checkbox("Store raw scraped data", value=True)
            
            retention_days = st.slider(
                "Data Retention (days)",
                min_value=7,
                max_value=365,
                value=90
            )
        
        with col2:
            st.markdown("### Display Settings")
            
            st.selectbox("Theme", ["Light", "Dark", "Auto"])
            st.slider("Chart Animation Speed", 0.0, 2.0, 1.0, 0.1)
            st.checkbox("Show data quality warnings", value=True)
            st.checkbox("Auto-refresh dashboard", value=False)
        
        st.markdown("---")
        st.markdown("### System Information")
        
        info_cols = st.columns(3)
        
        with info_cols[0]:
            st.metric("Python Version", sys.version.split()[0])
            st.metric("Pandas Version", pd.__version__)
        
        with info_cols[1]:
            st.metric("Streamlit Version", st.__version__)
            st.metric("Plotly Version", '5.18.0')
        
        with info_cols[2]:
            if PIPELINE_AVAILABLE:
                st.success("Pipeline: Available")
            else:
                st.warning("Pipeline: Not Available")
            
            if MODELS_AVAILABLE:
                st.success("Models: Available")
            else:
                st.warning("Models: Not Available")
        
        # Clear cache button
        if st.button("Clear Cache", type="secondary"):
            st.cache_data.clear()
            st.success("Cache cleared successfully!")
    
    def run(self):
        """Main dashboard execution."""
        try:
            # Render header
            self.render_header()
            
            # Render sidebar
            self.render_sidebar()
            
            # Create tabs for different views
            tabs = st.tabs([
                "📊 Dashboard",
                "📈 Scraped Data",
                "🔍 Analytics",
                "⚙️ Settings"
            ])
            
            with tabs[0]:
                self.render_dashboard_tab()
            
            with tabs[1]:
                self.render_scraped_data_tab()
            
            with tabs[2]:
                self.render_analytics_tab()
            
            with tabs[3]:
                self.render_settings_tab()
            
            # Footer
            st.markdown("---")
            st.markdown("""
            <div style='text-align: center; color: #666; font-size: 0.9em; padding: 20px;'>
                Market Risk Intelligence Dashboard • Version 2.0.0 • 
                <span style='color: #3f51b5;'>Professional Financial Analytics Platform</span><br>
                Data updates every 15 minutes • Last updated: {}
            </div>
            """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Dashboard error: {str(e)}")
            with st.expander("Error Details"):
                st.code(traceback.format_exc())


def main():
    """Main entry point."""
    # Initialize dashboard
    dashboard = MarketRiskDashboard()
    
    # Run dashboard
    dashboard.run()


if __name__ == "__main__":
    main()
