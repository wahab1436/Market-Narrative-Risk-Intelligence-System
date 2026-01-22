"""
Professional Market Narrative Risk Intelligence Dashboard.
Modern, clean interface with comprehensive data visualization.
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

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import project modules with proper error handling
try:
    # Import configuration and logging first
    from src.utils.config_loader import config_loader
    print("✓ Config loader imported successfully")
except ImportError as e:
    st.error(f"Failed to import config_loader: {e}")
    config_loader = None

try:
    # Import logger with simplified approach - FIXED
    import logging
    from src.utils.logger import get_dashboard_logger
    
    # Create dashboard logger
    logger = get_dashboard_logger()
    print("✓ Logger imported successfully")
except ImportError as e:
    st.error(f"Failed to import logger: {e}")
    # Create a minimal logger as fallback
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

try:
    # Try to import EDA visualizer
    from src.eda.visualization import EDAVisualizer
    print("✓ EDAVisualizer imported successfully")
except ImportError as e:
    logger.warning(f"EDAVisualizer not available: {e}")
    EDAVisualizer = None

try:
    # Try to import models (optional for dashboard)
    from src.models.regression.linear_regression import LinearRegressionModel
    from src.models.regression.ridge_regression import RidgeRegressionModel
    from src.models.regression.lasso_regression import LassoRegressionModel
    print("✓ Model imports successful")
except ImportError as e:
    logger.info(f"Model imports optional, continuing without: {e}")
    LinearRegressionModel = RidgeRegressionModel = LassoRegressionModel = None


class MarketRiskDashboard:
    """
    Professional dashboard for market narrative risk intelligence.
    """
    
    def __init__(self):
        """Initialize dashboard with professional settings."""
        # Page configuration
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
        
        # Initialize logger - FIXED
        self.logger = logger
        
        # Initialize configuration
        try:
            if config_loader:
                self.config = config_loader.get_dashboard_config()
                self.colors = self.config.get('color_palette', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
                print("✓ Configuration loaded successfully")
            else:
                self.config = {}
                self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        except Exception as e:
            logger.warning(f"Failed to load dashboard config: {e}")
            self.config = {}
            self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # Initialize EDA visualizer if available
        if EDAVisualizer:
            self.eda_visualizer = EDAVisualizer(theme="plotly_white")
        else:
            self.eda_visualizer = None
        
        # Initialize session state
        self._init_session_state()
        
        # Load data
        self.load_data()
        
        # Log initialization
        self.logger.info("MarketRiskDashboard initialized successfully")
    
    def _init_session_state(self):
        """Initialize session state variables."""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'current_view' not in st.session_state:
            st.session_state.current_view = 'overview'
        if 'filtered_data' not in st.session_state:
            st.session_state.filtered_data = None
        if 'model_predictions' not in st.session_state:
            st.session_state.model_predictions = {}
    
    def load_data(self):
        """Load prediction data from gold layer."""
        try:
            gold_dir = Path("data/gold")
            
            if not gold_dir.exists():
                gold_dir.mkdir(parents=True, exist_ok=True)
            
            # Look for prediction files first
            prediction_files = list(gold_dir.glob("*predictions*.parquet"))
            
            if prediction_files:
                # Load the most recent predictions
                latest_file = max(prediction_files, key=lambda x: x.stat().st_mtime)
                self.df = pd.read_parquet(latest_file)
                
                # Ensure timestamp is datetime
                if 'timestamp' in self.df.columns:
                    self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
                
                # Set date range for filters
                if not self.df.empty and 'timestamp' in self.df.columns:
                    self.min_date = self.df['timestamp'].min().date()
                    self.max_date = self.df['timestamp'].max().date()
                
                self.logger.info(f"✓ Loaded {len(self.df)} records from {latest_file}")
                st.session_state.data_loaded = True
                
            else:
                # Try to load any gold data (features)
                gold_files = list(gold_dir.glob("features_*.parquet"))
                if gold_files:
                    latest_file = max(gold_files, key=lambda x: x.stat().st_mtime)
                    self.df = pd.read_parquet(latest_file)
                    
                    if 'timestamp' in self.df.columns:
                        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
                        self.min_date = self.df['timestamp'].min().date()
                        self.max_date = self.df['timestamp'].max().date()
                    
                    self.logger.info(f"✓ Loaded {len(self.df)} records from {latest_file}")
                    st.session_state.data_loaded = True
                else:
                    # Check silver layer
                    silver_dir = Path("data/silver")
                    if silver_dir.exists():
                        silver_files = list(silver_dir.glob("*.parquet"))
                        if silver_files:
                            latest_file = max(silver_files, key=lambda x: x.stat().st_mtime)
                            self.df = pd.read_parquet(latest_file)
                            
                            if 'timestamp' in self.df.columns:
                                self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
                                self.min_date = self.df['timestamp'].min().date()
                                self.max_date = self.df['timestamp'].max().date()
                            
                            self.logger.info(f"✓ Loaded {len(self.df)} records from silver layer: {latest_file}")
                            st.session_state.data_loaded = True
                        else:
                            self._create_sample_data()
                            self.logger.warning("No data files found, using sample data")
                    else:
                        self._create_sample_data()
                        self.logger.warning("No data directories found, using sample data")
                    
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}", exc_info=True)
            self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample data for demonstration."""
        dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
        
        # Create realistic sample data
        np.random.seed(42)
        base_trend = np.linspace(0, 2, len(dates))
        seasonal = 0.5 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)
        noise = np.random.normal(0, 0.3, len(dates))
        
        stress_scores = base_trend + seasonal + noise
        
        self.df = pd.DataFrame({
            'timestamp': dates,
            'headline': [f"Market update {i}" for i in range(len(dates))],
            'sentiment_polarity': np.random.uniform(-0.8, 0.8, len(dates)),
            'keyword_stress_score': np.random.exponential(0.3, len(dates)),
            'weighted_stress_score': stress_scores,
            'linear_regression_prediction': stress_scores + np.random.normal(0, 0.2, len(dates)),
            'ridge_regression_prediction': stress_scores + np.random.normal(0, 0.15, len(dates)),
            'lasso_regression_prediction': stress_scores + np.random.normal(0, 0.18, len(dates)),
            'neural_network_prediction': stress_scores + np.random.normal(0, 0.12, len(dates)),
            'xgboost_risk_regime': np.random.choice(['low', 'medium', 'high'], len(dates), p=[0.3, 0.5, 0.2]),
            'prob_low': np.random.uniform(0, 1, len(dates)),
            'prob_medium': np.random.uniform(0, 1, len(dates)),
            'prob_high': np.random.uniform(0, 1, len(dates)),
            'is_anomaly': np.random.choice([0, 1], len(dates), p=[0.92, 0.08]),
            'anomaly_score': np.random.uniform(-0.5, 0.5, len(dates)),
            'daily_article_count': np.random.poisson(50, len(dates)),
            'market_breadth': np.random.uniform(0, 1, len(dates))
        })
        
        # Ensure probabilities sum to 1
        probs = self.df[['prob_low', 'prob_medium', 'prob_high']].values
        probs = probs / probs.sum(axis=1, keepdims=True)
        self.df[['prob_low', 'prob_medium', 'prob_high']] = probs
        
        self.min_date = self.df['timestamp'].min().date()
        self.max_date = self.df['timestamp'].max().date()
        
        st.session_state.data_loaded = True
        self.logger.info("Created sample data for demonstration")
    
    def render_header(self):
        """Render professional header."""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
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
            # System Information
            st.markdown("""
            <div style='padding: 10px; background-color: #f8f9fa; border-radius: 5px; margin-bottom: 20px;'>
                <p style='font-size: 0.9em; color: #6c757d; margin: 0;'>
                <strong>System Status:</strong> Operational<br>
                <strong>Last Updated:</strong> {date}<br>
                <strong>Records Loaded:</strong> {records:,}
                </p>
            </div>
            """.format(
                date=datetime.now().strftime("%Y-%m-%d %H:%M"),
                records=len(self.df) if hasattr(self, 'df') else 0
            ), unsafe_allow_html=True)
            
            st.markdown("### Navigation")
            
            # View selection
            view_options = {
                'overview': 'System Overview',
                'stress_analysis': 'Stress Score Analysis',
                'risk_regimes': 'Risk Regime Classification',
                'anomaly_detection': 'Anomaly Detection',
                'historical_similarity': 'Historical Similarity',
                'model_performance': 'Model Performance',
                'feature_analysis': 'Feature Analysis',
                'data_export': 'Data Export'
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
            
            # Date range filter
            if hasattr(self, 'min_date') and hasattr(self, 'max_date'):
                date_range = st.date_input(
                    "Analysis Period",
                    value=(self.max_date - timedelta(days=30), self.max_date),
                    min_value=self.min_date,
                    max_value=self.max_date
                )
                
                if len(date_range) == 2:
                    self.start_date, self.end_date = date_range
                else:
                    self.start_date = self.end_date = date_range[0] if date_range else self.max_date
            else:
                self.start_date = self.end_date = datetime.now().date()
            
            # Risk regime filter
            if 'xgboost_risk_regime' in self.df.columns:
                risk_regimes = ['All'] + sorted(self.df['xgboost_risk_regime'].dropna().unique().tolist())
                selected_regime = st.selectbox(
                    "Risk Regime Filter",
                    risk_regimes,
                    index=0
                )
                self.selected_regime = selected_regime if selected_regime != 'All' else None
            else:
                self.selected_regime = None
            
            # Anomaly filter
            if 'is_anomaly' in self.df.columns:
                anomaly_filter = st.radio(
                    "Anomaly Filter",
                    ['All Data', 'Anomalies Only', 'Exclude Anomalies'],
                    index=0
                )
                self.anomaly_filter = anomaly_filter
            else:
                self.anomaly_filter = 'All Data'
            
            # Confidence threshold
            confidence_threshold = st.slider(
                "Minimum Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="Filter predictions by model confidence"
            )
            self.confidence_threshold = confidence_threshold
            
            st.markdown("---")
            
            # Data statistics
            with st.expander("Data Statistics", expanded=False):
                if st.session_state.data_loaded:
                    total_records = len(self.df)
                    filtered_df = self._apply_filters(self.df)
                    filtered_records = len(filtered_df)
                    
                    st.metric("Total Records", f"{total_records:,}")
                    st.metric("Filtered Records", f"{filtered_records:,}")
                    
                    if 'xgboost_risk_regime' in filtered_df.columns:
                        regime_counts = filtered_df['xgboost_risk_regime'].value_counts()
                        for regime, count in regime_counts.items():
                            st.metric(f"{regime.title()} Risk", f"{count:,}")
    
    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all sidebar filters to data."""
        filtered_df = df.copy()
        
        # Date filter
        if 'timestamp' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['timestamp'].dt.date >= self.start_date) &
                (filtered_df['timestamp'].dt.date <= self.end_date)
            ]
        
        # Risk regime filter
        if self.selected_regime and 'xgboost_risk_regime' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['xgboost_risk_regime'] == self.selected_regime]
        
        # Anomaly filter
        if 'is_anomaly' in filtered_df.columns:
            if self.anomaly_filter == 'Anomalies Only':
                filtered_df = filtered_df[filtered_df['is_anomaly'] == 1]
            elif self.anomaly_filter == 'Exclude Anomalies':
                filtered_df = filtered_df[filtered_df['is_anomaly'] == 0]
        
        # Confidence filter for classification models
        prob_cols = [col for col in filtered_df.columns if col.startswith('prob_')]
        if prob_cols:
            max_probs = filtered_df[prob_cols].max(axis=1)
            filtered_df = filtered_df[max_probs >= self.confidence_threshold]
        
        return filtered_df
    
    # [REST OF THE METHODS REMAIN THE SAME - keeping them for completeness]
    def render_overview(self):
        """Render system overview dashboard."""
        st.markdown("## System Overview")
        
        filtered_df = self._apply_filters(self.df)
        
        # Show data status
        if filtered_df.empty:
            st.warning("No data matches current filters. Please adjust your filter settings.")
            return
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'weighted_stress_score' in filtered_df.columns:
                current_stress = filtered_df['weighted_stress_score'].iloc[-1] if len(filtered_df) > 0 else 0
                avg_stress = filtered_df['weighted_stress_score'].mean()
                st.metric(
                    "Current Stress Score",
                    f"{current_stress:.2f}",
                    f"{current_stress - avg_stress:+.2f} vs avg"
                )
            else:
                st.info("Stress score data not available")
        
        with col2:
            if 'xgboost_risk_regime' in filtered_df.columns:
                current_regime = filtered_df['xgboost_risk_regime'].iloc[-1] if len(filtered_df) > 0 else "Unknown"
                regime_color = {
                    'low': 'green',
                    'medium': 'orange',
                    'high': 'red'
                }.get(current_regime, 'gray')
                
                st.markdown(f"""
                <div style='text-align: center; padding: 10px;'>
                    <div style='font-size: 0.9em; color: #6c757d;'>Current Risk Regime</div>
                    <div style='font-size: 1.5em; font-weight: bold; color: {regime_color};'>{current_regime.upper()}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Risk regime data not available")
        
        with col3:
            if 'is_anomaly' in filtered_df.columns:
                anomaly_count = int(filtered_df['is_anomaly'].sum())
                anomaly_rate = (anomaly_count / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
                st.metric(
                    "Anomalies Detected",
                    f"{anomaly_count}",
                    f"{anomaly_rate:.1f}% rate"
                )
            else:
                st.info("Anomaly data not available")
        
        with col4:
            total_articles = len(filtered_df)
            date_range = (filtered_df['timestamp'].max() - filtered_df['timestamp'].min()).days if 'timestamp' in filtered_df.columns else 1
            avg_daily = total_articles / max(date_range, 1)
            st.metric(
                "Articles Analyzed",
                f"{total_articles:,}",
                f"{avg_daily:.0f}/day avg"
            )
        
        st.markdown("---")
        
        # Show simple data table
        st.markdown("### Recent Articles")
        display_cols = ['timestamp', 'headline', 'sentiment_polarity', 'weighted_stress_score']
        display_cols = [col for col in display_cols if col in filtered_df.columns]
        
        if display_cols:
            recent_data = filtered_df[display_cols].tail(10).copy()
            if 'timestamp' in recent_data.columns:
                recent_data['timestamp'] = recent_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(recent_data, use_container_width=True, hide_index=True)
        else:
            st.info("No displayable columns available in the data")
    
    # Placeholder methods for other views
    def render_stress_analysis(self):
        st.markdown("## Stress Analysis")
        st.info("Detailed stress analysis view - Coming soon after model training completes")
    
    def render_risk_regimes(self):
        st.markdown("## Risk Regimes")
        st.info("Risk regime analysis - Coming soon after model training completes")
    
    def render_anomaly_detection(self):
        st.markdown("## Anomaly Detection")
        st.info("Anomaly detection view - Coming soon after model training completes")
    
    def render_historical_similarity(self):
        st.markdown("## Historical Similarity")
        st.info("Historical similarity analysis - Coming soon after model training completes")
    
    def render_model_performance(self):
        st.markdown("## Model Performance")
        st.info("Model performance metrics - Coming soon after model training completes")
    
    def render_feature_analysis(self):
        st.markdown("## Feature Analysis")
        st.info("Feature importance analysis - Coming soon after model training completes")
    
    def render_data_export(self):
        st.markdown("## Data Export")
        st.info("Data export functionality - Coming soon")
    
    def render_footer(self):
        """Render professional footer."""
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #6c757d; font-size: 0.9em; padding: 20px;'>
        Market Narrative Risk Intelligence System v1.0.0 © 2024<br>
        ✨ Real-time market analysis powered by machine learning
        </div>
        """, unsafe_allow_html=True)
    
    def run(self):
        """Run the dashboard application."""
        try:
            # Render header
            self.render_header()
            
            # Render sidebar (navigation and filters)
            self.render_sidebar()
            
            # Render main content based on selected view
            view_handlers = {
                'overview': self.render_overview,
                'stress_analysis': self.render_stress_analysis,
                'risk_regimes': self.render_risk_regimes,
                'anomaly_detection': self.render_anomaly_detection,
                'historical_similarity': self.render_historical_similarity,
                'model_performance': self.render_model_performance,
                'feature_analysis': self.render_feature_analysis,
                'data_export': self.render_data_export
            }
            
            current_view = st.session_state.get('current_view', 'overview')
            handler = view_handlers.get(current_view, self.render_overview)
            
            # Execute the view handler
            handler()
            
            # Render footer
            self.render_footer()
            
            self.logger.info(f"Dashboard view '{current_view}' rendered successfully")
            
        except Exception as e:
            self.logger.error(f"Dashboard error: {e}", exc_info=True)
            st.error(f"⚠️ An error occurred: {str(e)}")
            with st.expander("Error Details"):
                st.code(traceback.format_exc())


def main():
    """Main entry point for the dashboard."""
    try:
        dashboard = MarketRiskDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"❌ Failed to initialize dashboard: {e}")
        st.code(traceback.format_exc())
        st.stop()


if __name__ == "__main__":
    main()
