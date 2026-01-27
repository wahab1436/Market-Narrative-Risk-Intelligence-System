"""
Market Narrative Risk Intelligence Dashboard
Professional analytics platform for market stress detection and risk monitoring
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

# Global timestamp for consistent file naming
from src.pipeline.utils.run_id import RUN_ID

# Page configuration
st.set_page_config(
    page_title="Market Risk Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import project modules with error handling
try:
    from src.utils.config_loader import config_loader
    print("Configuration loader imported")
except ImportError:
    config_loader = None

try:
    from src.utils.logger import get_dashboard_logger
    logger = get_dashboard_logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from src.scraper import scrape_and_save
    from src.preprocessing.clean_data import clean_and_save
    from src.preprocessing.feature_engineering import engineer_and_save
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

try:
    from src.models.regression.linear_regression import LinearRegressionModel
    from src.models.regression.ridge_regression import RidgeRegressionModel
    from src.models.regression.lasso_regression import LassoRegressionModel
    from src.models.neural_network import NeuralNetworkModel
    from src.models.xgboost_model import XGBoostModel
    from src.models.isolation_forest import IsolationForestModel
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False


class MarketRiskDashboard:
    """Professional market risk intelligence dashboard."""
    
    def __init__(self):
        """Initialize dashboard with professional settings."""
        self.logger = logger
        self.run_id = RUN_ID
        
        # Load configuration
        self.config = self._load_config()
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Initialize session state
        self._init_session_state()
        
        # Load data
        self.load_data()
        
        self.logger.info("Dashboard initialized successfully")
    
    def _load_config(self):
        """Load dashboard configuration."""
        try:
            if config_loader:
                return config_loader.get_config("config", {})
        except Exception as e:
            self.logger.warning(f"Config load failed: {e}")
        return {}
    
    def _init_session_state(self):
        """Initialize Streamlit session state."""
        defaults = {
            'data_loaded': False,
            'current_view': 'overview',
            'start_date': None,
            'end_date': None,
            'selected_regime': None,
            'anomaly_filter': 'All Data',
            'confidence_threshold': 0.7
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def load_data(self):
        """Load the most recent prediction data."""
        try:
            gold_dir = Path("data/gold")
            gold_dir.mkdir(parents=True, exist_ok=True)
            
            # Priority 1: Current run predictions
            current_file = gold_dir / f"predictions_{self.run_id}.parquet"
            if current_file.exists() and current_file.stat().st_size > 100:
                self.df = pd.read_parquet(current_file)
                self._finalize_data_load(str(current_file))
                return
            
            # Priority 2: Latest predictions file
            pred_files = list(gold_dir.glob("*predictions*.parquet"))
            if pred_files:
                latest_file = max(pred_files, key=lambda x: x.stat().st_mtime)
                if latest_file.stat().st_size > 100:
                    self.df = pd.read_parquet(latest_file)
                    self._finalize_data_load(str(latest_file))
                    return
            
            # Priority 3: Features file
            feature_files = list(gold_dir.glob("features_*.parquet"))
            if feature_files:
                latest_feature = max(feature_files, key=lambda x: x.stat().st_mtime)
                if latest_feature.stat().st_size > 100:
                    self.df = pd.read_parquet(latest_feature)
                    self._finalize_data_load(str(latest_feature))
                    return
            
            # Fallback: Create sample data
            self._create_sample_data()
            
        except Exception as e:
            self.logger.error(f"Data loading failed: {e}")
            self._create_sample_data()
    
    def _finalize_data_load(self, source_path):
        """Finalize data loading process."""
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        if not self.df.empty and 'timestamp' in self.df.columns:
            self.min_date = self.df['timestamp'].min().date()
            self.max_date = self.df['timestamp'].max().date()
            st.session_state.start_date = self.max_date - timedelta(days=30)
            st.session_state.end_date = self.max_date
        
        self.logger.info(f"Loaded {len(self.df)} records from {source_path}")
        st.session_state.data_loaded = True
    
    def _create_sample_data(self):
        """Create sample data for demonstration."""
        dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
        
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
        
        # Normalize probabilities
        probs = self.df[['prob_low', 'prob_medium', 'prob_high']].values
        probs = probs / probs.sum(axis=1, keepdims=True)
        self.df[['prob_low', 'prob_medium', 'prob_high']] = probs
        
        self.min_date = self.df['timestamp'].min().date()
        self.max_date = self.df['timestamp'].max().date()
        st.session_state.start_date = self.max_date - timedelta(days=30)
        st.session_state.end_date = self.max_date
        
        st.session_state.data_loaded = True
        self.logger.info("Created sample data for demonstration")
    
    def _run_full_pipeline(self):
        """Execute complete data pipeline."""
        if not PIPELINE_AVAILABLE:
            st.error("Pipeline components unavailable")
            return False
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Scraping phase
            status_text.text("Phase 1: Data Collection")
            progress_bar.progress(10)
            
            result = scrape_and_save()
            bronze_path = self._handle_scraper_result(result)
            if not bronze_path:
                return False
            
            progress_bar.progress(25)
            
            # Cleaning phase
            status_text.text("Phase 2: Data Validation")
            progress_bar.progress(30)
            
            silver_path = clean_and_save(bronze_path)
            if not silver_path:
                st.error("Data cleaning failed")
                return False
            
            progress_bar.progress(50)
            
            # Feature engineering
            status_text.text("Phase 3: Feature Engineering")
            progress_bar.progress(55)
            
            gold_path = engineer_and_save(silver_path)
            if not gold_path:
                st.error("Feature engineering failed")
                return False
            
            progress_bar.progress(70)
            
            # Model training
            status_text.text("Phase 4: Model Training")
            progress_bar.progress(75)
            
            df = pd.read_parquet(gold_path)
            success = self._train_models(df)
            
            if success:
                progress_bar.progress(100)
                status_text.text("Pipeline completed successfully")
                return True
            else:
                st.error("Model training failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}")
            st.error(f"Pipeline execution failed: {e}")
            return False
        finally:
            progress_bar.empty()
            status_text.empty()
    
    def _handle_scraper_result(self, result):
        """Handle scraper result (DataFrame or path)."""
        if result is None:
            st.error("Data collection failed")
            return None
        
        if hasattr(result, 'shape'):  # DataFrame
            bronze_path = Path("data/bronze") / f"yahoo_market_{self.run_id}.parquet"
            bronze_path.parent.mkdir(parents=True, exist_ok=True)
            result.to_parquet(bronze_path)
            return bronze_path
        else:  # Path object
            return Path(result) if result else None
    
    def _train_models(self, df):
        """Train all available models."""
        models = {
            'linear_regression': LinearRegressionModel(),
            'ridge_regression': RidgeRegressionModel(),
            'lasso_regression': LassoRegressionModel(),
            'neural_network': NeuralNetworkModel(),
            'xgboost': XGBoostModel(),
            'isolation_forest': IsolationForestModel()
        }
        
        predictions_dfs = []
        
        for model_name, model in models.items():
            try:
                model.train(df)
                predictions = model.predict(df)
                
                pred_cols = [col for col in predictions.columns 
                           if any(x in col for x in ['prediction', 'regime', 'anomaly'])]
                
                if pred_cols:
                    predictions_subset = predictions[['timestamp'] + pred_cols]
                    predictions_dfs.append(predictions_subset)
                
                # Save model
                model_dir = Path("models")
                model_dir.mkdir(exist_ok=True)
                model.save(model_dir / f"{model_name}_{self.run_id}.joblib")
                
            except Exception as e:
                self.logger.error(f"{model_name} failed: {e}")
                continue
        
        if predictions_dfs:
            final_predictions = df[['timestamp']].copy()
            for pred_df in predictions_dfs:
                final_predictions = final_predictions.merge(pred_df, on='timestamp', how='left')
            
            # Add original features
            feature_cols = [col for col in df.columns if col != 'timestamp']
            final_predictions = final_predictions.merge(df[['timestamp'] + feature_cols], on='timestamp', how='left')
            
            # Save predictions
            predictions_path = Path("data/gold") / f"predictions_{self.run_id}.parquet"
            final_predictions.to_parquet(predictions_path, index=False)
            return True
        
        return False
    
    def _run_quick_update(self):
        """Perform quick model update using existing data."""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            if not hasattr(self, 'df') or self.df.empty:
                st.error("No data available")
                return False
            
            df = self.df.copy().tail(500)
            
            status_text.text("Processing data...")
            progress_bar.progress(30)
            
            # Simple feature engineering
            if 'sentiment_polarity' in df.columns:
                df['sentiment_ma_7'] = df['sentiment_polarity'].rolling(7, min_periods=1).mean()
            if 'keyword_stress_score' in df.columns:
                df['stress_ma_7'] = df['keyword_stress_score'].rolling(7, min_periods=1).mean()
            
            status_text.text("Training models...")
            progress_bar.progress(60)
            
            from sklearn.linear_model import LinearRegression
            
            feature_cols = [col for col in df.columns if col not in ['timestamp', 'headline']]
            X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
            y = df.get('weighted_stress_score', np.random.rand(len(df)))
            
            lr = LinearRegression()
            lr.fit(X, y)
            df['quick_prediction'] = lr.predict(X)
            
            status_text.text("Saving results...")
            progress_bar.progress(90)
            
            output_file = Path("data/gold") / f"predictions_{self.run_id}.parquet"
            df.to_parquet(output_file, index=False)
            
            progress_bar.progress(100)
            status_text.text("Update completed")
            return True
            
        except Exception as e:
            st.error(f"Quick update failed: {e}")
            return False
        finally:
            progress_bar.empty()
            status_text.empty()
    
    def render_header(self):
        """Render professional header section."""
        st.markdown("""
        <div style='text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;'>
            <h1 style='color: white; margin-bottom: 0.5rem; font-size: 2.5rem;'>Market Risk Intelligence</h1>
            <p style='color: white; font-size: 1.2rem; opacity: 0.9;'>Advanced Analytics for Market Stress Detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render navigation sidebar."""
        with st.sidebar:
            # System status
            st.markdown("### System Status")
            
            status_cols = st.columns(2)
            with status_cols[0]:
                st.metric("Data Records", f"{len(self.df):,}" if hasattr(self, 'df') else "0")
            with status_cols[1]:
                st.metric("Pipeline", "Ready" if PIPELINE_AVAILABLE else "Offline")
            
            st.markdown("---")
            
            # Pipeline controls
            st.markdown("### Pipeline Control")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Full Analysis", type="primary", use_container_width=True,
                           disabled=not PIPELINE_AVAILABLE):
                    with st.spinner("Running full analysis..."):
                        if self._run_full_pipeline():
                            self.load_data()
                            st.success("Analysis completed")
                            st.rerun()
            
            with col2:
                if st.button("Quick Update", use_container_width=True):
                    with st.spinner("Updating models..."):
                        if self._run_quick_update():
                            self.load_data()
                            st.success("Update completed")
                            st.rerun()
            
            st.markdown("---")
            
            # Navigation
            st.markdown("### Analysis Views")
            views = {
                'overview': 'Dashboard Overview',
                'stress_analysis': 'Stress Analytics',
                'risk_regimes': 'Risk Classification',
                'anomaly_detection': 'Anomaly Detection',
                'model_performance': 'Model Performance',
                'feature_analysis': 'Feature Analysis'
            }
            
            selected_view = st.selectbox(
                "Select View",
                options=list(views.keys()),
                format_func=lambda x: views[x]
            )
            st.session_state.current_view = selected_view
            
            st.markdown("---")
            
            # Filters
            st.markdown("### Data Filters")
            
            if hasattr(self, 'min_date') and hasattr(self, 'max_date'):
                col1, col2 = st.columns(2)
                with col1:
                    st.session_state.start_date = st.date_input(
                        "From Date",
                        value=st.session_state.start_date,
                        min_value=self.min_date,
                        max_value=self.max_date
                    )
                with col2:
                    st.session_state.end_date = st.date_input(
                        "To Date", 
                        value=st.session_state.end_date,
                        min_value=self.min_date,
                        max_value=self.max_date
                    )
            
            if 'xgboost_risk_regime' in self.df.columns:
                regimes = ['All'] + self.df['xgboost_risk_regime'].dropna().unique().tolist()
                st.session_state.selected_regime = st.selectbox(
                    "Risk Level",
                    regimes,
                    index=0
                )
            
            st.session_state.confidence_threshold = st.slider(
                "Confidence Threshold",
                0.0, 1.0, 0.7, 0.05
            )
    
    def _apply_filters(self):
        """Apply filters to current data."""
        df = self.df.copy()
        
        # Date filter
        if 'timestamp' in df.columns:
            mask = (df['timestamp'].dt.date >= st.session_state.start_date) & \
                   (df['timestamp'].dt.date <= st.session_state.end_date)
            df = df[mask]
        
        # Risk regime filter
        if (st.session_state.selected_regime and 
            st.session_state.selected_regime != 'All' and 
            'xgboost_risk_regime' in df.columns):
            df = df[df['xgboost_risk_regime'] == st.session_state.selected_regime]
        
        return df
    
    def render_overview(self):
        """Render main dashboard overview."""
        st.markdown("## Dashboard Overview")
        
        filtered_df = self._apply_filters()
        
        if filtered_df.empty:
            st.warning("No data available for selected filters")
            return
        
        # Key metrics
        metrics_cols = st.columns(4)
        
        with metrics_cols[0]:
            if 'weighted_stress_score' in filtered_df.columns:
                current = filtered_df['weighted_stress_score'].iloc[-1]
                avg = filtered_df['weighted_stress_score'].mean()
                st.metric("Stress Score", f"{current:.2f}", f"{current-avg:+.2f}")
        
        with metrics_cols[1]:
            if 'xgboost_risk_regime' in filtered_df.columns:
                regime = filtered_df['xgboost_risk_regime'].iloc[-1]
                color = {'low': '#10b981', 'medium': '#f59e0b', 'high': '#ef4444'}.get(regime, '#6b7280')
                st.markdown(f"<h3 style='color: {color}; text-align: center;'>{regime.upper()}</h3>", 
                           unsafe_allow_html=True)
                st.markdown("<p style='text-align: center;'>Risk Level</p>", unsafe_allow_html=True)
        
        with metrics_cols[2]:
            if 'is_anomaly' in filtered_df.columns:
                anomalies = filtered_df['is_anomaly'].sum()
                rate = (anomalies / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
                st.metric("Anomalies", f"{anomalies:.0f}", f"{rate:.1f}%")
        
        with metrics_cols[3]:
            st.metric("Articles", f"{len(filtered_df):,}")
        
        # Stress score chart
        if 'weighted_stress_score' in filtered_df.columns:
            fig = px.line(
                filtered_df, 
                x='timestamp', 
                y='weighted_stress_score',
                title='Market Stress Over Time'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent data table
        st.markdown("### Recent Market Data")
        display_cols = [c for c in ['timestamp', 'headline', 'weighted_stress_score', 'xgboost_risk_regime'] 
                       if c in filtered_df.columns]
        
        if display_cols:
            recent_data = filtered_df[display_cols].tail(10).copy()
            if 'timestamp' in recent_data.columns:
                recent_data['timestamp'] = recent_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(recent_data, use_container_width=True)
    
    def render_stress_analysis(self):
        """Render stress analysis view."""
        st.markdown("## Stress Score Analysis")
        
        filtered_df = self._apply_filters()
        
        if filtered_df.empty or 'weighted_stress_score' not in filtered_df.columns:
            st.info("Stress score data not available")
            return
        
        # Time series with trend
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=filtered_df['timestamp'],
            y=filtered_df['weighted_stress_score'],
            mode='lines',
            name='Stress Score',
            line=dict(color=self.colors[0], width=3)
        ))
        
        # Add 7-day moving average
        if len(filtered_df) > 7:
            ma_7 = filtered_df['weighted_stress_score'].rolling(7).mean()
            fig.add_trace(go.Scatter(
                x=filtered_df['timestamp'],
                y=ma_7,
                mode='lines',
                name='7-Day MA',
                line=dict(color=self.colors[1], width=2, dash='dash')
            ))
        
        fig.update_layout(
            title='Stress Score Analysis',
            height=500,
            xaxis_title="Date",
            yaxis_title="Stress Score"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        stats_cols = st.columns(4)
        stats = filtered_df['weighted_stress_score']
        
        with stats_cols[0]:
            st.metric("Current", f"{stats.iloc[-1]:.2f}")
        with stats_cols[1]:
            st.metric("Average", f"{stats.mean():.2f}")
        with stats_cols[2]:
            st.metric("Maximum", f"{stats.max():.2f}")
        with stats_cols[3]:
            st.metric("Volatility", f"{stats.std():.2f}")
    
    def render_risk_regimes(self):
        """Render risk regime analysis."""
        st.markdown("## Risk Regime Analysis")
        
        filtered_df = self._apply_filters()
        
        if filtered_df.empty:
            st.info("No data available")
            return
        
        if 'xgboost_risk_regime' in filtered_df.columns:
            # Distribution
            regime_counts = filtered_df['xgboost_risk_regime'].value_counts()
            
            fig = px.pie(
                values=regime_counts.values,
                names=regime_counts.index,
                title='Risk Regime Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Timeline
            if 'weighted_stress_score' in filtered_df.columns:
                fig2 = px.scatter(
                    filtered_df,
                    x='timestamp',
                    y='weighted_stress_score',
                    color='xgboost_risk_regime',
                    title='Risk Regimes Over Time',
                    color_discrete_map={
                        'low': '#10b981',
                        'medium': '#f59e0b', 
                        'high': '#ef4444'
                    }
                )
                st.plotly_chart(fig2, use_container_width=True)
    
    def render_anomaly_detection(self):
        """Render anomaly detection view."""
        st.markdown("## Anomaly Detection")
        
        filtered_df = self._apply_filters()
        
        if filtered_df.empty:
            st.info("No data available")
            return
        
        if 'is_anomaly' in filtered_df.columns and 'weighted_stress_score' in filtered_df.columns:
            # Scatter plot
            normal = filtered_df[filtered_df['is_anomaly'] == 0]
            anomalies = filtered_df[filtered_df['is_anomaly'] == 1]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=normal['timestamp'],
                y=normal['weighted_stress_score'],
                mode='markers',
                name='Normal',
                marker=dict(color='lightblue', size=6)
            ))
            
            fig.add_trace(go.Scatter(
                x=anomalies['timestamp'],
                y=anomalies['weighted_stress_score'],
                mode='markers',
                name='Anomaly',
                marker=dict(color='red', size=10, symbol='diamond')
            ))
            
            fig.update_layout(
                title='Anomaly Detection',
                height=500,
                xaxis_title="Date",
                yaxis_title="Stress Score"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Stats
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Anomalies Detected", int(filtered_df['is_anomaly'].sum()))
            with col2:
                rate = (filtered_df['is_anomaly'].sum() / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
                st.metric("Detection Rate", f"{rate:.1f}%")
    
    def render_model_performance(self):
        """Render model performance comparison."""
        st.markdown("## Model Performance")
        
        filtered_df = self._apply_filters()
        
        if filtered_df.empty:
            st.info("No data available")
            return
        
        pred_cols = [col for col in filtered_df.columns if 'prediction' in col.lower()]
        
        if pred_cols and 'weighted_stress_score' in filtered_df.columns:
            # Comparison chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=filtered_df['timestamp'],
                y=filtered_df['weighted_stress_score'],
                mode='lines',
                name='Actual',
                line=dict(color='black', width=3)
            ))
            
            for i, col in enumerate(pred_cols[:4]):
                fig.add_trace(go.Scatter(
                    x=filtered_df['timestamp'],
                    y=filtered_df[col],
                    mode='lines',
                    name=col.replace('_', ' ').title(),
                    line=dict(width=2, dash='dash')
                ))
            
            fig.update_layout(
                title='Model Predictions vs Actual',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_feature_analysis(self):
        """Render feature correlation analysis."""
        st.markdown("## Feature Analysis")
        
        filtered_df = self._apply_filters()
        
        if filtered_df.empty:
            st.info("No data available")
            return
        
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 1:
            # Correlation matrix
            corr_matrix = filtered_df[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                title='Feature Correlation Matrix',
                color_continuous_scale='RdBu',
                aspect='auto'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature statistics
            st.markdown("### Feature Statistics")
            stats_df = filtered_df[numeric_cols].describe().round(3)
            st.dataframe(stats_df, use_container_width=True)
    
    def render_footer(self):
        """Render professional footer."""
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #6b7280; padding: 2rem 0;'>"
            "Market Risk Intelligence System v1.0.0 | Professional Analytics Platform"
            "</div>",
            unsafe_allow_html=True
        )
    
    def run(self):
        """Execute dashboard application."""
        try:
            self.render_header()
            self.render_sidebar()
            
            # Route to appropriate view
            view_handlers = {
                'overview': self.render_overview,
                'stress_analysis': self.render_stress_analysis,
                'risk_regimes': self.render_risk_regimes,
                'anomaly_detection': self.render_anomaly_detection,
                'model_performance': self.render_model_performance,
                'feature_analysis': self.render_feature_analysis
            }
            
            current_view = st.session_state.current_view
            handler = view_handlers.get(current_view, self.render_overview)
            handler()
            
            self.render_footer()
            
        except Exception as e:
            st.error("Application error occurred")
            with st.expander("Error Details"):
                st.code(traceback.format_exc())


def main():
    """Main entry point."""
    try:
        dashboard = MarketRiskDashboard()
        dashboard.run()
    except Exception as e:
        st.error("Failed to initialize dashboard")
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
