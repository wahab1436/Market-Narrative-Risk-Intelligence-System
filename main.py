"""
UNIFIED MAIN ENTRY POINT
Market Narrative Risk Intelligence System
Backend Pipeline + Frontend Dashboard in One File

Usage:
    python main.py                    # Run complete pipeline (backend)
    python main.py --dashboard        # Run dashboard (frontend)
    streamlit run main.py             # Run dashboard (frontend)
    python main.py --scrape-only      # Run only scraping
    python main.py --clean-only       # Run only cleaning
    python main.py --features-only    # Run only feature engineering
    python main.py --train-only       # Run only model training
"""
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import contextmanager

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import critical dependencies first
import pandas as pd
import numpy as np

# Check if running under Streamlit
try:
    import streamlit.runtime.scriptrunner.script_run_context as script_run_context
    IS_STREAMLIT = script_run_context.get_script_run_ctx() is not None
except:
    IS_STREAMLIT = False

# Import Streamlit if available
if IS_STREAMLIT or '--dashboard' in sys.argv:
    try:
        import streamlit as st
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        STREAMLIT_AVAILABLE = True
    except ImportError:
        STREAMLIT_AVAILABLE = False
        print("Warning: Streamlit not available. Install with: pip install streamlit plotly")
else:
    STREAMLIT_AVAILABLE = False

# Import backend modules
from src.utils.config_loader import config_loader
from src.utils.logger import get_pipeline_logger, get_dashboard_logger, setup_pipeline_logging
from src.scraper import scrape_and_save
from src.preprocessing.clean_data import clean_and_save
from src.preprocessing.feature_engineering import engineer_and_save
from src.models.regression.linear_regression import LinearRegressionModel
from src.models.regression.ridge_regression import RidgeRegressionModel
from src.models.regression.lasso_regression import LassoRegressionModel
from src.models.regression.polynomial_regression import PolynomialRegressionModel
from src.models.neural_network import NeuralNetworkModel
from src.models.xgboost_model import XGBoostModel
from src.models.knn_model import KNNModel
from src.models.isolation_forest import IsolationForestModel


# =============================================================================
# BACKEND: PIPELINE ORCHESTRATOR
# =============================================================================

@contextmanager
def LoggingContext(logger, context_name: str):
    """Context manager for logging."""
    logger.info(f"Starting {context_name}")
    try:
        yield
    finally:
        logger.info(f"Completed {context_name}")


class PipelineOrchestrator:
    """
    Backend Pipeline Orchestrator.
    Handles data collection, cleaning, feature engineering, and model training.
    """
    
    def __init__(self, run_all: bool = True):
        """Initialize pipeline orchestrator."""
        self.config = config_loader.get_config("config")
        
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        setup_pipeline_logging(level=log_level)
        
        self.logger = get_pipeline_logger()
        self.run_all = run_all
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.bronze_path = None
        self.silver_path = None
        self.gold_path = None
        self.execution_metrics = {}
        
        print("=" * 80)
        print("MARKET NARRATIVE RISK INTELLIGENCE SYSTEM")
        print("=" * 80)
        self.logger.info("PipelineOrchestrator initialized")
        
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary directories."""
        with LoggingContext(self.logger, "directory_setup"):
            directories = [
                Path("data/bronze"),
                Path("data/silver"),
                Path("data/gold"),
                Path("logs"),
                Path("models"),
                Path("config"),
            ]
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
    
    def run_scraping(self) -> bool:
        """Execute scraping phase."""
        with LoggingContext(self.logger, "scraping_phase"):
            try:
                self.logger.info("Starting scraping phase")
                scraping_config = config_loader.get_scraping_config()
                use_selenium = scraping_config.get("use_selenium", False)
                self.logger.info(f"Scraping configuration: use_selenium={use_selenium}")
                
                self.bronze_path = scrape_and_save()
                
                if self.bronze_path:
                    self.logger.info(f"Scraping completed successfully: {self.bronze_path}")
                    df = pd.read_parquet(self.bronze_path)
                    self.execution_metrics['scraping'] = {
                        'articles_collected': len(df),
                        'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
                        'file_size_mb': Path(self.bronze_path).stat().st_size / 1024 / 1024
                    }
                    print(f"✓ Scraping completed: {self.bronze_path}")
                    return True
                else:
                    self.logger.warning("Scraping completed but no data was collected")
                    return False
                    
            except Exception as e:
                self.logger.error(f"Scraping failed: {e}", exc_info=True)
                print(f"✗ Scraping failed: {e}")
                return False
    
    def run_cleaning(self) -> bool:
        """Execute data cleaning phase."""
        with LoggingContext(self.logger, "cleaning_phase"):
            try:
                self.logger.info("Starting data cleaning phase")
                
                if self.bronze_path is None:
                    bronze_dir = Path("data/bronze")
                    files = list(bronze_dir.glob("*.parquet"))
                    if not files:
                        self.logger.error("No bronze files found for cleaning")
                        return False
                    self.bronze_path = max(files, key=lambda x: x.stat().st_mtime)
                    self.logger.info(f"Using latest bronze file: {self.bronze_path}")
                
                self.silver_path = clean_and_save(self.bronze_path)
                
                if self.silver_path:
                    self.logger.info(f"Data cleaning completed successfully: {self.silver_path}")
                    df = pd.read_parquet(self.silver_path)
                    original_df = pd.read_parquet(self.bronze_path)
                    
                    self.execution_metrics['cleaning'] = {
                        'original_records': len(original_df),
                        'cleaned_records': len(df),
                        'records_removed': len(original_df) - len(df),
                    }
                    print(f"✓ Cleaning completed: {self.silver_path}")
                    return True
                else:
                    self.logger.warning("Cleaning completed but no valid data")
                    return False
                    
            except Exception as e:
                self.logger.error(f"Data cleaning failed: {e}", exc_info=True)
                print(f"✗ Cleaning failed: {e}")
                return False
    
    def run_feature_engineering(self) -> bool:
        """Execute feature engineering phase."""
        with LoggingContext(self.logger, "feature_engineering_phase"):
            try:
                self.logger.info("Starting feature engineering phase")
                
                if self.silver_path is None:
                    silver_dir = Path("data/silver")
                    files = list(silver_dir.glob("*.parquet"))
                    if not files:
                        self.logger.error("No silver files found for feature engineering")
                        return False
                    self.silver_path = max(files, key=lambda x: x.stat().st_mtime)
                    self.logger.info(f"Using latest silver file: {self.silver_path}")
                
                self.gold_path = engineer_and_save(self.silver_path)
                
                if self.gold_path:
                    self.logger.info(f"Feature engineering completed successfully: {self.gold_path}")
                    df = pd.read_parquet(self.gold_path)
                    
                    self.execution_metrics['feature_engineering'] = {
                        'records_processed': len(df),
                        'features_created': len(df.columns),
                    }
                    print(f"✓ Feature engineering completed: {self.gold_path}")
                    return True
                else:
                    self.logger.warning("Feature engineering completed but no features created")
                    return False
                    
            except Exception as e:
                self.logger.error(f"Feature engineering failed: {e}", exc_info=True)
                print(f"✗ Feature engineering failed: {e}")
                return False
    
    def run_model_training(self) -> pd.DataFrame:
        """Execute model training phase."""
        with LoggingContext(self.logger, "model_training_phase"):
            try:
                self.logger.info("Starting model training phase")
                
                if self.gold_path is None:
                    gold_dir = Path("data/gold")
                    files = list(gold_dir.glob("features_*.parquet"))
                    if not files:
                        self.logger.error("No gold files found for model training")
                        return pd.DataFrame()
                    self.gold_path = max(files, key=lambda x: x.stat().st_mtime)
                    self.logger.info(f"Using latest gold file: {self.gold_path}")
                
                self.logger.info(f"Loading data from {self.gold_path}")
                df = pd.read_parquet(self.gold_path)
                self.logger.info(f"Loaded {len(df)} records with {len(df.columns)} features for model training")
                
                models = {
                    'linear_regression': LinearRegressionModel(),
                    'ridge_regression': RidgeRegressionModel(),
                    'lasso_regression': LassoRegressionModel(),
                    'polynomial_regression': PolynomialRegressionModel(),
                    'neural_network': NeuralNetworkModel(),
                    'xgboost': XGBoostModel(),
                    'knn': KNNModel(),
                    'isolation_forest': IsolationForestModel()
                }
                
                predictions_dfs = []
                model_metrics = {}
                
                for model_name, model in models.items():
                    with LoggingContext(self.logger, f"{model_name}_training"):
                        try:
                            self.logger.info(f"Training {model_name}")
                            results = model.train(df)
                            predictions = model.predict(df)
                            
                            pred_cols = [col for col in predictions.columns 
                                       if any(x in col for x in ['prediction', 'regime', 'anomaly', 'similarity', 'forecast'])]
                            
                            if pred_cols:
                                predictions_subset = predictions[['timestamp'] + pred_cols]
                                predictions_dfs.append(predictions_subset)
                                self.logger.info(f"{model_name} trained successfully")
                                model_metrics[model_name] = {'status': 'success', 'predictions': len(predictions_subset)}
                                
                                model_dir = Path("models")
                                model_dir.mkdir(exist_ok=True)
                                model.save(model_dir / f"{model_name}_{self.timestamp}.joblib")
                                self.logger.info(f"Model saved to models/{model_name}_{self.timestamp}.joblib")
                            else:
                                self.logger.warning(f"{model_name} produced no predictions")
                                model_metrics[model_name] = {'status': 'no_predictions'}
                                
                        except Exception as e:
                            self.logger.error(f"{model_name} training failed: {e}", exc_info=True)
                            model_metrics[model_name] = {'status': 'failed', 'error': str(e)}
                            continue
                
                if predictions_dfs:
                    final_predictions = df[['timestamp']].copy()
                    
                    for pred_df in predictions_dfs:
                        final_predictions = final_predictions.merge(pred_df, on='timestamp', how='left')
                    
                    feature_cols = [col for col in df.columns if col != 'timestamp']
                    final_predictions = final_predictions.merge(df[['timestamp'] + feature_cols], on='timestamp', how='left')
                    
                    predictions_path = Path("data/gold") / f"predictions_{self.timestamp}.parquet"
                    final_predictions.to_parquet(predictions_path, index=False)
                    
                    self.execution_metrics['model_training'] = {
                        'models_trained': len([m for m in model_metrics.values() if m['status'] == 'success']),
                        'total_predictions': len(final_predictions),
                        'predictions_file': str(predictions_path)
                    }
                    
                    self.logger.info(f"All model predictions saved to: {predictions_path}")
                    print(f"✓ Model training completed: {predictions_path}")
                    return final_predictions
                else:
                    self.logger.warning("No models were successfully trained")
                    return pd.DataFrame()
                    
            except Exception as e:
                self.logger.error(f"Model training phase failed: {e}", exc_info=True)
                print(f"✗ Model training failed: {e}")
                return pd.DataFrame()
    
    def run_pipeline(self):
        """Execute the complete pipeline."""
        with LoggingContext(self.logger, "complete_pipeline"):
            try:
                self.logger.info("Starting complete pipeline execution")
                
                pipeline_steps = [
                    ("Scraping", self.run_scraping),
                    ("Cleaning", self.run_cleaning),
                    ("Feature Engineering", self.run_feature_engineering),
                    ("Model Training", self.run_model_training)
                ]
                
                results = {}
                predictions_df = None
                
                for step_name, step_func in pipeline_steps:
                    self.logger.info(f"Executing step: {step_name}")
                    
                    if step_name == "Model Training":
                        predictions_df = step_func()
                        results[step_name] = not predictions_df.empty
                    else:
                        results[step_name] = step_func()
                
                self._generate_execution_summary(results, predictions_df)
                return all(results.values())
                
            except Exception as e:
                self.logger.error(f"Pipeline execution failed: {e}", exc_info=True)
                print(f"✗ Pipeline execution failed: {e}")
                return False
    
    def _generate_execution_summary(self, results: dict, predictions_df: pd.DataFrame = None):
        """Generate and display execution summary."""
        total_steps = len(results)
        successful_steps = sum(results.values())
        success_rate = (successful_steps / total_steps) * 100 if total_steps > 0 else 0
        
        summary = f"""
Pipeline Execution Summary
{"=" * 40}

Steps Completed: {successful_steps}/{total_steps} ({success_rate:.1f}%)
Execution Timestamp: {self.timestamp}

Step Results:
"""
        for step_name, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            summary += f"  {step_name:25} {status}\n"
        
        if predictions_df is not None and not predictions_df.empty:
            pred_cols = [col for col in predictions_df.columns 
                       if any(x in col for x in ['prediction', 'regime', 'anomaly'])]
            summary += f"\nPredictions Generated: {len(predictions_df)} records"
            summary += f"\nPrediction Features: {len(pred_cols)}"
        
        summary += f"\n\nData Files Generated:"
        if self.bronze_path:
            summary += f"\n  Bronze: {self.bronze_path}"
        if self.silver_path:
            summary += f"\n  Silver: {self.silver_path}"
        if self.gold_path:
            summary += f"\n  Gold: {self.gold_path}"
        
        summary += f"\n\n{'=' * 40}"
        
        print(summary)
        
        summary_path = Path(f"logs/pipeline_summary_{self.timestamp}.txt")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        self.logger.info(f"Execution summary saved to {summary_path}")


# =============================================================================
# FRONTEND: STREAMLIT DASHBOARD
# =============================================================================

class MarketRiskDashboard:
    """
    Frontend Streamlit Dashboard.
    Visualizes pipeline results and model predictions.
    """
    
    def __init__(self):
        """Initialize dashboard."""
        st.set_page_config(
            page_title="Market Narrative Risk Intelligence",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        self.logger = get_dashboard_logger()
        
        try:
            if config_loader:
                self.config = config_loader.get_dashboard_config()
                self.colors = self.config.get('color_palette', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            else:
                self.config = {}
                self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        except Exception as e:
            self.logger.warning(f"Failed to load dashboard config: {e}")
            self.config = {}
            self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        self._init_session_state()
        self.load_data()
        
        # Initialize filter attributes
        self.selected_regime = None
        self.anomaly_filter = 'All Data'
        self.confidence_threshold = 0.7
        self.start_date = None
        self.end_date = None
        
        self.logger.info("MarketRiskDashboard initialized")
    
    def _init_session_state(self):
        """Initialize session state."""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'current_view' not in st.session_state:
            st.session_state.current_view = 'overview'
    
    def load_data(self):
        """Load data from gold layer."""
        try:
            gold_dir = Path("data/gold")
            
            if not gold_dir.exists():
                gold_dir.mkdir(parents=True, exist_ok=True)
            
            prediction_files = list(gold_dir.glob("*predictions*.parquet"))
            
            if prediction_files:
                latest_file = max(prediction_files, key=lambda x: x.stat().st_mtime)
                self.df = pd.read_parquet(latest_file)
                
                if 'timestamp' in self.df.columns:
                    self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
                    self.min_date = self.df['timestamp'].min().date()
                    self.max_date = self.df['timestamp'].max().date()
                
                self.logger.info(f"✓ Loaded {len(self.df)} records from {latest_file}")
                st.session_state.data_loaded = True
            else:
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
                    self._create_sample_data()
                    
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}", exc_info=True)
            self._create_sample_data()
    
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
            'xgboost_risk_regime': np.random.choice(['low', 'medium', 'high'], len(dates), p=[0.3, 0.5, 0.2]),
            'is_anomaly': np.random.choice([0, 1], len(dates), p=[0.92, 0.08]),
            'daily_article_count': np.random.poisson(50, len(dates)),
        })
        
        self.min_date = self.df['timestamp'].min().date()
        self.max_date = self.df['timestamp'].max().date()
        st.session_state.data_loaded = True
    
    def render_header(self):
        """Render header."""
        st.markdown("""
        <div style='text-align: center; padding: 20px 0;'>
            <h1 style='margin-bottom: 5px; color: #2c3e50;'>Market Narrative Risk Intelligence System</h1>
            <p style='color: #7f8c8d; font-size: 1.1em;'>Advanced analytics for market stress detection</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
    
    def render_sidebar(self):
        """Render sidebar with navigation and filters."""
        with st.sidebar:
            st.markdown(f"""
            <div style='padding: 10px; background-color: #f8f9fa; border-radius: 5px; margin-bottom: 20px;'>
                <p style='font-size: 0.9em; color: #6c757d; margin: 0;'>
                <strong>System Status:</strong> Operational<br>
                <strong>Last Updated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M")}<br>
                <strong>Records Loaded:</strong> {len(self.df):,}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🔧 Pipeline Control")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Run Pipeline", use_container_width=True):
                    with st.spinner("Running pipeline..."):
                        try:
                            orchestrator = PipelineOrchestrator()
                            success = orchestrator.run_pipeline()
                            if success:
                                st.success("Pipeline completed!")
                                st.rerun()
                            else:
                                st.error("Pipeline failed")
                        except Exception as e:
                            st.error(f"Error: {e}")
            
            with col2:
                if st.button("🔃 Reload", use_container_width=True):
                    st.rerun()
            
            st.markdown("---")
            st.markdown("### Navigation")
            
            view_options = {
                'overview': '📊 System Overview',
                'data_explorer': '🔍 Data Explorer',
                'model_metrics': '📈 Model Performance'
            }
            
            selected_view = st.selectbox(
                "Select View",
                options=list(view_options.keys()),
                format_func=lambda x: view_options[x]
            )
            st.session_state.current_view = selected_view
            
            st.markdown("---")
            st.markdown("### Data Filters")
            
            if hasattr(self, 'min_date') and hasattr(self, 'max_date'):
                if self.min_date == self.max_date:
                    default_start = self.min_date
                    default_end = self.max_date
                else:
                    date_diff = (self.max_date - self.min_date).days
                    if date_diff > 30:
                        default_start = self.max_date - timedelta(days=30)
                    else:
                        default_start = self.min_date
                    default_end = self.max_date
                
                date_range = st.date_input(
                    "Analysis Period",
                    value=(default_start, default_end),
                    min_value=self.min_date,
                    max_value=self.max_date
                )
                
                if len(date_range) == 2:
                    self.start_date, self.end_date = date_range
                else:
                    self.start_date = self.end_date = date_range[0] if date_range else self.max_date
    
    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply filters to data."""
        filtered_df = df.copy()
        
        if 'timestamp' in filtered_df.columns and self.start_date and self.end_date:
            filtered_df = filtered_df[
                (filtered_df['timestamp'].dt.date >= self.start_date) &
                (filtered_df['timestamp'].dt.date <= self.end_date)
            ]
        
        return filtered_df
    
    def render_overview(self):
        """Render system overview."""
        st.markdown("## 📊 System Overview")
        
        filtered_df = self._apply_filters(self.df)
        
        if filtered_df.empty:
            st.warning("No data matches current filters.")
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'weighted_stress_score' in filtered_df.columns:
                current_stress = filtered_df['weighted_stress_score'].iloc[-1]
                avg_stress = filtered_df['weighted_stress_score'].mean()
                st.metric("Current Stress", f"{current_stress:.2f}", f"{current_stress - avg_stress:+.2f}")
            else:
                st.info("Stress data unavailable")
        
        with col2:
            if 'xgboost_risk_regime' in filtered_df.columns:
                current_regime = filtered_df['xgboost_risk_regime'].iloc[-1]
                st.metric("Risk Regime", current_regime.upper())
            else:
                st.info("Regime data unavailable")
        
        with col3:
            if 'is_anomaly' in filtered_df.columns:
                anomaly_count = int(filtered_df['is_anomaly'].sum())
                st.metric("Anomalies", anomaly_count)
            else:
                st.info("Anomaly data unavailable")
        
        with col4:
            st.metric("Total Articles", f"{len(filtered_df):,}")
        
        st.markdown("---")
        st.markdown("### Recent Articles")
        
        display_cols = ['timestamp', 'headline', 'sentiment_polarity', 'weighted_stress_score']
        display_cols = [col for col in display_cols if col in filtered_df.columns]
        
        if display_cols:
            recent_data = filtered_df[display_cols].tail(10).copy()
            if 'timestamp' in recent_data.columns:
                recent_data['timestamp'] = recent_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(recent_data, use_container_width=True, hide_index=True)
    
    def render_data_explorer(self):
        """Render data explorer."""
        st.markdown("## 🔍 Data Explorer")
        
        filtered_df = self._apply_filters(self.df)
        
        st.markdown(f"**Total Records:** {len(filtered_df):,}")
        st.markdown(f"**Total Columns:** {len(filtered_df.columns)}")
        
        st.markdown("### Column Information")
        st.write(filtered_df.dtypes)
        
        st.markdown("### Data Preview")
        st.dataframe(filtered_df.head(20), use_container_width=True)
        
        st.markdown("### Download Data")
        csv = filtered_df.to_csv(index=False)
        st.download_button("Download as CSV", csv, "data.csv", "text/csv")
    
    def render_model_metrics(self):
        """Render model performance metrics."""
        st.markdown("## 📈 Model Performance")
        
        st.info("Model performance metrics will be displayed here after training completes.")
    
    def run(self):
        """Run the dashboard."""
        try:
            self.render_header()
            self.render_sidebar()
            
            view_handlers = {
                'overview': self.render_overview,
                'data_explorer': self.render_data_explorer,
                'model_metrics': self.render_model_metrics
            }
            
            current_view = st.session_state.get('current_view', 'overview')
            handler = view_handlers.get(current_view, self.render_overview)
            handler()
            
        except Exception as e:
            self.logger.error(f"Dashboard error: {e}", exc_info=True)
            st.error(f"An error occurred: {str(e)}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main entry point - handles both backend pipeline and frontend dashboard.
    """
    
    # Check if running under Streamlit
    if IS_STREAMLIT:
        # Running as Streamlit app (frontend)
        if STREAMLIT_AVAILABLE
