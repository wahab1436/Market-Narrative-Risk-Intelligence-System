"""
Main pipeline orchestrator with integrated dashboard support.
Can run as pipeline (backend) or Streamlit app (frontend).
"""
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import critical dependencies first
import pandas as pd
import numpy as np

# IMPORTANT: Initialize config_loader FIRST before any other imports
from src.utils.config_loader import config_loader

# Now import everything else
from src.utils.logger import get_pipeline_logger, setup_pipeline_logging
from src.scraper import scrape_and_save
from src.preprocessing.clean_data import clean_and_save
from src.preprocessing.feature_engineering import engineer_and_save
from src.models.regression.linear_regression import LinearRegressionModel
from src.models.regression.ridge_regression import RidgeRegressionModel
from src.models.regression.lasso_regression import LassoRegressionModel
from src.models.regression.polynomial_regression import PolynomialRegressionModel
from src.models.regression.time_lagged_regression import TimeLaggedRegressionModel
from src.models.neural_network import NeuralNetworkModel
from src.models.xgboost_model import XGBoostModel
from src.models.knn_model import KNNModel
from src.models.isolation_forest import IsolationForestModel
from src.explainability.shap_analysis import SHAPAnalyzer

# Import Streamlit components (will only work when running as Streamlit app)
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


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
    Orchestrates the complete data pipeline.
    """
    
    def __init__(self, run_all: bool = True):
        """
        Initialize pipeline orchestrator.
        
        Args:
            run_all: Whether to run all pipeline steps
        """
        # Get config
        self.config = config_loader.get_config("config")
        
        # Setup logging based on config
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        setup_pipeline_logging(level=log_level)
        
        self.logger = get_pipeline_logger()
        self.run_all = run_all
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # File paths
        self.bronze_path = None
        self.silver_path = None
        self.gold_path = None
        
        # Execution metrics
        self.execution_metrics = {}
        
        print("=" * 80)
        print("MARKET NARRATIVE RISK INTELLIGENCE SYSTEM")
        print("=" * 80)
        self.logger.info("PipelineOrchestrator initialized")
        
        # Setup directories
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary directories for the pipeline."""
        with LoggingContext(self.logger, "directory_setup"):
            directories = [
                Path("data/bronze"),
                Path("data/silver"),
                Path("data/gold"),
                Path("logs"),
                Path("models"),
                Path("config"),
                Path("tests"),
                Path("docs")
            ]
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Ensured directory exists: {directory}")
    
    def run_scraping(self) -> bool:
        """
        Execute scraping phase to collect news data.
        
        Returns:
            True if successful, False otherwise
        """
        with LoggingContext(self.logger, "scraping_phase"):
            try:
                self.logger.info("Starting scraping phase")
                
                # Get scraping configuration
                scraping_config = config_loader.get_scraping_config()
                use_selenium = scraping_config.get("use_selenium", False)
                
                self.logger.info(f"Scraping configuration: use_selenium={use_selenium}")
                
                # Execute scraping
                self.bronze_path = scrape_and_save()
                
                if self.bronze_path:
                    self.logger.info(f"Scraping completed successfully: {self.bronze_path}")
                    
                    # Log metrics
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
        """
        Execute data cleaning and validation phase.
        
        Returns:
            True if successful, False otherwise
        """
        with LoggingContext(self.logger, "cleaning_phase"):
            try:
                self.logger.info("Starting data cleaning phase")
                
                # If no bronze path specified, find latest
                if self.bronze_path is None:
                    bronze_dir = Path("data/bronze")
                    files = list(bronze_dir.glob("*.parquet"))
                    if not files:
                        self.logger.error("No bronze files found for cleaning")
                        return False
                    self.bronze_path = max(files, key=lambda x: x.stat().st_mtime)
                    self.logger.info(f"Using latest bronze file: {self.bronze_path}")
                
                # Execute cleaning
                self.silver_path = clean_and_save(self.bronze_path)
                
                if self.silver_path:
                    self.logger.info(f"Data cleaning completed successfully: {self.silver_path}")
                    
                    # Log metrics
                    df = pd.read_parquet(self.silver_path)
                    original_df = pd.read_parquet(self.bronze_path)
                    
                    self.execution_metrics['cleaning'] = {
                        'original_records': len(original_df),
                        'cleaned_records': len(df),
                        'records_removed': len(original_df) - len(df),
                        'removal_percentage': ((len(original_df) - len(df)) / len(original_df)) * 100,
                        'file_size_mb': Path(self.silver_path).stat().st_size / 1024 / 1024
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
        """
        Execute feature engineering phase.
        
        Returns:
            True if successful, False otherwise
        """
        with LoggingContext(self.logger, "feature_engineering_phase"):
            try:
                self.logger.info("Starting feature engineering phase")
                
                # If no silver path specified, find latest
                if self.silver_path is None:
                    silver_dir = Path("data/silver")
                    files = list(silver_dir.glob("*.parquet"))
                    if not files:
                        self.logger.error("No silver files found for feature engineering")
                        return False
                    self.silver_path = max(files, key=lambda x: x.stat().st_mtime)
                    self.logger.info(f"Using latest silver file: {self.silver_path}")
                
                # Execute feature engineering
                self.gold_path = engineer_and_save(self.silver_path)
                
                if self.gold_path:
                    self.logger.info(f"Feature engineering completed successfully: {self.gold_path}")
                    
                    # Log metrics
                    df = pd.read_parquet(self.gold_path)
                    
                    self.execution_metrics['feature_engineering'] = {
                        'records_processed': len(df),
                        'features_created': len(df.columns),
                        'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
                        'file_size_mb': Path(self.gold_path).stat().st_size / 1024 / 1024
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
        """
        Execute model training phase for all models.
        
        Returns:
            DataFrame with all predictions
        """
        with LoggingContext(self.logger, "model_training_phase"):
            try:
                self.logger.info("Starting model training phase")
                
                # If no gold path specified, find latest
                if self.gold_path is None:
                    gold_dir = Path("data/gold")
                    files = list(gold_dir.glob("features_*.parquet"))
                    if not files:
                        self.logger.error("No gold files found for model training")
                        return pd.DataFrame()
                    self.gold_path = max(files, key=lambda x: x.stat().st_mtime)
                    self.logger.info(f"Using latest gold file: {self.gold_path}")
                
                # Load gold layer data
                self.logger.info(f"Loading data from {self.gold_path}")
                df = pd.read_parquet(self.gold_path)
                self.logger.info(f"Loaded {len(df)} records with {len(df.columns)} features for model training")
                
                # Initialize all models
                models = {
                    'linear_regression': LinearRegressionModel(),
                    'ridge_regression': RidgeRegressionModel(),
                    'lasso_regression': LassoRegressionModel(),
                    'polynomial_regression': PolynomialRegressionModel(),
                    # 'time_lagged_regression': TimeLaggedRegressionModel(),  # Skip for small datasets
                    'neural_network': NeuralNetworkModel(),
                    'xgboost': XGBoostModel(),
                    'knn': KNNModel(),
                    'isolation_forest': IsolationForestModel()
                }
                
                # Train models and collect predictions
                predictions_dfs = []
                model_metrics = {}
                
                for model_name, model in models.items():
                    with LoggingContext(self.logger, f"{model_name}_training"):
                        try:
                            self.logger.info(f"Training {model_name}")
                            
                            # Train model
                            results = model.train(df)
                            
                            # Make predictions
                            predictions = model.predict(df)
                            
                            # Extract prediction columns
                            pred_cols = [col for col in predictions.columns 
                                       if any(x in col for x in ['prediction', 'regime', 'anomaly', 'similarity', 'forecast'])]
                            
                            if pred_cols:
                                predictions_subset = predictions[['timestamp'] + pred_cols]
                                predictions_dfs.append(predictions_subset)
                                self.logger.info(f"{model_name} trained successfully")
                                
                                # Store model metrics
                                model_metrics[model_name] = {
                                    'status': 'success',
                                    'predictions': len(predictions_subset)
                                }
                                
                                # Save model
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
                
                # Merge all predictions
                if predictions_dfs:
                    # Start with timestamp
                    final_predictions = df[['timestamp']].copy()
                    
                    # Merge all prediction columns
                    for pred_df in predictions_dfs:
                        final_predictions = final_predictions.merge(
                            pred_df,
                            on='timestamp',
                            how='left'
                        )
                    
                    # Add original features
                    feature_cols = [col for col in df.columns if col != 'timestamp']
                    final_predictions = final_predictions.merge(
                        df[['timestamp'] + feature_cols],
                        on='timestamp',
                        how='left'
                    )
                    
                    # Save predictions
                    predictions_path = Path("data/gold") / f"predictions_{self.timestamp}.parquet"
                    final_predictions.to_parquet(predictions_path, index=False)
                    
                    # Store execution metrics
                    self.execution_metrics['model_training'] = {
                        'models_trained': len([m for m in model_metrics.values() if m['status'] == 'success']),
                        'models_failed': len([m for m in model_metrics.values() if m['status'] == 'failed']),
                        'total_predictions': len(final_predictions),
                        'prediction_columns': len([c for c in final_predictions.columns if any(x in c for x in ['prediction', 'regime', 'anomaly'])]),
                        'predictions_file': str(predictions_path)
                    }
                    
                    self.logger.info(f"All model predictions saved to: {predictions_path}")
                    print(f"✓ Model training completed: {predictions_path}")
                    
                    return final_predictions
                else:
                    self.logger.warning("No models were successfully trained")
                    self.execution_metrics['model_training'] = {
                        'models_trained': 0,
                        'models_failed': len(model_metrics),
                        'total_predictions': 0
                    }
                    return pd.DataFrame()
                    
            except Exception as e:
                self.logger.error(f"Model training phase failed: {e}", exc_info=True)
                print(f"✗ Model training failed: {e}")
                return pd.DataFrame()
    
    def run_pipeline(self):
        """
        Execute the complete pipeline from scraping to models.
        """
        with LoggingContext(self.logger, "complete_pipeline"):
            try:
                self.logger.info("Starting complete pipeline execution")
                
                # Execution sequence
                pipeline_steps = [
                    ("Scraping", self.run_scraping),
                    ("Cleaning", self.run_cleaning),
                    ("Feature Engineering", self.run_feature_engineering),
                    ("Model Training", self.run_model_training)
                ]
                
                # Execute pipeline steps
                results = {}
                predictions_df = None
                
                for step_name, step_func in pipeline_steps:
                    self.logger.info(f"Executing step: {step_name}")
                    
                    if step_name == "Model Training":
                        # Model training returns predictions DataFrame
                        predictions_df = step_func()
                        results[step_name] = not predictions_df.empty
                    else:
                        results[step_name] = step_func()
                
                # Generate execution summary
                self._generate_execution_summary(results, predictions_df)
                
                return all(results.values())
                
            except Exception as e:
                self.logger.error(f"Pipeline execution failed: {e}", exc_info=True)
                print(f"✗ Pipeline execution failed: {e}")
                return False
    
    def _generate_execution_summary(self, results: dict, predictions_df: pd.DataFrame = None):
        """
        Generate and display execution summary.
        
        Args:
            results: Dictionary of step results
            predictions_df: Predictions DataFrame if available
        """
        # Calculate statistics
        total_steps = len(results)
        successful_steps = sum(results.values())
        success_rate = (successful_steps / total_steps) * 100 if total_steps > 0 else 0
        
        # Generate summary
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
        
        # Add model predictions info
        if predictions_df is not None and not predictions_df.empty:
            pred_cols = [col for col in predictions_df.columns 
                       if any(x in col for x in ['prediction', 'regime', 'anomaly'])]
            summary += f"\nPredictions Generated: {len(predictions_df)} records"
            summary += f"\nPrediction Features: {len(pred_cols)}"
        
        # Add file locations
        summary += f"\n\nData Files Generated:"
        if self.bronze_path:
            summary += f"\n  Bronze: {self.bronze_path}"
        if self.silver_path:
            summary += f"\n  Silver: {self.silver_path}"
        if self.gold_path:
            summary += f"\n  Gold: {self.gold_path}"
        
        summary += f"\n\n{'=' * 40}"
        
        print(summary)
        
        # Save summary to file
        summary_path = Path(f"logs/pipeline_summary_{self.timestamp}.txt")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        self.logger.info(f"Execution summary saved to {summary_path}")


def run_streamlit_app():
    """Run the Streamlit dashboard application."""
    if not STREAMLIT_AVAILABLE:
        print("Error: Streamlit not available. Install with: pip install streamlit")
        return
    
    # Import dashboard
    from src.dashboard.app import MarketRiskDashboard
    
    # Add pipeline control in sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔧 Pipeline Control")
        
        # Check data status
        gold_dir = Path("data/gold")
        if gold_dir.exists():
            files = list(gold_dir.glob("*.parquet"))
            if files:
                latest = max(files, key=lambda x: x.stat().st_mtime)
                mod_time = datetime.fromtimestamp(latest.stat().st_mtime)
                st.success(f"Data available: {mod_time.strftime('%Y-%m-%d %H:%M')}")
            else:
                st.warning("No data files found")
        else:
            st.warning("No data directory")
        
        # Pipeline buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Refresh", use_container_width=True):
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
    
    # Run dashboard
    try:
        dashboard = MarketRiskDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Dashboard error: {e}")
        
        # Offer to run pipeline if no data
        if "No data" in str(e) or not Path("data/gold").exists():
            st.warning("No data available. Run the pipeline first.")
            if st.button("Run Pipeline Now", type="primary"):
                with st.spinner("Running pipeline... (2-3 minutes)"):
                    orchestrator = PipelineOrchestrator()
                    orchestrator.run_pipeline()
                    st.rerun()


def main():
    """
    Main entry point - detects if running as Streamlit or CLI.
    """
    # Check if running under Streamlit
    try:
        # Streamlit sets this when running
        import streamlit.runtime.scriptrunner.script_run_context as script_run_context
        if script_run_context.get_script_run_ctx() is not None:
            # Running as Streamlit app
            run_streamlit_app()
            return
    except:
        pass
    
    # Running as CLI (normal Python)
    parser = argparse.ArgumentParser(description="Market Narrative Risk Intelligence System")
    
    parser.add_argument("--scrape-only", action="store_true", help="Run only scraping step")
    parser.add_argument("--clean-only", action="store_true", help="Run only cleaning step")
    parser.add_argument("--features-only", action="store_true", help="Run only feature engineering")
    parser.add_argument("--train-only", action="store_true", help="Run only model training")
    parser.add_argument("--dashboard", action="store_true", help="Run Streamlit dashboard")
    
    args = parser.parse_args()
    
    # Run dashboard if requested
    if args.dashboard:
        if STREAMLIT_AVAILABLE:
            print("Starting Streamlit dashboard...")
            import subprocess
            subprocess.run(["streamlit", "run", __file__])
        else:
            print("Error: Streamlit not installed. Run: pip install streamlit")
        return
    
    # Initialize orchestrator
    orchestrator = PipelineOrchestrator()
    
    # Execute based on arguments
    if args.scrape_only:
        orchestrator.run_scraping()
    elif args.clean_only:
        orchestrator.run_cleaning()
    elif args.features_only:
        orchestrator.run_feature_engineering()
    elif args.train_only:
        orchestrator.run_model_training()
    else:
        # Run complete pipeline
        success = orchestrator.run_pipeline()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
