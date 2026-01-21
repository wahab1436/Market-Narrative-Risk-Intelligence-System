"""
Main pipeline orchestrator for Market Narrative Risk Intelligence System.
Full integration with configuration management and structured logging.
"""
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add src to path for module imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import configuration and logging utilities
from src.utils.config_loader import config_loader, get_value
from src.utils.logger import loggers, get_pipeline_logger, LoggingContext

# Import pipeline components
from src.scraper.investing_scraper import InvestingScraper, scrape_and_save
from src.preprocessing.clean_data import DataCleaner, clean_and_save
from src.preprocessing.feature_engineering import FeatureEngineer, engineer_and_save

# Import all models
from src.models.regression.linear_regression import LinearRegressionModel
from src.models.regression.ridge_regression import RidgeRegressionModel
from src.models.regression.lasso_regression import LassoRegressionModel
from src.models.regression.polynomial_regression import PolynomialRegressionModel
from src.models.regression.time_lagged_regression import TimeLaggedRegressionModel
from src.models.neural_network import NeuralNetworkModel
from src.models.xgboost_model import XGBoostModel
from src.models.knn_model import KNNModel
from src.models.isolation_forest import IsolationForestModel

# Import explainability and EDA
from src.explainability.shap_analysis import SHAPAnalyzer
from src.eda.visualization import EDAVisualizer


class PipelineOrchestrator:
    """
    Orchestrates the complete data pipeline with integrated logging and configuration.
    """
    
    def __init__(self, run_all: bool = True):
        """
        Initialize pipeline orchestrator.
        
        Args:
            run_all: Whether to run all pipeline steps by default
        """
        self.config = config_loader.get_config("config")
        self.run_all = run_all
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize logger
        self.logger = get_pipeline_logger()
        
        # File paths for data persistence
        self.bronze_path = None
        self.silver_path = None
        self.gold_path = None
        
        # Performance metrics
        self.execution_metrics = {}
        
        # Create necessary directories
        self._setup_directories()
        
        print("=" * 80)
        print("MARKET NARRATIVE RISK INTELLIGENCE SYSTEM")
        print("=" * 80)
    
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
                    
                    print(f"Scraping completed: {self.bronze_path}")
                    return True
                else:
                    self.logger.warning("Scraping completed but no data was collected")
                    return False
                    
            except Exception as e:
                self.logger.error(f"Scraping failed: {e}", exc_info=True)
                print(f"Scraping failed: {e}")
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
                    
                    print(f"Cleaning completed: {self.silver_path}")
                    return True
                else:
                    self.logger.warning("Cleaning completed but no valid data")
                    return False
                    
            except Exception as e:
                self.logger.error(f"Data cleaning failed: {e}", exc_info=True)
                print(f"Cleaning failed: {e}")
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
                    
                    print(f"Feature engineering completed: {self.gold_path}")
                    return True
                else:
                    self.logger.warning("Feature engineering completed but no features created")
                    return False
                    
            except Exception as e:
                self.logger.error(f"Feature engineering failed: {e}", exc_info=True)
                print(f"Feature engineering failed: {e}")
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
                    'time_lagged_regression': TimeLaggedRegressionModel(),
                    'neural_network': NeuralNetworkModel(),
                    'xgboost': XGBoostModel(),
                    'knn': KNNModel(),
                    'isolation_forest': IsolationForestModel()
                }
                
                # Train models and collect predictions
                predictions_dfs = []
                model_metrics = {}
                
                for model_name, model in models.items():
                    model_logger = getattr(loggers, model_name.split('_')[0])() if hasattr(loggers, model_name.split('_')[0]) else self.logger
                    
                    with LoggingContext(model_logger, f"{model_name}_training"):
                        try:
                            model_logger.info(f"Training {model_name}")
                            
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
                                model_logger.info(f"{model_name} trained successfully")
                                
                                # Store model metrics
                                model_metrics[model_name] = {
                                    'status': 'success',
                                    'predictions': len(predictions_subset)
                                }
                                
                                # Save model
                                model_dir = Path("models")
                                model_dir.mkdir(exist_ok=True)
                                model.save(model_dir / f"{model_name}_{self.timestamp}.joblib")
                                model_logger.info(f"Model saved to models/{model_name}_{self.timestamp}.joblib")
                                
                                # Log specific metrics
                                if 'mse' in results:
                                    model_logger.metric(f"{model_name}_mse", results['mse'])
                                if 'r2' in results:
                                    model_logger.metric(f"{model_name}_r2", results['r2'])
                                if 'accuracy' in results:
                                    model_logger.metric(f"{model_name}_accuracy", results['accuracy'])
                                    
                            else:
                                model_logger.warning(f"{model_name} produced no predictions")
                                model_metrics[model_name] = {'status': 'no_predictions'}
                                
                        except Exception as e:
                            model_logger.error(f"{model_name} training failed: {e}", exc_info=True)
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
                    print(f"Model training completed: {predictions_path}")
                    
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
                print(f"Model training failed: {e}")
                return pd.DataFrame()
    
    def run_explainability(self, df: pd.DataFrame):
        """
        Execute model explainability analysis.
        
        Args:
            df: DataFrame with predictions
        """
        with LoggingContext(self.logger, "explainability_phase"):
            try:
                self.logger.info("Starting model explainability analysis")
                
                if df.empty:
                    self.logger.warning("No data available for explainability analysis")
                    return
                
                # Initialize SHAP analyzer
                shap_analyzer = SHAPAnalyzer()
                
                # Load trained models for SHAP analysis
                models_dir = Path("models")
                model_files = list(models_dir.glob("*.joblib"))
                
                if not model_files:
                    self.logger.warning("No trained models found for SHAP analysis")
                    return
                
                # Prepare feature data for SHAP
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                exclude_cols = ['weighted_stress_score', 'sentiment_polarity', 'vader_compound']
                feature_cols = [col for col in numeric_cols if col not in exclude_cols]
                
                if not feature_cols:
                    self.logger.warning("No features available for SHAP analysis")
                    return
                
                X = df[feature_cols].fillna(0)
                
                # Analyze each model
                shap_results = {}
                for model_file in model_files[:3]:  # Limit to 3 models for performance
                    model_name = model_file.stem.split('_')[0]
                    
                    try:
                        model_logger = getattr(loggers, model_name)() if hasattr(loggers, model_name) else self.logger
                        
                        with LoggingContext(model_logger, f"{model_name}_shap_analysis"):
                            model_logger.info(f"Performing SHAP analysis for {model_name}")
                            
                            # Load model
                            if model_name == 'linear':
                                model_obj = LinearRegressionModel()
                            elif model_name == 'ridge':
                                model_obj = RidgeRegressionModel()
                            elif model_name == 'lasso':
                                model_obj = LassoRegressionModel()
                            elif model_name == 'polynomial':
                                model_obj = PolynomialRegressionModel()
                            elif model_name == 'timelagged':
                                model_obj = TimeLaggedRegressionModel()
                            elif model_name == 'xgboost':
                                model_obj = XGBoostModel()
                            else:
                                model_logger.warning(f"SHAP analysis not implemented for {model_name}")
                                continue
                            
                            model_obj.load(model_file)
                            
                            # Perform SHAP analysis
                            if model_name in ['linear', 'ridge', 'lasso']:
                                shap_results[model_name] = shap_analyzer.analyze_linear_model(
                                    model_obj.model,
                                    X,
                                    model_name
                                )
                            elif model_name == 'xgboost':
                                shap_results[model_name] = shap_analyzer.analyze_xgboost(
                                    model_obj.model,
                                    X,
                                    model_name
                                )
                            elif model_name == 'neuralnetwork':
                                # Neural network SHAP analysis
                                X_array = X.values
                                shap_results[model_name] = shap_analyzer.analyze_neural_network(
                                    model_obj.model,
                                    X_array,
                                    X.columns.tolist(),
                                    model_name
                                )
                            
                            # Generate report
                            report_path = Path(f"logs/shap_report_{model_name}_{self.timestamp}.html")
                            shap_analyzer.generate_report(shap_results[model_name], report_path)
                            
                            model_logger.info(f"SHAP analysis completed for {model_name}")
                            
                    except Exception as e:
                        self.logger.error(f"SHAP analysis failed for {model_name}: {e}", exc_info=True)
                        continue
                
                self.logger.info("Explainability analysis completed")
                self.execution_metrics['explainability'] = {
                    'models_analyzed': len(shap_results),
                    'reports_generated': len(list(Path("logs").glob("shap_report_*.html")))
                }
                print("Explainability analysis completed")
                
            except Exception as e:
                self.logger.error(f"Explainability phase failed: {e}", exc_info=True)
                print(f"Explainability failed: {e}")
    
    def run_eda_analysis(self, df: pd.DataFrame):
        """
        Execute exploratory data analysis.
        
        Args:
            df: DataFrame for EDA
        """
        with LoggingContext(self.logger, "eda_phase"):
            try:
                self.logger.info("Starting exploratory data analysis")
                
                if df.empty:
                    self.logger.warning("No data available for EDA")
                    return
                
                # Initialize EDA visualizer
                eda_visualizer = EDAVisualizer(theme="plotly_white")
                
                # Create EDA report
                eda_dir = Path("logs/eda_reports")
                eda_dir.mkdir(exist_ok=True)
                
                report_path = eda_visualizer.create_eda_report(
                    df,
                    eda_dir,
                    report_name=f"eda_report_{self.timestamp}"
                )
                
                # Generate individual visualizations
                viz_dir = eda_dir / f"visualizations_{self.timestamp}"
                viz_dir.mkdir(exist_ok=True)
                
                # Feature distributions
                eda_visualizer.plot_feature_distributions(
                    df,
                    output_path=viz_dir / 'feature_distributions.png'
                )
                
                # Correlation matrix
                eda_visualizer.plot_correlation_matrix(
                    df,
                    output_path=viz_dir / 'correlation_matrix.png'
                )
                
                # Time series decomposition if available
                if 'timestamp' in df.columns and 'weighted_stress_score' in df.columns:
                    eda_visualizer.plot_time_series_decomposition(
                        df,
                        output_path=viz_dir / 'time_series_decomposition.png'
                    )
                
                # Stress score evolution
                if 'timestamp' in df.columns and 'weighted_stress_score' in df.columns:
                    eda_visualizer.plot_stress_score_evolution(
                        df,
                        output_path=viz_dir / 'stress_score_evolution.png'
                    )
                
                self.logger.info(f"EDA report generated: {report_path}")
                self.execution_metrics['eda'] = {
                    'report_generated': str(report_path),
                    'visualizations_created': len(list(viz_dir.glob("*.png")))
                }
                print(f"EDA analysis completed: {report_path}")
                
            except Exception as e:
                self.logger.error(f"EDA analysis failed: {e}", exc_info=True)
                print(f"EDA analysis failed: {e}")
    
    def run_dashboard(self):
        """
        Launch the Streamlit dashboard.
        Note: This method provides instructions and can optionally launch the dashboard.
        """
        with LoggingContext(self.logger, "dashboard_launch"):
            try:
                self.logger.info("Preparing dashboard launch")
                
                # Check for prediction files
                gold_dir = Path("data/gold")
                prediction_files = list(gold_dir.glob("*_predictions.parquet"))
                
                if not prediction_files:
                    self.logger.warning("No prediction files found for dashboard")
                    print("No prediction data available. Run the pipeline first.")
                    return
                
                latest_predictions = max(prediction_files, key=lambda x: x.stat().st_mtime)
                self.logger.info(f"Latest predictions for dashboard: {latest_predictions}")
                
                # Dashboard launch information
                dashboard_info = """
                Dashboard is ready to launch.
                
                To launch the dashboard, run:
                  streamlit run src/dashboard/app.py
                
                Alternatively, use the provided quick start script:
                  python quick_start.py
                
                The dashboard will be available at: http://localhost:8501
                """
                
                print(dashboard_info)
                self.logger.info("Dashboard launch instructions displayed")
                
                # Optional: Auto-launch dashboard
                auto_launch = get_value("dashboard.auto_launch", False)
                if auto_launch:
                    self.logger.info("Auto-launching dashboard")
                    subprocess.Popen([
                        "streamlit", "run", "src/dashboard/app.py",
                        "--server.port", "8501",
                        "--server.address", "0.0.0.0"
                    ])
                    print("Dashboard auto-launched at http://localhost:8501")
                
            except Exception as e:
                self.logger.error(f"Dashboard setup failed: {e}", exc_info=True)
                print(f"Dashboard setup failed: {e}")
    
    def run_pipeline(self):
        """
        Execute the complete pipeline from scraping to dashboard.
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
                        
                        # Run explainability and EDA if training successful
                        if results[step_name]:
                            self.run_explainability(predictions_df)
                            self.run_eda_analysis(predictions_df)
                    else:
                        results[step_name] = step_func()
                
                # Provide dashboard instructions
                self.run_dashboard()
                
                # Generate execution summary
                self._generate_execution_summary(results, predictions_df)
                
                return all(results.values())
                
            except Exception as e:
                self.logger.error(f"Pipeline execution failed: {e}", exc_info=True)
                print(f"Pipeline execution failed: {e}")
                return False
    
    def _generate_execution_summary(self, results: dict, predictions_df: pd.DataFrame = None):
        """
        Generate and display execution summary.
        
        Args:
            results: Dictionary of step results
            predictions_df: Predictions DataFrame if available
        """
        with LoggingContext(self.logger, "execution_summary"):
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
                status = "SUCCESS" if success else "FAILED"
                summary += f"  {step_name:25} {status}\n"
            
            # Add model predictions info
            if predictions_df is not None and not predictions_df.empty:
                pred_cols = [col for col in predictions_df.columns 
                           if any(x in col for x in ['prediction', 'regime', 'anomaly'])]
                summary += f"\nPredictions Generated: {len(predictions_df)} records"
                summary += f"\nPrediction Features: {len(pred_cols)}"
                
                # Risk regime distribution if available
                if 'xgboost_risk_regime' in predictions_df.columns:
                    regime_counts = predictions_df['xgboost_risk_regime'].value_counts()
                    summary += "\n\nRisk Regime Distribution:"
                    for regime, count in regime_counts.items():
                        percentage = (count / len(predictions_df)) * 100
                        summary += f"\n  {regime:10} {count:4} records ({percentage:.1f}%)"
                
                # Anomaly detection if available
                if 'is_anomaly' in predictions_df.columns:
                    anomaly_count = predictions_df['is_anomaly'].sum()
                    anomaly_percentage = (anomaly_count / len(predictions_df)) * 100
                    summary += f"\n\nAnomalies Detected: {anomaly_count} ({anomaly_percentage:.1f}%)"
            
            # Add file locations
            summary += f"\n\nData Files Generated:"
            if self.bronze_path:
                summary += f"\n  Bronze: {self.bronze_path}"
            if self.silver_path:
                summary += f"\n  Silver: {self.silver_path}"
            if self.gold_path:
                summary += f"\n  Gold: {self.gold_path}"
            
            # Add next steps
            summary += f"""
            
            Next Steps:
              1. Review logs in 'logs/' directory
              2. Check model artifacts in 'models/' directory
              3. Launch dashboard: streamlit run src/dashboard/app.py
              4. View EDA reports in 'logs/eda_reports/'
            
            {"=" * 40}
            """
            
            print(summary)
            
            # Save summary to file
            summary_path = Path(f"logs/pipeline_summary_{self.timestamp}.txt")
            with open(summary_path, 'w') as f:
                f.write(summary)
            
            self.logger.info(f"Execution summary saved to {summary_path}")
            
            # Log overall metrics
            self.logger.metric("pipeline_success_rate", success_rate)
            self.logger.metric("total_steps", total_steps)
            self.logger.metric("successful_steps", successful_steps)
            
            # Store final execution metrics
            self.execution_metrics['pipeline_summary'] = {
                'success_rate': success_rate,
                'total_steps': total_steps,
                'successful_steps': successful_steps,
                'summary_file': str(summary_path)
            }


def main():
    """
    Main entry point for the pipeline.
    Provides command-line interface for executing specific pipeline components.
    """
    parser = argparse.ArgumentParser(
        description="Market Narrative Risk Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run complete pipeline
  %(prog)s --scrape-only      # Run only scraping
  %(prog)s --clean-only       # Run only data cleaning
  %(prog)s --features-only    # Run only feature engineering
  %(prog)s --train-only       # Run only model training
  %(prog)s --dashboard        # Launch dashboard
  %(prog)s --help             # Show this help message
        """
    )
    
    # Execution mode arguments
    parser.add_argument(
        "--scrape-only",
        action="store_true",
        help="Run only scraping step"
    )
    
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Run only cleaning step"
    )
    
    parser.add_argument(
        "--features-only",
        action="store_true",
        help="Run only feature engineering"
    )
    
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Run only model training"
    )
    
    parser.add_argument(
        "--explain-only",
        action="store_true",
        help="Run only explainability analysis"
    )
    
    parser.add_argument(
        "--eda-only",
        action="store_true",
        help="Run only exploratory data analysis"
    )
    
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch dashboard only"
    )
    
    parser.add_argument(
        "--quick-start",
        action="store_true",
        help="Run quick start setup"
    )
    
    parser.add_argument(
        "--config-dir",
        default="config",
        help="Directory containing configuration files"
    )
    
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level"
    )
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = PipelineOrchestrator()
    
    # Execute based on arguments
    if args.scrape_only:
        orchestrator.run_scraping()
    
    elif args.clean_only:
        orchestrator.bronze_path = max(Path("data/bronze").glob("*.parquet"), 
                                      key=lambda x: x.stat().st_mtime, default=None)
        orchestrator.run_cleaning()
    
    elif args.features_only:
        orchestrator.silver_path = max(Path("data/silver").glob("*.parquet"), 
                                      key=lambda x: x.stat().st_mtime, default=None)
        orchestrator.run_feature_engineering()
    
    elif args.train_only:
        orchestrator.gold_path = max(Path("data/gold").glob("features_*.parquet"), 
                                    key=lambda x: x.stat().st_mtime, default=None)
        predictions = orchestrator.run_model_training()
        if not predictions.empty:
            orchestrator.run_explainability(predictions)
            orchestrator.run_eda_analysis(predictions)
    
    elif args.explain_only:
        # Find latest predictions for explainability
        gold_dir = Path("data/gold")
        prediction_files = list(gold_dir.glob("*_predictions.parquet"))
        if prediction_files:
            latest_predictions = max(prediction_files, key=lambda x: x.stat().st_mtime)
            df = pd.read_parquet(latest_predictions)
            orchestrator.run_explainability(df)
        else:
            print("No prediction files found for explainability analysis")
    
    elif args.eda_only:
        # Find latest data for EDA
        gold_dir = Path("data/gold")
        data_files = list(gold_dir.glob("*.parquet"))
        if data_files:
            latest_data = max(data_files, key=lambda x: x.stat().st_mtime)
            df = pd.read_parquet(latest_data)
            orchestrator.run_eda_analysis(df)
        else:
            print("No data files found for EDA analysis")
    
    elif args.dashboard:
        orchestrator.run_dashboard()
    
    elif args.quick_start:
        # Run quick start script if available
        quick_start_path = Path("quick_start.py")
        if quick_start_path.exists():
            subprocess.run([sys.executable, "quick_start.py"])
        else:
            print("Quick start script not found. Running complete pipeline instead.")
            success = orchestrator.run_pipeline()
            sys.exit(0 if success else 1)
    
    else:
        # Run complete pipeline
        success = orchestrator.run_pipeline()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
