"""
Main pipeline orchestrator for Market Narrative Risk Intelligence System.
"""
import sys
import subprocess
import threading
import webbrowser
import time
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.logger import loggers, get_pipeline_logger, LoggingContext, setup_pipeline_logging
from src.utils.config_loader import config_loader, get_value

# Import pipeline components
from src.scraper.investing_scraper import InvestingScraper, scrape_and_save
from src.preprocessing.clean_data import DataCleaner, clean_and_save
from src.preprocessing.feature_engineering import FeatureEngineer, engineer_and_save

# Import models
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
from src.eda.visualization import EDAVisualizer


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
        self.config = config_loader.get_config("config")
        self.run_all = run_all
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = get_pipeline_logger()
        
        # File paths
        self.bronze_path = None
        self.silver_path = None
        self.gold_path = None
        self.predictions_path = None
        
        print("=" * 80)
        print("MARKET NARRATIVE RISK INTELLIGENCE SYSTEM")
        print("=" * 80)
        
        with LoggingContext(self.logger, "pipeline_initialization"):
            self.logger.info("PipelineOrchestrator initialized")
    
    def setup_directories(self):
        """Create necessary directories."""
        with LoggingContext(self.logger, "directory_setup"):
            directories = [
                'data/bronze',
                'data/silver', 
                'data/gold',
                'logs',
                'models',
                'config'
            ]
            
            for directory in directories:
                Path(directory).mkdir(parents=True, exist_ok=True)
            
            self.logger.info("Directories created")
    
    def run_scraping(self) -> bool:
        """
        Run scraping step.
        
        Returns:
            True if successful, False otherwise
        """
        with LoggingContext(self.logger, "scraping_phase"):
            print("\n[1/6] SCRAPING")
            print("-" * 40)
            
            self.logger.info("Starting scraping phase")
            
            # Get scraping configuration
            scraping_config = config_loader.get_scraping_config()
            use_selenium = scraping_config.get('use_selenium', False)
            self.logger.info(f"Scraping configuration: use_selenium={use_selenium}")
            
            # Run scraping
            scraper = InvestingScraper(use_selenium=use_selenium)
            
            try:
                articles = scraper.scrape_latest_news()
                
                if articles:
                    scraper.save_to_bronze(articles)
                    
                    # Return path to latest file
                    bronze_dir = Path("data/bronze")
                    files = list(bronze_dir.glob("*.parquet"))
                    if files:
                        self.bronze_path = max(files, key=lambda x: x.stat().st_mtime)
                        self.logger.info(f"Scraping completed successfully: {self.bronze_path}")
                        print(f"✓ Scraping completed: {self.bronze_path}")
                        return True
                else:
                    self.logger.warning("No articles collected")
                    print("✗ Scraping failed: No articles collected")
                    return False
                    
            except Exception as e:
                self.logger.error(f"Scraping failed: {e}")
                print(f"✗ Scraping failed: {e}")
                return False
            finally:
                scraper.close()
    
    def run_cleaning(self) -> bool:
        """
        Run data cleaning step.
        
        Returns:
            True if successful, False otherwise
        """
        with LoggingContext(self.logger, "cleaning_phase"):
            print("\n[2/6] DATA CLEANING")
            print("-" * 40)
            
            self.logger.info("Starting data cleaning phase")
            
            try:
                cleaner = DataCleaner()
                
                if self.bronze_path is None:
                    # Find latest bronze file
                    bronze_dir = Path("data/bronze")
                    files = list(bronze_dir.glob("*.parquet"))
                    if not files:
                        self.logger.error("No bronze files found")
                        return False
                    self.bronze_path = max(files, key=lambda x: x.stat().st_mtime)
                
                df = cleaner.load_bronze_data(self.bronze_path)
                valid_df, invalid_df = cleaner.validate_data(df)
                cleaned_df = cleaner.clean_dataframe(valid_df)
                self.silver_path = cleaner.save_to_silver(cleaned_df)
                
                self.logger.info(f"Data cleaning completed successfully: {self.silver_path}")
                print(f"✓ Cleaning completed: {self.silver_path}")
                return True
                
            except Exception as e:
                self.logger.error(f"Data cleaning failed: {e}")
                print(f"✗ Cleaning failed: {e}")
                return False
    
    def run_feature_engineering(self) -> bool:
        """
        Run feature engineering step.
        
        Returns:
            True if successful, False otherwise
        """
        with LoggingContext(self.logger, "feature_engineering_phase"):
            print("\n[3/6] FEATURE ENGINEERING")
            print("-" * 40)
            
            self.logger.info("Starting feature engineering phase")
            
            try:
                engineer = FeatureEngineer()
                
                if self.silver_path is None:
                    # Find latest silver file
                    silver_dir = Path("data/silver")
                    files = list(silver_dir.glob("*.parquet"))
                    if not files:
                        self.logger.error("No silver files found")
                        return False
                    self.silver_path = max(files, key=lambda x: x.stat().st_mtime)
                
                df = engineer.load_silver_data(self.silver_path)
                feature_df = engineer.engineer_features(df)
                self.gold_path = engineer.save_to_gold(feature_df)
                
                self.logger.info(f"Feature engineering completed successfully: {self.gold_path}")
                print(f"✓ Feature engineering completed: {self.gold_path}")
                return True
                
            except Exception as e:
                self.logger.error(f"Feature engineering failed: {e}")
                print(f"✗ Feature engineering failed: {e}")
                return False
    
    def run_model_training(self) -> pd.DataFrame:
        """
        Run all model training steps.
        
        Returns:
            DataFrame with all predictions
        """
        with LoggingContext(self.logger, "model_training_phase"):
            print("\n[4/6] MODEL TRAINING")
            print("-" * 40)
            
            self.logger.info("Starting model training phase")
            
            # Load gold layer data
            if self.gold_path is None:
                # Find latest gold file
                gold_dir = Path("data/gold")
                files = list(gold_dir.glob("*.parquet"))
                if not files:
                    self.logger.error("No gold files found")
                    return pd.DataFrame()
                self.gold_path = max(files, key=lambda x: x.stat().st_mtime)
            
            self.logger.info(f"Loading data from {self.gold_path}")
            df = pd.read_parquet(self.gold_path)
            print(f"✓ Loaded {len(df)} records with {len(df.columns)} features for model training")
            self.logger.info(f"Loaded {len(df)} records with {len(df.columns)} features for model training")
            
            # Initialize models
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
            model_results = {}
            
            for model_name, model in models.items():
                with LoggingContext(self.logger, f"{model_name}_training"):
                    print(f"\n  Training {model_name.replace('_', ' ').title()}...")
                    self.logger.info(f"Training {model_name}")
                    
                    try:
                        # Train model
                        results = model.train(df)
                        model_results[model_name] = results
                        
                        # Make predictions
                        predictions = model.predict(df)
                        
                        # Extract relevant prediction columns
                        pred_cols = [col for col in predictions.columns 
                                   if any(x in col for x in ['prediction', 'regime', 'anomaly', 'similarity', 'residual'])]
                        
                        # Save predictions
                        if pred_cols:
                            predictions_subset = predictions[['timestamp'] + pred_cols]
                            predictions_dfs.append(predictions_subset)
                            print(f"  ✓ {model_name} trained successfully")
                            self.logger.info(f"{model_name} trained successfully")
                        
                        # Save model
                        model_path = Path(f"models/{model_name}_{self.timestamp}.joblib")
                        model.save(model_path)
                        self.logger.info(f"Model saved to {model_path}")
                        
                    except Exception as e:
                        self.logger.error(f"{model_name} training failed: {e}")
                        print(f"  ✗ {model_name} failed: {e}")
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
                feature_cols = [col for col in df.columns if col not in ['timestamp']]
                final_predictions = final_predictions.merge(
                    df[['timestamp'] + feature_cols],
                    on='timestamp',
                    how='left'
                )
                
                # Save predictions
                self.predictions_path = Path("data/gold") / f"predictions_{self.timestamp}.parquet"
                final_predictions.to_parquet(self.predictions_path, index=False)
                
                print(f"\n✓ All model predictions saved to: {self.predictions_path}")
                self.logger.info(f"Predictions saved to {self.predictions_path}")
                
                return final_predictions
            else:
                print("\n✗ No models were successfully trained")
                self.logger.warning("No models were successfully trained")
                return pd.DataFrame()
    
    def run_explainability(self, df: pd.DataFrame):
        """
        Run SHAP explainability analysis.
        
        Args:
            df: DataFrame with predictions
        """
        if df.empty:
            self.logger.warning("No data for explainability analysis")
            return
        
        with LoggingContext(self.logger, "explainability_phase"):
            print("\n[5/6] MODEL EXPLAINABILITY")
            print("-" * 40)
            
            self.logger.info("Starting SHAP analysis")
            
            try:
                shap_analyzer = SHAPAnalyzer()
                
                # Load trained models for SHAP analysis
                models_dir = Path("models")
                model_files = list(models_dir.glob(f"*_{self.timestamp}.joblib"))
                
                if not model_files:
                    self.logger.warning("No trained models found for SHAP analysis")
                    print("✗ No trained models found for SHAP analysis")
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
                for model_file in model_files[:3]:  # Limit to 3 models for performance
                    model_name = model_file.stem.replace(f"_{self.timestamp}", "")
                    
                    try:
                        # Run SHAP analysis based on model type
                        if 'linear' in model_name or 'ridge' in model_name or 'lasso' in model_name:
                            # Load linear model
                            if 'linear' in model_name:
                                model_obj = LinearRegressionModel()
                            elif 'ridge' in model_name:
                                model_obj = RidgeRegressionModel()
                            else:
                                model_obj = LassoRegressionModel()
                            
                            model_obj.load(model_file)
                            
                            # Run SHAP analysis
                            shap_results = shap_analyzer.analyze_linear_model(
                                model_obj.model,
                                X,
                                model_name
                            )
                            
                        elif 'xgboost' in model_name:
                            # Load XGBoost model
                            model_obj = XGBoostModel()
                            model_obj.load(model_file)
                            
                            # Run SHAP analysis
                            shap_results = shap_analyzer.analyze_xgboost(
                                model_obj.model,
                                X,
                                model_name
                            )
                        
                        # Generate report
                        report_path = Path(f"logs/shap_report_{model_name}_{self.timestamp}.html")
                        shap_analyzer.generate_report(shap_results, report_path)
                        
                        print(f"  ✓ SHAP analysis completed for {model_name}")
                        
                    except Exception as e:
                        self.logger.error(f"SHAP analysis failed for {model_name}: {e}")
                        print(f"  ✗ SHAP analysis failed for {model_name}: {e}")
                
                print("\n✓ SHAP analysis completed")
                self.logger.info("SHAP analysis completed")
                
            except Exception as e:
                self.logger.error(f"Explainability phase failed: {e}")
                print(f"\n✗ Explainability failed: {e}")
    
    def run_dashboard(self):
        """
        Launch the Streamlit dashboard.
        """
        with LoggingContext(self.logger, "dashboard_launch"):
            print("\n[6/6] DASHBOARD")
            print("-" * 40)
            
            self.logger.info("Starting dashboard launch")
            
            try:
                # Check if predictions exist
                if self.predictions_path is None:
                    # Find latest predictions
                    gold_dir = Path("data/gold")
                    prediction_files = list(gold_dir.glob("predictions_*.parquet"))
                    if prediction_files:
                        self.predictions_path = max(prediction_files, key=lambda x: x.stat().st_mtime)
                
                if self.predictions_path and self.predictions_path.exists():
                    print(f"\n✓ Predictions available: {self.predictions_path}")
                    print(f"✓ Dashboard ready with {pd.read_parquet(self.predictions_path).shape[0]} records")
                else:
                    print("\n⚠ No predictions found. Dashboard will use sample data.")
                
                print("\n" + "=" * 60)
                print("DASHBOARD LAUNCH INSTRUCTIONS")
                print("=" * 60)
                print("\nTo launch the dashboard, open a NEW terminal and run:")
                print("\n  streamlit run src/dashboard/app.py")
                print("\nOr use the quick start script:")
                print("\n  python quick_start.py")
                print("\nThe dashboard will open at: http://localhost:8501")
                print("\n" + "=" * 60)
                
                # Offer to launch dashboard automatically
                launch = input("\nDo you want to launch the dashboard now? (y/n): ")
                if launch.lower() == 'y':
                    self._launch_dashboard()
                
                self.logger.info("Dashboard instructions displayed")
                
            except Exception as e:
                self.logger.error(f"Dashboard setup failed: {e}")
                print(f"\n✗ Dashboard setup failed: {e}")
    
    def _launch_dashboard(self):
        """Launch the dashboard in a subprocess."""
        try:
            print("\n🚀 Launching dashboard...")
            
            # Start dashboard in background
            import threading
            
            def run_streamlit():
                subprocess.run([
                    sys.executable, '-m', 'streamlit', 'run',
                    'src/dashboard/app.py',
                    '--server.port', '8501',
                    '--server.address', '0.0.0.0',
                    '--server.headless', 'true'
                ])
            
            # Start dashboard thread
            dashboard_thread = threading.Thread(target=run_streamlit, daemon=True)
            dashboard_thread.start()
            
            # Wait for dashboard to start
            time.sleep(3)
            
            # Open browser
            webbrowser.open('http://localhost:8501')
            
            print("✅ Dashboard launched at http://localhost:8501")
            print("📊 Dashboard is running in the background")
            print("🛑 Press Ctrl+C in this terminal to stop the pipeline (dashboard will continue)")
            
            # Keep the main thread alive
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Pipeline stopped. Dashboard is still running.")
                print("   To stop the dashboard, close the browser tab and stop the Streamlit process.")
            
        except Exception as e:
            print(f"❌ Failed to launch dashboard: {e}")
            print("\nYou can still launch it manually with:")
            print("  streamlit run src/dashboard/app.py")
    
    def run_eda(self, df: pd.DataFrame):
        """
        Run Exploratory Data Analysis.
        
        Args:
            df: DataFrame for EDA
        """
        if df.empty:
            return
        
        with LoggingContext(self.logger, "eda_analysis"):
            print("\n[+1] EXPLORATORY DATA ANALYSIS")
            print("-" * 40)
            
            self.logger.info("Starting EDA")
            
            try:
                eda = EDAVisualizer()
                output_dir = Path(f"logs/eda_{self.timestamp}")
                report_path = eda.create_eda_report(df, output_dir)
                
                print(f"✓ EDA report generated: {report_path}")
                self.logger.info(f"EDA report generated: {report_path}")
                
            except Exception as e:
                self.logger.error(f"EDA failed: {e}")
                print(f"✗ EDA failed: {e}")
    
    def run_pipeline(self):
        """
        Run the complete pipeline.
        """
        with LoggingContext(self.logger, "complete_pipeline"):
            print("\nStarting complete pipeline execution...")
            
            # Setup directories
            self.setup_directories()
            
            # Define pipeline steps
            steps = [
                ("Scraping", self.run_scraping),
                ("Cleaning", self.run_cleaning),
                ("Feature Engineering", self.run_feature_engineering),
                ("Model Training", self.run_model_training)
            ]
            
            # Execute steps
            results = {}
            predictions = None
            
            for step_name, step_func in steps:
                print(f"\nExecuting step: {step_name}")
                
                if step_name == "Model Training":
                    predictions = step_func()
                    results[step_name] = not predictions.empty
                    
                    if results[step_name] and not predictions.empty:
                        # Run EDA on predictions
                        self.run_eda(predictions)
                        
                        # Run explainability
                        self.run_explainability(predictions)
                else:
                    results[step_name] = step_func()
            
            # Generate summary
            self._generate_summary(results, predictions)
            
            # Offer dashboard launch
            if any(results.values()):
                self.run_dashboard()
    
    def _generate_summary(self, results: dict, predictions: pd.DataFrame):
        """Generate execution summary."""
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 80)
        
        completed_steps = sum(results.values())
        total_steps = len(results)
        
        print(f"\nSteps Completed: {completed_steps}/{total_steps} ({completed_steps/total_steps*100:.1f}%)")
        print(f"Execution Timestamp: {self.timestamp}")
        print("\nStep Results:")
        
        for step_name, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"  {step_name:20} {status}")
        
        if predictions is not None and not predictions.empty:
            print(f"\nPredictions Generated: {len(predictions)} records")
            prediction_cols = [col for col in predictions.columns if any(x in col for x in ['prediction', 'regime', 'anomaly'])]
            print(f"Prediction Features: {len(prediction_cols)}")
        
        # List generated files
        print("\nData Files Generated:")
        if self.bronze_path:
            print(f"  Bronze: {self.bronze_path}")
        if self.silver_path:
            print(f"  Silver: {self.silver_path}")
        if self.gold_path:
            print(f"  Gold: {self.gold_path}")
        if self.predictions_path:
            print(f"  Predictions: {self.predictions_path}")
        
        print("\n" + "=" * 80)
        
        # Save summary to log file
        summary_file = Path(f"logs/pipeline_summary_{self.timestamp}.txt")
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MARKET NARRATIVE RISK INTELLIGENCE SYSTEM\n")
            f.write("Pipeline Execution Summary\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Steps Completed: {completed_steps}/{total_steps}\n\n")
            
            for step_name, success in results.items():
                status = "SUCCESS" if success else "FAILED"
                f.write(f"{step_name}: {status}\n")
        
        self.logger.info(f"Execution summary saved to {summary_file}")


def main():
    """
    Main entry point for the pipeline.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Market Narrative Risk Intelligence System"
    )
    
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
        "--dashboard-only",
        action="store_true",
        help="Launch dashboard only"
    )
    
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Run pipeline without launching dashboard"
    )
    
    args = parser.parse_args()
    
    # Initialize logging
    setup_pipeline_logging()
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator()
    
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
        orchestrator.gold_path = max(Path("data/gold").glob("*.parquet"), 
                                    key=lambda x: x.stat().st_mtime, default=None)
        predictions = orchestrator.run_model_training()
        if not predictions.empty:
            orchestrator.run_explainability(predictions)
            orchestrator.run_eda(predictions)
    elif args.dashboard_only:
        orchestrator.run_dashboard()
    else:
        # Run complete pipeline
        orchestrator.run_pipeline()


if __name__ == "__main__":
    main()
