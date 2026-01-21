"""
Main pipeline orchestrator for Market Narrative Risk Intelligence System.
"""
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.logger import (
    scraper_logger, preprocessing_logger, model_logger, dashboard_logger
)
from src.utils.config_loader import config_loader

# Import pipeline components
from src.scraper.investing_scraper import scrape_and_save
from src.preprocessing.clean_data import clean_and_save
from src.preprocessing.feature_engineering import engineer_and_save

# Import models
from src.models.regression.linear_regression import LinearRegressionModel
from src.models.regression.ridge_regression import RidgeRegressionModel
from src.models.regression.lasso_regression import LassoRegressionModel
from src.models.neural_network import NeuralNetworkModel
from src.models.xgboost_model import XGBoostModel
from src.models.knn_model import KNNModel
from src.models.isolation_forest import IsolationForestModel

from src.explainability.shap_analysis import SHAPAnalyzer


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
        
        # Initialize loggers
        self.loggers = {
            'scraper': scraper_logger,
            'preprocessing': preprocessing_logger,
            'model': model_logger,
            'dashboard': dashboard_logger
        }
        
        # File paths
        self.bronze_path = None
        self.silver_path = None
        self.gold_path = None
        
        print("=" * 80)
        print("MARKET NARRATIVE RISK INTELLIGENCE SYSTEM")
        print("=" * 80)
    
    def run_scraping(self) -> bool:
        """
        Run scraping step.
        
        Returns:
            True if successful, False otherwise
        """
        print("\n[1/6] SCRAPING")
        print("-" * 40)
        
        try:
            self.loggers['scraper'].info("Starting scraping phase")
            self.bronze_path = scrape_and_save()
            
            if self.bronze_path:
                print(f"✓ Scraping completed: {self.bronze_path}")
                self.loggers['scraper'].info(f"Scraping completed: {self.bronze_path}")
                return True
            else:
                print("✗ Scraping failed: No articles collected")
                return False
                
        except Exception as e:
            self.loggers['scraper'].error(f"Scraping failed: {e}")
            print(f"✗ Scraping failed: {e}")
            return False
    
    def run_cleaning(self) -> bool:
        """
        Run data cleaning step.
        
        Returns:
            True if successful, False otherwise
        """
        print("\n[2/6] DATA CLEANING")
        print("-" * 40)
        
        try:
            self.loggers['preprocessing'].info("Starting cleaning phase")
            self.silver_path = clean_and_save(self.bronze_path)
            
            if self.silver_path:
                print(f"✓ Cleaning completed: {self.silver_path}")
                self.loggers['preprocessing'].info(f"Cleaning completed: {self.silver_path}")
                return True
            else:
                print("✗ Cleaning failed: No valid data")
                return False
                
        except Exception as e:
            self.loggers['preprocessing'].error(f"Cleaning failed: {e}")
            print(f"✗ Cleaning failed: {e}")
            return False
    
    def run_feature_engineering(self) -> bool:
        """
        Run feature engineering step.
        
        Returns:
            True if successful, False otherwise
        """
        print("\n[3/6] FEATURE ENGINEERING")
        print("-" * 40)
        
        try:
            self.loggers['preprocessing'].info("Starting feature engineering phase")
            self.gold_path = engineer_and_save(self.silver_path)
            
            if self.gold_path:
                print(f"✓ Feature engineering completed: {self.gold_path}")
                self.loggers['preprocessing'].info(f"Feature engineering completed: {self.gold_path}")
                return True
            else:
                print("✗ Feature engineering failed")
                return False
                
        except Exception as e:
            self.loggers['preprocessing'].error(f"Feature engineering failed: {e}")
            print(f"✗ Feature engineering failed: {e}")
            return False
    
    def run_model_training(self) -> pd.DataFrame:
        """
        Run all model training steps.
        
        Returns:
            DataFrame with all predictions
        """
        print("\n[4/6] MODEL TRAINING")
        print("-" * 40)
        
        try:
            # Load gold layer data
            self.loggers['model'].info("Loading gold layer data for model training")
            df = pd.read_parquet(self.gold_path)
            print(f"✓ Loaded {len(df)} records for model training")
            
            # Initialize models
            models = {
                'linear_regression': LinearRegressionModel(),
                'ridge_regression': RidgeRegressionModel(),
                'lasso_regression': LassoRegressionModel(),
                'neural_network': NeuralNetworkModel(),
                'xgboost': XGBoostModel(),
                'knn': KNNModel(),
                'isolation_forest': IsolationForestModel()
            }
            
            # Train models and collect predictions
            predictions_dfs = []
            
            for model_name, model in models.items():
                print(f"\n  Training {model_name.replace('_', ' ').title()}...")
                self.loggers['model'].info(f"Training {model_name}")
                
                try:
                    # Train model
                    results = model.train(df)
                    
                    # Make predictions
                    predictions = model.predict(df)
                    
                    # Extract relevant prediction columns
                    pred_cols = [col for col in predictions.columns 
                               if any(x in col for x in ['prediction', 'regime', 'anomaly', 'similarity'])]
                    
                    # Save predictions
                    if pred_cols:
                        predictions_subset = predictions[['timestamp'] + pred_cols]
                        predictions_dfs.append(predictions_subset)
                        print(f"  ✓ {model_name} trained successfully")
                    
                    # Save model
                    model.save(Path(f"models/{model_name}_{self.timestamp}.joblib"))
                    
                except Exception as e:
                    self.loggers['model'].error(f"{model_name} training failed: {e}")
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
                feature_cols = [col for col in df.columns if col != 'timestamp']
                final_predictions = final_predictions.merge(
                    df[['timestamp'] + feature_cols],
                    on='timestamp',
                    how='left'
                )
                
                # Save predictions
                predictions_path = Path("data/gold") / f"predictions_{self.timestamp}.parquet"
                final_predictions.to_parquet(predictions_path, index=False)
                
                print(f"\n✓ All model predictions saved to: {predictions_path}")
                self.loggers['model'].info(f"Predictions saved to {predictions_path}")
                
                return final_predictions
            else:
                print("\n✗ No models were successfully trained")
                return pd.DataFrame()
                
        except Exception as e:
            self.loggers['model'].error(f"Model training phase failed: {e}")
            print(f"\n✗ Model training failed: {e}")
            return pd.DataFrame()
    
    def run_explainability(self, df: pd.DataFrame):
        """
        Run SHAP explainability analysis.
        
        Args:
            df: DataFrame with predictions
        """
        print("\n[5/6] MODEL EXPLAINABILITY")
        print("-" * 40)
        
        try:
            self.loggers['model'].info("Starting SHAP analysis")
            
            # Initialize SHAP analyzer
            shap_analyzer = SHAPAnalyzer()
            
            # Load trained models for SHAP analysis
            models_dir = Path("models")
            model_files = list(models_dir.glob("*.joblib"))
            
            if not model_files:
                print("✗ No trained models found for SHAP analysis")
                return
            
            # Prepare feature data for SHAP
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ['weighted_stress_score', 'sentiment_polarity', 'vader_compound']
            feature_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            X = df[feature_cols].fillna(0)
            
            # Analyze each model
            for model_file in model_files[:3]:  # Limit to 3 models for performance
                model_name = model_file.stem.split('_')[0]
                
                try:
                    # Load model
                    if model_name == 'linear':
                        from src.models.regression.linear_regression import LinearRegressionModel
                        model_obj = LinearRegressionModel()
                        model_obj.load(model_file)
                        
                        # Run SHAP analysis
                        shap_results = shap_analyzer.analyze_linear_model(
                            model_obj.model,
                            X,
                            model_name
                        )
                        
                    elif model_name == 'xgboost':
                        from src.models.xgboost_model import XGBoostModel
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
                    self.loggers['model'].error(f"SHAP analysis failed for {model_name}: {e}")
                    print(f"  ✗ SHAP analysis failed for {model_name}: {e}")
            
            print("\n✓ SHAP analysis completed")
            self.loggers['model'].info("SHAP analysis completed")
            
        except Exception as e:
            self.loggers['model'].error(f"Explainability phase failed: {e}")
            print(f"\n✗ Explainability failed: {e}")
    
    def run_dashboard(self):
        """
        Launch the Streamlit dashboard.
        """
        print("\n[6/6] DASHBOARD")
        print("-" * 40)
        
        try:
            self.loggers['dashboard'].info("Starting dashboard")
            
            print("\nDashboard is ready to run!")
            print("\nTo launch the dashboard, run:")
            print("  streamlit run src/dashboard/app.py")
            
            # Note: We don't actually launch Streamlit here to avoid blocking
            # In production, this would be run separately
            
            self.loggers['dashboard'].info("Dashboard instructions displayed")
            
        except Exception as e:
            self.loggers['dashboard'].error(f"Dashboard setup failed: {e}")
            print(f"\n✗ Dashboard setup failed: {e}")
    
    def run_pipeline(self):
        """
        Run the complete pipeline.
        """
        print("\nStarting complete pipeline execution...")
        
        # Create necessary directories
        Path("data/bronze").mkdir(parents=True, exist_ok=True)
        Path("data/silver").mkdir(parents=True, exist_ok=True)
        Path("data/gold").mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(parents=True, exist_ok=True)
        Path("models").mkdir(parents=True, exist_ok=True)
        
        # Run pipeline steps
        steps = [
            ("Scraping", self.run_scraping),
            ("Cleaning", self.run_cleaning),
            ("Feature Engineering", self.run_feature_engineering),
            ("Model Training", lambda: self.run_model_training())
        ]
        
        results = {}
        
        for step_name, step_func in steps:
            if step_name == "Model Training":
                # Model training returns predictions DataFrame
                predictions = step_func()
                results[step_name] = not predictions.empty
                if results[step_name]:
                    # Run explainability on successful predictions
                    self.run_explainability(predictions)
            else:
                results[step_name] = step_func()
        
        # Run dashboard instructions
        self.run_dashboard()
        
        # Summary
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 80)
        
        for step_name, success in results.items():
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"{step_name:25} {status}")
        
        print("\n" + "=" * 80)
        
        if all(results.values()):
            print("Pipeline completed successfully!")
            self.loggers['model'].info("Pipeline completed successfully")
        else:
            print("Pipeline completed with errors.")
            self.loggers['model'].warning("Pipeline completed with errors")


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
    
    args = parser.parse_args()
    
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
    else:
        # Run complete pipeline
        orchestrator.run_pipeline()


if __name__ == "__main__":
    main()
