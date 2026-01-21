"""
Market Narrative Risk Intelligence System
"""
__version__ = "1.0.0"
__author__ = "Market Intelligence Team"

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

# Import explainability
from src.explainability.shap_analysis import SHAPAnalyzer

# Import dashboard
from src.dashboard.app import RiskIntelligenceDashboard, main as run_dashboard

# Import utils
from src.utils.logger import setup_logger, PipelineLogger
from src.utils.config_loader import ConfigLoader, config_loader

__all__ = [
    # Scraper
    'InvestingScraper',
    'scrape_and_save',
    
    # Preprocessing
    'DataCleaner',
    'clean_and_save',
    'FeatureEngineer',
    'engineer_and_save',
    
    # Models
    'LinearRegressionModel',
    'RidgeRegressionModel',
    'LassoRegressionModel',
    'PolynomialRegressionModel',
    'TimeLaggedRegressionModel',
    'NeuralNetworkModel',
    'XGBoostModel',
    'KNNModel',
    'IsolationForestModel',
    
    # Explainability
    'SHAPAnalyzer',
    
    # Dashboard
    'RiskIntelligenceDashboard',
    'run_dashboard',
    
    # Utils
    'setup_logger',
    'PipelineLogger',
    'ConfigLoader',
    'config_loader'
]
