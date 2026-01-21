"""
Linear regression model for stress score prediction.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from pathlib import Path
from typing import Tuple, Optional

from src.utils.logger import model_logger
from src.utils.config_loader import config_loader


class LinearRegressionModel:
    """
    Linear regression model for continuous stress score prediction.
    """
    
    def __init__(self):
        """Initialize linear regression model."""
        self.config = config_loader.get_config("config")
        self.model_config = self.config.get("models", {}).get("regression", {}).get("linear", {})
        self.model = LinearRegression(**self.model_config)
        self.feature_columns = None
        self.target_column = 'weighted_stress_score'
        model_logger.info("LinearRegressionModel initialized")
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for training.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Tuple of (features, target)
        """
        # Select numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target and irrelevant columns
        exclude_cols = [self.target_column, 'sentiment_polarity', 'vader_compound']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        self.feature_columns = feature_cols
        
        X = df[feature_cols].fillna(0)
        y = df[self.target_column].fillna(0)
        
        return X, y
    
    def train(self, df: pd.DataFrame) -> dict:
        """
        Train linear regression model.
        
        Args:
            df: Training DataFrame
        
        Returns:
            Dictionary with training results
        """
        model_logger.info("Training linear regression model")
        
        # Prepare data
        X, y = self.prepare_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Feature importance (coefficients)
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'coefficient': self.model.coef_
        }).sort_values('coefficient', key=abs, ascending=False)
        
        results = {
            'mse': mse,
            'r2': r2,
            'feature_importance': feature_importance,
            'model_params': self.model.get_params()
        }
        
        model_logger.info(f"Linear regression trained: MSE={mse:.4f}, R2={r2:.4f}")
        return results
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on new data.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with predictions
        """
        if self.feature_columns is None:
            raise ValueError("Model must be trained before prediction")
        
        # Prepare features
        X = df[self.feature_columns].fillna(0)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Create results DataFrame
        results = df.copy()
        results['linear_regression_prediction'] = predictions
        results['linear_regression_residual'] = results.get(
            self.target_column, 0
        ) - predictions
        
        return results
    
    def save(self, filepath: Path):
        """
        Save model to disk.
        
        Args:
            filepath: Path to save model
        """
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }, filepath)
        model_logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: Path):
        """
        Load model from disk.
        
        Args:
            filepath: Path to model file
        """
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_columns = data['feature_columns']
        self.target_column = data['target_column']
        model_logger.info(f"Model loaded from {filepath}")
