"""
Polynomial regression model for non-linear stress score prediction.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path
from typing import Tuple, Dict

from src.utils.logger import model_logger
from src.utils.config_loader import config_loader


class PolynomialRegressionModel:
    """
    Polynomial regression model for capturing non-linear relationships.
    """
    
    def __init__(self):
        """Initialize polynomial regression model."""
        self.config = config_loader.get_config("config")
        self.model_config = self.config.get("models", {}).get("regression", {}).get("polynomial", {})
        self.degree = self.model_config.get('degree', 2)
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=self.degree)),
            ('linear', LinearRegression())
        ])
        
        self.feature_columns = None
        self.target_column = 'weighted_stress_score'
        model_logger.info("PolynomialRegressionModel initialized")
    
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
    
    def train(self, df: pd.DataFrame, tune_degree: bool = False) -> Dict:
        """
        Train polynomial regression model.
        
        Args:
            df: Training DataFrame
            tune_degree: Whether to tune polynomial degree
        
        Returns:
            Dictionary with training results
        """
        model_logger.info("Training polynomial regression model")
        
        # Prepare data
        X, y = self.prepare_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Tune polynomial degree
        if tune_degree:
            param_grid = {
                'poly__degree': [1, 2, 3, 4]
            }
            grid_search = GridSearchCV(
                self.pipeline,
                param_grid,
                cv=5,
                scoring='neg_mean_squared_error'
            )
            grid_search.fit(X_train, y_train)
            self.pipeline = grid_search.best_estimator_
            self.degree = grid_search.best_params_['poly__degree']
            model_logger.info(f"Best polynomial degree: {self.degree}")
        else:
            self.pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Get feature names after polynomial transformation
        poly_features = self.pipeline.named_steps['poly'].get_feature_names_out(self.feature_columns)
        coefficients = self.pipeline.named_steps['linear'].coef_
        
        # Feature importance (coefficients)
        feature_importance = pd.DataFrame({
            'feature': poly_features,
            'coefficient': coefficients
        }).sort_values('coefficient', key=abs, ascending=False)
        
        results = {
            'degree': self.degree,
            'mse': mse,
            'r2': r2,
            'feature_importance': feature_importance.head(20),  # Limit to top 20
            'n_features': len(poly_features)
        }
        
        model_logger.info(f"Polynomial regression (degree={self.degree}) trained: MSE={mse:.4f}, R2={r2:.4f}")
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
        predictions = self.pipeline.predict(X)
        
        # Create results DataFrame
        results = df.copy()
        results['polynomial_regression_prediction'] = predictions
        results['polynomial_regression_residual'] = results.get(
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
            'pipeline': self.pipeline,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'degree': self.degree
        }, filepath)
        model_logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: Path):
        """
        Load model from disk.
        
        Args:
            filepath: Path to model file
        """
        data = joblib.load(filepath)
        self.pipeline = data['pipeline']
        self.feature_columns = data['feature_columns']
        self.target_column = data['target_column']
        self.degree = data['degree']
        model_logger.info(f"Model loaded from {filepath}")
