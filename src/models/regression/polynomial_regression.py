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
from typing import Tuple, Dict, Optional

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
            ('poly', PolynomialFeatures(degree=self.degree, include_bias=False)),
            ('linear', LinearRegression(fit_intercept=True))
        ])
        
        self.feature_columns = None
        self.target_column = 'weighted_stress_score'
        model_logger.info(f"PolynomialRegressionModel initialized (degree={self.degree})")
    
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
        
        # Limit to top features to avoid explosion of polynomial terms
        if len(feature_cols) > 10:
            # Select top correlated features with target
            if self.target_column in df.columns:
                correlations = df[feature_cols].corrwith(df[self.target_column]).abs()
                feature_cols = correlations.nlargest(10).index.tolist()
            else:
                feature_cols = feature_cols[:10]
        
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
        
        # Tune polynomial degree if requested
        if tune_degree:
            model_logger.info("Tuning polynomial degree")
            param_grid = {
                'poly__degree': [1, 2, 3]
            }
            
            # Use smaller grid for faster tuning
            grid_search = GridSearchCV(
                self.pipeline,
                param_grid,
                cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            self.pipeline = grid_search.best_estimator_
            self.degree = grid_search.best_params_['poly__degree']
            model_logger.info(f"Best polynomial degree: {self.degree}")
        else:
            # Train with fixed degree
            self.pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Get feature names after polynomial transformation
        poly = self.pipeline.named_steps['poly']
        linear = self.pipeline.named_steps['linear']
        
        try:
            poly_features = poly.get_feature_names_out(self.feature_columns)
            coefficients = linear.coef_
            
            # Feature importance (coefficients)
            feature_importance = pd.DataFrame({
                'feature': poly_features,
                'coefficient': coefficients
            }).sort_values('coefficient', key=abs, ascending=False)
        except:
            # Fallback if feature names not available
            n_features = poly.n_output_features_
            feature_importance = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(n_features)],
                'coefficient': linear.coef_
            }).sort_values('coefficient', key=abs, ascending=False)
        
        # Calculate polynomial complexity metrics
        n_terms = len(feature_importance)
        significant_terms = (np.abs(feature_importance['coefficient']) > 0.01).sum()
        
        results = {
            'degree': self.degree,
            'mse': mse,
            'r2': r2,
            'feature_importance': feature_importance.head(15),  # Limit to top 15
            'n_polynomial_terms': n_terms,
            'significant_terms': significant_terms,
            'model_params': self.pipeline.get_params()
        }
        
        model_logger.info(
            f"Polynomial regression (degree={self.degree}) trained: "
            f"MSE={mse:.4f}, R2={r2:.4f}, "
            f"Terms={n_terms}, Significant={significant_terms}"
        )
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
        
        # Add residual if target exists
        if self.target_column in df.columns:
            results['polynomial_regression_residual'] = (
                df[self.target_column] - predictions
            )
        
        return results
    
    def analyze_nonlinearity(self, df: pd.DataFrame) -> Dict:
        """
        Analyze non-linear relationships between features and target.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Dictionary with non-linearity analysis
        """
        if self.feature_columns is None:
            raise ValueError("Model must be trained before analysis")
        
        analysis = {}
        
        # Prepare features
        X = df[self.feature_columns].fillna(0)
        y = df[self.target_column].fillna(0) if self.target_column in df.columns else None
        
        # Get polynomial transformer and linear model
        poly = self.pipeline.named_steps['poly']
        linear = self.pipeline.named_steps['linear']
        
        try:
            # Get polynomial feature names
            poly_features = poly.get_feature_names_out(self.feature_columns)
            
            # Identify interaction terms
            interaction_terms = []
            for feature in poly_features:
                if ' ' in feature or '*' in feature:  # Interaction term
                    interaction_terms.append(feature)
            
            # Calculate contribution of interaction terms
            interaction_coefs = []
            for term in interaction_terms:
                if term in poly_features:
                    idx = np.where(poly_features == term)[0][0]
                    interaction_coefs.append({
                        'term': term,
                        'coefficient': linear.coef_[idx],
                        'abs_contribution': np.abs(linear.coef_[idx])
                    })
            
            # Sort by absolute contribution
            interaction_coefs.sort(key=lambda x: x['abs_contribution'], reverse=True)
            
            analysis['interaction_terms'] = interaction_coefs[:10]  # Top 10
            analysis['n_interaction_terms'] = len(interaction_terms)
            
        except Exception as e:
            model_logger.warning(f"Non-linearity analysis failed: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
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


def create_polynomial_model():
    """Factory function to create polynomial regression model."""
    return PolynomialRegressionModel()
