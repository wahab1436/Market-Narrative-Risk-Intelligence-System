"""
Time-lagged regression model for stress score prediction.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from pathlib import Path
from typing import Tuple, Dict, List
from datetime import datetime, timedelta

from src.utils.logger import model_logger
from src.utils.config_loader import config_loader


class TimeLaggedRegressionModel:
    """
    Time-lagged regression model for time-series stress score prediction.
    """
    
    def __init__(self):
        """Initialize time-lagged regression model."""
        self.config = config_loader.get_config("config")
        self.model_config = self.config.get("models", {}).get("regression", {}).get("time_lagged", {})
        self.lag_window = self.model_config.get('lag_window', 7)
        self.model = LinearRegression()
        self.feature_columns = None
        self.target_column = 'weighted_stress_score'
        model_logger.info("TimeLaggedRegressionModel initialized")
    
    def create_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lagged features for time-series analysis.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with lagged features
        """
        if 'timestamp' not in df.columns:
            raise ValueError("DataFrame must have 'timestamp' column for time-lagged features")
        
        # Ensure sorted by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Create lagged features for key columns
        lag_columns = ['weighted_stress_score', 'keyword_stress_score', 'sentiment_polarity']
        available_columns = [col for col in lag_columns if col in df.columns]
        
        lagged_df = df.copy()
        
        for col in available_columns:
            for lag in range(1, self.lag_window + 1):
                lagged_df[f'{col}_lag_{lag}'] = lagged_df[col].shift(lag)
        
        # Add moving averages
        for col in available_columns:
            lagged_df[f'{col}_ma_3'] = lagged_df[col].rolling(3).mean()
            lagged_df[f'{col}_ma_7'] = lagged_df[col].rolling(7).mean()
        
        # Add day of week and month features
        lagged_df['day_of_week'] = lagged_df['timestamp'].dt.dayofweek
        lagged_df['month'] = lagged_df['timestamp'].dt.month
        
        return lagged_df.dropna()
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for training.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Tuple of (features, target)
        """
        # Create lagged features
        lagged_df = self.create_lagged_features(df)
        
        # Select lagged feature columns
        lagged_cols = [col for col in lagged_df.columns if '_lag_' in col or '_ma_' in col]
        time_cols = ['day_of_week', 'month']
        
        feature_cols = lagged_cols + time_cols
        self.feature_columns = feature_cols
        
        X = lagged_df[feature_cols].fillna(0)
        y = lagged_df[self.target_column]
        
        return X, y
    
    def train(self, df: pd.DataFrame) -> Dict:
        """
        Train time-lagged regression model.
        
        Args:
            df: Training DataFrame
        
        Returns:
            Dictionary with training results
        """
        model_logger.info("Training time-lagged regression model")
        
        # Prepare data
        X, y = self.prepare_data(df)
        
        # Use time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        cv_scores = {'mse': [], 'r2': []}
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train on fold
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            cv_scores['mse'].append(mean_squared_error(y_test, y_pred))
            cv_scores['r2'].append(r2_score(y_test, y_pred))
        
        # Final training on all data
        self.model.fit(X, y)
        
        # Cross-validation results
        avg_mse = np.mean(cv_scores['mse'])
        avg_r2 = np.mean(cv_scores['r2'])
        
        # Feature importance (coefficients)
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'coefficient': self.model.coef_
        }).sort_values('coefficient', key=abs, ascending=False)
        
        results = {
            'lag_window': self.lag_window,
            'cv_mse': avg_mse,
            'cv_r2': avg_r2,
            'feature_importance': feature_importance,
            'cv_scores': cv_scores
        }
        
        model_logger.info(f"Time-lagged regression (window={self.lag_window}) trained: CV MSE={avg_mse:.4f}, CV R2={avg_r2:.4f}")
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
        
        # Create lagged features
        lagged_df = self.create_lagged_features(df)
        
        # Prepare features
        X = lagged_df[self.feature_columns].fillna(0)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Create results DataFrame
        results = df.copy()
        
        # Align predictions with original DataFrame
        aligned_predictions = pd.Series(index=df.index, dtype=float)
        aligned_residuals = pd.Series(index=df.index, dtype=float)
        
        # Match by timestamp
        for idx, timestamp in enumerate(lagged_df['timestamp']):
            mask = df['timestamp'] == timestamp
            if mask.any():
                aligned_predictions[mask] = predictions[idx]
                if self.target_column in df.columns:
                    aligned_residuals[mask] = df.loc[mask, self.target_column].iloc[0] - predictions[idx]
        
        results['time_lagged_prediction'] = aligned_predictions
        results['time_lagged_residual'] = aligned_residuals
        
        return results
    
    def forecast(self, df: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
        """
        Forecast future stress scores.
        
        Args:
            df: Historical DataFrame
            horizon: Number of days to forecast
        
        Returns:
            DataFrame with forecasts
        """
        if self.feature_columns is None:
            raise ValueError("Model must be trained before forecasting")
        
        # Get the latest data
        latest_data = df.sort_values('timestamp').tail(self.lag_window).copy()
        
        # Create forecasts
        forecasts = []
        current_features = latest_data.copy()
        
        for i in range(horizon):
            # Create lagged features for current point
            lagged_current = self.create_lagged_features(current_features)
            
            if lagged_current.empty:
                break
            
            # Get features for prediction
            X_current = lagged_current[self.feature_columns].fillna(0).iloc[-1:]
            
            # Make prediction
            forecast = self.model.predict(X_current)[0]
            
            # Create new timestamp
            last_timestamp = current_features['timestamp'].iloc[-1]
            new_timestamp = last_timestamp + timedelta(days=1)
            
            # Store forecast
            forecasts.append({
                'timestamp': new_timestamp,
                'forecast': forecast,
                'horizon': i + 1
            })
            
            # Update current features with forecast for next iteration
            new_row = pd.DataFrame([{
                'timestamp': new_timestamp,
                'weighted_stress_score': forecast,
                'keyword_stress_score': current_features['keyword_stress_score'].iloc[-1],
                'sentiment_polarity': current_features['sentiment_polarity'].iloc[-1]
            }])
            current_features = pd.concat([current_features, new_row], ignore_index=True)
        
        return pd.DataFrame(forecasts)
    
    def save(self, filepath: Path):
        """
        Save model to disk.
        
        Args:
            filepath: Path to save model
        """
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'lag_window': self.lag_window
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
        self.lag_window = data['lag_window']
        model_logger.info(f"Model loaded from {filepath}")
