"""
Time-lagged regression model for stress score prediction.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Tuple, Dict
import joblib
from pathlib import Path

from src.utils.logger import model_logger
from src.utils.config_loader import config_loader


class TimeLaggedRegressionModel:
    """
    Time-lagged regression model using historical features to predict stress scores.
    """
    
    def __init__(self, lag_window: int = 7):
        """
        Initialize time-lagged regression model.
        
        Args:
            lag_window: Number of days to use for lagged features
        """
        self.config = config_loader.get_config("config")
        self.lag_window = lag_window
        self.model = Ridge(alpha=1.0)
        self.feature_columns = None
        model_logger.info(f"TimeLaggedRegressionModel initialized (lag_window={lag_window})")
    
    def create_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-lagged features from the dataset.
        
        Args:
            df: Input DataFrame with timestamp and features
        
        Returns:
            DataFrame with lagged features
        """
        # Ensure timestamp column exists and is datetime
        if 'timestamp' not in df.columns:
            model_logger.warning("No timestamp column found, using index")
            df = df.copy()
            df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='D')
        
        df_sorted = df.copy()
        df_sorted['timestamp'] = pd.to_datetime(df_sorted['timestamp'])
        df_sorted = df_sorted.sort_values('timestamp')
        df_sorted.set_index('timestamp', inplace=True)
        
        # **CRITICAL FIX: Select only numeric columns before resampling**
        numeric_cols = df_sorted.select_dtypes(include=[np.number]).columns.tolist()
        df_numeric = df_sorted[numeric_cols]
        
        # Resample to daily frequency and fill missing values
        try:
            df_resampled = df_numeric.resample('D').mean()
            df_resampled = df_resampled.fillna(method='ffill').fillna(0)
        except Exception as e:
            model_logger.warning(f"Resampling failed: {e}. Using original data.")
            df_resampled = df_numeric.fillna(0)
        
        # Create lagged features
        lagged_data = pd.DataFrame(index=df_resampled.index)
        
        # Get numeric feature columns (excluding target if present)
        feature_cols = [col for col in df_resampled.columns 
                       if col not in ['weighted_stress_score', 'stress_score']]
        
        if len(feature_cols) == 0:
            model_logger.warning("No numeric features available for lagging")
            # Return a minimal dataframe with just the target if available
            if 'weighted_stress_score' in df_resampled.columns:
                return df_resampled[['weighted_stress_score']].reset_index()
            else:
                return pd.DataFrame()
        
        # Add current values
        for col in feature_cols:
            lagged_data[f'{col}_current'] = df_resampled[col]
        
        # Add lagged values
        for lag in range(1, self.lag_window + 1):
            for col in feature_cols:
                lagged_data[f'{col}_lag_{lag}'] = df_resampled[col].shift(lag)
        
        # Add rolling statistics
        for col in feature_cols:
            lagged_data[f'{col}_rolling_mean'] = df_resampled[col].rolling(
                window=self.lag_window, min_periods=1
            ).mean()
            lagged_data[f'{col}_rolling_std'] = df_resampled[col].rolling(
                window=self.lag_window, min_periods=1
            ).std().fillna(0)
        
        # Add target variable
        if 'weighted_stress_score' in df_resampled.columns:
            lagged_data['target'] = df_resampled['weighted_stress_score']
        elif 'stress_score' in df_resampled.columns:
            lagged_data['target'] = df_resampled['stress_score']
        else:
            model_logger.warning("No target variable found")
            lagged_data['target'] = 0
        
        # Drop rows with NaN values from lagging
        lagged_data = lagged_data.dropna()
        
        return lagged_data.reset_index()
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target for training.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Tuple of (X, y) arrays
        """
        # Create lagged features
        lagged_df = self.create_lagged_features(df)
        
        if len(lagged_df) == 0:
            model_logger.error("No data available after creating lagged features")
            return np.array([]), np.array([])
        
        # Separate features and target
        if 'target' in lagged_df.columns:
            y = lagged_df['target'].values
            X_cols = [col for col in lagged_df.columns 
                     if col not in ['target', 'timestamp', 'index']]
            X = lagged_df[X_cols].fillna(0).values
            self.feature_columns = X_cols
        else:
            model_logger.error("No target column found")
            return np.array([]), np.array([])
        
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
        
        if len(X) == 0 or len(y) == 0:
            model_logger.error("Insufficient data for training")
            return {
                'error': 'Insufficient data',
                'mse': 0,
                'r2': 0,
                'mae': 0,
                'n_samples': 0
            }
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=min(3, len(X) - 1))
        
        mse_scores = []
        r2_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_val)
            
            mse_scores.append(mean_squared_error(y_val, y_pred))
            r2_scores.append(r2_score(y_val, y_pred))
        
        # Final training on all data
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        # Feature importance (coefficients)
        if self.feature_columns:
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'coefficient': self.model.coef_
            }).sort_values('coefficient', key=abs, ascending=False)
        else:
            feature_importance = pd.DataFrame()
        
        results = {
            'mse': mse,
            'r2': r2,
            'mae': mae,
            'cv_mse_mean': np.mean(mse_scores) if mse_scores else 0,
            'cv_r2_mean': np.mean(r2_scores) if r2_scores else 0,
            'n_samples': len(X),
            'n_features': X.shape[1] if len(X) > 0 else 0,
            'lag_window': self.lag_window,
            'feature_importance': feature_importance
        }
        
        model_logger.info(
            f"Time-lagged regression trained: MSE={mse:.4f}, R2={r2:.4f}, "
            f"Samples={len(X)}, Features={X.shape[1] if len(X) > 0 else 0}"
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
        # Check if model was actually trained (not just initialized)
        if self.model is None or self.feature_columns is None:
            model_logger.warning("Model not trained, returning original dataframe")
            results = df.copy()
            results['predicted_stress'] = 0
            results['prediction_error'] = 0
            return results
        
        # Prepare features
        X, _ = self.prepare_data(df)
        
        if len(X) == 0:
            model_logger.warning("No features available for prediction")
            results = df.copy()
            results['predicted_stress'] = 0
            return results
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Create results DataFrame
        lagged_df = self.create_lagged_features(df)
        results = lagged_df.copy()
        results['predicted_stress'] = predictions
        
        if 'target' in results.columns:
            results['prediction_error'] = results['target'] - results['predicted_stress']
        
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
        self.lag_window = data['lag_window']
        model_logger.info(f"Model loaded from {filepath}")
