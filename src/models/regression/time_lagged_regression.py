"""
Time-lagged regression model for stress score prediction.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Tuple, Dict
import joblib
from pathlib import Path

from src.utils.logger import model_logger
from src.utils.config_loader import config_loader

class TimeLaggedRegressionModel:
    def __init__(self, lag_window: int = 7):
        self.config = config_loader.get_config("config")
        self.lag_window = lag_window
        self.model = Ridge(alpha=1.0)
        self.feature_columns = None
        model_logger.info(f"TimeLaggedRegressionModel initialized (lag_window={lag_window})")
    
    def create_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-lagged features with a safety fallback for small datasets.
        """
        if df.empty:
            return pd.DataFrame()

        # Ensure timestamp exists
        df_sorted = df.copy()
        if 'timestamp' not in df_sorted.columns:
            df_sorted['timestamp'] = pd.to_datetime('today')
        
        df_sorted['timestamp'] = pd.to_datetime(df_sorted['timestamp'])
        df_sorted = df_sorted.sort_values('timestamp')
        
        # Isolate numeric features
        numeric_cols = df_sorted.select_dtypes(include=[np.number]).columns.tolist()
        # Keep timestamp for resampling/alignment
        df_numeric = df_sorted[['timestamp'] + numeric_cols].set_index('timestamp')
        
        try:
            # Try daily resampling
            df_resampled = df_numeric.resample('D').mean()
            # If resampling collapses data to too few points, use raw sequence
            if len(df_resampled.dropna(how='all')) < 2:
                model_logger.info("Daily resampling resulted in too few rows. Using raw article sequence.")
                df_resampled = df_numeric.copy()
            
            df_resampled = df_resampled.fillna(method='ffill').fillna(0)
        except Exception as e:
            model_logger.warning(f"Resampling failed: {e}. Using original data.")
            df_resampled = df_numeric.fillna(0)
        
        lagged_data = pd.DataFrame(index=df_resampled.index)
        feature_cols = [col for col in df_resampled.columns 
                       if col not in ['weighted_stress_score', 'stress_score', 'target']]
        
        # Generate Lags
        for col in feature_cols:
            lagged_data[f'{col}_current'] = df_resampled[col]
            for lag in range(1, self.lag_window + 1):
                lagged_data[f'{col}_lag_{lag}'] = df_resampled[col].shift(lag)
            
            # Rolling statistics
            lagged_data[f'{col}_rolling_mean'] = df_resampled[col].rolling(
                window=self.lag_window, min_periods=1).mean()

        # Target Logic
        if 'weighted_stress_score' in df_resampled.columns:
            lagged_data['target'] = df_resampled['weighted_stress_score']
        elif 'stress_score' in df_resampled.columns:
            lagged_data['target'] = df_resampled['stress_score']
        else:
            lagged_data['target'] = 0
            
        return lagged_data.fillna(0).reset_index()

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Extracts features (X) and target (y).
        """
        lagged_df = self.create_lagged_features(df)
        if lagged_df.empty:
            return np.array([]), np.array([]), []
            
        X_cols = [col for col in lagged_df.columns if col not in ['target', 'timestamp', 'index', 'level_0']]
        X = lagged_df[X_cols].values
        y = lagged_df['target'].values
        return X, y, X_cols

    def train(self, df: pd.DataFrame) -> Dict:
        """
        Trains model and returns metrics PLUS predictions for the pipeline.
        """
        model_logger.info("Training time-lagged regression model")
        X, y, X_cols = self.prepare_data(df)
        self.feature_columns = X_cols
        
        if len(X) < 2:
            return {'error': 'Insufficient data', 'n_samples': len(X)}
        
        # Fit and Predict
        self.model.fit(X, y)
        predictions = self.model.predict(X)
        
        # Create a results dataframe for the pipeline
        results_df = self.create_lagged_features(df)
        results_df['prediction'] = predictions
        
        metrics = {
            'mse': mean_squared_error(y, predictions),
            'r2': r2_score(y, predictions),
            'n_samples': len(X),
            'predictions': results_df # Crucial: return the dataframe with predictions
        }
        
        model_logger.info(f"Trained: MSE={metrics['mse']:.4f}, Samples={len(X)}")
        return metrics

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame containing predictions.
        """
        X, _, _ = self.prepare_data(df)
        results_df = self.create_lagged_features(df)
        
        if len(X) == 0:
            results_df['prediction'] = 0
            return results_df
            
        results_df['prediction'] = self.model.predict(X)
        return results_df

    def save(self, filepath: Path):
        joblib.dump({'model': self.model, 'features': self.feature_columns, 'window': self.lag_window}, filepath)

    def load(self, filepath: Path):
        data = joblib.load(filepath)
        self.model, self.feature_columns, self.lag_window = data['model'], data['features'], data['window']
