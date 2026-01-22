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
    """
    Time-lagged regression model using historical features to predict stress scores.
    """
    
    def __init__(self, lag_window: int = 7):
        self.config = config_loader.get_config("config")
        self.lag_window = lag_window
        self.model = Ridge(alpha=1.0)
        self.feature_columns = None
        model_logger.info(f"TimeLaggedRegressionModel initialized (lag_window={lag_window})")
    
    def create_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-lagged features with safety for small/raw datasets.
        """
        if df.empty:
            return pd.DataFrame()

        df_sorted = df.copy()
        if 'timestamp' not in df_sorted.columns:
            # Fallback for raw sequences
            df_sorted['timestamp'] = pd.date_range(start=pd.Timestamp.now(), periods=len(df_sorted), freq='H')
        
        df_sorted['timestamp'] = pd.to_datetime(df_sorted['timestamp'])
        df_sorted = df_sorted.sort_values('timestamp')
        
        numeric_cols = df_sorted.select_dtypes(include=[np.number]).columns.tolist()
        df_numeric = df_sorted[['timestamp'] + numeric_cols].set_index('timestamp')
        
        try:
            # Attempt daily resampling
            df_resampled = df_numeric.resample('D').mean()
            # If resampling collapses data below lag threshold, use raw data
            if len(df_resampled.dropna(how='all')) < 2:
                model_logger.info("Resampling resulted in too few rows. Using original article sequence.")
                df_resampled = df_numeric.copy()
            
            # Fill missing history values instead of dropping them
            df_resampled = df_resampled.fillna(method='ffill').fillna(0)
        except Exception as e:
            model_logger.warning(f"Resampling failed: {e}. Using raw data.")
            df_resampled = df_numeric.fillna(0)
        
        lagged_data = pd.DataFrame(index=df_resampled.index)
        feature_cols = [col for col in df_resampled.columns 
                       if col not in ['weighted_stress_score', 'stress_score', 'target']]
        
        for col in feature_cols:
            lagged_data[f'{col}_current'] = df_resampled[col]
            for lag in range(1, self.lag_window + 1):
                # Shift creates NaNs, which we will fill later
                lagged_data[f'{col}_lag_{lag}'] = df_resampled[col].shift(lag)
            
            lagged_data[f'{col}_rolling_mean'] = df_resampled[col].rolling(
                window=self.lag_window, min_periods=1).mean()

        # Target mapping
        if 'weighted_stress_score' in df_resampled.columns:
            lagged_data['target'] = df_resampled['weighted_stress_score']
        elif 'stress_score' in df_resampled.columns:
            lagged_data['target'] = df_resampled['stress_score']
        else:
            lagged_data['target'] = 0
        
        # FIX: Fill NaNs from lags with 0 to maintain row count (e.g., 132 rows)
        return lagged_data.fillna(0).reset_index()
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
        lagged_df = self.create_lagged_features(df)
        if lagged_df.empty:
            return np.array([]), np.array([]), []
        
        X_cols = [col for col in lagged_df.columns if col not in ['target', 'timestamp', 'index']]
        X = lagged_df[X_cols].values
        y = lagged_df['target'].values
        return X, y, X_cols
    
    def train(self, df: pd.DataFrame) -> Dict:
        model_logger.info("Training time-lagged regression model")
        X, y, X_cols = self.prepare_data(df)
        self.feature_columns = X_cols
        
        if len(X) < 2:
            return {'error': 'Insufficient data', 'n_samples': len(X)}
        
        # Training
        self.model.fit(X, y)
        preds = self.model.predict(X)
        
        # Prepare final dataframe for the dashboard
        results_df = self.create_lagged_features(df)
        # Use a standard column name like 'predicted_stress'
        results_df['predicted_stress'] = preds
        
        results = {
            'mse': mean_squared_error(y, preds),
            'r2': r2_score(y, preds),
            'n_samples': len(X),
            'predictions': results_df # Ensure predictions are sent back to main pipeline
        }
        
        model_logger.info(f"Trained: MSE={results['mse']:.4f}, Samples={results['n_samples']}")
        return results
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X, _, _ = self.prepare_data(df)
        results = self.create_lagged_features(df)
        
        if len(X) == 0:
            results['predicted_stress'] = 0
            return results
        
        results['predicted_stress'] = self.model.predict(X)
        return results
    
    def save(self, filepath: Path):
        joblib.dump({'model': self.model, 'features': self.feature_columns, 'window': self.lag_window}, filepath)

    def load(self, filepath: Path):
        data = joblib.load(filepath)
        self.model, self.feature_columns, self.lag_window = data['model'], data['features'], data['window']
