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
        """
        self.config = config_loader.get_config("config")
        self.lag_window = lag_window
        self.model = Ridge(alpha=1.0)
        self.feature_columns = None
        model_logger.info(f"TimeLaggedRegressionModel initialized (lag_window={lag_window})")
    
    def create_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-lagged features from the dataset with fallback for small datasets.
        """
        if 'timestamp' not in df.columns:
            model_logger.warning("No timestamp column found, using index")
            df = df.copy()
            df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='D')
        
        df_sorted = df.copy()
        df_sorted['timestamp'] = pd.to_datetime(df_sorted['timestamp'])
        df_sorted = df_sorted.sort_values('timestamp')
        df_sorted.set_index('timestamp', inplace=True)
        
        numeric_cols = df_sorted.select_dtypes(include=[np.number]).columns.tolist()
        df_numeric = df_sorted[numeric_cols]
        
        # Attempt daily resampling
        try:
            df_resampled = df_numeric.resample('D').mean()
            # FIX: If daily resampling collapses data to too few rows, don't resample
            if len(df_resampled.dropna(how='all')) < self.lag_window + 1:
                model_logger.info("Daily resampling resulted in too few rows. Using raw article sequence.")
                df_resampled = df_numeric.copy()
            
            df_resampled = df_resampled.fillna(method='ffill').fillna(0)
        except Exception as e:
            model_logger.warning(f"Resampling failed: {e}. Using original data.")
            df_resampled = df_numeric.fillna(0)
        
        lagged_data = pd.DataFrame(index=df_resampled.index)
        feature_cols = [col for col in df_resampled.columns 
                       if col not in ['weighted_stress_score', 'stress_score']]
        
        if len(feature_cols) == 0:
            model_logger.warning("No numeric features available for lagging")
            return pd.DataFrame()

        # Add current and lagged values
        for col in feature_cols:
            lagged_data[f'{col}_current'] = df_resampled[col]
            for lag in range(1, self.lag_window + 1):
                lagged_data[f'{col}_lag_{lag}'] = df_resampled[col].shift(lag)
            
            # Rolling stats
            lagged_data[f'{col}_rolling_mean'] = df_resampled[col].rolling(
                window=self.lag_window, min_periods=1).mean()
        
        # Target assignment
        if 'weighted_stress_score' in df_resampled.columns:
            lagged_data['target'] = df_resampled['weighted_stress_score']
        elif 'stress_score' in df_resampled.columns:
            lagged_data['target'] = df_resampled['stress_score']
        else:
            lagged_data['target'] = 0
        
        # FIX: Instead of dropna(), use fillna to allow training on partial history
        # This prevents the "0 rows" error when history is missing
        lagged_data = lagged_data.fillna(0)
        
        return lagged_data.reset_index()
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target for training.
        """
        lagged_df = self.create_lagged_features(df)
        
        if lagged_df.empty:
            model_logger.error("No data available after creating lagged features")
            return np.array([]), np.array([])
        
        if 'target' in lagged_df.columns:
            y = lagged_df['target'].values
            X_cols = [col for col in lagged_df.columns 
                     if col not in ['target', 'timestamp', 'index']]
            X = lagged_df[X_cols].fillna(0).values
            self.feature_columns = X_cols
            return X, y
        
        return np.array([]), np.array([])
    
    def train(self, df: pd.DataFrame) -> Dict:
        """
        Train time-lagged regression model with safety checks for small data.
        """
        model_logger.info("Training time-lagged regression model")
        X, y = self.prepare_data(df)
        
        # FIX: Reduced sample requirement to 2 to allow basic training
        if len(X) < 2:
            model_logger.error(f"Insufficient samples for training: {len(X)}")
            return {'error': 'Insufficient data', 'mse': 0, 'r2': 0, 'n_samples': len(X)}
        
        # Time series split safety check
        n_splits = min(3, len(X) - 1)
        mse_scores, r2_scores = [], []
        
        if n_splits > 1:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            for train_idx, val_idx in tscv.split(X):
                self.model.fit(X[train_idx], y[train_idx])
                preds = self.model.predict(X[val_idx])
                mse_scores.append(mean_squared_error(y[val_idx], preds))
                r2_scores.append(r2_score(y[val_idx], preds))
        
        # Final Fit
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        
        results = {
            'mse': mean_squared_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'n_samples': len(X),
            'n_features': X.shape[1]
        }
        
        model_logger.info(f"Trained: MSE={results['mse']:.4f}, Samples={results['n_samples']}")
        return results

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X, _ = self.prepare_data(df)
        if len(X) == 0:
            results = df.copy()
            results['predicted_stress'] = 0
            return results
        
        predictions = self.model.predict(X)
        lagged_df = self.create_lagged_features(df)
        lagged_df['predicted_stress'] = predictions
        return lagged_df

    def save(self, filepath: Path):
        joblib.dump({'model': self.model, 'feature_columns': self.feature_columns, 'lag_window': self.lag_window}, filepath)

    def load(self, filepath: Path):
        data = joblib.load(filepath)
        self.model, self.feature_columns, self.lag_window = data['model'], data['feature_columns'], data['lag_window']
