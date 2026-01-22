"""
Time-lagged regression model for stress score prediction.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from pathlib import Path
from typing import Tuple, Dict, List, Optional
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
        self.model = Ridge(alpha=1.0)  # Use Ridge for stability with correlated lags
        self.feature_columns = None
        self.target_column = 'weighted_stress_score'
        self.scaler = None
        model_logger.info(f"TimeLaggedRegressionModel initialized (lag_window={self.lag_window})")
    
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
        
        # Ensure sorted by timestamp and set as index
        df_sorted = df.sort_values('timestamp').copy()
        df_sorted.set_index('timestamp', inplace=True)
        
        # **FIX: Select only numeric columns before resampling**
        numeric_cols = df_sorted.select_dtypes(include=[np.number]).columns.tolist()
        df_numeric = df_sorted[numeric_cols]
        
        # Resample to daily frequency if needed (now only on numeric data)
        if len(df_numeric) > 100:  # If we have enough data
            df_resampled = df_numeric.resample('D').mean()
        else:
            df_resampled = df_numeric
        
        # Select key features for lagging
        lag_candidates = [
            'weighted_stress_score',
            'keyword_stress_score', 
            'sentiment_polarity',
            'daily_article_count',
            'rolling_7d_mean'
        ]
        
        available_features = [f for f in lag_candidates if f in df_resampled.columns]
        
        if not available_features:
            # Fallback to numeric columns
            numeric_cols_list = df_resampled.select_dtypes(include=[np.number]).columns.tolist()
            available_features = numeric_cols_list[:3] if len(numeric_cols_list) >= 3 else numeric_cols_list
        
        if not available_features:
            raise ValueError("No numeric features available for time-lagged model")
        
        # Create lagged features
        lagged_data = {}
        
        for feature in available_features:
            # Create lags
            for lag in range(1, self.lag_window + 1):
                lagged_data[f'{feature}_lag_{lag}'] = df_resampled[feature].shift(lag)
            
            # Create moving averages
            lagged_data[f'{feature}_ma_3'] = df_resampled[feature].rolling(window=3, min_periods=1).mean()
            lagged_data[f'{feature}_ma_7'] = df_resampled[feature].rolling(window=7, min_periods=1).mean()
        
        # Combine all features
        lagged_df = pd.DataFrame(lagged_data)
        
        # Add time-based features
        lagged_df['day_of_week'] = lagged_df.index.dayofweek
        lagged_df['day_of_month'] = lagged_df.index.day
        lagged_df['month'] = lagged_df.index.month
        lagged_df['quarter'] = lagged_df.index.quarter
        
        # Add cyclical encoding for day of week
        lagged_df['day_of_week_sin'] = np.sin(2 * np.pi * lagged_df['day_of_week'] / 7)
        lagged_df['day_of_week_cos'] = np.cos(2 * np.pi * lagged_df['day_of_week'] / 7)
        
        # Add target variable (shifted for prediction)
        if 'weighted_stress_score' in df_resampled.columns:
            # Predict next day's stress score
            lagged_df['target'] = df_resampled['weighted_stress_score'].shift(-1)
        
        # Drop rows with NaN (from lagging)
        lagged_df = lagged_df.dropna()
        
        return lagged_df.reset_index()
    
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
        
        if 'target' not in lagged_df.columns:
            raise ValueError("Could not create target variable. Ensure 'weighted_stress_score' exists in data.")
        
        # Separate features and target
        feature_cols = [col for col in lagged_df.columns if col not in ['timestamp', 'target']]
        self.feature_columns = feature_cols
        
        X = lagged_df[feature_cols].fillna(0)
        y = lagged_df['target']
        
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
        
        if len(X) < 20:
            model_logger.warning(f"Insufficient data for time-lagged model: {len(X)} samples")
            return {
                'error': 'Insufficient data',
                'n_samples': len(X)
            }
        
        # Use time series cross-validation
        tscv = TimeSeriesSplit(n_splits=min(5, len(X) // 4))
        
        cv_scores = {
            'mse': [],
            'rmse': [],
            'mae': [],
            'r2': []
        }
        
        feature_importances = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train on fold
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            
            cv_scores['mse'].append(mean_squared_error(y_test, y_pred))
            cv_scores['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
            cv_scores['mae'].append(mean_absolute_error(y_test, y_pred))
            cv_scores['r2'].append(r2_score(y_test, y_pred))
            
            # Collect feature importances
            feature_importances.append(np.abs(self.model.coef_))
        
        # Final training on all data
        self.model.fit(X, y)
        
        # Calculate average metrics
        avg_metrics = {k: np.mean(v) for k, v in cv_scores.items()}
        
        # Calculate feature importance
        avg_feature_importance = np.mean(feature_importances, axis=0)
        
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': avg_feature_importance,
            'coefficient': self.model.coef_
        }).sort_values('importance', ascending=False)
        
        # Identify most important lags
        lag_features = feature_importance_df[feature_importance_df['feature'].str.contains('_lag_')]
        ma_features = feature_importance_df[feature_importance_df['feature'].str.contains('_ma_')]
        
        results = {
            'lag_window': self.lag_window,
            'n_samples': len(X),
            'cv_metrics': avg_metrics,
            'cv_details': cv_scores,
            'feature_importance': feature_importance_df,
            'top_lag_features': lag_features.head(5).to_dict('records'),
            'top_ma_features': ma_features.head(5).to_dict('records'),
            'model_params': self.model.get_params()
        }
        
        model_logger.info(
            f"Time-lagged regression trained: "
            f"RMSE={avg_metrics['rmse']:.4f}, R2={avg_metrics['r2']:.4f}, "
            f"Samples={len(X)}"
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
        
        # Create lagged features
        lagged_df = self.create_lagged_features(df)
        
        if lagged_df.empty:
            model_logger.warning("No data for prediction after lagging")
            return df.copy()
        
        # Prepare features
        X = lagged_df[self.feature_columns].fillna(0)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Create results DataFrame
        results = df.copy()
        
        # Initialize prediction columns
        results['time_lagged_prediction'] = np.nan
        results['time_lagged_residual'] = np.nan
        
        # Match predictions to original timestamps
        for idx, row in lagged_df.iterrows():
            timestamp = row['timestamp']
            
            # Find matching row in original data (next day prediction)
            if 'target' in row:
                # This is prediction for next timestamp
                target_date = timestamp + timedelta(days=1)
                mask = results['timestamp'].dt.date == target_date.date()
                
                if mask.any():
                    results.loc[mask, 'time_lagged_prediction'] = predictions[idx]
                    
                    # Calculate residual if actual value exists
                    if self.target_column in results.columns:
                        actual = results.loc[mask, self.target_column].iloc[0]
                        results.loc[mask, 'time_lagged_residual'] = actual - predictions[idx]
        
        # Forward fill predictions for visualization
        results['time_lagged_prediction'] = results['time_lagged_prediction'].ffill()
        
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
        latest_data = df.sort_values('timestamp').tail(self.lag_window * 2).copy()
        
        if latest_data.empty:
            model_logger.warning("No data for forecasting")
            return pd.DataFrame()
        
        # Create initial lagged features
        current_lagged = self.create_lagged_features(latest_data)
        
        if current_lagged.empty or 'target' not in current_lagged.columns:
            model_logger.warning("Could not create lagged features for forecasting")
            return pd.DataFrame()
        
        # Create forecasts
        forecasts = []
        current_features = current_lagged.copy()
        
        for i in range(horizon):
            # Get the most recent row
            last_row = current_features.iloc[-1:][self.feature_columns].fillna(0)
            
            if last_row.empty:
                break
            
            # Make prediction
            forecast = self.model.predict(last_row)[0]
            
            # Create forecast timestamp
            last_timestamp = current_features.iloc[-1]['timestamp']
            forecast_timestamp = last_timestamp + timedelta(days=1)
            
            # Store forecast
            forecasts.append({
                'timestamp': forecast_timestamp,
                'forecast': forecast,
                'horizon': i + 1,
                'forecast_date': datetime.now().date()
            })
            
            # Create new data point for next iteration
            new_point = latest_data.iloc[-1:].copy()
            new_point['timestamp'] = forecast_timestamp
            new_point['weighted_stress_score'] = forecast
            
            # Update latest_data for next iteration
            latest_data = pd.concat([latest_data, new_point], ignore_index=True)
            
            # Recreate lagged features with new data
            current_lagged = self.create_lagged_features(latest_data)
            if not current_lagged.empty:
                current_features = current_lagged.copy()
        
        forecast_df = pd.DataFrame(forecasts)
        
        # Calculate confidence intervals
        if not forecast_df.empty:
            # Simple confidence intervals based on historical residuals
            if hasattr(self, 'last_cv_rmse'):
                std_error = self.last_cv_rmse
            else:
                std_error = forecast_df['forecast'].std()
            
            forecast_df['forecast_lower'] = forecast_df['forecast'] - 1.96 * std_error
            forecast_df['forecast_upper'] = forecast_df['forecast'] + 1.96 * std_error
        
        return forecast_df
    
    def analyze_autocorrelation(self, df: pd.DataFrame) -> Dict:
        """
        Analyze autocorrelation in the time series.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Dictionary with autocorrelation analysis
        """
        if 'timestamp' not in df.columns or self.target_column not in df.columns:
            return {}
        
        # Prepare time series
        ts_data = df.sort_values('timestamp').set_index('timestamp')[self.target_column]
        
        # Calculate autocorrelation
        max_lag = min(30, len(ts_data) - 1)
        autocorrelations = []
        
        for lag in range(1, max_lag + 1):
            if len(ts_data) > lag:
                autocorr = ts_data.autocorr(lag=lag)
                if not pd.isna(autocorr):
                    autocorrelations.append({
                        'lag': lag,
                        'autocorrelation': autocorr,
                        'significant': abs(autocorr) > 2 / np.sqrt(len(ts_data))
                    })
        
        # Find significant lags
        significant_lags = [ac['lag'] for ac in autocorrelations if ac['significant']]
        
        analysis = {
            'autocorrelations': autocorrelations[:20],  # First 20 lags
            'significant_lags': significant_lags,
            'max_autocorrelation': max([abs(ac['autocorrelation']) for ac in autocorrelations]) if autocorrelations else 0,
            'n_samples': len(ts_data)
        }
        
        return analysis
    
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


def create_time_lagged_model():
    """Factory function to create time-lagged regression model."""
    return TimeLaggedRegressionModel()
