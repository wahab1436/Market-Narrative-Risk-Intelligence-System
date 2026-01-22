"""
Time-Lagged Regression Model - FIXED VERSION
Captures temporal dependencies with proper data handling for small datasets.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, List, Optional
import joblib
from pathlib import Path

from src.utils.logger import model_logger


class TimeLaggedRegressionModel:
    """
    Time-lagged regression model with improved small dataset handling.
    """
    
    def __init__(self, lag_window: int = 3, min_samples: int = 10):
        """
        Initialize time-lagged regression model.
        
        Args:
            lag_window: Number of time steps to look back (reduced from 7 to 3)
            min_samples: Minimum samples required for training
        """
        self.lag_window = lag_window
        self.min_samples = min_samples
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        
        model_logger.info(f"TimeLaggedRegressionModel initialized (lag_window={lag_window})")
    
    def _create_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lagged features from time series data.
        
        Args:
            df: Input dataframe with timestamp
            
        Returns:
            DataFrame with lagged features
        """
        try:
            # Sort by timestamp
            df_sorted = df.sort_values('timestamp').copy()
            
            # Reset index to ensure sequential indexing
            df_sorted = df_sorted.reset_index(drop=True)
            
            # Select numeric columns for lagging (exclude timestamp)
            numeric_cols = df_sorted.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove any prediction columns that might exist
            numeric_cols = [col for col in numeric_cols if not any(
                pred in col for pred in ['prediction', 'regime', 'anomaly', 'prob_']
            )]
            
            # Limit to most important features if too many
            if len(numeric_cols) > 10:
                # Use variance to select top features
                variances = df_sorted[numeric_cols].var().sort_values(ascending=False)
                numeric_cols = variances.head(10).index.tolist()
            
            # Create lagged features
            lagged_df = df_sorted[['timestamp']].copy()
            
            # Add current values
            for col in numeric_cols:
                lagged_df[f'{col}_current'] = df_sorted[col].values
            
            # Add lagged values (reduce lags for small datasets)
            actual_lags = min(self.lag_window, len(df_sorted) // 4)  # Use at most 1/4 of data
            actual_lags = max(1, actual_lags)  # At least 1 lag
            
            for lag in range(1, actual_lags + 1):
                for col in numeric_cols:
                    lagged_df[f'{col}_lag{lag}'] = df_sorted[col].shift(lag)
            
            # Add rolling statistics (with smaller windows for small datasets)
            roll_window = min(3, len(df_sorted) // 3)
            roll_window = max(2, roll_window)
            
            for col in numeric_cols:
                if col in df_sorted.columns:
                    # Rolling mean
                    lagged_df[f'{col}_roll_mean'] = df_sorted[col].rolling(
                        window=roll_window, min_periods=1
                    ).mean()
                    
                    # Rolling std
                    lagged_df[f'{col}_roll_std'] = df_sorted[col].rolling(
                        window=roll_window, min_periods=1
                    ).std().fillna(0)
            
            # Drop rows with NaN values
            initial_rows = len(lagged_df)
            lagged_df = lagged_df.dropna()
            
            model_logger.info(
                f"Created lagged features: {initial_rows} -> {len(lagged_df)} rows, "
                f"{len(lagged_df.columns)} features (using {actual_lags} lags, {roll_window} roll window)"
            )
            
            return lagged_df
            
        except Exception as e:
            model_logger.error(f"Error creating lagged features: {e}", exc_info=True)
            return pd.DataFrame()
    
    def train(self, df: pd.DataFrame) -> Dict:
        """
        Train time-lagged regression model.
        
        Args:
            df: Training dataframe
            
        Returns:
            Dictionary with training metrics
        """
        try:
            model_logger.info("Training time-lagged regression model")
            
            # Check if we have enough data
            if len(df) < self.min_samples:
                model_logger.warning(
                    f"Insufficient data for training: {len(df)} < {self.min_samples}. "
                    "Skipping time-lagged model."
                )
                return {
                    'status': 'skipped',
                    'reason': 'insufficient_data',
                    'samples': len(df)
                }
            
            # Create lagged features
            lagged_df = self._create_lagged_features(df)
            
            if lagged_df.empty or len(lagged_df) < self.min_samples:
                model_logger.warning(
                    f"After creating lags, insufficient samples: {len(lagged_df)}"
                )
                return {
                    'status': 'skipped',
                    'reason': 'insufficient_lagged_data',
                    'samples': len(lagged_df)
                }
            
            # Prepare features and target
            feature_cols = [col for col in lagged_df.columns if col != 'timestamp']
            
            # Target is the current weighted_stress_score
            target_col = None
            for col in feature_cols:
                if 'weighted_stress_score_current' in col:
                    target_col = col
                    break
            
            if target_col is None:
                model_logger.warning("No target column found (weighted_stress_score)")
                return {
                    'status': 'skipped',
                    'reason': 'no_target_column'
                }
            
            # Remove target from features
            feature_cols = [col for col in feature_cols if col != target_col]
            
            if len(feature_cols) == 0:
                model_logger.warning("No feature columns available")
                return {
                    'status': 'skipped',
                    'reason': 'no_features'
                }
            
            self.feature_columns = feature_cols
            
            # Extract features and target
            X = lagged_df[feature_cols].values
            y = lagged_df[target_col].values
            
            # Check for invalid values
            if np.any(np.isnan(X)) or np.any(np.isnan(y)):
                model_logger.warning("NaN values detected in features or target")
                return {
                    'status': 'skipped',
                    'reason': 'nan_values'
                }
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            
            # Make predictions
            y_pred = self.model.predict(X_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            self.is_trained = True
            
            model_logger.info(
                f"Trained successfully: MSE={mse:.4f}, R²={r2:.4f}, "
                f"Features={len(feature_cols)}, Samples={len(X)}"
            )
            
            return {
                'status': 'success',
                'mse': mse,
                'r2': r2,
                'samples': len(X),
                'features': len(feature_cols),
                'lags_used': self.lag_window
            }
            
        except Exception as e:
            model_logger.error(f"Training failed: {e}", exc_info=True)
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions using time-lagged model.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with predictions
        """
        try:
            if not self.is_trained:
                model_logger.warning("Model not trained, cannot make predictions")
                return df.copy()
            
            # Create lagged features
            lagged_df = self._create_lagged_features(df)
            
            if lagged_df.empty:
                model_logger.warning("Could not create lagged features for prediction")
                return df.copy()
            
            # Check if we have all required features
            missing_features = set(self.feature_columns) - set(lagged_df.columns)
            if missing_features:
                model_logger.warning(f"Missing features for prediction: {missing_features}")
                return df.copy()
            
            # Extract features
            X = lagged_df[self.feature_columns].values
            
            # Check for NaN values
            if np.any(np.isnan(X)):
                model_logger.warning("NaN values in prediction features")
                # Fill NaN with column means
                col_means = np.nanmean(X, axis=0)
                for i in range(X.shape[1]):
                    X[np.isnan(X[:, i]), i] = col_means[i]
            
            # Scale and predict
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            
            # Create results dataframe
            results = df.copy()
            
            # Map predictions back to original dataframe
            # Create a mapping based on timestamps
            timestamp_to_pred = dict(zip(lagged_df['timestamp'], predictions))
            
            results['time_lagged_prediction'] = results['timestamp'].map(timestamp_to_pred)
            
            # Fill any missing predictions with the mean
            if results['time_lagged_prediction'].isna().any():
                mean_pred = results['time_lagged_prediction'].mean()
                results['time_lagged_prediction'].fillna(mean_pred, inplace=True)
            
            model_logger.info(f"Generated {len(predictions)} predictions")
            
            return results
            
        except Exception as e:
            model_logger.error(f"Prediction failed: {e}", exc_info=True)
            return df.copy()
    
    def save(self, filepath: Path):
        """Save model to disk."""
        try:
            if not self.is_trained:
                model_logger.warning("Model not trained, skipping save")
                return
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'lag_window': self.lag_window,
                'is_trained': self.is_trained
            }
            
            joblib.dump(model_data, filepath)
            model_logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            model_logger.error(f"Failed to save model: {e}", exc_info=True)
    
    def load(self, filepath: Path):
        """Load model from disk."""
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.lag_window = model_data['lag_window']
            self.is_trained = model_data['is_trained']
            
            model_logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            model_logger.error(f"Failed to load model: {e}", exc_info=True)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance based on coefficients.
        
        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained:
            return pd.DataFrame()
        
        try:
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'coefficient': self.model.coef_,
                'abs_coefficient': np.abs(self.model.coef_)
            })
            
            importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
            
            return importance_df
            
        except Exception as e:
            model_logger.error(f"Failed to get feature importance: {e}")
            return pd.DataFrame()


# Test function
if __name__ == "__main__":
    # Create sample data for testing
    dates = pd.date_range(start='2024-01-01', periods=50, freq='H')
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'weighted_stress_score': np.random.randn(50).cumsum() + 10,
        'sentiment_polarity': np.random.randn(50) * 0.5,
        'keyword_stress_score': np.random.exponential(0.3, 50),
        'daily_article_count': np.random.poisson(50, 50)
    })
    
    # Test model
    model = TimeLaggedRegressionModel(lag_window=3)
    
    print("\n=== Testing Time-Lagged Regression ===")
    print(f"Sample data: {len(sample_data)} rows")
    
    # Train
    results = model.train(sample_data)
    print(f"\nTraining results: {results}")
    
    # Predict
    if results.get('status') == 'success':
        predictions = model.predict(sample_data)
        print(f"\nPredictions shape: {predictions.shape}")
        print(f"Prediction columns: {[col for col in predictions.columns if 'prediction' in col]}")
        
        # Feature importance
        importance = model.get_feature_importance()
        print("\nTop 5 Important Features:")
        print(importance.head())
    else:
        print("\nModel training was skipped or failed")
