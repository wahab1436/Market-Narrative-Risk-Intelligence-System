"""
KNN model for historical similarity analysis.
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import joblib
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from datetime import datetime, timedelta

from src.utils.logger import model_logger
from src.utils.config_loader import config_loader


class KNNModel:
    """
    K-Nearest Neighbors model for finding similar historical periods.
    """
    
    def __init__(self):
        """Initialize KNN model."""
        self.config = config_loader.get_config("config")
        self.model_config = self.config.get("models", {}).get("knn", {})
        self.model = NearestNeighbors(**self.model_config)
        self.scaler = StandardScaler()
        self.feature_columns = None
        model_logger.info("KNNModel initialized")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for similarity search.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with aggregated features
        """
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Aggregate by day
        df_agg = df.copy()
        df_agg['date'] = df_agg['timestamp'].dt.date
        
        # Group by date and aggregate
        daily_features = df_agg.groupby('date').agg({
            'sentiment_polarity': 'mean',
            'keyword_stress_score': 'mean',
            'daily_article_count': 'first',
            'rolling_7d_mean': 'first',
            'rolling_7d_volatility': 'first',
            'weighted_stress_score': 'mean',
            'market_breadth': 'mean'
        }).reset_index()
        
        return daily_features
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare features for KNN training.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Tuple of (features DataFrame, feature array)
        """
        # Prepare aggregated features
        daily_features = self.prepare_features(df)
        
        # Select numeric columns for similarity
        numeric_cols = daily_features.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_columns = numeric_cols
        
        # Extract feature matrix
        X = daily_features[numeric_cols].fillna(0).values
        
        return daily_features, X
    
    def train(self, df: pd.DataFrame) -> Dict:
        """
        Train KNN model on historical data.
        
        Args:
            df: Training DataFrame
        
        Returns:
            Dictionary with training results
        """
        model_logger.info("Training KNN model for historical similarity")
        
        # Prepare data
        daily_features, X = self.prepare_data(df)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit KNN model
        self.model.fit(X_scaled)
        
        # Find nearest neighbors for each point (excluding itself)
        distances, indices = self.model.kneighbors(X_scaled, n_neighbors=min(6, len(X)))
        
        # Calculate similarity metrics
        avg_similarity = np.mean(distances[:, 1:])  # Exclude self
        
        results = {
            'n_samples': len(X),
            'avg_similarity': avg_similarity,
            'feature_columns': self.feature_columns
        }
        
        model_logger.info(f"KNN trained on {len(X)} samples, avg similarity={avg_similarity:.4f}")
        return results
    
    def find_similar_periods(self, df: pd.DataFrame, target_date: datetime = None,
                           n_neighbors: int = 5) -> pd.DataFrame:
        """
        Find historically similar periods.
        
        Args:
            df: Input DataFrame
            target_date: Target date to find similar periods for
            n_neighbors: Number of similar periods to find
        
        Returns:
            DataFrame with similar periods and similarity scores
        """
        if self.model is None or self.feature_columns is None:
            raise ValueError("Model must be trained before similarity search")
        
        # Prepare data
        daily_features, X = self.prepare_data(df)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        if target_date is None:
            # Use the most recent period
            target_idx = -1
            target_date = daily_features.iloc[-1]['date']
        else:
            # Find index for target date
            target_date = pd.Timestamp(target_date).date()
            target_idx = daily_features[daily_features['date'] == target_date].index
        
        if len(target_idx) == 0:
            model_logger.warning(f"Target date {target_date} not found in data")
            return pd.DataFrame()
        
        # Find nearest neighbors
        target_features = X_scaled[target_idx].reshape(1, -1)
        distances, indices = self.model.kneighbors(
            target_features,
            n_neighbors=min(n_neighbors + 1, len(X))
        )
        
        # Prepare results
        similar_periods = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if i == 0 and idx == target_idx[0]:
                continue  # Skip self
            
            period_data = daily_features.iloc[idx].to_dict()
            period_data['similarity_score'] = 1 / (1 + dist)  # Convert distance to similarity
            period_data['days_apart'] = (pd.Timestamp(target_date) - pd.Timestamp(period_data['date'])).days
            similar_periods.append(period_data)
        
        results_df = pd.DataFrame(similar_periods)
        
        # Add feature contributions to similarity
        if len(results_df) > 0:
            target_feature_values = X[target_idx].flatten()
            
            for period_idx, row in results_df.iterrows():
                period_feature_values = X[daily_features['date'] == row['date']].flatten()
                feature_diffs = np.abs(target_feature_values - period_feature_values)
                
                # Normalize differences
                feature_contributions = feature_diffs / np.sum(feature_diffs)
                
                # Add top contributing features
                top_features_idx = np.argsort(feature_contributions)[-3:][::-1]
                top_features = []
                for feat_idx in top_features_idx:
                    if feat_idx < len(self.feature_columns):
                        top_features.append({
                            'feature': self.feature_columns[feat_idx],
                            'contribution': float(feature_contributions[feat_idx])
                        })
                
                results_df.at[period_idx, 'top_similarity_features'] = str(top_features)
        
        return results_df
    
    def save(self, filepath: Path):
        """
        Save model to disk.
        
        Args:
            filepath: Path to save model
        """
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
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
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        model_logger.info(f"Model loaded from {filepath}")
