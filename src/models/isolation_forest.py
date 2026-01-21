"""
Isolation Forest model for anomaly detection.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
from pathlib import Path
from typing import Tuple, Dict, Optional

from src.utils.logger import model_logger
from src.utils.config_loader import config_loader


class IsolationForestModel:
    """
    Isolation Forest model for detecting anomalous market periods.
    """
    
    def __init__(self):
        """Initialize Isolation Forest model."""
        self.config = config_loader.get_config("config")
        self.model_config = self.config.get("models", {}).get("isolation_forest", {})
        self.model = IsolationForest(**self.model_config)
        self.scaler = StandardScaler()
        self.feature_columns = None
        model_logger.info("IsolationForestModel initialized")
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare features for anomaly detection.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Tuple of (features DataFrame, feature array)
        """
        # Select numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove irrelevant columns
        exclude_cols = ['sentiment_polarity', 'vader_compound']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        self.feature_columns = feature_cols
        
        X = df[feature_cols].fillna(0).values
        
        return df, X
    
    def create_ground_truth(self, df: pd.DataFrame, X: np.ndarray) -> np.ndarray:
        """
        Create synthetic ground truth for evaluation.
        
        Args:
            df: Input DataFrame
            X: Feature array
        
        Returns:
            Array with ground truth labels
        """
        # Use extreme stress scores as proxy for anomalies
        stress_scores = df.get('weighted_stress_score', 0)
        
        if len(stress_scores) > 0:
            # Define anomalies as periods with extreme stress scores
            threshold_high = stress_scores.quantile(0.95)
            threshold_low = stress_scores.quantile(0.05)
            
            y_true = np.zeros(len(X))
            y_true[(stress_scores > threshold_high) | (stress_scores < threshold_low)] = 1
        else:
            y_true = np.zeros(len(X))
        
        return y_true
    
    def train(self, df: pd.DataFrame) -> Dict:
        """
        Train Isolation Forest model.
        
        Args:
            df: Training DataFrame
        
        Returns:
            Dictionary with training results
        """
        model_logger.info("Training Isolation Forest model")
        
        # Prepare data
        df_features, X = self.prepare_data(df)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create synthetic ground truth for evaluation
        y_true = self.create_ground_truth(df_features, X_scaled)
        
        # Train model
        self.model.fit(X_scaled)
        
        # Evaluate
        y_pred = self.model.predict(X_scaled)
        y_pred_binary = np.where(y_pred == -1, 1, 0)  # Convert to binary
        
        # Calculate metrics if we have anomalies
        if np.sum(y_true) > 0:
            precision = precision_score(y_true, y_pred_binary, zero_division=0)
            recall = recall_score(y_true, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        else:
            precision = recall = f1 = 0
        
        # Get anomaly scores
        anomaly_scores = self.model.decision_function(X_scaled)
        
        # Feature importance for anomalies
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': np.std(X_scaled, axis=0) * np.mean(np.abs(anomaly_scores))
        }).sort_values('importance', ascending=False)
        
        results = {
            'n_anomalies_detected': np.sum(y_pred_binary),
            'anomaly_rate': np.mean(y_pred_binary),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'feature_importance': feature_importance,
            'model_params': self.model.get_params()
        }
        
        model_logger.info(f"Isolation Forest trained: {results['n_anomalies_detected']} anomalies detected")
        return results
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in new data.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with anomaly flags and scores
        """
        if self.model is None or self.feature_columns is None:
            raise ValueError("Model must be trained before detection")
        
        # Prepare features
        df_features, X = self.prepare_data(df)
        X_scaled = self.scaler.transform(X)
        
        # Detect anomalies
        predictions = self.model.predict(X_scaled)
        anomaly_scores = self.model.decision_function(X_scaled)
        
        # Create results DataFrame
        results = df.copy()
        results['is_anomaly'] = np.where(predictions == -1, 1, 0)
        results['anomaly_score'] = anomaly_scores
        
        # Add feature contributions to anomaly score
        feature_contributions = []
        for i, row in df_features.iterrows():
            contributions = {}
            for j, feature in enumerate(self.feature_columns):
                feature_value = row[feature] if feature in row else 0
                scaled_value = X_scaled[i, j] if j < X_scaled.shape[1] else 0
                contributions[feature] = {
                    'original_value': float(feature_value),
                    'scaled_value': float(scaled_value),
                    'contribution': float(np.abs(scaled_value))
                }
            
            # Sort by contribution
            sorted_contributions = sorted(
                contributions.items(),
                key=lambda x: x[1]['contribution'],
                reverse=True
            )[:3]  # Top 3 features
            
            feature_contributions.append(str(dict(sorted_contributions)))
        
        results['top_anomaly_features'] = feature_contributions
        
        return results
    
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
