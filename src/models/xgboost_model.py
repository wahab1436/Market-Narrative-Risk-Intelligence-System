"""
XGBoost model for risk regime classification.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib
from pathlib import Path
from typing import Tuple, Dict, Optional

from src.utils.logger import model_logger
from src.utils.config_loader import config_loader


class XGBoostModel:
    """
    XGBoost model for multi-class risk regime classification.
    """
    
    def __init__(self):
        """Initialize XGBoost model."""
        self.config = config_loader.get_config("config")
        self.model_config = self.config.get("models", {}).get("xgboost", {})
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        model_logger.info("XGBoostModel initialized")
    
    def create_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Create risk regime labels based on stress scores.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Series with risk regime labels
        """
        # Use weighted stress score to create risk regimes
        stress_scores = df.get('weighted_stress_score', pd.Series(dtype=float))
        
        if isinstance(stress_scores, pd.DataFrame):
            stress_scores = stress_scores.iloc[:, 0]
        
        # Create quantile-based thresholds
        if len(stress_scores) > 0 and stress_scores.notna().any():
            q_low = stress_scores.quantile(0.33)
            q_high = stress_scores.quantile(0.67)
        else:
            q_low, q_high = -0.5, 0.5
        
        # Assign labels
        labels = pd.Series('medium', index=df.index)
        labels[stress_scores < q_low] = 'low'
        labels[stress_scores > q_high] = 'high'
        
        return labels
    
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
        exclude_cols = ['weighted_stress_score', 'sentiment_polarity', 'vader_compound']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if not feature_cols:
            raise ValueError("No valid features found for XGBoost model")
        
        self.feature_columns = feature_cols
        
        X = df[feature_cols].fillna(0)
        y = self.create_labels(df)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        return X, y_encoded
    
    def train(self, df: pd.DataFrame) -> Dict:
        """
        Train XGBoost model.
        
        Args:
            df: Training DataFrame
        
        Returns:
            Dictionary with training results
        """
        model_logger.info("Training XGBoost model")
        
        # Prepare data
        X, y = self.prepare_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model = xgb.XGBClassifier(**self.model_config)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Classification report
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        y_test_labels = self.label_encoder.inverse_transform(y_test)
        
        class_report = classification_report(
            y_test_labels,
            y_pred_labels,
            output_dict=True,
            zero_division=0
        )
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        results = {
            'accuracy': accuracy,
            'classification_report': class_report,
            'feature_importance': feature_importance,
            'model_params': self.model.get_params()
        }
        
        model_logger.info(f"XGBoost trained: Accuracy={accuracy:.4f}")
        return results
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on new data.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with predictions
        """
        if self.model is None or self.feature_columns is None:
            raise ValueError("Model must be trained before prediction")
        
        # Prepare features - ensure we only use rows that have the required features
        X = df[self.feature_columns].fillna(0)
        
        # Make predictions
        predictions_encoded = self.model.predict(X)
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(X)
        
        # **FIXED: Create results DataFrame**
        results = df.copy()
        results['xgboost_risk_regime'] = predictions
        
        # **FIXED: Add probability columns - simple direct assignment**
        # Get all possible classes from the label encoder
        all_classes = self.label_encoder.classes_
        
        # Add probability for each class - probabilities and results have same length
        for i, class_name in enumerate(all_classes):
            results[f'prob_{class_name}'] = probabilities[:, i]
        
        return results
    
    def save(self, filepath: Path):
        """
        Save model to disk.
        
        Args:
            filepath: Path to save model
        """
        joblib.dump({
            'model': self.model,
            'label_encoder': self.label_encoder,
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
        self.label_encoder = data['label_encoder']
        self.feature_columns = data['feature_columns']
        model_logger.info(f"Model loaded from {filepath}")
