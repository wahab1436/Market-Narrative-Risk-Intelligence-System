"""
XGBoost model for multi-class risk regime classification.
FIXED: Handles untrained model gracefully in predict()
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from src.utils.logger import model_logger
from src.utils.config_loader import config_loader


class XGBoostModel:
    """
    XGBoost model for risk regime classification (low / medium / high)
    """

    CLASSES = np.array(["low", "medium", "high"])

    def __init__(self):
        self.config = config_loader.get_config("config")
        self.model_config = self.config.get("models", {}).get("xgboost", {})
        self.model = None
        self.feature_columns = None

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.CLASSES)

        model_logger.info("XGBoostModel initialized")

    def create_labels(self, df: pd.DataFrame) -> pd.Series:
        """Create risk regime labels from stress scores."""
        stress = df.get("weighted_stress_score")

        if stress is None or stress.isna().all():
            labels = pd.Series("medium", index=df.index)
            if len(labels) >= 3:
                labels.iloc[: len(labels) // 3] = "low"
                labels.iloc[-len(labels) // 3 :] = "high"
            return labels

        q_low = stress.quantile(0.33)
        q_high = stress.quantile(0.67)

        labels = pd.Series("medium", index=df.index)
        labels[stress < q_low] = "low"
        labels[stress > q_high] = "high"
        return labels

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare features and labels for training."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        exclude_cols = {
            "weighted_stress_score",
            "sentiment_polarity",
            "vader_compound",
        }

        self.feature_columns = [c for c in numeric_cols if c not in exclude_cols]

        if not self.feature_columns:
            raise ValueError("No numeric features available for XGBoost")

        X = df[self.feature_columns].fillna(0)
        y = self.create_labels(df)

        y_encoded = self.label_encoder.transform(y)
        return X, y_encoded

    def train(self, df: pd.DataFrame) -> Dict:
        """Train XGBoost model with safety checks."""
        model_logger.info("Training XGBoost model")

        X, y = self.prepare_data(df)
        
        # Check if we have enough data for train-test split
        min_samples_for_split = 10
        unique_classes = np.unique(y)
        
        # Check if we have multiple classes
        if len(unique_classes) == 1:
            model_logger.warning(f"Only one class ({unique_classes[0]}) present in data. Creating synthetic samples for other classes.")
            
            # Create synthetic samples for missing classes
            missing_classes = [c for c in range(3) if c not in unique_classes]
            
            # Add one sample for each missing class (duplicate existing data with different label)
            X_synthetic = []
            y_synthetic = []
            
            for missing_class in missing_classes:
                # Use first sample as template, add small noise
                sample = X.iloc[0].copy()
                # Add small random noise to make it slightly different
                sample = sample + np.random.randn(len(sample)) * 0.01
                X_synthetic.append(sample)
                y_synthetic.append(missing_class)
            
            # Combine original and synthetic data
            X_combined = pd.concat([X, pd.DataFrame(X_synthetic, columns=X.columns)], ignore_index=True)
            y_combined = np.concatenate([y, y_synthetic])
            
            model_logger.info(f"Added {len(missing_classes)} synthetic samples. Total: {len(X_combined)}")
            
            # Now train with all classes present
            model_params = self.model_config.copy()
            model_params["objective"] = "multi:softprob"
            model_params["num_class"] = 3
            
            self.model = xgb.XGBClassifier(**model_params)
            self.model.fit(X_combined, y_combined)
            
            # Predict on original data only
            y_pred = self.model.predict(X)
            accuracy = accuracy_score(y, y_pred)
            
            model_logger.info(f"XGBoost trained with synthetic data: accuracy={accuracy:.4f}")
            
            return {
                "accuracy": accuracy,
                "classification_report": {},
                "model_params": self.model.get_params(),
                "training_samples": len(X),
                "note": "Trained with synthetic samples for missing classes"
            }
        
        if len(X) < min_samples_for_split:
            model_logger.warning(f"Only {len(X)} samples available. Training without test split.")
            
            # Train on all data without splitting
            model_params = self.model_config.copy()
            model_params["objective"] = "multi:softprob"
            model_params["num_class"] = 3
            
            self.model = xgb.XGBClassifier(**model_params)
            
            try:
                self.model.fit(X, y)
            except ValueError as e:
                if "Invalid classes" in str(e):
                    model_logger.error(f"XGBoost error: {e}")
                    model_logger.warning("Attempting to fix by ensuring all classes are present...")
                    
                    # Can't train with missing classes in XGBoost multi-class
                    self.model = None  # Reset to None
                    return {
                        "accuracy": 0.0,
                        "classification_report": {},
                        "model_params": {},
                        "training_samples": len(X),
                        "note": f"Cannot train: Only {len(unique_classes)} classes present, need 3 for multi-class"
                    }
                else:
                    raise
            
            # Predict on training data
            y_pred = self.model.predict(X)
            accuracy = accuracy_score(y, y_pred)
            
            report = classification_report(
                self.label_encoder.inverse_transform(y),
                self.label_encoder.inverse_transform(y_pred),
                output_dict=True,
                zero_division=0,
            )
            
            model_logger.info(f"XGBoost trained (small dataset): accuracy={accuracy:.4f}, samples={len(X)}")
            
            return {
                "accuracy": accuracy,
                "classification_report": report,
                "model_params": self.model.get_params(),
                "training_samples": len(X),
                "note": "Trained without test split due to small dataset"
            }
        
        # Check if stratification is possible
        min_class_count = min(np.bincount(y))
        
        if min_class_count < 2:
            model_logger.warning(f"Insufficient samples per class for stratification. Training without stratify.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            # Normal stratified split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

        # Train model with proper parameters
        model_params = self.model_config.copy()
        model_params["objective"] = "multi:softprob"
        model_params["num_class"] = 3

        self.model = xgb.XGBClassifier(**model_params)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        report = classification_report(
            self.label_encoder.inverse_transform(y_test),
            self.label_encoder.inverse_transform(y_pred),
            output_dict=True,
            zero_division=0,
        )

        model_logger.info(f"XGBoost trained successfully (accuracy={accuracy:.4f})")

        return {
            "accuracy": accuracy,
            "classification_report": report,
            "model_params": self.model.get_params(),
        }

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict risk regimes.
        FIXED: Handles untrained model gracefully.
        """
        # Check if model is trained
        if self.model is None:
            model_logger.warning("XGBoost model is not trained, returning dummy predictions")
            results = df.copy()
            results["xgboost_risk_regime"] = "medium"
            results["prob_low"] = 0.33
            results["prob_medium"] = 0.34
            results["prob_high"] = 0.33
            return results
        
        if self.feature_columns is None:
            model_logger.error("Feature columns not set")
            results = df.copy()
            results["xgboost_risk_regime"] = "medium"
            results["prob_low"] = 0.33
            results["prob_medium"] = 0.34
            results["prob_high"] = 0.33
            return results

        # Model is trained, proceed with prediction
        X = df[self.feature_columns].fillna(0)
        n = len(X)

        preds_encoded = self.model.predict(X)
        preds = self.label_encoder.inverse_transform(preds_encoded)

        probs = self.model.predict_proba(X)

        # Handle probability shape
        if probs.shape == (3, n):
            probs = probs.T
        elif probs.shape != (n, 3):
            model_logger.error(f"Invalid probability shape: {probs.shape}")
            results = df.copy()
            results["xgboost_risk_regime"] = "medium"
            results["prob_low"] = 0.33
            results["prob_medium"] = 0.34
            results["prob_high"] = 0.33
            return results

        results = df.copy()
        results["xgboost_risk_regime"] = preds

        for i, cls in enumerate(self.CLASSES):
            results[f"prob_{cls}"] = probs[:, i]

        return results

    def save(self, path: Path):
        """Save model to disk."""
        joblib.dump(
            {
                "model": self.model,
                "feature_columns": self.feature_columns,
            },
            path,
        )
        model_logger.info(f"XGBoost model saved to {path}")

    def load(self, path: Path):
        """Load model from disk."""
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_columns = data["feature_columns"]
        model_logger.info(f"XGBoost model loaded from {path}")
