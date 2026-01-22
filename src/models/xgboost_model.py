"""
XGBoost model for multi-class risk regime classification.
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

    # --------------------------------------------------
    def create_labels(self, df: pd.DataFrame) -> pd.Series:
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

    # --------------------------------------------------
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
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

    # --------------------------------------------------
    def train(self, df: pd.DataFrame) -> Dict:
        model_logger.info("Training XGBoost model")

        X, y = self.prepare_data(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # ---- SAFE PARAM HANDLING (NO DUPLICATES) ----
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

    # --------------------------------------------------
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None or self.feature_columns is None:
            raise RuntimeError("XGBoost model is not trained")

        X = df[self.feature_columns].fillna(0)
        n = len(X)

        preds_encoded = self.model.predict(X)
        preds = self.label_encoder.inverse_transform(preds_encoded)

        probs = self.model.predict_proba(X)

        # ---- HARD SHAPE SAFETY ----
        if probs.shape == (3, n):
            probs = probs.T
        elif probs.shape != (n, 3):
            raise RuntimeError(f"Invalid probability shape: {probs.shape}")

        results = df.copy()
        results["xgboost_risk_regime"] = preds

        for i, cls in enumerate(self.CLASSES):
            results[f"prob_{cls}"] = probs[:, i]

        return results

    # --------------------------------------------------
    def save(self, path: Path):
        joblib.dump(
            {
                "model": self.model,
                "feature_columns": self.feature_columns,
            },
            path,
        )
        model_logger.info(f"XGBoost model saved to {path}")

    def load(self, path: Path):
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_columns = data["feature_columns"]
        model_logger.info(f"XGBoost model loaded from {path}")
