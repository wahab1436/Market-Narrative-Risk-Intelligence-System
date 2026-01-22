"""
XGBoost model for risk regime classification.
"""

from pathlib import Path
from typing import Tuple, Dict

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.utils.config_loader import config_loader
from src.utils.logger import model_logger


class XGBoostModel:
    """
    XGBoost model for multi-class risk regime classification.
    """

    def __init__(self):
        self.config = config_loader.get_config("config")
        self.model_config = self.config.get("models", {}).get("xgboost", {})
        self.model: xgb.XGBClassifier | None = None
        self.label_encoder = LabelEncoder()
        self.feature_columns: list[str] | None = None

        model_logger.info("XGBoostModel initialized")

    # ------------------------------------------------------------------
    # LABEL CREATION
    # ------------------------------------------------------------------
    def create_labels(self, df: pd.DataFrame) -> pd.Series:
        """Create quantile-based risk regime labels."""
        stress = df.get("weighted_stress_score")

        if stress is None or stress.isna().all():
            model_logger.warning("Missing stress score — defaulting to 'medium'")
            return pd.Series("medium", index=df.index)

        q_low = stress.quantile(0.33)
        q_high = stress.quantile(0.67)

        labels = pd.Series("medium", index=df.index)
        labels.loc[stress < q_low] = "low"
        labels.loc[stress > q_high] = "high"

        return labels

    # ------------------------------------------------------------------
    # DATA PREP
    # ------------------------------------------------------------------
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        exclude = {
            "weighted_stress_score",
            "sentiment_polarity",
            "vader_compound",
        }

        self.feature_columns = [c for c in numeric_cols if c not in exclude]

        if not self.feature_columns:
            raise ValueError("No numeric features available for XGBoost")

        X = df[self.feature_columns].fillna(0)
        y = self.create_labels(df)

        y_encoded = self.label_encoder.fit_transform(y)

        return X, y_encoded

    # ------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------
    def train(self, df: pd.DataFrame) -> Dict:
        model_logger.info("Training XGBoost model")

        X, y = self.prepare_data(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        self.model = xgb.XGBClassifier(**self.model_config)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        report = classification_report(
            self.label_encoder.inverse_transform(y_test),
            self.label_encoder.inverse_transform(y_pred),
            output_dict=True,
            zero_division=0,
        )

        model_logger.info(f"XGBoost training complete — accuracy={acc:.4f}")

        return {
            "accuracy": acc,
            "classification_report": report,
            "model_params": self.model.get_params(),
            "feature_importance": pd.DataFrame(
                {
                    "feature": self.feature_columns,
                    "importance": self.model.feature_importances_,
                }
            ).sort_values("importance", ascending=False),
        }

    # ------------------------------------------------------------------
    # PREDICT (FULLY FIXED)
    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None or self.feature_columns is None:
            raise RuntimeError("Model must be trained or loaded before prediction")

        X = df[self.feature_columns].fillna(0)

        model_logger.info(f"XGBoost predict: X shape = {X.shape}")

        preds_encoded = self.model.predict(X)
        preds = self.label_encoder.inverse_transform(preds_encoded)

        # --- PROBABILITY FIX (CORE BUG) ---
        probs = self.model.predict_proba(X)
        model_logger.info(f"XGBoost raw probs shape = {probs.shape}")

        n_rows = len(X)
        n_classes = len(self.label_encoder.classes_)

        # Fix transposed / flattened outputs
        if probs.shape == (n_classes, n_rows):
            probs = probs.T

        elif probs.ndim == 1:
            probs = probs.reshape(-1, 1)

        elif probs.shape[0] != n_rows:
            probs = probs.reshape(n_rows, n_classes)

        if probs.shape != (n_rows, n_classes):
            raise RuntimeError(
                f"XGBoost probability shape invalid after fix: {probs.shape}"
            )

        model_logger.info(f"XGBoost normalized probs shape = {probs.shape}")

        # Build output
        results = df.copy()
        results["xgboost_risk_regime"] = preds

        for i, cls in enumerate(self.label_encoder.classes_):
            results[f"prob_{cls}"] = probs[:, i]

        return results

    # ------------------------------------------------------------------
    # SAVE / LOAD
    # ------------------------------------------------------------------
    def save(self, path: Path):
        joblib.dump(
            {
                "model": self.model,
                "label_encoder": self.label_encoder,
                "feature_columns": self.feature_columns,
            },
            path,
        )
        model_logger.info(f"XGBoost model saved → {path}")

    def load(self, path: Path):
        data = joblib.load(path)
        self.model = data["model"]
        self.label_encoder = data["label_encoder"]
        self.feature_columns = data["feature_columns"]

        model_logger.info(f"XGBoost model loaded ← {path}")
