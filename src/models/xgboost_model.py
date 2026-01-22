"""
XGBoost model for risk regime classification.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from src.utils.logger import model_logger
from src.utils.config_loader import config_loader


class XGBoostModel:

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
            if len(labels) > 3:
                labels.iloc[: len(labels)//3] = "low"
                labels.iloc[-len(labels)//3 :] = "high"
            return labels

        q1, q2 = stress.quantile([0.33, 0.67])
        labels = pd.Series("medium", index=df.index)
        labels[stress < q1] = "low"
        labels[stress > q2] = "high"
        return labels

    # --------------------------------------------------
    def prepare_data(self, df: pd.DataFrame):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude = {"weighted_stress_score", "sentiment_polarity", "vader_compound"}

        self.feature_columns = [c for c in numeric_cols if c not in exclude]
        X = df[self.feature_columns].fillna(0)

        y = self.create_labels(df)
        y_enc = self.label_encoder.transform(y)

        return X, y_enc

    # --------------------------------------------------
    def train(self, df: pd.DataFrame):
        X, y = self.prepare_data(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            **self.model_config
        )

        self.model.fit(X_train, y_train)

        acc = accuracy_score(y_test, self.model.predict(X_test))
        model_logger.info(f"XGBoost trained — accuracy={acc:.4f}")

        return {"accuracy": acc}

    # --------------------------------------------------
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feature_columns].fillna(0)
        n = len(X)

        preds = self.label_encoder.inverse_transform(self.model.predict(X))
        probs = self.model.predict_proba(X)

        # ✅ HARD NORMALIZATION (NO CRASH)
        if probs.shape[0] == 3 and probs.shape[1] == n * 2:
            probs = probs[:, :n].T
        elif probs.shape == (3, n):
            probs = probs.T
        elif probs.shape != (n, 3):
            raise RuntimeError(f"Bad probability shape: {probs.shape}")

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

    def load(self, path: Path):
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_columns = data["feature_columns"]
