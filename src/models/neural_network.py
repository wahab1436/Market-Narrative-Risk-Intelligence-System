"""
Neural network model for non-linear stress estimation.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import joblib
from pathlib import Path
from typing import Tuple, Dict, Optional

from src.utils.logger import model_logger
from src.utils.config_loader import config_loader


class NeuralNetworkModel:
    """
    Neural network model for stress score prediction.
    """
    
    def __init__(self):
        """Initialize neural network model."""
        self.config = config_loader.get_config("config")
        self.model_config = self.config.get("models", {}).get("neural_network", {})
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = 'weighted_stress_score'
        model_logger.info("NeuralNetworkModel initialized")
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target for training.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Tuple of (features, target) arrays
        """
        # Select numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target and irrelevant columns
        exclude_cols = [self.target_column, 'sentiment_polarity', 'vader_compound']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        self.feature_columns = feature_cols
        
        X = df[feature_cols].fillna(0).values
        y = df[self.target_column].fillna(0).values
        
        return X, y
    
    def build_model(self, input_dim: int) -> keras.Model:
        """
        Build neural network architecture.
        
        Args:
            input_dim: Number of input features
        
        Returns:
            Compiled Keras model
        """
        hidden_layers = self.model_config.get('hidden_layers', [64, 32, 16])
        activation = self.model_config.get('activation', 'relu')
        dropout = self.model_config.get('dropout', 0.2)
        learning_rate = self.model_config.get('learning_rate', 0.001)
        
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(input_dim,)))
        
        # Hidden layers
        for units in hidden_layers:
            model.add(layers.Dense(units, activation=activation))
            model.add(layers.Dropout(dropout))
        
        # Output layer
        model.add(layers.Dense(1, activation='linear'))
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', keras.metrics.RootMeanSquaredError()]
        )
        
        return model
    
    def train(self, df: pd.DataFrame) -> Dict:
        """
        Train neural network model.
        
        Args:
            df: Training DataFrame
        
        Returns:
            Dictionary with training results
        """
        model_logger.info("Training neural network model")
        
        # Prepare data
        X, y = self.prepare_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build model
        self.model = self.build_model(X_train_scaled.shape[1])
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Train model
        history = self.model.fit(
            X_train_scaled,
            y_train,
            validation_split=0.2,
            epochs=self.model_config.get('epochs', 100),
            batch_size=self.model_config.get('batch_size', 32),
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled, verbose=0).flatten()
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Training history
        history_dict = {
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'mae': history.history['mae'],
            'val_mae': history.history['val_mae']
        }
        
        results = {
            'mse': mse,
            'r2': r2,
            'training_history': history_dict,
            'model_summary': self.model.summary()
        }
        
        model_logger.info(f"Neural network trained: MSE={mse:.4f}, R2={r2:.4f}")
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
        
        # Prepare features
        X = df[self.feature_columns].fillna(0).values
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled, verbose=0).flatten()
        
        # Create results DataFrame
        results = df.copy()
        results['neural_network_prediction'] = predictions
        results['neural_network_residual'] = results.get(
            self.target_column, 0
        ) - predictions
        
        return results
    
    def save(self, filepath: Path):
        """
        Save model to disk.
        
        Args:
            filepath: Path to save model
        """
        # Save Keras model
        model_path = filepath.with_suffix('.keras')
        self.model.save(model_path)
        
        # Save scaler and metadata
        joblib.dump({
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }, filepath)
        
        model_logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: Path):
        """
        Load model from disk.
        
        Args:
            filepath: Path to model file
        """
        # Load Keras model
        model_path = filepath.with_suffix('.keras')
        self.model = keras.models.load_model(model_path)
        
        # Load scaler and metadata
        data = joblib.load(filepath)
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        self.target_column = data['target_column']
        
        model_logger.info(f"Model loaded from {filepath}")
