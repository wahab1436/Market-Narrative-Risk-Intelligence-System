"""
Feature engineering module for market narrative analysis.
FIXED: Added proper target variable creation for regression models
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import re
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

from src.utils.logger import preprocessing_logger
from src.utils.config_loader import config_loader


class FeatureEngineer:
    """
    Create features from cleaned news data.
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        self.config = config_loader.get_config("config")
        self.feature_config = config_loader.get_config("feature_config")
        self.sia = SentimentIntensityAnalyzer()
        preprocessing_logger.info("FeatureEngineer initialized")
    
    def load_silver_data(self, filepath: Optional[Path] = None) -> pd.DataFrame:
        """
        Load data from silver layer.
        
        Args:
            filepath: Path to silver file
        
        Returns:
            Loaded DataFrame
        """
        if filepath is None:
            silver_dir = Path("data/silver")
            files = list(silver_dir.glob("*.parquet"))
            if not files:
                raise FileNotFoundError("No silver files found")
            filepath = max(files, key=lambda x: x.stat().st_mtime)
        
        preprocessing_logger.info(f"Loading silver data from {filepath}")
        df = pd.read_parquet(filepath)
        return df
    
    def extract_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from text columns.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with text features
        """
        df = df.copy()
        
        # Sentiment analysis
        df['sentiment_polarity'] = df['headline'].apply(
            lambda x: TextBlob(x).sentiment.polarity if isinstance(x, str) else 0
        )
        df['sentiment_subjectivity'] = df['headline'].apply(
            lambda x: TextBlob(x).sentiment.subjectivity if isinstance(x, str) else 0
        )
        
        # VADER sentiment
        df['vader_compound'] = df['headline'].apply(
            lambda x: self.sia.polarity_scores(x)['compound'] if isinstance(x, str) else 0
        )
        
        # Keyword stress score
        stress_keywords = self.feature_config.get('stress_keywords', {})
        
        def calculate_stress_score(text):
            if not isinstance(text, str):
                return 0
            
            score = 0
            text_lower = text.lower()
            
            for keyword in stress_keywords.get('high_risk', []):
                if keyword in text_lower:
                    score += 3
            
            for keyword in stress_keywords.get('medium_risk', []):
                if keyword in text_lower:
                    score += 2
            
            for keyword in stress_keywords.get('low_risk', []):
                if keyword in text_lower:
                    score += 1
            
            # Normalize by text length
            word_count = len(text.split())
            if word_count > 0:
                score = score / word_count
            
            return score
        
        df['keyword_stress_score'] = df['headline'].apply(calculate_stress_score)
        
        # Text complexity features
        df['headline_length'] = df['headline'].str.len()
        df['word_count'] = df['headline'].apply(
            lambda x: len(x.split()) if isinstance(x, str) else 0
        )
        
        # Asset category features
        asset_categories = self.feature_config.get('asset_categories', {})
        for category, keywords in asset_categories.items():
            df[f'mentions_{category}'] = df['headline'].apply(
                lambda x: 1 if any(keyword.lower() in x.lower() for keyword in keywords) else 0
                if isinstance(x, str) else 0
            )
        
        preprocessing_logger.info("Extracted text features")
        return df
    
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with temporal features
        """
        df = df.copy()
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Set timestamp as index for resampling
        df_indexed = df.set_index('timestamp')
        
        # Daily article count
        daily_counts = df_indexed.resample('D').size()
        daily_counts.name = 'daily_article_count'
        
        # Create feature DataFrame with complete date range
        feature_df = pd.DataFrame(index=pd.date_range(
            start=df['timestamp'].min(),
            end=df['timestamp'].max(),
            freq='D'
        ))
        
        # Merge counts
        feature_df = feature_df.join(daily_counts)
        feature_df['daily_article_count'] = feature_df['daily_article_count'].fillna(0)
        
        # Rolling statistics
        feature_df['rolling_7d_mean'] = feature_df['daily_article_count'].rolling(7, min_periods=1).mean()
        feature_df['rolling_7d_volatility'] = feature_df['daily_article_count'].rolling(7, min_periods=1).std()
        
        # Lag features
        for lag in [1, 3, 7]:
            feature_df[f'lag_{lag}d'] = feature_df['daily_article_count'].shift(lag)
        
        # Fill NaN in lag features
        feature_df = feature_df.fillna(0)
        
        # Day of week features
        feature_df['day_of_week'] = feature_df.index.dayofweek
        feature_df['is_weekend'] = feature_df['day_of_week'].isin([5, 6]).astype(int)
        
        # Merge back with original data
        df['date'] = df['timestamp'].dt.date
        feature_df['date'] = feature_df.index.date
        
        df = df.merge(feature_df.reset_index(drop=True), on='date', how='left')
        
        preprocessing_logger.info("Extracted temporal features")
        return df
    
    def create_composite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create composite features from base features.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with composite features
        """
        df = df.copy()
        
        # Weighted stress score
        weights = self.feature_config.get('weights', {})
        
        # Normalize features with better handling of edge cases
        for col in ['sentiment_polarity', 'keyword_stress_score', 'daily_article_count']:
            if col in df.columns:
                col_std = df[col].std()
                if col_std > 0:
                    df[f'{col}_norm'] = (df[col] - df[col].mean()) / col_std
                else:
                    df[f'{col}_norm'] = 0
        
        # Calculate weighted stress score
        if all(f'{col}_norm' in df.columns for col in ['sentiment_polarity', 'keyword_stress_score']):
            # Calculate recency score
            days_old = (datetime.now() - df['timestamp']).dt.days
            max_days = days_old.max() if days_old.max() > 0 else 1
            recency_score = 1 - (days_old / max_days)
            
            df['weighted_stress_score'] = (
                weights.get('sentiment', 0.3) * df['sentiment_polarity_norm'] +
                weights.get('keyword_stress', 0.4) * df['keyword_stress_score_norm'] +
                weights.get('volume', 0.2) * df.get('daily_article_count_norm', 0) +
                weights.get('recency', 0.1) * recency_score
            )
        
        # Market breadth indicator
        asset_columns = [col for col in df.columns if col.startswith('mentions_')]
        if asset_columns:
            df['market_breadth'] = df[asset_columns].sum(axis=1)
        
        # Sentiment volatility
        if 'sentiment_polarity' in df.columns:
            df['sentiment_volatility_7d'] = df['sentiment_polarity'].rolling(7, min_periods=1).std()
        
        preprocessing_logger.info("Created composite features")
        return df
    
    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        **FIX: Create proper target variable for regression models**
        
        The target should be a FUTURE value we want to predict.
        Here we create a 'future_stress' target that represents
        the stress score 1 day ahead.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with target variable
        """
        df = df.copy()
        
        # Sort by timestamp to ensure proper ordering
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Create target: Next day's weighted stress score
        if 'weighted_stress_score' in df.columns:
            # Shift the stress score backward (so current row has next day's value)
            df['target_stress_next_day'] = df['weighted_stress_score'].shift(-1)
            
            # For the last row, use current value (no future data available)
            df['target_stress_next_day'].fillna(df['weighted_stress_score'], inplace=True)
            
            preprocessing_logger.info("Created target variable: target_stress_next_day")
        else:
            preprocessing_logger.warning("Cannot create target: weighted_stress_score not found")
            # Create a dummy target based on sentiment
            if 'sentiment_polarity' in df.columns:
                df['target_stress_next_day'] = df['sentiment_polarity'].abs()
            else:
                df['target_stress_next_day'] = 0
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with all features
        """
        preprocessing_logger.info("Starting feature engineering")
        
        # Extract different feature types
        df = self.extract_text_features(df)
        df = self.extract_temporal_features(df)
        df = self.create_composite_features(df)
        
        # **FIX: Create target variable for supervised learning**
        df = self.create_target_variable(df)
        
        # Drop intermediate columns
        drop_cols = [col for col in df.columns if col.endswith('_norm')]
        df = df.drop(columns=drop_cols, errors='ignore')
        
        # Log feature summary
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'headline', 'source', 'url', 'date']]
        preprocessing_logger.info(f"Final feature set: {len(df.columns)} columns ({len(feature_cols)} features)")
        preprocessing_logger.info(f"Target variable: target_stress_next_day")
        
        return df
    
    def save_to_gold(self, df: pd.DataFrame) -> Path:
        """
        Save engineered features to gold layer.
        
        Args:
            df: Feature-engineered DataFrame
        
        Returns:
            Path to saved gold file
        """
        if df.empty:
            preprocessing_logger.warning("No data to save to gold")
            return None
        
        # Create timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"features_{timestamp}.parquet"
        filepath = Path("data/gold") / filename
        
        # Save as Parquet
        df.to_parquet(filepath, index=False)
        preprocessing_logger.info(f"Saved {len(df)} records with {len(df.columns)} features to {filepath}")
        return filepath


def engineer_and_save(input_file: Optional[Path] = None) -> Path:
    """
    Main feature engineering function.
    
    Args:
        input_file: Path to silver file
    
    Returns:
        Path to saved gold file
    """
    engineer = FeatureEngineer()
    
    try:
        df = engineer.load_silver_data(input_file)
        feature_df = engineer.engineer_features(df)
        gold_path = engineer.save_to_gold(feature_df)
        return gold_path
        
    except Exception as e:
        preprocessing_logger.error(f"Feature engineering failed: {e}")
        raise
