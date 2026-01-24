"""
Feature engineering module for market narrative analysis.
FIXED: Works with or without config files, has sensible defaults
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

# Try to import config, but have defaults
try:
    from src.utils.config_loader import config_loader
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    preprocessing_logger.warning("Config loader not available, using defaults")


class FeatureEngineer:
    """
    Create features from cleaned market/news data.
    Works with market data from Investing.com scraper.
    """
    
    # Default configuration if config files not available
    DEFAULT_STRESS_KEYWORDS = {
        'high_risk': [
            'crash', 'plunge', 'collapse', 'panic', 'crisis', 'turmoil',
            'recession', 'bear market', 'sell-off', 'volatility spike',
            'default', 'bankruptcy', 'emergency'
        ],
        'medium_risk': [
            'decline', 'fall', 'drop', 'weakness', 'concern', 'risk',
            'uncertainty', 'warning', 'pressure', 'correction', 'loss'
        ],
        'low_risk': [
            'caution', 'watch', 'monitor', 'attention', 'mixed',
            'fluctuation', 'variation', 'change'
        ]
    }
    
    DEFAULT_ASSET_CATEGORIES = {
        'equities': ['S&P', 'Dow', 'NASDAQ', 'stock', 'equity', 'index', 'Russell'],
        'commodities': ['gold', 'silver', 'oil', 'crude', 'copper', 'gas', 'metal'],
        'forex': ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'currency', 'forex', 'dollar'],
        'crypto': ['bitcoin', 'BTC', 'ethereum', 'ETH', 'crypto'],
        'volatility': ['VIX', 'volatility', 'fear', 'stress']
    }
    
    DEFAULT_WEIGHTS = {
        'sentiment': 0.3,
        'keyword_stress': 0.4,
        'volume': 0.2,
        'recency': 0.1
    }
    
    def __init__(self):
        """Initialize feature engineer with config or defaults."""
        self.sia = SentimentIntensityAnalyzer()
        
        # Load config or use defaults
        if HAS_CONFIG:
            try:
                self.config = config_loader.get_config("config")
                self.feature_config = config_loader.get_config("feature_config")
                preprocessing_logger.info("[preprocessing] FeatureEngineer initialized with config")
            except Exception as e:
                preprocessing_logger.warning(f"[preprocessing] Config loading failed: {e}, using defaults")
                self.config = {}
                self.feature_config = {}
        else:
            self.config = {}
            self.feature_config = {}
        
        # Get configuration with fallbacks
        self.stress_keywords = self.feature_config.get(
            'stress_keywords',
            self.DEFAULT_STRESS_KEYWORDS
        )
        self.asset_categories = self.feature_config.get(
            'asset_categories',
            self.DEFAULT_ASSET_CATEGORIES
        )
        self.weights = self.feature_config.get(
            'weights',
            self.DEFAULT_WEIGHTS
        )
        
        preprocessing_logger.info("[preprocessing] FeatureEngineer initialized")
    
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
            if not silver_dir.exists():
                raise FileNotFoundError(f"Silver directory not found: {silver_dir}")
            
            files = list(silver_dir.glob("*.parquet"))
            if not files:
                raise FileNotFoundError("No silver files found")
            filepath = max(files, key=lambda x: x.stat().st_mtime)
        
        preprocessing_logger.info(f"[preprocessing] Loading silver data from {filepath}")
        df = pd.read_parquet(filepath)
        preprocessing_logger.info(f"[preprocessing] Loaded {len(df)} records with {len(df.columns)} columns")
        
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
        
        # Ensure headline column exists
        if 'headline' not in df.columns:
            preprocessing_logger.warning("[preprocessing] No headline column found")
            df['headline'] = df.get('snippet', '')
        
        # Sentiment analysis
        df['sentiment_polarity'] = df['headline'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity if pd.notna(x) else 0
        )
        df['sentiment_subjectivity'] = df['headline'].apply(
            lambda x: TextBlob(str(x)).sentiment.subjectivity if pd.notna(x) else 0
        )
        
        # VADER sentiment
        df['vader_compound'] = df['headline'].apply(
            lambda x: self.sia.polarity_scores(str(x))['compound'] if pd.notna(x) else 0
        )
        
        # For market data, incorporate price change into sentiment
        if 'change_percent' in df.columns:
            # Normalize change_percent to -1 to 1 range
            change_normalized = df['change_percent'].clip(-10, 10) / 10
            # Combine with text sentiment
            df['combined_sentiment'] = (df['vader_compound'] + change_normalized) / 2
        else:
            df['combined_sentiment'] = df['vader_compound']
        
        # Keyword stress score
        def calculate_stress_score(text):
            if not isinstance(text, str) or pd.isna(text):
                return 0
            
            score = 0
            text_lower = str(text).lower()
            
            for keyword in self.stress_keywords.get('high_risk', []):
                if keyword in text_lower:
                    score += 3
            
            for keyword in self.stress_keywords.get('medium_risk', []):
                if keyword in text_lower:
                    score += 2
            
            for keyword in self.stress_keywords.get('low_risk', []):
                if keyword in text_lower:
                    score += 1
            
            # For market data, add price-based stress
            # Large moves = high stress
            return score
        
        df['keyword_stress_score'] = df['headline'].apply(calculate_stress_score)
        
        # Add price-based stress for market data
        if 'change_percent' in df.columns:
            # Absolute change > 2% adds stress
            df['price_stress'] = df['change_percent'].abs().apply(
                lambda x: min(x / 2, 5) if x > 2 else 0
            )
            df['keyword_stress_score'] = df['keyword_stress_score'] + df['price_stress']
        
        # Text complexity features
        df['headline_length'] = df['headline'].astype(str).str.len()
        df['word_count'] = df['headline'].apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )
        
        # Asset category features
        for category, keywords in self.asset_categories.items():
            df[f'mentions_{category}'] = df['headline'].apply(
                lambda x: 1 if any(keyword.lower() in str(x).lower() for keyword in keywords) else 0
                if pd.notna(x) else 0
            )
        
        # For market data, also check asset_type column
        if 'type' in df.columns:
            for category in self.asset_categories.keys():
                if category in df['type'].values:
                    df.loc[df['type'] == category, f'mentions_{category}'] = 1
        
        preprocessing_logger.info("[preprocessing] Extracted text features")
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
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Remove any rows with invalid timestamps
        df = df.dropna(subset=['timestamp'])
        
        if df.empty:
            preprocessing_logger.warning("[preprocessing] No valid timestamps found")
            return df
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Basic time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['month'] = df['timestamp'].dt.month
        
        # Try to create aggregated temporal features
        try:
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
            feature_df['rolling_7d_mean'] = feature_df['daily_article_count'].rolling(
                7, min_periods=1
            ).mean()
            feature_df['rolling_7d_volatility'] = feature_df['daily_article_count'].rolling(
                7, min_periods=1
            ).std().fillna(0)
            
            # Lag features
            for lag in [1, 3, 7]:
                feature_df[f'lag_{lag}d'] = feature_df['daily_article_count'].shift(lag)
            
            # Fill NaN in lag features
            feature_df = feature_df.fillna(0)
            
            # Merge back with original data
            df['date'] = df['timestamp'].dt.date
            feature_df['date'] = feature_df.index.date
            
            df = df.merge(feature_df.reset_index(drop=True), on='date', how='left')
            
        except Exception as e:
            preprocessing_logger.warning(f"[preprocessing] Could not create aggregated temporal features: {e}")
            # Add simple defaults
            df['daily_article_count'] = len(df)
            df['rolling_7d_mean'] = len(df)
            df['rolling_7d_volatility'] = 0
            for lag in [1, 3, 7]:
                df[f'lag_{lag}d'] = 0
        
        preprocessing_logger.info("[preprocessing] Extracted temporal features")
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
        
        # Normalize features with better handling of edge cases
        for col in ['sentiment_polarity', 'keyword_stress_score', 'combined_sentiment']:
            if col in df.columns:
                col_std = df[col].std()
                if col_std > 0:
                    df[f'{col}_norm'] = (df[col] - df[col].mean()) / col_std
                else:
                    df[f'{col}_norm'] = 0
        
        # Calculate weighted stress score
        if 'combined_sentiment' in df.columns:
            # Calculate recency score
            if df['timestamp'].notna().any():
                max_timestamp = df['timestamp'].max()
                days_old = (max_timestamp - df['timestamp']).dt.total_seconds() / 86400
                max_days = days_old.max() if days_old.max() > 0 else 1
                recency_score = 1 - (days_old / max_days)
                recency_score = recency_score.clip(0, 1)
            else:
                recency_score = 0.5
            
            # Build weighted stress score
            df['weighted_stress_score'] = 0
            
            if 'combined_sentiment_norm' in df.columns:
                df['weighted_stress_score'] += self.weights['sentiment'] * df['combined_sentiment_norm']
            
            if 'keyword_stress_score_norm' in df.columns:
                df['weighted_stress_score'] += self.weights['keyword_stress'] * df['keyword_stress_score_norm']
            
            if 'daily_article_count' in df.columns:
                count_std = df['daily_article_count'].std()
                if count_std > 0:
                    count_norm = (df['daily_article_count'] - df['daily_article_count'].mean()) / count_std
                    df['weighted_stress_score'] += self.weights['volume'] * count_norm
            
            df['weighted_stress_score'] += self.weights['recency'] * recency_score
        
        # Market breadth indicator
        asset_columns = [col for col in df.columns if col.startswith('mentions_')]
        if asset_columns:
            df['market_breadth'] = df[asset_columns].sum(axis=1)
        
        # Sentiment volatility
        if 'sentiment_polarity' in df.columns and len(df) >= 7:
            df['sentiment_volatility_7d'] = df['sentiment_polarity'].rolling(
                7, min_periods=1
            ).std().fillna(0)
        else:
            df['sentiment_volatility_7d'] = 0
        
        preprocessing_logger.info("[preprocessing] Created composite features")
        return df
    
    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create proper target variable for regression models.
        
        The target should be a FUTURE value we want to predict.
        For market data, we predict next observation's stress.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with target variable
        """
        df = df.copy()
        
        # Sort by timestamp to ensure proper ordering
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Create target: Next observation's weighted stress score
        if 'weighted_stress_score' in df.columns:
            # Shift the stress score backward (so current row has next value)
            df['target_stress_next'] = df['weighted_stress_score'].shift(-1)
            
            # For the last row, use current value (no future data available)
            df['target_stress_next'].fillna(df['weighted_stress_score'], inplace=True)
            
            preprocessing_logger.info("[preprocessing] Created target variable: target_stress_next")
        else:
            preprocessing_logger.warning("[preprocessing] Cannot create target: weighted_stress_score not found")
            # Create a dummy target based on sentiment
            if 'combined_sentiment' in df.columns:
                df['target_stress_next'] = df['combined_sentiment'].abs()
            elif 'sentiment_polarity' in df.columns:
                df['target_stress_next'] = df['sentiment_polarity'].abs()
            else:
                df['target_stress_next'] = 0
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with all features
        """
        preprocessing_logger.info("[preprocessing] Starting feature engineering")
        
        if df.empty:
            preprocessing_logger.error("[preprocessing] Input DataFrame is empty")
            return df
        
        # Log input
        preprocessing_logger.info(f"[preprocessing] Input: {len(df)} records, {len(df.columns)} columns")
        
        # Extract different feature types
        df = self.extract_text_features(df)
        df = self.extract_temporal_features(df)
        df = self.create_composite_features(df)
        df = self.create_target_variable(df)
        
        # Drop intermediate normalized columns
        drop_cols = [col for col in df.columns if col.endswith('_norm')]
        df = df.drop(columns=drop_cols, errors='ignore')
        
        # Also drop temporary columns
        temp_cols = ['date', 'price_stress']
        df = df.drop(columns=[c for c in temp_cols if c in df.columns], errors='ignore')
        
        # Log feature summary
        feature_cols = [
            col for col in df.columns 
            if col not in ['timestamp', 'headline', 'snippet', 'source', 'url', 'asset', 'type']
        ]
        preprocessing_logger.info(
            f"[preprocessing] Output: {len(df)} records, {len(df.columns)} total columns, "
            f"{len(feature_cols)} features"
        )
        preprocessing_logger.info("[preprocessing] Target variable: target_stress_next")
        
        # Log sample feature names
        if feature_cols:
            sample_features = feature_cols[:10]
            preprocessing_logger.info(f"[preprocessing] Sample features: {', '.join(sample_features)}")
        
        return df
    
    def save_to_gold(self, df: pd.DataFrame) -> Optional[Path]:
        """
        Save engineered features to gold layer.
        
        Args:
            df: Feature-engineered DataFrame
        
        Returns:
            Path to saved gold file or None
        """
        if df.empty:
            preprocessing_logger.warning("[preprocessing] No data to save to gold")
            return None
        
        # Create timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"features_{timestamp}.parquet"
        filepath = Path("data/gold") / filename
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as Parquet
        df.to_parquet(filepath, index=False)
        preprocessing_logger.info(
            f"[preprocessing] Saved {len(df)} records with {len(df.columns)} features to {filepath}"
        )
        return filepath


def engineer_and_save(input_file: Optional[Path] = None) -> Optional[Path]:
    """
    Main feature engineering function.
    
    Args:
        input_file: Path to silver file
    
    Returns:
        Path to saved gold file or None
    """
    engineer = FeatureEngineer()
    
    try:
        df = engineer.load_silver_data(input_file)
        
        if df.empty:
            preprocessing_logger.error("[preprocessing] No data to process")
            return None
        
        feature_df = engineer.engineer_features(df)
        
        if feature_df.empty:
            preprocessing_logger.error("[preprocessing] Feature engineering produced no data")
            return None
        
        gold_path = engineer.save_to_gold(feature_df)
        return gold_path
        
    except Exception as e:
        preprocessing_logger.error(f"[preprocessing] Feature engineering failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    print("\n" + "="*70)
    print("FEATURE ENGINEERING TEST")
    print("="*70 + "\n")
    
    try:
        result = engineer_and_save()
        
        if result:
            print(f"✓ SUCCESS: Features saved to {result}")
            
            # Show summary
            df = pd.read_parquet(result)
            print(f"\nFeature summary:")
            print(f"  Records: {len(df)}")
            print(f"  Columns: {len(df.columns)}")
            print(f"\nFeature columns:")
            for col in df.columns[:15]:
                print(f"  • {col}")
        else:
            print("✗ FAILED: No output produced")
            
    except Exception as e:
        print(f"✗ ERROR: {e}")
