"""
Data cleaning and validation module.
FINAL FIXED VERSION - All column name issues resolved
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List, Optional
from pathlib import Path
import re

from src.utils.logger import preprocessing_logger
from src.utils.config_loader import config_loader


class DataCleaner:
    """
    Clean and validate raw scraped data.
    Handles both news articles and market data.
    """
    
    def __init__(self):
        """Initialize data cleaner."""
        self.config = config_loader.get_config("config")
        self.processing_config = self.config.get("processing", {})
        preprocessing_logger.info("[preprocessing] DataCleaner initialized")
    
    def load_bronze_data(self, filepath: Optional[Path] = None) -> pd.DataFrame:
        """
        Load data from bronze layer.
        
        Args:
            filepath: Path to bronze file, uses latest if None
        
        Returns:
            Loaded DataFrame
        """
        if filepath is None:
            bronze_dir = Path("data/bronze")
            files = list(bronze_dir.glob("*.parquet"))
            if not files:
                raise FileNotFoundError("No bronze files found")
            filepath = max(files, key=lambda x: x.stat().st_mtime)
        
        preprocessing_logger.info(f"[preprocessing] Loading bronze data from {filepath}")
        df = pd.read_parquet(filepath)
        
        # Log data summary
        preprocessing_logger.info(f"[preprocessing] Loaded {len(df)} records with {len(df.columns)} columns")
        if not df.empty:
            preprocessing_logger.info(f"[preprocessing] Columns: {', '.join(df.columns.tolist())}")
        
        return df
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text
        
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation and numbers
        text = re.sub(r'[^\w\s.,!?$%€£¥+-]', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'([.,!?]){2,}', r'\1', text)
        
        return text.strip()
    
    def detect_data_type(self, df: pd.DataFrame) -> str:
        """
        Detect whether this is market data or news data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            'market' or 'news'
        """
        # Check for market data indicators
        market_indicators = ['price', 'change_percent', 'asset_type', 'market_stress_score']
        has_market_cols = any(col in df.columns for col in market_indicators)
        
        # Check source
        has_investing_source = False
        if 'source' in df.columns:
            has_investing_source = df['source'].str.contains('investing.com|yahoo_finance', na=False).any()
        
        if has_market_cols or has_investing_source:
            preprocessing_logger.info("[preprocessing] Detected data type: MARKET DATA")
            return 'market'
        else:
            preprocessing_logger.info("[preprocessing] Detected data type: NEWS DATA")
            return 'news'
    
    def validate_market_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Validate market data.
        Market data has different validation rules than news articles.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Tuple of (valid_df, invalid_df)
        """
        masks = {}
        
        # Required fields for market data
        if 'headline' in df.columns:
            masks['missing_headline'] = df['headline'].isna() | (df['headline'] == '')
            # Market headlines can be shorter
            masks['short_headline'] = df['headline'].str.len() < 10
        
        if 'timestamp' in df.columns:
            masks['missing_timestamp'] = df['timestamp'].isna()
            masks['invalid_date'] = ~pd.to_datetime(df['timestamp'], errors='coerce').notna()
        
        # Price data validation (if present)
        if 'price' in df.columns:
            masks['invalid_price'] = (df['price'] <= 0) | df['price'].isna()
        
        # Combine masks - be lenient with market data
        invalid_mask = pd.Series(False, index=df.index)
        critical_masks = ['missing_headline', 'missing_timestamp', 'invalid_date']
        
        for mask_name, mask in masks.items():
            invalid_count = mask.sum()
            if invalid_count > 0:
                preprocessing_logger.warning(f"[preprocessing] Found {invalid_count} records with {mask_name}")
            
            # Only invalidate on critical issues
            if mask_name in critical_masks:
                invalid_mask = invalid_mask | mask
        
        valid_df = df[~invalid_mask].copy()
        invalid_df = df[invalid_mask].copy()
        
        preprocessing_logger.info(
            f"[preprocessing] Market data validation: {len(valid_df)} valid, {len(invalid_df)} invalid"
        )
        return valid_df, invalid_df
    
    def validate_news_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Validate news article data.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Tuple of (valid_df, invalid_df)
        """
        masks = {
            'missing_headline': df['headline'].isna(),
            'missing_timestamp': df['timestamp'].isna(),
            'short_text': df['headline'].str.len() < self.processing_config.get('min_text_length', 20),
            'long_text': df['headline'].str.len() > self.processing_config.get('max_text_length', 1000),
            'invalid_date': ~pd.to_datetime(df['timestamp'], errors='coerce').notna()
        }
        
        # Combine masks
        invalid_mask = pd.Series(False, index=df.index)
        for mask_name, mask in masks.items():
            invalid_count = mask.sum()
            if invalid_count > 0:
                preprocessing_logger.warning(f"[preprocessing] Found {invalid_count} records with {mask_name}")
                invalid_mask = invalid_mask | mask
        
        valid_df = df[~invalid_mask].copy()
        invalid_df = df[invalid_mask].copy()
        
        preprocessing_logger.info(
            f"[preprocessing] News data validation: {len(valid_df)} valid, {len(invalid_df)} invalid"
        )
        return valid_df, invalid_df
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Validate data with type detection.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Tuple of (valid_df, invalid_df)
        """
        # Detect data type and use appropriate validation
        data_type = self.detect_data_type(df)
        
        if data_type == 'market':
            return self.validate_market_data(df)
        else:
            return self.validate_news_data(df)
    
    def enrich_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived fields to market data for feature engineering.
        FIXED: Safe handling of all column name variations.
        
        Args:
            df: Market data DataFrame
            
        Returns:
            Enriched DataFrame
        """
        df = df.copy()
        
        # Create snippet from headline if missing
        if 'snippet' not in df.columns or df['snippet'].isna().all():
            df['snippet'] = df['headline']
        
        # Add market context to snippet
        if 'price' in df.columns:
            df['snippet'] = df.apply(
                lambda row: f"{row['snippet']} [Price: ${row.get('price', 0):.2f}]",
                axis=1
            )
        
        # Extract asset tags - SAFE VERSION
        if 'asset_tags' not in df.columns:
            if 'asset' in df.columns:
                df['asset_tags'] = df['asset'].apply(lambda x: [x] if pd.notna(x) else [])
            else:
                # Fallback: extract from headline
                df['asset_tags'] = df['headline'].apply(
                    lambda x: [] if pd.isna(x) else [x.split()[0]] if len(x.split()) > 0 else []
                )
        
        # Ensure market_tags exists - COMPLETELY SAFE VERSION
        if 'market_tags' not in df.columns:
            # Try multiple column name variations
            type_col = None
            
            if 'asset_type' in df.columns:
                type_col = 'asset_type'
            elif 'type' in df.columns:
                type_col = 'type'
            
            if type_col:
                df['market_tags'] = df[type_col].apply(
                    lambda x: [x] if pd.notna(x) else []
                )
            else:
                # Ultimate fallback: empty list
                df['market_tags'] = [[] for _ in range(len(df))]
                preprocessing_logger.warning("[preprocessing] Neither 'asset_type' nor 'type' column found, using empty market_tags")
        
        return df
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply cleaning operations to DataFrame.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Detect data type
        data_type = self.detect_data_type(df)
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Clean headline (required for all data)
        if 'headline' in df.columns:
            df['headline'] = df['headline'].apply(self.clean_text)
        
        # Clean snippet if present
        if 'snippet' in df.columns:
            df['snippet'] = df['snippet'].apply(self.clean_text)
        
        # Enrich market data
        if data_type == 'market':
            df = self.enrich_market_data(df)
        
        # Handle categorical columns - SAFE VERSION
        cat_cols = self.processing_config.get('categorical_columns', ['asset_tags', 'market_tags'])
        for col in cat_cols:
            if col in df.columns:
                # Convert list strings to actual lists
                def safe_convert_to_list(x):
                    if isinstance(x, list):
                        return x
                    elif isinstance(x, str):
                        if x.startswith('['):
                            try:
                                return eval(x)
                            except:
                                return [x]
                        else:
                            return [x] if x else []
                    else:
                        return []
                
                df[col] = df[col].apply(safe_convert_to_list)
        
        # Remove duplicates based on headline and timestamp
        df = df.drop_duplicates(subset=['headline', 'timestamp'], keep='first')
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        preprocessing_logger.info(f"[preprocessing] Cleaned {len(df)} records")
        
        # Log sample
        if not df.empty:
            preprocessing_logger.info(f"[preprocessing] Sample headline: {df.iloc[0]['headline'][:100]}")
        
        return df
    
    def save_to_silver(self, df: pd.DataFrame) -> Optional[Path]:
        """
        Save cleaned data to silver layer.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            Path to saved file or None
        """
        if df.empty:
            preprocessing_logger.warning("[preprocessing] No data to save to silver")
            return None
        
        # Create timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cleaned_{timestamp}.parquet"
        filepath = Path("data/silver") / filename
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as Parquet
        df.to_parquet(filepath, index=False)
        preprocessing_logger.info(
            f"[preprocessing] Saved {len(df)} records with {len(df.columns)} columns to {filepath}"
        )
        return filepath


def clean_and_save(input_file: Optional[Path] = None) -> Optional[Path]:
    """
    Main cleaning function to be called from pipeline.
    
    Args:
        input_file: Path to bronze file
    
    Returns:
        Path to saved silver file or None
    """
    cleaner = DataCleaner()
    
    try:
        # Load data
        df = cleaner.load_bronze_data(input_file)
        
        if df.empty:
            preprocessing_logger.warning("[preprocessing] Bronze data is empty")
            return None
        
        # Validate
        valid_df, invalid_df = cleaner.validate_data(df)
        
        # Log invalid records
        if not invalid_df.empty:
            preprocessing_logger.warning(
                f"[preprocessing] Skipping {len(invalid_df)} invalid records"
            )
        
        # Clean valid data
        if valid_df.empty:
            preprocessing_logger.error("[preprocessing] No valid records after validation")
            return None
        
        cleaned_df = cleaner.clean_dataframe(valid_df)
        
        # Save
        silver_path = cleaner.save_to_silver(cleaned_df)
        return silver_path
        
    except Exception as e:
        preprocessing_logger.error(f"[preprocessing] Data cleaning failed: {e}", exc_info=True)
        raise
