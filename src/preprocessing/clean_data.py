"""
Data cleaning and validation module.
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
    """
    
    def __init__(self):
        """Initialize data cleaner."""
        self.config = config_loader.get_config("config")
        self.processing_config = self.config.get("processing", {})
        preprocessing_logger.info("DataCleaner initialized")
    
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
        
        preprocessing_logger.info(f"Loading bronze data from {filepath}")
        df = pd.read_parquet(filepath)
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
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        return text.strip()
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Validate data and separate valid/invalid records.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Tuple of (valid_df, invalid_df)
        """
        # Create validation masks
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
                preprocessing_logger.warning(f"Found {invalid_count} records with {mask_name}")
                invalid_mask = invalid_mask | mask
        
        valid_df = df[~invalid_mask].copy()
        invalid_df = df[invalid_mask].copy()
        
        preprocessing_logger.info(f"Validation: {len(valid_df)} valid, {len(invalid_df)} invalid records")
        return valid_df, invalid_df
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply cleaning operations to DataFrame.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Clean text columns
        text_cols = self.processing_config.get('text_columns', ['headline', 'snippet'])
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].apply(self.clean_text)
        
        # Handle categorical columns
        cat_cols = self.processing_config.get('categorical_columns', ['asset_tags', 'market_tags'])
        for col in cat_cols:
            if col in df.columns:
                # Convert list strings to actual lists
                if df[col].dtype == 'object':
                    df[col] = df[col].apply(
                        lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else []
                    )
        
        # Remove duplicates based on headline and timestamp
        df = df.drop_duplicates(subset=['headline', 'timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        preprocessing_logger.info(f"Cleaned {len(df)} records")
        return df
    
    def save_to_silver(self, df: pd.DataFrame):
        """
        Save cleaned data to silver layer.
        
        Args:
            df: Cleaned DataFrame
        """
        if df.empty:
            preprocessing_logger.warning("No data to save to silver")
            return
        
        # Create timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cleaned_news_{timestamp}.parquet"
        filepath = Path("data/silver") / filename
        
        # Save as Parquet
        df.to_parquet(filepath, index=False)
        preprocessing_logger.info(f"Saved {len(df)} records to {filepath}")
        return filepath


def clean_and_save(input_file: Optional[Path] = None) -> Path:
    """
    Main cleaning function to be called from pipeline.
    
    Args:
        input_file: Path to bronze file
    
    Returns:
        Path to saved silver file
    """
    cleaner = DataCleaner()
    
    try:
        df = cleaner.load_bronze_data(input_file)
        valid_df, invalid_df = cleaner.validate_data(df)
        cleaned_df = cleaner.clean_dataframe(valid_df)
        silver_path = cleaner.save_to_silver(cleaned_df)
        return silver_path
        
    except Exception as e:
        preprocessing_logger.error(f"Data cleaning failed: {e}")
        raise
