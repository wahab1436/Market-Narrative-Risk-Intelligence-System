"""
Investing.com news scraper - Updated imports to avoid circular imports.
"""
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

# Import logger directly to avoid circular imports
from src.utils.logger import get_scraper_logger
# Import config_loader but be careful
from src.utils.config_loader import config_loader

class InvestingScraper:
    """
    Scraper for Investing.com financial news.
    """
    
    def __init__(self, use_selenium: bool = False):
        """
        Initialize scraper.
        
        Args:
            use_selenium: Whether to use Selenium for JavaScript content
        """
        self.logger = get_scraper_logger()
        
        try:
            # Get config - handle case where config_loader might not be initialized
            self.config = config_loader.get_config("config")
            self.scraping_config = self.config.get("scraping", {})
        except:
            # Fallback config if config_loader fails
            self.scraping_config = {
                'base_url': "https://www.investing.com",
                'news_url': "https://www.investing.com/news/latest-news",
                'max_retries': 3,
                'timeout': 30,
                'user_agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                'max_articles': 100
            }
        
        self.use_selenium = use_selenium
        
        if use_selenium:
            self._setup_selenium()
    
    # ... rest of the class remains the same ...
