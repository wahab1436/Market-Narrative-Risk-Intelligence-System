"""
Scraper module - Real Market Data Only (Yahoo Finance)
"""
from src.scraper.investing_market_scraper import scrape_market_data

def scrape_and_save():
    """
    Scrape real market data from Yahoo Finance.
    Single source - reliable and fast.
    
    Returns:
        Path to saved bronze file
    """
    return scrape_market_data()

__all__ = ['scrape_and_save']
