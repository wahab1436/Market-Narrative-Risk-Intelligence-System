"""
Scraper module - Market Data Scraper
"""
from src.scraper import YahooFinanceScraper, scrape_yahoo_finance_data


def scrape_and_save():
    """
    Scrape real market data from Yahoo Finance.
    Uses priority 2 (critical + important assets) for balanced speed/coverage.
    
    Returns:
        Path to saved bronze file
    """
    # Use priority 2: Critical + Important assets (recommended)
    # Priority 1 = Only critical (fastest)
    # Priority 2 = Critical + Important (recommended)
    # Priority 3 = All assets (slowest)
    return scrape_yahoo_finance_data(priority_filter=2)


__all__ = ['scrape_and_save', 'YahooFinanceScraper', 'scrape_yahoo_finance_data']
