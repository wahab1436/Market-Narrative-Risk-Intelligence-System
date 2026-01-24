"""
Scraper module - Investing.com Market Data Scraper
"""
from src.scraper.investing_scraper import SafeInvestingScraper, scrape_investing_data

def scrape_and_save():
    """
    Scrape real market data from Investing.com.
    Uses priority 2 (critical + important assets) for balanced speed/coverage.
    
    Returns:
        Path to saved bronze file
    """
    # Use priority 2: Critical + Important assets (balanced approach)
    # Priority 1 = Only critical (fastest)
    # Priority 2 = Critical + Important (recommended)
    # Priority 3 = All assets (slowest)
    return scrape_investing_data(priority_filter=2)

__all__ = ['scrape_and_save', 'SafeInvestingScraper', 'scrape_investing_data']
