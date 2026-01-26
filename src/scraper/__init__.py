"""
Scraper module - Market Data Scraper
"""
from src.scraper.yahoo_scraper import YahooFinanceScraper, scrape_yahoo_finance_data


def scrape_and_save(priority_filter=2):
    """
    Scrape real market data from Yahoo Finance.
    Uses priority 2 (critical + important assets) for balanced speed/coverage.
    
    Args:
        priority_filter: Priority level (1=critical, 2=balanced, 3=all)
    
    Returns:
        Path to saved bronze file
    """
    return scrape_yahoo_finance_data(priority_filter=priority_filter)


# Make sure these are exported
__all__ = [
    'scrape_and_save',
    'YahooFinanceScraper', 
    'scrape_yahoo_finance_data'
]
