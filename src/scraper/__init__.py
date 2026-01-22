
"""
Scraper module for collecting financial news data.
"""
from src.scraper.investing_scraper import RSSNewsScraper, scrape_and_save

__all__ = ['RSSNewsScraper', 'scrape_and_save']
