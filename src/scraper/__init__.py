"""
Scraper module for collecting financial news data.
"""
from src.scraper.investing_scraper import InvestingScraper, scrape_and_save

__all__ = ['InvestingScraper', 'scrape_and_save']
