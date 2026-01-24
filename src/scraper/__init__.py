"""
Scraper module for collecting financial news data.
"""
from src.scraper.investing_scraper import FastRSSNewsScraper, scrape_and_save

__all__ = ['FastRSSNewsScraper', 'scrape_and_save']
