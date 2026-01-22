"""
Scraper module for collecting financial news data.
"""
from src.scraper.rss_news_scraper import RSSNewsScraper, scrape_and_save

__all__ = ['RSSNewsScraper', 'scrape_and_save']
