"""
RSS Feed scraper for financial news - works reliably without blocking.
Supports multiple news sources: Reuters, MarketWatch, CNBC, Bloomberg, etc.
"""
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import xml.etree.ElementTree as ET

import requests
import pandas as pd
from bs4 import BeautifulSoup

from src.utils.logger import scraper_logger
from src.utils.config_loader import config_loader


class RSSNewsScraper:
    """
    RSS feed scraper for financial news from multiple sources.
    """
    
    def __init__(self):
        """Initialize RSS scraper with multiple news sources."""
        self.config = config_loader.get_config("config")
        self.scraping_config = self.config.get("scraping", {})
        
        # RSS feed URLs - these are public and won't block
        self.rss_feeds = {
            'reuters_business': 'https://news.google.com/rss/search?q=when:24h+allinurl:reuters.com&ceid=US:en&hl=en-US&gl=US',
            'marketwatch': 'https://feeds.marketwatch.com/marketwatch/topstories/',
            'cnbc': 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10001147',
            'bloomberg': 'https://news.google.com/rss/search?q=when:24h+allinurl:bloomberg.com&ceid=US:en&hl=en-US&gl=US',
            'wsj': 'https://feeds.a.dj.com/rss/RSSMarketsMain.xml',
            'financial_times': 'https://news.google.com/rss/search?q=when:24h+allinurl:ft.com&ceid=US:en&hl=en-US&gl=US',
            'forbes': 'https://news.google.com/rss/search?q=when:24h+allinurl:forbes.com+finance&ceid=US:en&hl=en-US&gl=US',
            'seeking_alpha': 'https://news.google.com/rss/search?q=when:24h+allinurl:seekingalpha.com&ceid=US:en&hl=en-US&gl=US'
        }
    
    def _fetch_rss_feed(self, url: str, source: str) -> List[Dict]:
        """
        Fetch and parse RSS feed.
        
        Args:
            url: RSS feed URL
            source: Source name
            
        Returns:
            List of article dictionaries
        """
        articles = []
        
        try:
            scraper_logger.info(f"Fetching RSS feed from {source}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Parse XML
            root = ET.fromstring(response.content)
            
            # Handle different RSS formats
            items = root.findall('.//item') or root.findall('.//{http://www.w3.org/2005/Atom}entry')
            
            scraper_logger.info(f"Found {len(items)} items in {source}")
            
            for item in items:
                try:
                    article = self._parse_rss_item(item, source)
                    if article:
                        articles.append(article)
                except Exception as e:
                    scraper_logger.debug(f"Error parsing item: {e}")
                    continue
            
            scraper_logger.info(f"Successfully parsed {len(articles)} articles from {source}")
            
        except Exception as e:
            scraper_logger.error(f"Error fetching RSS feed from {source}: {e}")
        
        return articles
    
    def _parse_rss_item(self, item, source: str) -> Optional[Dict]:
        """
        Parse individual RSS item.
        
        Args:
            item: XML item element
            source: Source name
            
        Returns:
            Dictionary with article data or None
        """
        try:
            # Extract title (headline)
            title_elem = item.find('title')
            if title_elem is None:
                title_elem = item.find('{http://www.w3.org/2005/Atom}title')
            headline = title_elem.text if title_elem is not None else None
            
            # Extract description (snippet)
            desc_elem = item.find('description') or item.find('{http://www.w3.org/2005/Atom}summary')
            snippet = None
            if desc_elem is not None and desc_elem.text:
                # Clean HTML tags from description
                soup = BeautifulSoup(desc_elem.text, 'html.parser')
                snippet = soup.get_text(strip=True)[:500]  # Limit length
            
            # Extract URL
            link_elem = item.find('link')
            if link_elem is None:
                link_elem = item.find('{http://www.w3.org/2005/Atom}link')
                article_url = link_elem.get('href') if link_elem is not None else None
            else:
                article_url = link_elem.text if link_elem.text else link_elem.get('href')
            
            # Extract timestamp
            pub_date_elem = (
                item.find('pubDate') or 
                item.find('{http://www.w3.org/2005/Atom}published') or
                item.find('{http://www.w3.org/2005/Atom}updated')
            )
            
            timestamp = None
            if pub_date_elem is not None and pub_date_elem.text:
                try:
                    # Try parsing different date formats
                    timestamp = pub_date_elem.text
                except Exception:
                    timestamp = datetime.now().isoformat()
            else:
                timestamp = datetime.now().isoformat()
            
            # Extract categories/tags
            tags = []
            category_elems = item.findall('category') or item.findall('{http://www.w3.org/2005/Atom}category')
            for cat in category_elems:
                tag_text = cat.text if cat.text else cat.get('term')
                if tag_text:
                    tags.append(tag_text)
            
            if not headline:
                return None
            
            # If no snippet, use headline as snippet
            if not snippet:
                snippet = headline
            
            return {
                'headline': headline,
                'snippet': snippet,
                'timestamp': timestamp,
                'asset_tags': tags,
                'url': article_url,
                'source': source,
                'scraped_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            scraper_logger.debug(f"Error parsing RSS item: {e}")
            return None
    
    def scrape_all_sources(self, max_articles_per_source: int = 50) -> List[Dict]:
        """
        Scrape news from all RSS sources.
        
        Args:
            max_articles_per_source: Maximum articles per source
            
        Returns:
            List of all articles from all sources
        """
        all_articles = []
        
        scraper_logger.info(f"Starting RSS scraping from {len(self.rss_feeds)} sources")
        
        for source_name, feed_url in self.rss_feeds.items():
            try:
                articles = self._fetch_rss_feed(feed_url, source_name)
                
                # Limit articles per source
                articles = articles[:max_articles_per_source]
                all_articles.extend(articles)
                
                # Be polite - small delay between sources
                time.sleep(1)
                
            except Exception as e:
                scraper_logger.error(f"Error scraping {source_name}: {e}")
                continue
        
        scraper_logger.info(f"Total articles scraped: {len(all_articles)}")
        return all_articles
    
    def save_to_bronze(self, articles: List[Dict]):
        """
        Save scraped data to bronze layer.
        
        Args:
            articles: List of article dictionaries
        """
        if not articles:
            scraper_logger.warning("No articles to save")
            return
        
        df = pd.DataFrame(articles)
        
        # Create timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rss_news_{timestamp}.parquet"
        filepath = Path("data/bronze") / filename
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as Parquet
        df.to_parquet(filepath, index=False)
        scraper_logger.info(f"Saved {len(df)} articles to {filepath}")
        
        return filepath


def scrape_and_save():
    """
    Main scraping function using RSS feeds.
    
    Returns:
        Path to saved bronze file
    """
    scraper = RSSNewsScraper()
    
    try:
        scraper_logger.info("Starting RSS news scraping")
        
        # Scrape from all sources
        articles = scraper.scrape_all_sources(max_articles_per_source=20)
        
        if articles:
            filepath = scraper.save_to_bronze(articles)
            scraper_logger.info(f"RSS scraping completed successfully: {filepath}")
            return filepath
        else:
            scraper_logger.warning("No articles scraped from any source")
            return None
        
    except Exception as e:
        scraper_logger.error(f"RSS scraping failed: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    # Test the scraper
    result = scrape_and_save()
    if result:
        print(f"Successfully scraped news to: {result}")
        df = pd.read_parquet(result)
        print(f"\nScraped {len(df)} articles")
        print(f"\nSources: {df['source'].value_counts()}")
        print(f"\nSample headlines:")
        for headline in df['headline'].head(5):
            print(f"  - {headline}")
    else:
        print("Scraping failed")
