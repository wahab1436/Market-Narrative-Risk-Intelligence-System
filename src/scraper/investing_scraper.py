"""
Investing.com news scraper with improved anti-blocking measures.
"""
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import pandas as pd

from src.utils.logger import scraper_logger
from src.utils.config_loader import config_loader


class InvestingScraper:
    """
    Scraper for Investing.com financial news with improved anti-blocking.
    """
    
    def __init__(self, use_selenium: bool = False):
        """
        Initialize scraper.
        
        Args:
            use_selenium: Whether to use Selenium for JavaScript content
        """
        self.config = config_loader.get_config("config")
        self.scraping_config = self.config.get("scraping", {})
        self.use_selenium = use_selenium
        self.session = requests.Session()
        
    def _get_realistic_headers(self) -> dict:
        """
        Get realistic browser headers to avoid blocking.
        
        Returns:
            Dictionary of HTTP headers
        """
        return {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
            'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"'
        }
    
    def _make_request(self, url: str, retry_count: int = 0) -> Optional[requests.Response]:
        """
        Make HTTP request with retry logic and better headers.
        
        Args:
            url: URL to request
            retry_count: Current retry attempt
        
        Returns:
            Response object or None
        """
        headers = self._get_realistic_headers()
        
        try:
            # Add random delay to appear more human-like
            time.sleep(random.uniform(1, 3))
            
            response = self.session.get(
                url,
                headers=headers,
                timeout=self.scraping_config.get('timeout', 30),
                allow_redirects=True
            )
            
            scraper_logger.info(f"Response status: {response.status_code}")
            
            # Check if we got blocked
            if response.status_code == 403:
                scraper_logger.warning("Received 403 - website blocking scraper")
                
                # Try with different approach
                if retry_count < self.scraping_config.get('max_retries', 3):
                    delay = self.scraping_config.get('retry_delay', 5) * (2 ** retry_count)
                    scraper_logger.info(f"Retrying with delay: {delay}s")
                    time.sleep(delay)
                    
                    # Try to get cookies first
                    self._get_cookies(url)
                    return self._make_request(url, retry_count + 1)
                else:
                    scraper_logger.error("Max retries exceeded - website blocking access")
                    return None
            
            response.raise_for_status()
            return response
            
        except requests.exceptions.RequestException as e:
            if retry_count < self.scraping_config.get('max_retries', 3):
                delay = self.scraping_config.get('retry_delay', 5) * (2 ** retry_count)
                scraper_logger.warning(f"Request failed, retrying in {delay}s: {e}")
                time.sleep(delay)
                return self._make_request(url, retry_count + 1)
            else:
                scraper_logger.error(f"Max retries exceeded for URL: {url}")
                return None
    
    def _get_cookies(self, url: str):
        """
        Get cookies from the main page first.
        
        Args:
            url: URL to get cookies from
        """
        try:
            base_url = "/".join(url.split("/")[:3])
            response = self.session.get(
                base_url,
                headers=self._get_realistic_headers(),
                timeout=10
            )
            scraper_logger.info(f"Got cookies from {base_url}")
        except Exception as e:
            scraper_logger.warning(f"Failed to get cookies: {e}")
    
    def _parse_article(self, article_soup) -> Optional[Dict]:
        """
        Parse individual article from HTML.
        
        Args:
            article_soup: BeautifulSoup object of article
        
        Returns:
            Dictionary with article data or None
        """
        try:
            # Extract headline
            title_elem = article_soup.find('a', class_='title')
            headline = title_elem.get_text(strip=True) if title_elem else None
            
            # Extract snippet
            snippet_elem = article_soup.find('p', class_='summary')
            snippet = snippet_elem.get_text(strip=True) if snippet_elem else None
            
            # Extract timestamp
            time_elem = article_soup.find('time')
            timestamp = time_elem.get('datetime') if time_elem else None
            
            # Extract asset tags
            tags = []
            tag_elems = article_soup.find_all('a', class_='relatedInstrument')
            for tag in tag_elems:
                tags.append(tag.get_text(strip=True))
            
            # Extract article URL
            article_url = None
            if title_elem and title_elem.get('href'):
                article_url = self.scraping_config['base_url'] + title_elem.get('href')
            
            if not all([headline, snippet, timestamp]):
                return None
            
            return {
                'headline': headline,
                'snippet': snippet,
                'timestamp': timestamp,
                'asset_tags': tags,
                'url': article_url,
                'scraped_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            scraper_logger.error(f"Error parsing article: {e}")
            return None
    
    def scrape_latest_news(self) -> List[Dict]:
        """
        Scrape latest news from Investing.com.
        
        Returns:
            List of article dictionaries
        """
        scraper_logger.info("Starting news scraping")
        
        url = self.scraping_config.get('news_url')
        
        # First, get cookies
        self._get_cookies(url)
        
        # Now try to scrape
        response = self._make_request(url)
        
        if not response:
            scraper_logger.error("Failed to fetch news page")
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Try different selectors as Investing.com structure may vary
        article_elements = (
            soup.find_all('article', class_='js-article-item') or
            soup.find_all('article') or
            soup.find_all('div', class_='largeTitle') or
            soup.find_all('div', class_='article')
        )
        
        scraper_logger.info(f"Found {len(article_elements)} article elements")
        
        articles = []
        max_articles = self.scraping_config.get('max_articles', 100)
        
        for article_elem in article_elements[:max_articles]:
            article_data = self._parse_article(article_elem)
            if article_data:
                articles.append(article_data)
        
        scraper_logger.info(f"Successfully scraped {len(articles)} articles")
        return articles
    
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
        filename = f"investing_news_{timestamp}.parquet"
        filepath = Path("data/bronze") / filename
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as Parquet
        df.to_parquet(filepath, index=False)
        scraper_logger.info(f"Saved {len(df)} articles to {filepath}")
    
    def close(self):
        """Clean up resources."""
        if self.session:
            self.session.close()


def scrape_and_save():
    """
    Main scraping function to be called from pipeline.
    
    Returns:
        Path to saved bronze file
    """
    scraper = InvestingScraper(use_selenium=False)
    
    try:
        articles = scraper.scrape_latest_news()
        if articles:
            scraper.save_to_bronze(articles)
            
            # Return path to latest file
            bronze_dir = Path("data/bronze")
            files = list(bronze_dir.glob("*.parquet"))
            if files:
                return max(files, key=lambda x: x.stat().st_mtime)
        return None
        
    except Exception as e:
        scraper_logger.error(f"Scraping failed: {e}")
        return None
    finally:
        scraper.close()
