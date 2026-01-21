"""
Investing.com news scraper with error handling and logging.
"""
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import pandas as pd

# DON'T import selenium at module level - only when needed
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.common.exceptions import TimeoutException, WebDriverException

from src.utils.logger import scraper_logger
from src.utils.config_loader import config_loader


class InvestingScraper:
    """
    Scraper for Investing.com financial news.
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
        self.driver = None
        
        if use_selenium:
            self._setup_selenium()
    
    def _setup_selenium(self):
        """Setup Selenium WebDriver with options."""
        try:
            # Import selenium only when actually needed
            from selenium import webdriver
            from selenium.common.exceptions import WebDriverException
            
            options = webdriver.ChromeOptions()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument(f'user-agent={self.scraping_config.get("user_agent")}')
            self.driver = webdriver.Chrome(options=options)
            scraper_logger.info("Selenium WebDriver initialized")
        except ImportError as e:
            scraper_logger.warning(f"Selenium not available: {e}")
            self.use_selenium = False
        except Exception as e:
            scraper_logger.error(f"Failed to initialize Selenium: {e}")
            self.use_selenium = False
    
    def _make_request(self, url: str, retry_count: int = 0) -> Optional[requests.Response]:
        """
        Make HTTP request with retry logic.
        
        Args:
            url: URL to request
            retry_count: Current retry attempt
        
        Returns:
            Response object or None
        """
        headers = {
            'User-Agent': self.scraping_config.get('user_agent'),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=self.scraping_config.get('timeout', 30)
            )
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
        
        articles = []
        url = self.scraping_config.get('news_url')
        
        if self.use_selenium:
            articles = self._scrape_with_selenium(url)
        else:
            articles = self._scrape_with_requests(url)
        
        scraper_logger.info(f"Scraped {len(articles)} articles")
        return articles
    
    def _scrape_with_requests(self, url: str) -> List[Dict]:
        """Scrape using requests and BeautifulSoup."""
        response = self._make_request(url)
        if not response:
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        article_elements = soup.find_all('article', class_='js-article-item')
        
        articles = []
        max_articles = self.scraping_config.get('max_articles', 100)
        
        for article_elem in article_elements[:max_articles]:
            article_data = self._parse_article(article_elem)
            if article_data:
                articles.append(article_data)
        
        return articles
    
    def _scrape_with_selenium(self, url: str) -> List[Dict]:
        """Scrape using Selenium for JavaScript-rendered content."""
        try:
            # Import selenium exceptions only when needed
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from selenium.common.exceptions import TimeoutException
            
            if not self.driver:
                scraper_logger.error("Selenium driver not initialized")
                return []
            
            self.driver.get(url)
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "js-article-item"))
            )
            
            # Scroll to load more content
            for _ in range(3):
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            article_elements = soup.find_all('article', class_='js-article-item')
            
            articles = []
            max_articles = self.scraping_config.get('max_articles', 100)
            
            for article_elem in article_elements[:max_articles]:
                article_data = self._parse_article(article_elem)
                if article_data:
                    articles.append(article_data)
            
            return articles
            
        except Exception as e:
            scraper_logger.error(f"Selenium scraping error: {e}")
            return []
    
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
        if self.driver:
            try:
                self.driver.quit()
            except Exception as e:
                scraper_logger.warning(f"Error closing driver: {e}")


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
