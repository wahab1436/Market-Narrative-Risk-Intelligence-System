"""
Investing.com news scraper - Fixed import issues.
"""
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import sys

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    HAS_DEPENDENCIES = True
except ImportError as e:
    print(f"Warning: Missing dependency: {e}")
    HAS_DEPENDENCIES = False

# Try to import Selenium but don't fail if not installed
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException
    HAS_SELENIUM = True
except ImportError:
    HAS_SELENIUM = False
    print("Warning: Selenium not installed. Will use requests only.")

# Import local modules with error handling
try:
    from src.utils.logger import get_scraper_logger
except ImportError:
    # Fallback logging if logger can't be imported
    import logging
    logging.basicConfig(level=logging.INFO)
    class FallbackLogger:
        def info(self, msg): print(f"[INFO] {msg}")
        def warning(self, msg): print(f"[WARNING] {msg}")
        def error(self, msg): print(f"[ERROR] {msg}")
    get_scraper_logger = lambda: FallbackLogger()

try:
    from src.utils.config_loader import config_loader
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    print("Warning: config_loader not available, using default config")


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
        self.logger = get_scraper_logger()
        
        # Default configuration
        self.scraping_config = {
            'base_url': "https://www.investing.com",
            'news_url': "https://www.investing.com/news/latest-news",
            'max_retries': 3,
            'retry_delay': 5,
            'timeout': 30,
            'user_agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            'max_articles': 100
        }
        
        # Try to load from config if available
        if HAS_CONFIG:
            try:
                config = config_loader.get_config("config")
                if 'scraping' in config:
                    self.scraping_config.update(config['scraping'])
            except Exception as e:
                self.logger.warning(f"Could not load config: {e}")
        
        # Check if Selenium is available
        if use_selenium and not HAS_SELENIUM:
            self.logger.warning("Selenium requested but not available. Falling back to requests.")
            use_selenium = False
        
        self.use_selenium = use_selenium and HAS_SELENIUM
        
        if self.use_selenium:
            self._setup_selenium()
        else:
            self.driver = None
    
    def _setup_selenium(self):
        """Setup Selenium WebDriver with options."""
        try:
            options = webdriver.ChromeOptions()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument(f'user-agent={self.scraping_config.get("user_agent")}')
            self.driver = webdriver.Chrome(options=options)
            self.logger.info("Selenium WebDriver initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize Selenium: {e}")
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
        if not HAS_DEPENDENCIES:
            self.logger.error("Requests dependency not available")
            return None
        
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
                self.logger.warning(f"Request failed, retrying in {delay}s: {e}")
                time.sleep(delay)
                return self._make_request(url, retry_count + 1)
            else:
                self.logger.error(f"Max retries exceeded for URL: {url}")
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
            timestamp = time_elem.get('datetime') if time_elem else datetime.now().isoformat()
            
            # Extract asset tags
            tags = []
            tag_elems = article_soup.find_all('a', class_='relatedInstrument')
            for tag in tag_elems:
                tags.append(tag.get_text(strip=True))
            
            # Extract article URL
            article_url = None
            if title_elem and title_elem.get('href'):
                href = title_elem.get('href')
                article_url = self.scraping_config['base_url'] + href if not href.startswith('http') else href
            
            if not all([headline, snippet]):
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
            self.logger.error(f"Error parsing article: {e}")
            return None
    
    def scrape_latest_news(self) -> List[Dict]:
        """
        Scrape latest news from Investing.com.
        
        Returns:
            List of article dictionaries
        """
        if not HAS_DEPENDENCIES:
            self.logger.error("Required dependencies not available for scraping")
            return []
        
        self.logger.info("Starting news scraping")
        
        articles = []
        url = self.scraping_config.get('news_url')
        
        if self.use_selenium:
            articles = self._scrape_with_selenium(url)
        else:
            articles = self._scrape_with_requests(url)
        
        self.logger.info(f"Scraped {len(articles)} articles")
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
        if not HAS_SELENIUM:
            return []
        
        try:
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
            
        except TimeoutException:
            self.logger.error("Timeout waiting for page to load")
            return []
        except Exception as e:
            self.logger.error(f"Selenium scraping error: {e}")
            return []
    
    def save_to_bronze(self, articles: List[Dict]):
        """
        Save scraped data to bronze layer.
        
        Args:
            articles: List of article dictionaries
        """
        if not articles:
            self.logger.warning("No articles to save")
            return
        
        if not HAS_DEPENDENCIES:
            self.logger.error("Pandas not available for saving data")
            return
        
        df = pd.DataFrame(articles)
        
        # Create timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"investing_news_{timestamp}.parquet"
        filepath = Path("data/bronze") / filename
        
        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as Parquet
        try:
            df.to_parquet(filepath, index=False)
            self.logger.info(f"Saved {len(df)} articles to {filepath}")
            return filepath
        except Exception as e:
            # Fallback to CSV if Parquet fails
            try:
                csv_path = filepath.with_suffix('.csv')
                df.to_csv(csv_path, index=False)
                self.logger.info(f"Saved {len(df)} articles to {csv_path} (CSV fallback)")
                return csv_path
            except Exception as e2:
                self.logger.error(f"Failed to save data: {e2}")
                return None
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'driver') and self.driver:
            self.driver.quit()


def scrape_and_save(use_selenium: bool = False):
    """
    Main scraping function to be called from pipeline.
    
    Args:
        use_selenium: Whether to use Selenium
    
    Returns:
        Path to saved bronze file or None
    """
    scraper = InvestingScraper(use_selenium=use_selenium)
    
    try:
        articles = scraper.scrape_latest_news()
        if articles:
            filepath = scraper.save_to_bronze(articles)
            return filepath
        else:
            print("No articles scraped")
            return None
            
    except Exception as e:
        print(f"Scraping failed: {e}")
        return None
    finally:
        scraper.close()


# Simple test function
def test_scraper():
    """Test the scraper."""
    print("Testing scraper...")
    result = scrape_and_save(use_selenium=False)
    if result:
        print(f"Scraping successful. Saved to: {result}")
    else:
        print("Scraping failed")


if __name__ == "__main__":
    test_scraper()
