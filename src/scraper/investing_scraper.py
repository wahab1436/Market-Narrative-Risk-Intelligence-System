"""
Investing.com Real Market Data Scraper - SELENIUM BROWSER AUTOMATION
Bypasses Cloudflare by simulating a REAL browser
100% Real Investing.com data - NO mock data, NO Yahoo Finance
"""
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd
import json
import re

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("ERROR: Selenium not installed. Install with: pip install selenium")

try:
    from src.utils.logger import scraper_logger
except ImportError:
    import logging
    scraper_logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


class SafeInvestingScraper:
    """
    Investing.com scraper using Selenium to bypass Cloudflare.
    Simulates real browser behavior for 100% real data extraction.
    """
    
    def __init__(self, delay_range: Tuple[int, int] = (2, 4), max_retries: int = 2):
        """Initialize Selenium-based Investing.com scraper."""
        
        if not SELENIUM_AVAILABLE:
            raise ImportError("Selenium is required. Install with: pip install selenium")
        
        self.base_url = "https://www.investing.com"
        self.delay_range = delay_range
        self.max_retries = max_retries
        self.request_count = 0
        self.failed_count = 0
        
        # Browser driver (will be initialized per session)
        self.driver = None
        
        # Market instruments to track
        self.instruments = {
            # Major Indices
            'S&P 500': {'url': '/indices/us-spx-500', 'type': 'index', 'priority': 1},
            'Dow Jones': {'url': '/indices/us-30', 'type': 'index', 'priority': 1},
            'NASDAQ': {'url': '/indices/nasdaq-composite', 'type': 'index', 'priority': 1},
            'VIX': {'url': '/indices/volatility-s-p-500', 'type': 'volatility', 'priority': 1},
            'Russell 2000': {'url': '/indices/smallcap-2000', 'type': 'index', 'priority': 2},
            
            # International Indices
            'FTSE 100': {'url': '/indices/uk-100', 'type': 'index', 'priority': 2},
            'DAX': {'url': '/indices/germany-30', 'type': 'index', 'priority': 2},
            'Nikkei 225': {'url': '/indices/japan-ni225', 'type': 'index', 'priority': 2},
            
            # Commodities
            'Gold': {'url': '/commodities/gold', 'type': 'commodity', 'priority': 1},
            'Crude Oil': {'url': '/commodities/crude-oil', 'type': 'commodity', 'priority': 1},
            'Silver': {'url': '/commodities/silver', 'type': 'commodity', 'priority': 2},
            'Natural Gas': {'url': '/commodities/natural-gas', 'type': 'commodity', 'priority': 2},
            'Copper': {'url': '/commodities/copper', 'type': 'commodity', 'priority': 3},
            
            # Currencies (Forex)
            'EUR/USD': {'url': '/currencies/eur-usd', 'type': 'forex', 'priority': 1},
            'GBP/USD': {'url': '/currencies/gbp-usd', 'type': 'forex', 'priority': 2},
            'USD/JPY': {'url': '/currencies/usd-jpy', 'type': 'forex', 'priority': 2},
            'USD/CHF': {'url': '/currencies/usd-chf', 'type': 'forex', 'priority': 3},
            
            # Cryptocurrencies
            'Bitcoin': {'url': '/crypto/bitcoin/usd', 'type': 'crypto', 'priority': 1},
            'Ethereum': {'url': '/crypto/ethereum/usd', 'type': 'crypto', 'priority': 2},
        }
        
        scraper_logger.info(f"Initialized Selenium-based Investing.com scraper")
        scraper_logger.info(f"Instruments to scrape: {len(self.instruments)}")
        scraper_logger.info(f"Delay range: {delay_range[0]}-{delay_range[1]}s")
    
    def _create_driver(self) -> webdriver.Chrome:
        """Create a Selenium WebDriver with anti-detection settings."""
        
        chrome_options = Options()
        
        # Essential options for Streamlit Cloud
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--disable-software-rasterizer')
        
        # Anti-detection measures
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Realistic browser settings
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--start-maximized')
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        # Language and encoding
        chrome_options.add_argument('--lang=en-US')
        chrome_options.add_argument('--accept-language=en-US,en;q=0.9')
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            
            # Inject anti-detection JavaScript
            driver.execute_cdp_cmd('Network.setUserAgentOverride', {
                "userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            })
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            driver.set_page_load_timeout(30)
            driver.implicitly_wait(10)
            
            scraper_logger.info("✓ Selenium WebDriver initialized successfully")
            return driver
            
        except Exception as e:
            scraper_logger.error(f"Failed to create WebDriver: {e}")
            raise
    
    def _respectful_delay(self):
        """Wait between requests to be respectful."""
        delay = random.uniform(self.delay_range[0], self.delay_range[1])
        scraper_logger.debug(f"Waiting {delay:.1f}s...")
        time.sleep(delay)
    
    def _parse_price(self, text: str) -> float:
        """Safely parse price from text."""
        if not text:
            return 0.0
        
        try:
            cleaned = str(text).strip()
            cleaned = cleaned.replace(',', '').replace('$', '').replace('€', '').replace('£', '')
            cleaned = cleaned.replace('%', '').replace('+', '').replace(' ', '').replace('\n', '')
            
            if '(' in cleaned and ')' in cleaned:
                cleaned = '-' + cleaned.replace('(', '').replace(')', '')
            
            match = re.search(r'-?\d+\.?\d*', cleaned)
            if match:
                return float(match.group())
            
            return 0.0
        except (ValueError, AttributeError):
            return 0.0
    
    def _extract_price_from_page(self, driver: webdriver.Chrome) -> Dict[str, float]:
        """
        Extract price data from Investing.com page using multiple strategies.
        Returns dict with price, change, change_percent, etc.
        """
        data = {}
        wait = WebDriverWait(driver, 15)
        
        # Strategy 1: Try data-test attributes (most reliable)
        try:
            price_elem = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "[data-test='instrument-price-last']"))
            )
            price_text = price_elem.text
            data['price'] = self._parse_price(price_text)
            scraper_logger.debug(f"Found price via data-test: {price_text}")
        except:
            scraper_logger.debug("data-test price element not found")
        
        # Try to get change
        try:
            change_elem = driver.find_element(By.CSS_SELECTOR, "[data-test='instrument-price-change']")
            data['change'] = self._parse_price(change_elem.text)
        except:
            pass
        
        # Try to get change percent
        try:
            change_pct_elem = driver.find_element(By.CSS_SELECTOR, "[data-test='instrument-price-change-percent']")
            data['change_percent'] = self._parse_price(change_pct_elem.text)
        except:
            pass
        
        # Strategy 2: Try large text elements if data-test failed
        if 'price' not in data or data['price'] == 0:
            try:
                large_price_selectors = [
                    "span.text-5xl",
                    "span.text-4xl",
                    "div.instrument-price_instrument-price__3uw4E span",
                    "span[class*='text-']"
                ]
                
                for selector in large_price_selectors:
                    try:
                        elem = driver.find_element(By.CSS_SELECTOR, selector)
                        price = self._parse_price(elem.text)
                        if price > 0:
                            data['price'] = price
                            scraper_logger.debug(f"Found price via {selector}: {price}")
                            break
                    except:
                        continue
            except:
                pass
        
        # Strategy 3: Parse page source for JSON-LD
        if 'price' not in data or data['price'] == 0:
            try:
                page_source = driver.page_source
                
                # Look for JSON-LD structured data
                json_ld_pattern = r'<script type="application/ld\+json">(.*?)</script>'
                matches = re.findall(json_ld_pattern, page_source, re.DOTALL)
                
                for match in matches:
                    try:
                        json_data = json.loads(match)
                        
                        if isinstance(json_data, dict):
                            if 'price' in json_data:
                                data['price'] = float(json_data['price'])
                                scraper_logger.debug(f"Found price in JSON-LD: {data['price']}")
                                break
                            elif 'offers' in json_data and isinstance(json_data['offers'], dict):
                                if 'price' in json_data['offers']:
                                    data['price'] = float(json_data['offers']['price'])
                                    scraper_logger.debug(f"Found price in JSON-LD offers: {data['price']}")
                                    break
                    except:
                        continue
            except:
                pass
        
        # Strategy 4: Try XPath for price elements
        if 'price' not in data or data['price'] == 0:
            xpath_selectors = [
                "//span[contains(@class, 'text-5xl')]",
                "//div[@data-test='instrument-price-last']",
                "//span[contains(text(), '$') or contains(text(), '.')]"
            ]
            
            for xpath in xpath_selectors:
                try:
                    elem = driver.find_element(By.XPATH, xpath)
                    price = self._parse_price(elem.text)
                    if price > 0:
                        data['price'] = price
                        scraper_logger.debug(f"Found price via XPath: {price}")
                        break
                except:
                    continue
        
        # Calculate derived values
        if 'price' in data and 'change' in data and data['change'] != 0:
            data['prev_close'] = data['price'] - data['change']
        
        if 'price' in data and 'change_percent' in data and data['change_percent'] != 0:
            if 'prev_close' not in data:
                data['prev_close'] = data['price'] / (1 + data['change_percent'] / 100)
            if 'change' not in data:
                data['change'] = data['price'] - data['prev_close']
        
        return data
    
    def scrape_instrument(self, name: str, info: Dict) -> Optional[Dict]:
        """
        Scrape data for a single instrument from Investing.com.
        Uses active Selenium session.
        """
        url = f"{self.base_url}{info['url']}"
        
        scraper_logger.info(f"Scraping: {name} ({info['type']}) from Investing.com")
        
        for attempt in range(self.max_retries):
            try:
                # Navigate to page
                scraper_logger.debug(f"Loading: {url} (attempt {attempt + 1}/{self.max_retries})")
                self.driver.get(url)
                self.request_count += 1
                
                # Wait for page to load and extract data
                time.sleep(2)  # Give Cloudflare time to process
                
                price_data = self._extract_price_from_page(self.driver)
                
                if 'price' not in price_data or price_data['price'] == 0:
                    scraper_logger.warning(f"Could not extract valid price for {name} (attempt {attempt + 1})")
                    if attempt < self.max_retries - 1:
                        time.sleep(3)
                        continue
                    else:
                        self.failed_count += 1
                        return None
                
                # Build result
                result = {
                    'asset': name,
                    'type': info['type'],
                    'priority': info.get('priority', 3),
                    'url': url,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'investing.com',
                    'price': price_data['price'],
                    'change': price_data.get('change', 0.0),
                    'change_percent': price_data.get('change_percent', 0.0),
                    'prev_close': price_data.get('prev_close', price_data['price']),
                    'day_low': price_data.get('day_low'),
                    'day_high': price_data.get('day_high'),
                }
                
                scraper_logger.info(
                    f"✓ {name}: ${result['price']:.2f} "
                    f"({result['change_percent']:+.2f}%) "
                    f"[Investing.com]"
                )
                
                return result
                
            except TimeoutException:
                scraper_logger.warning(f"Timeout loading {name} (attempt {attempt + 1})")
                if attempt < self.max_retries - 1:
                    time.sleep(5)
                continue
                
            except Exception as e:
                scraper_logger.error(f"Error scraping {name}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(3)
                    continue
                
        self.failed_count += 1
        return None
    
    def scrape_all(self, priority_filter: Optional[int] = None) -> List[Dict]:
        """
        Scrape all instruments from Investing.com using Selenium.
        Maintains single browser session for efficiency.
        """
        scraper_logger.info("="*70)
        scraper_logger.info("STARTING INVESTING.COM SCRAPING WITH SELENIUM")
        scraper_logger.info("="*70)
        scraper_logger.info("Source: Investing.com (real browser automation)")
        scraper_logger.info("Method: Selenium WebDriver with anti-detection")
        scraper_logger.info("="*70)
        
        # Filter by priority
        instruments = self.instruments
        if priority_filter:
            instruments = {
                name: info for name, info in self.instruments.items()
                if info.get('priority', 3) <= priority_filter
            }
            scraper_logger.info(f"Filtering to priority <= {priority_filter}: {len(instruments)} instruments")
        
        market_data = []
        start_time = time.time()
        
        try:
            # Create browser session
            scraper_logger.info("Initializing Selenium WebDriver...")
            self.driver = self._create_driver()
            
            # Scrape each instrument
            for i, (name, info) in enumerate(instruments.items(), 1):
                scraper_logger.info(f"\n[{i}/{len(instruments)}] Processing: {name}")
                
                data = self.scrape_instrument(name, info)
                
                if data:
                    market_data.append(data)
                
                # Respectful delay between requests
                if i < len(instruments):
                    self._respectful_delay()
            
        finally:
            # Always close browser
            if self.driver:
                try:
                    self.driver.quit()
                    scraper_logger.info("Browser session closed")
                except:
                    pass
        
        elapsed = time.time() - start_time
        success_rate = (len(market_data) / len(instruments) * 100) if instruments else 0
        
        scraper_logger.info("\n" + "="*70)
        scraper_logger.info("SCRAPING COMPLETE")
        scraper_logger.info("="*70)
        scraper_logger.info(f"Success: {len(market_data)}/{len(instruments)} ({success_rate:.1f}%)")
        scraper_logger.info(f"Failed: {self.failed_count}")
        scraper_logger.info(f"Total requests: {self.request_count}")
        scraper_logger.info(f"Time elapsed: {elapsed:.1f}s")
        scraper_logger.info(f"Avg time per request: {elapsed/self.request_count:.1f}s" if self.request_count > 0 else "N/A")
        scraper_logger.info("="*70)
        
        if not market_data:
            scraper_logger.error("NO DATA COLLECTED FROM INVESTING.COM")
        
        return market_data
    
    def calculate_market_stress(self, market_data: List[Dict]) -> float:
        """Calculate market stress score from real data."""
        if not market_data:
            return 0.0
        
        stress_factors = []
        
        for data in market_data:
            asset = data['asset']
            change_pct = data.get('change_percent', 0)
            price = data.get('price', 0)
            priority = data.get('priority', 3)
            
            if 'VIX' in asset:
                vix_stress = min(price / 10, 10)
                stress_factors.append(vix_stress * 3.0)
            
            move_stress = min(abs(change_pct) / 2, 10)
            weight = (4 - priority)
            stress_factors.append(move_stress * weight)
            
            if change_pct < -3:
                stress_factors.append(min(abs(change_pct), 10) * 1.5)
        
        if stress_factors:
            avg_stress = sum(stress_factors) / len(stress_factors)
            return min(round(avg_stress, 2), 10.0)
        
        return 0.0
    
    def create_articles(self, market_data: List[Dict]) -> List[Dict]:
        """Convert Investing.com market data to article format."""
        if not market_data:
            scraper_logger.warning("No market data to convert to articles")
            return []
        
        articles = []
        timestamp = datetime.now().isoformat()
        stress_score = self.calculate_market_stress(market_data)
        
        # Market overview
        overview_lines = [
            f"Market Stress Score: {stress_score:.1f}/10",
            f"Data Points: {len(market_data)} (Investing.com)",
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        
        # Group by type
        by_type = {}
        for data in market_data:
            asset_type = data['type']
            if asset_type not in by_type:
                by_type[asset_type] = []
            by_type[asset_type].append(data)
        
        # Add summary
        for asset_type, items in sorted(by_type.items()):
            overview_lines.append(f"{asset_type.upper()}:")
            items_sorted = sorted(items, key=lambda x: abs(x.get('change_percent', 0)), reverse=True)
            
            for item in items_sorted:
                direction = "↑" if item['change_percent'] > 0 else "↓"
                overview_lines.append(
                    f"  {direction} {item['asset']}: ${item['price']:.2f} "
                    f"({item['change_percent']:+.2f}%)"
                )
            overview_lines.append("")
        
        overview_snippet = '\n'.join(overview_lines)
        
        # Main overview article
        articles.append({
            'headline': f'Investing.com Market Overview - Stress {stress_score:.1f}/10',
            'snippet': overview_snippet[:2000],
            'timestamp': timestamp,
            'asset_tags': [d['asset'] for d in market_data],
            'url': 'https://www.investing.com',
            'source': 'investing.com',
            'scraped_at': timestamp,
            'market_stress_score': stress_score,
            'data_points': len(market_data),
            'asset_types': list(by_type.keys()),
            'scraper_stats': {
                'requests': self.request_count,
                'failures': self.failed_count,
                'success_rate': f"{((len(market_data))/(len(market_data)+self.failed_count)*100):.1f}%" if market_data else "0%"
            }
        })
        
        # Individual movers
        for data in market_data:
            change_pct = abs(data.get('change_percent', 0))
            
            should_create = (
                change_pct > 2.0 or
                (data.get('priority') == 1 and change_pct > 1.0) or
                'VIX' in data['asset']
            )
            
            if should_create:
                direction = "surges" if data['change_percent'] > 3 else \
                           "rises" if data['change_percent'] > 0 else \
                           "plunges" if data['change_percent'] < -3 else "falls"
                
                headline = f"{data['asset']} {direction} {change_pct:.1f}% to ${data['price']:.2f}"
                
                snippet = (
                    f"{data['asset']} ({data['type']}) is trading at ${data['price']:.2f}, "
                    f"{direction[:-1]}ing {change_pct:.2f}% from ${data['prev_close']:.2f}. "
                    f"Data from Investing.com. "
                )
                
                if data.get('day_low') and data.get('day_high'):
                    snippet += f"Range: ${data['day_low']:.2f} - ${data['day_high']:.2f}. "
                
                if 'VIX' in data['asset']:
                    if data['price'] > 30:
                        snippet += "High volatility detected. "
                    elif data['price'] < 15:
                        snippet += "Low volatility environment. "
                
                articles.append({
                    'headline': headline,
                    'snippet': snippet,
                    'timestamp': timestamp,
                    'asset_tags': [data['asset']],
                    'url': data['url'],
                    'source': 'investing.com',
                    'scraped_at': timestamp,
                    'price': data['price'],
                    'change': data['change'],
                    'change_percent': data['change_percent'],
                    'asset_type': data['type'],
                    'priority': data.get('priority', 3)
                })
        
        scraper_logger.info(f"Created {len(articles)} articles from {len(market_data)} Investing.com data points")
        return articles
    
    def save_to_bronze(self, articles: List[Dict]) -> Optional[Path]:
        """Save Investing.com articles to bronze layer."""
        if not articles:
            scraper_logger.error("No articles to save")
            return None
        
        df = pd.DataFrame(articles)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"investing_market_{timestamp}.parquet"
        filepath = Path("data/bronze") / filename
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(filepath, index=False)
        scraper_logger.info(f"✓ Saved {len(df)} Investing.com articles to {filepath}")
        
        return filepath


def scrape_investing_data(priority_filter: Optional[int] = None):
    """
    Scrape real data from Investing.com using Selenium.
    Returns None if scraping fails.
    """
    if not SELENIUM_AVAILABLE:
        print("ERROR: Selenium not installed!")
        print("Install with: pip install selenium")
        return None
    
    scraper = SafeInvestingScraper(delay_range=(2, 4), max_retries=2)
    
    try:
        market_data = scraper.scrape_all(priority_filter=priority_filter)
        
        if market_data:
            articles = scraper.create_articles(market_data)
            filepath = scraper.save_to_bronze(articles)
            return filepath
        else:
            scraper_logger.error("NO INVESTING.COM DATA COLLECTED")
            return None
            
    except KeyboardInterrupt:
        scraper_logger.warning("\nScraping interrupted by user")
        return None
    except Exception as e:
        scraper_logger.error(f"Scraping failed: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    print("\n" + "="*70)
    print("INVESTING.COM SELENIUM SCRAPER")
    print("="*70)
    print("Method: Real browser automation with Selenium WebDriver")
    print("Source: 100% Investing.com data")
    print("\nPriority levels:")
    print("  1 = Critical only - FASTEST")
    print("  2 = Critical + Important - BALANCED")
    print("  3 = All assets - COMPREHENSIVE")
    print("\n" + "="*70 + "\n")
    
    result = scrape_investing_data(priority_filter=2)
    
    if result:
        print(f"\n{'='*70}")
        print("✓ INVESTING.COM DATA COLLECTED!")
        print(f"{'='*70}")
        print(f"\nData saved to: {result}")
        
        df = pd.read_parquet(result)
        print(f"\nInvesting.com data points: {len(df)}")
        
        if len(df) > 0:
            print(f"\n{'='*70}")
            print("MARKET SUMMARY (INVESTING.COM)")
            print(f"{'='*70}\n")
            
            overview = df[df['headline'].str.contains('Overview', na=False)]
            if not overview.empty:
                print(overview.iloc[0]['snippet'][:500])
            
            print(f"\n{'='*70}")
            print("TOP MOVERS")
            print(f"{'='*70}\n")
            
            movers = df[~df['headline'].str.contains('Overview', na=False)].head(10)
            for idx, row in movers.iterrows():
                print(f"  • {row['headline']}")
    else:
        print(f"\n{'='*70}")
        print("❌ SCRAPING FAILED")
        print(f"{'='*70}")
        print("\nMake sure Selenium is installed:")
        print("  pip install selenium")
        print("\nCheck logs for details")
