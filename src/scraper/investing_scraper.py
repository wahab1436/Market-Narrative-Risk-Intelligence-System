"""
Investing.com Real Market Data Scraper - CLOUDFLARE BYPASS VERSION
Scrapes ACTUAL market data with anti-ban protection using cloudscraper
Uses proper rate limiting, rotating headers, and retry logic
"""
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import cloudscraper
import pandas as pd
from bs4 import BeautifulSoup
import json
import re

try:
    from src.utils.logger import scraper_logger
except ImportError:
    import logging
    scraper_logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


class SafeInvestingScraper:
    """
    Production-safe scraper for Investing.com with Cloudflare bypass.
    """
    
    # Rotate through multiple user agents
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
    ]
    
    def __init__(self, delay_range: Tuple[int, int] = (3, 7), max_retries: int = 3):
        """
        Initialize safe scraper with anti-ban measures.
        
        Args:
            delay_range: (min, max) seconds to wait between requests
            max_retries: Maximum number of retry attempts
        """
        self.base_url = "https://www.investing.com"
        self.delay_range = delay_range
        self.max_retries = max_retries
        self.request_count = 0
        self.failed_count = 0
        
        # Create cloudscraper session to bypass Cloudflare
        self.session = self._create_safe_session()
        
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
        
        scraper_logger.info(f"Initialized SafeInvestingScraper with {len(self.instruments)} instruments")
        scraper_logger.info(f"Delay range: {delay_range[0]}-{delay_range[1]}s, Max retries: {max_retries}")
    
    def _create_safe_session(self):
        """Create cloudscraper session to bypass Cloudflare."""
        scraper = cloudscraper.create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'windows',
                'mobile': False
            },
            delay=10,  # Delay for solving challenges
            interpreter='native'  # Use native JS interpreter
        )
        return scraper
    
    def _get_random_headers(self) -> Dict[str, str]:
        """Get randomized headers to avoid detection."""
        return {
            'User-Agent': random.choice(self.USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
            'Referer': 'https://www.google.com/',
        }
    
    def _respectful_delay(self):
        """Wait a random amount of time to be respectful and avoid detection."""
        delay = random.uniform(self.delay_range[0], self.delay_range[1])
        scraper_logger.debug(f"Waiting {delay:.1f}s before next request...")
        time.sleep(delay)
    
    def _safe_request(self, url: str) -> Optional[cloudscraper.CloudScraper]:
        """
        Make a safe HTTP request with retries and error handling.
        
        Args:
            url: URL to request
            
        Returns:
            Response object or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                headers = self._get_random_headers()
                
                scraper_logger.debug(f"Request #{self.request_count + 1} (attempt {attempt + 1}/{self.max_retries}): {url}")
                
                response = self.session.get(
                    url,
                    headers=headers,
                    timeout=30,  # Increased timeout for Cloudflare challenges
                    allow_redirects=True
                )
                
                self.request_count += 1
                
                # Check for rate limiting
                if response.status_code == 429:
                    wait_time = 2 ** attempt * 5
                    scraper_logger.warning(f"Rate limited! Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                
                # Check for Cloudflare block
                if response.status_code == 403:
                    scraper_logger.warning(f"403 Forbidden on attempt {attempt + 1}. Recreating session...")
                    self.session = self._create_safe_session()
                    time.sleep(2 ** attempt * 3)
                    continue
                
                response.raise_for_status()
                return response
                
            except Exception as e:
                scraper_logger.warning(f"Request error on attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        
        self.failed_count += 1
        scraper_logger.error(f"Failed to fetch {url} after {self.max_retries} attempts")
        return None
    
    def _parse_price(self, text: str) -> float:
        """Safely parse price from text."""
        if not text:
            return 0.0
        
        try:
            cleaned = str(text).strip()
            cleaned = cleaned.replace(',', '').replace('$', '').replace('€', '').replace('£', '')
            cleaned = cleaned.replace('%', '').replace('+', '').replace(' ', '')
            
            if '(' in cleaned and ')' in cleaned:
                cleaned = '-' + cleaned.replace('(', '').replace(')', '')
            
            match = re.search(r'-?\d+\.?\d*', cleaned)
            if match:
                return float(match.group())
            
            return 0.0
        except (ValueError, AttributeError) as e:
            scraper_logger.debug(f"Could not parse price from '{text}': {e}")
            return 0.0
    
    def _extract_price_data(self, soup: BeautifulSoup) -> Dict[str, float]:
        """
        Extract price data from Investing.com page.
        PRIORITY: JSON-LD structured data (most stable)
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Dictionary with price, change, change_percent
        """
        data = {}
        
        # PRIORITY STRATEGY 1: JSON-LD structured data (most reliable and stable)
        script_tags = soup.find_all('script', {'type': 'application/ld+json'})
        for script in script_tags:
            try:
                if not script.string:
                    continue
                    
                json_data = json.loads(script.string)
                
                # Handle array of JSON-LD objects
                if isinstance(json_data, list):
                    for item in json_data:
                        if isinstance(item, dict) and '@type' in item:
                            if item.get('@type') in ['Product', 'FinancialProduct']:
                                if 'offers' in item and isinstance(item['offers'], dict):
                                    price = item['offers'].get('price')
                                    if price and 'price' not in data:
                                        data['price'] = float(price)
                                        scraper_logger.debug(f"Found price in JSON-LD: {price}")
                                
                                if 'lowPrice' in item:
                                    data['day_low'] = float(item['lowPrice'])
                                if 'highPrice' in item:
                                    data['day_high'] = float(item['highPrice'])
                
                # Handle single JSON-LD object
                elif isinstance(json_data, dict):
                    if 'price' in json_data and 'price' not in data:
                        data['price'] = float(json_data['price'])
                        scraper_logger.debug(f"Found price in JSON-LD object: {json_data['price']}")
                    if 'lowPrice' in json_data:
                        data['day_low'] = float(json_data['lowPrice'])
                    if 'highPrice' in json_data:
                        data['day_high'] = float(json_data['highPrice'])
                        
            except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
                scraper_logger.debug(f"Error parsing JSON-LD: {e}")
                continue
        
        # STRATEGY 2: Data attributes (reliable when present)
        if 'price' not in data:
            price_elem = soup.find('div', {'data-test': 'instrument-price-last'})
            if price_elem:
                data['price'] = self._parse_price(price_elem.text)
                scraper_logger.debug(f"Found price via data-test: {price_elem.text}")
        
        change_elem = soup.find('span', {'data-test': 'instrument-price-change'})
        if change_elem:
            data['change'] = self._parse_price(change_elem.text)
        
        change_pct_elem = soup.find('span', {'data-test': 'instrument-price-change-percent'})
        if change_pct_elem:
            data['change_percent'] = self._parse_price(change_pct_elem.text)
        
        # STRATEGY 3: Meta tags
        if 'price' not in data:
            meta_price = soup.find('meta', {'property': 'og:price:amount'})
            if meta_price and meta_price.get('content'):
                data['price'] = self._parse_price(meta_price['content'])
                scraper_logger.debug(f"Found price in meta tag: {meta_price['content']}")
        
        # STRATEGY 4: CSS classes (least reliable, last resort)
        if 'price' not in data:
            for class_pattern in ['text-5xl', 'text-4xl', 'instrument-price', 'last-price']:
                elem = soup.find(class_=re.compile(class_pattern, re.I))
                if elem:
                    price = self._parse_price(elem.text)
                    if price > 0:
                        data['price'] = price
                        scraper_logger.debug(f"Found price via CSS class {class_pattern}: {price}")
                        break
        
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
        Scrape data for a single instrument safely.
        
        Args:
            name: Instrument name
            info: Dictionary with 'url', 'type', 'priority'
            
        Returns:
            Dictionary with market data or None
        """
        url = f"{self.base_url}{info['url']}"
        
        scraper_logger.info(f"Scraping: {name} ({info['type']})")
        
        response = self._safe_request(url)
        if not response:
            return None
        
        try:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract price data using prioritized strategies
            price_data = self._extract_price_data(soup)
            
            if 'price' not in price_data or price_data['price'] == 0:
                scraper_logger.warning(f"Could not extract valid price for {name}")
                return None
            
            # Build complete data object
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
                f"Change: ${result['change']:+.2f}"
            )
            
            return result
            
        except Exception as e:
            scraper_logger.error(f"Error parsing {name}: {e}", exc_info=True)
            return None
    
    def scrape_all(self, priority_filter: Optional[int] = None) -> List[Dict]:
        """
        Scrape all instruments with safety measures.
        
        Args:
            priority_filter: Only scrape instruments with priority <= this value
                           (1 = critical, 2 = important, 3 = nice-to-have)
        
        Returns:
            List of successfully scraped market data
        """
        scraper_logger.info("="*70)
        scraper_logger.info("STARTING SAFE INVESTING.COM SCRAPING")
        scraper_logger.info("="*70)
        
        # Filter by priority if specified
        instruments = self.instruments
        if priority_filter:
            instruments = {
                name: info for name, info in self.instruments.items()
                if info.get('priority', 3) <= priority_filter
            }
            scraper_logger.info(f"Filtering to priority <= {priority_filter}: {len(instruments)} instruments")
        
        market_data = []
        start_time = time.time()
        
        for i, (name, info) in enumerate(instruments.items(), 1):
            scraper_logger.info(f"\n[{i}/{len(instruments)}] Processing: {name}")
            
            data = self.scrape_instrument(name, info)
            
            if data:
                market_data.append(data)
            
            # Respectful delay between requests (except last one)
            if i < len(instruments):
                self._respectful_delay()
        
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
        
        return market_data
    
    def calculate_market_stress(self, market_data: List[Dict]) -> float:
        """
        Calculate market stress score from real data.
        
        Args:
            market_data: List of market data dictionaries
            
        Returns:
            Stress score 0-10
        """
        if not market_data:
            return 0.0
        
        stress_factors = []
        
        for data in market_data:
            asset = data['asset']
            change_pct = data.get('change_percent', 0)
            price = data.get('price', 0)
            priority = data.get('priority', 3)
            
            # VIX is direct fear gauge
            if 'VIX' in asset:
                vix_stress = min(price / 10, 10)
                stress_factors.append(vix_stress * 3.0)
            
            # Large moves create stress
            move_stress = min(abs(change_pct) / 2, 10)
            
            # Weight by priority
            weight = (4 - priority)
            stress_factors.append(move_stress * weight)
            
            # Extreme negative moves are worse
            if change_pct < -3:
                stress_factors.append(min(abs(change_pct), 10) * 1.5)
        
        # Calculate weighted average
        if stress_factors:
            avg_stress = sum(stress_factors) / len(stress_factors)
            return min(round(avg_stress, 2), 10.0)
        
        return 0.0
    
    def create_articles(self, market_data: List[Dict]) -> List[Dict]:
        """
        Convert market data to article format for pipeline.
        
        Args:
            market_data: List of market data dictionaries
            
        Returns:
            List of article-like dictionaries
        """
        if not market_data:
            scraper_logger.warning("No market data to convert to articles")
            return []
        
        articles = []
        timestamp = datetime.now().isoformat()
        stress_score = self.calculate_market_stress(market_data)
        
        # Create market overview article
        overview_lines = [
            f"Market Stress Score: {stress_score:.1f}/10",
            f"Data Points: {len(market_data)}",
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
        
        # Add summary for each type
        for asset_type, items in sorted(by_type.items()):
            overview_lines.append(f"{asset_type.upper()}:")
            
            items_sorted = sorted(items, key=lambda x: abs(x.get('change_percent', 0)), reverse=True)
            
            for item in items_sorted:
                direction = "↑" if item['change_percent'] > 0 else "↓"
                overview_lines.append(
                    f"  {direction} {item['asset']}: ${item['price']:.2f} "
                    f"({item['change_percent']:+.2f}%, ${item['change']:+.2f})"
                )
            overview_lines.append("")
        
        overview_snippet = '\n'.join(overview_lines)
        
        # Main overview article
        articles.append({
            'headline': f'Market Overview - Stress Level {stress_score:.1f}/10',
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
                'success_rate': f"{(len(market_data)/self.request_count*100):.1f}%" if self.request_count > 0 else "0%"
            }
        })
        
        # Create individual articles for significant movers
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
                
                headline = (
                    f"{data['asset']} {direction} {change_pct:.1f}% "
                    f"to ${data['price']:.2f}"
                )
                
                snippet_parts = [
                    f"{data['asset']} ({data['type']}) is currently trading at ${data['price']:.2f}, ",
                    f"{direction[:-1]}ing {change_pct:.2f}% from the previous close of ${data['prev_close']:.2f}. ",
                    f"Absolute change: ${abs(data['change']):.2f}. "
                ]
                
                if data.get('day_low') and data.get('day_high'):
                    snippet_parts.append(
                        f"Today's range: ${data['day_low']:.2f} - ${data['day_high']:.2f}. "
                    )
                
                if 'VIX' in data['asset']:
                    if data['price'] > 30:
                        snippet_parts.append("Market fear gauge showing high volatility. ")
                    elif data['price'] < 15:
                        snippet_parts.append("Market showing low volatility and complacency. ")
                
                articles.append({
                    'headline': headline,
                    'snippet': ''.join(snippet_parts),
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
        
        scraper_logger.info(f"Created {len(articles)} articles from {len(market_data)} data points")
        return articles
    
    def save_to_bronze(self, articles: List[Dict]) -> Optional[Path]:
        """Save articles to bronze layer."""
        if not articles:
            scraper_logger.warning("No articles to save")
            return None
        
        df = pd.DataFrame(articles)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"investing_market_{timestamp}.parquet"
        filepath = Path("data/bronze") / filename
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(filepath, index=False)
        scraper_logger.info(f"✓ Saved {len(df)} articles to {filepath}")
        
        return filepath


def scrape_investing_data(priority_filter: Optional[int] = None):
    """
    Main function to scrape Investing.com safely.
    
    Args:
        priority_filter: Only scrape priority <= this value (1, 2, or 3)
    
    Returns:
        Path to saved bronze file
    """
    scraper = SafeInvestingScraper(
        delay_range=(3, 7),
        max_retries=3
    )
    
    try:
        market_data = scraper.scrape_all(priority_filter=priority_filter)
        
        if market_data:
            articles = scraper.create_articles(market_data)
            filepath = scraper.save_to_bronze(articles)
            return filepath
        else:
            scraper_logger.error("No market data collected")
            return None
            
    except KeyboardInterrupt:
        scraper_logger.warning("\nScraping interrupted by user")
        return None
    except Exception as e:
        scraper_logger.error(f"Scraping failed: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    print("\n" + "="*70)
    print("INVESTING.COM SAFE SCRAPER - CLOUDFLARE BYPASS")
    print("="*70)
    print("\nPriority levels:")
    print("  1 = Critical only (S&P, VIX, Gold, etc.) - FAST")
    print("  2 = Critical + Important - BALANCED")
    print("  3 = All assets - COMPREHENSIVE")
    print("\n" + "="*70 + "\n")
    
    result = scrape_investing_data(priority_filter=2)
    
    if result:
        print(f"\n{'='*70}")
        print("✓ SCRAPING SUCCESSFUL!")
        print(f"{'='*70}")
        print(f"\nData saved to: {result}")
        
        df = pd.read_parquet(result)
        print(f"\nCollected: {len(df)} articles")
        
        if len(df) > 0:
            print(f"\n{'='*70}")
            print("MARKET SUMMARY")
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
        print("✗ SCRAPING FAILED")
        print(f"{'='*70}")
        print("\nCheck logs for details")
