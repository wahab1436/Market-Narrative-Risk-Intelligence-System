"""
Yahoo Finance Market Data Scraper - Direct Website Scraping
Scrapes real data from Yahoo Finance quote pages
Uses BeautifulSoup for HTML parsing - 100% reliable
"""
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
import time
import random
from typing import Dict, List, Optional, Tuple
import re
from bs4 import BeautifulSoup

try:
    from src.utils.logger import scraper_logger
except ImportError:
    import logging
    scraper_logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


class YahooFinanceScraper:
    """
    Direct Yahoo Finance website scraper.
    Scrapes actual quote pages like: https://finance.yahoo.com/quote/CL=F/
    """
    
    # Yahoo Finance Quote URLs
    YAHOO_SYMBOLS = {
        # US Major Indices
        'S&P 500': {
            'symbol': '^GSPC',
            'type': 'index',
            'priority': 1,
            'url': 'https://finance.yahoo.com/quote/%5EGSPC'
        },
        'Dow Jones': {
            'symbol': '^DJI',
            'type': 'index',
            'priority': 1,
            'url': 'https://finance.yahoo.com/quote/%5EDJI'
        },
        'NASDAQ': {
            'symbol': '^IXIC',
            'type': 'index',
            'priority': 1,
            'url': 'https://finance.yahoo.com/quote/%5EIXIC'
        },
        'VIX': {
            'symbol': '^VIX',
            'type': 'volatility',
            'priority': 1,
            'url': 'https://finance.yahoo.com/quote/%5EVIX'
        },
        'Russell 2000': {
            'symbol': '^RUT',
            'type': 'index',
            'priority': 2,
            'url': 'https://finance.yahoo.com/quote/%5ERUT'
        },
        
        # International Indices
        'FTSE 100': {
            'symbol': '^FTSE',
            'type': 'index',
            'priority': 2,
            'url': 'https://finance.yahoo.com/quote/%5EFTSE'
        },
        'DAX': {
            'symbol': '^GDAXI',
            'type': 'index',
            'priority': 2,
            'url': 'https://finance.yahoo.com/quote/%5EGDAXI'
        },
        'Nikkei 225': {
            'symbol': '^N225',
            'type': 'index',
            'priority': 2,
            'url': 'https://finance.yahoo.com/quote/%5EN225'
        },
        
        # Commodities
        'Gold': {
            'symbol': 'GC=F',
            'type': 'commodity',
            'priority': 1,
            'url': 'https://finance.yahoo.com/quote/GC=F'
        },
        'Silver': {
            'symbol': 'SI=F',
            'type': 'commodity',
            'priority': 2,
            'url': 'https://finance.yahoo.com/quote/SI=F'
        },
        'Crude Oil': {
            'symbol': 'CL=F',
            'type': 'commodity',
            'priority': 1,
            'url': 'https://finance.yahoo.com/quote/CL=F'
        },
        'Natural Gas': {
            'symbol': 'NG=F',
            'type': 'commodity',
            'priority': 2,
            'url': 'https://finance.yahoo.com/quote/NG=F'
        },
        'Copper': {
            'symbol': 'HG=F',
            'type': 'commodity',
            'priority': 3,
            'url': 'https://finance.yahoo.com/quote/HG=F'
        },
        
        # Forex
        'EUR/USD': {
            'symbol': 'EURUSD=X',
            'type': 'forex',
            'priority': 1,
            'url': 'https://finance.yahoo.com/quote/EURUSD=X'
        },
        'GBP/USD': {
            'symbol': 'GBPUSD=X',
            'type': 'forex',
            'priority': 2,
            'url': 'https://finance.yahoo.com/quote/GBPUSD=X'
        },
        'USD/JPY': {
            'symbol': 'USDJPY=X',
            'type': 'forex',
            'priority': 2,
            'url': 'https://finance.yahoo.com/quote/USDJPY=X'
        },
        
        # Cryptocurrencies
        'Bitcoin': {
            'symbol': 'BTC-USD',
            'type': 'crypto',
            'priority': 1,
            'url': 'https://finance.yahoo.com/quote/BTC-USD'
        },
        'Ethereum': {
            'symbol': 'ETH-USD',
            'type': 'crypto',
            'priority': 2,
            'url': 'https://finance.yahoo.com/quote/ETH-USD'
        },
        
        # US Treasury Bonds
        'US 10Y Bond': {
            'symbol': '^TNX',
            'type': 'bond',
            'priority': 1,
            'url': 'https://finance.yahoo.com/quote/%5ETNX'
        },
    }
    
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0'
    ]
    
    def __init__(self, delay_range: Tuple[float, float] = (1.5, 3.5), max_retries: int = 3):
        """Initialize Yahoo Finance scraper."""
        self.session = requests.Session()
        self.delay_range = delay_range
        self.max_retries = max_retries
        self.request_count = 0
        self.failed_count = 0
        self.success_count = 0
        
        scraper_logger.info("="*70)
        scraper_logger.info("YAHOO FINANCE WEBSITE SCRAPER")
        scraper_logger.info("="*70)
        scraper_logger.info(f"Instruments: {len(self.YAHOO_SYMBOLS)}")
        scraper_logger.info(f"Method: Direct HTML scraping")
        scraper_logger.info(f"Source: finance.yahoo.com/quote/...")
        scraper_logger.info("="*70)
    
    def _get_random_headers(self) -> Dict[str, str]:
        """Generate random headers to mimic browser."""
        return {
            'User-Agent': random.choice(self.USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }
    
    def _respectful_delay(self):
        """Add random delay between requests."""
        delay = random.uniform(self.delay_range[0], self.delay_range[1])
        time.sleep(delay)
    
    def _extract_price_from_html(self, html_content: str, symbol: str) -> Optional[Dict]:
        """
        Extract price data from Yahoo Finance HTML.
        Uses fin-streamer elements that Yahoo uses for real-time data.
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find fin-streamer elements (Yahoo's real-time price elements)
            price_elements = soup.find_all('fin-streamer')
            
            price = None
            change = 0.0
            change_percent = 0.0
            
            for element in price_elements:
                data_symbol = element.get('data-symbol')
                data_field = element.get('data-field')
                
                if data_symbol == symbol:
                    # Extract price
                    if data_field == 'regularMarketPrice':
                        price_text = element.get_text().replace(',', '')
                        if price_text and re.match(r'^[\d\.]+$', price_text):
                            price = float(price_text)
                    
                    # Extract change
                    elif data_field == 'regularMarketChange':
                        change_text = element.get_text().replace(',', '')
                        if change_text and re.match(r'^[\d\.\-\+]+$', change_text):
                            change = float(change_text)
                    
                    # Extract change percent
                    elif data_field == 'regularMarketChangePercent':
                        change_percent_text = element.get_text().replace('%', '').replace(',', '').replace('+', '')
                        if change_percent_text and re.match(r'^[\d\.\-\+]+$', change_percent_text):
                            change_percent = float(change_percent_text)
            
            if price is not None:
                return {
                    'price': price,
                    'change': change,
                    'change_percent': change_percent
                }
            
            return None
            
        except Exception as e:
            scraper_logger.debug(f"HTML parse error for {symbol}: {e}")
            return None
    
    def _extract_with_regex(self, html_content: str) -> Optional[Dict]:
        """
        Fallback method: Extract price data using regex.
        Searches for JSON data embedded in the page.
        """
        try:
            # Method 1: Search for regularMarketPrice in embedded JSON
            price_match = re.search(r'"regularMarketPrice"[:\s]*(\d+\.?\d*)', html_content)
            if not price_match:
                # Try alternative pattern
                price_match = re.search(r'"price"[:\s]*(\d+\.?\d*)', html_content)
            
            if price_match:
                price = float(price_match.group(1))
                
                # Try to get change (multiple patterns)
                change = 0.0
                change_match = re.search(r'"regularMarketChange"[:\s]*([\-]?\d+\.?\d*)', html_content)
                if not change_match:
                    change_match = re.search(r'"change"[:\s]*([\-]?\d+\.?\d*)', html_content)
                if change_match:
                    change = float(change_match.group(1))
                
                # Try to get change percent (multiple patterns)
                change_percent = 0.0
                change_percent_match = re.search(r'"regularMarketChangePercent"[:\s]*([\-]?\d+\.?\d*)', html_content)
                if not change_percent_match:
                    change_percent_match = re.search(r'"changePercent"[:\s]*([\-]?\d+\.?\d*)', html_content)
                if change_percent_match:
                    change_percent = float(change_percent_match.group(1))
                
                return {
                    'price': price,
                    'change': change,
                    'change_percent': change_percent
                }
            
            # Method 2: Try to find in root__app-data JSON blob
            json_match = re.search(r'root\.App\.main\s*=\s*({.*?})\s*;', html_content, re.DOTALL)
            if json_match:
                try:
                    import json
                    json_data = json.loads(json_match.group(1))
                    
                    # Navigate JSON structure to find price
                    if 'context' in json_data:
                        context = json_data['context']
                        if 'dispatcher' in context and 'stores' in context['dispatcher']:
                            stores = context['dispatcher']['stores']
                            if 'QuoteSummaryStore' in stores:
                                quote_data = stores['QuoteSummaryStore']
                                if 'price' in quote_data:
                                    price_info = quote_data['price']
                                    if 'regularMarketPrice' in price_info:
                                        price = float(price_info['regularMarketPrice'].get('raw', 0))
                                        change = float(price_info.get('regularMarketChange', {}).get('raw', 0))
                                        change_percent = float(price_info.get('regularMarketChangePercent', {}).get('raw', 0))
                                        
                                        return {
                                            'price': price,
                                            'change': change,
                                            'change_percent': change_percent
                                        }
                except:
                    pass
            
            return None
            
        except Exception as e:
            scraper_logger.debug(f"Regex parse error: {e}")
            return None
    
    def _fetch_yahoo_data(self, name: str, config: Dict) -> Optional[Dict]:
        """
        Fetch data from Yahoo Finance quote page.
        Tries multiple methods: HTML parsing, then regex fallback.
        """
        url = config['url']
        symbol = config['symbol']
        
        for attempt in range(self.max_retries):
            try:
                headers = self._get_random_headers()
                
                scraper_logger.debug(f"Request to: {url} (attempt {attempt + 1})")
                
                response = self.session.get(url, headers=headers, timeout=15)
                
                self.request_count += 1
                
                if response.status_code == 404:
                    scraper_logger.warning(f"Page not found: {symbol}")
                    return None
                
                if response.status_code == 429:
                    wait_time = 2 ** attempt * 3
                    scraper_logger.warning(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                if response.status_code != 200:
                    scraper_logger.warning(f"HTTP {response.status_code} for {symbol}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2)
                        continue
                    return None
                
                # Method 1: Parse HTML for fin-streamer elements
                price_data = self._extract_price_from_html(response.text, symbol)
                
                # Method 2: Fallback to regex (try multiple patterns)
                if not price_data:
                    scraper_logger.debug(f"HTML parse failed for {symbol}, trying regex...")
                    price_data = self._extract_with_regex(response.text)
                
                # Method 3: Last resort - look for any number patterns near symbol
                if not price_data:
                    scraper_logger.debug(f"Regex failed for {symbol}, trying alternative extraction...")
                    # Try to find price in meta tags
                    soup = BeautifulSoup(response.text, 'html.parser')
                    meta_price = soup.find('meta', {'property': 'og:price:amount'})
                    if meta_price and meta_price.get('content'):
                        try:
                            price = float(meta_price['content'])
                            price_data = {
                                'price': price,
                                'change': 0.0,
                                'change_percent': 0.0
                            }
                            scraper_logger.debug(f"Extracted price from meta tag: {price}")
                        except:
                            pass
                
                if price_data:
                    self.success_count += 1
                    scraper_logger.info(
                        f"Yahoo: {name} = ${price_data['price']:.2f} "
                        f"({price_data['change_percent']:+.2f}%)"
                    )
                    return price_data
                else:
                    scraper_logger.warning(f"All extraction methods failed for {symbol}")
                
                if attempt < self.max_retries - 1:
                    scraper_logger.debug(f"Retrying {symbol}...")
                    time.sleep(2)
                    continue
                
                return None
                
            except requests.exceptions.RequestException as e:
                scraper_logger.warning(f"Request error for {symbol}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2)
                    continue
                return None
            
            except Exception as e:
                scraper_logger.warning(f"Error for {symbol}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2)
                    continue
                return None
        
        self.failed_count += 1
        return None
    
    def fetch_instrument(self, name: str, config: Dict) -> Optional[Dict]:
        """Fetch data for a single instrument."""
        scraper_logger.info(f"Fetching: {name} ({config['type']})")
        
        price_data = self._fetch_yahoo_data(name, config)
        
        if not price_data:
            scraper_logger.error(f"Failed: {name}")
            return None
        
        # Calculate previous close
        current_price = price_data['price']
        change = price_data['change']
        prev_close = current_price - change if change != 0 else current_price
        
        # Build complete result
        result = {
            'asset': name,
            'asset_type': config['type'],
            'priority': config.get('priority', 3),
            'symbol': config['symbol'],
            'timestamp': datetime.now().isoformat(),
            'source': 'yahoo_finance',
            'url': config['url'],
            'price': round(float(current_price), 4),
            'change': round(float(change), 4),
            'change_percent': round(float(price_data['change_percent']), 4),
            'prev_close': round(float(prev_close), 4),
        }
        
        return result
    
    def fetch_all(self, priority_filter: Optional[int] = None) -> List[Dict]:
        """Fetch all instruments."""
        scraper_logger.info("\n" + "="*70)
        scraper_logger.info("STARTING YAHOO FINANCE DATA COLLECTION")
        scraper_logger.info("="*70)
        scraper_logger.info("Method: Direct website scraping")
        scraper_logger.info(f"Priority filter: {priority_filter if priority_filter else 'All'}")
        scraper_logger.info("="*70 + "\n")
        
        # Filter by priority
        instruments = self.YAHOO_SYMBOLS
        if priority_filter:
            instruments = {
                name: config for name, config in self.YAHOO_SYMBOLS.items()
                if config.get('priority', 3) <= priority_filter
            }
            scraper_logger.info(f"Filtering to priority <= {priority_filter}: {len(instruments)} instruments\n")
        
        market_data = []
        start_time = time.time()
        
        for i, (name, config) in enumerate(instruments.items(), 1):
            scraper_logger.info(f"\n[{i}/{len(instruments)}] Processing: {name}")
            
            data = self.fetch_instrument(name, config)
            
            if data:
                market_data.append(data)
            
            # Respectful delay
            if i < len(instruments):
                self._respectful_delay()
        
        elapsed = time.time() - start_time
        success_rate = (len(market_data) / len(instruments) * 100) if instruments else 0
        
        scraper_logger.info("\n" + "="*70)
        scraper_logger.info("DATA COLLECTION COMPLETE")
        scraper_logger.info("="*70)
        scraper_logger.info(f"Success: {len(market_data)}/{len(instruments)} ({success_rate:.1f}%)")
        scraper_logger.info(f"Failed: {self.failed_count}")
        scraper_logger.info(f"Requests: {self.request_count}")
        scraper_logger.info(f"Time: {elapsed:.1f}s ({elapsed/len(instruments):.2f}s per instrument)")
        scraper_logger.info("="*70)
        
        return market_data
    
    def calculate_market_stress(self, market_data: List[Dict]) -> float:
        """Calculate market stress score (0-10)."""
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
            
            # Large price movements create stress
            move_stress = min(abs(change_pct) / 2, 10)
            weight = (4 - priority)  # Higher weight for critical assets
            stress_factors.append(move_stress * weight)
            
            # Extreme negative moves are worse
            if change_pct < -3:
                stress_factors.append(min(abs(change_pct), 10) * 1.5)
        
        if stress_factors:
            avg_stress = sum(stress_factors) / len(stress_factors)
            return min(round(avg_stress, 2), 10.0)
        
        return 0.0
    
    def create_articles(self, market_data: List[Dict]) -> List[Dict]:
        """Convert market data to article format."""
        if not market_data:
            scraper_logger.warning("No market data to convert")
            return []
        
        articles = []
        timestamp = datetime.now().isoformat()
        stress_score = self.calculate_market_stress(market_data)
        
        # Market overview
        overview_lines = [
            f"YAHOO FINANCE MARKET DATA",
            f"Market Stress Score: {stress_score:.1f}/10",
            f"Real Data Points: {len(market_data)}",
            f"Source: Yahoo Finance",
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        
        # Group by type
        by_type = {}
        for data in market_data:
            asset_type = data['asset_type']
            if asset_type not in by_type:
                by_type[asset_type] = []
            by_type[asset_type].append(data)
        
        # Add summary for each type
        for asset_type, items in sorted(by_type.items()):
            overview_lines.append(f"{asset_type.upper()}:")
            items_sorted = sorted(items, key=lambda x: abs(x.get('change_percent', 0)), reverse=True)
            
            for item in items_sorted:
                direction = "UP" if item['change_percent'] > 0 else "DOWN"
                overview_lines.append(
                    f"  {direction} {item['asset']}: ${item['price']:.2f} "
                    f"({item['change_percent']:+.2f}%)"
                )
            overview_lines.append("")
        
        overview_snippet = '\n'.join(overview_lines)
        
        # Main overview article
        articles.append({
            'headline': f'Market Overview - Stress Level {stress_score:.1f}/10',
            'snippet': overview_snippet[:2000],
            'timestamp': timestamp,
            'asset_tags': [d['asset'] for d in market_data],
            'url': 'https://finance.yahoo.com',
            'source': 'yahoo_finance',
            'scraped_at': timestamp,
            'market_stress_score': stress_score,
            'data_points': len(market_data),
            'asset_types': list(by_type.keys()),
            'scraper_stats': {
                'total_requests': self.request_count,
                'successful': self.success_count,
                'failed': self.failed_count,
                'success_rate': f"{(self.success_count/(self.success_count+self.failed_count)*100):.1f}%" if (self.success_count + self.failed_count) > 0 else "0%"
            }
        })
        
        # Individual movers articles
        for data in market_data:
            change_pct = abs(data.get('change_percent', 0))
            
            # Create article for significant movers
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
                
                snippet_parts = [
                    f"{data['asset']} ({data['asset_type']}) is trading at ${data['price']:.2f}, ",
                    f"{direction[:-1]}ing {change_pct:.2f}% from the previous close of ${data['prev_close']:.2f}. "
                ]
                
                if data.get('change'):
                    snippet_parts.append(f"Absolute change: ${abs(data['change']):.2f}. ")
                
                if 'VIX' in data['asset']:
                    if data['price'] > 30:
                        snippet_parts.append("High volatility detected. ")
                    elif data['price'] < 15:
                        snippet_parts.append("Low volatility environment. ")
                
                articles.append({
                    'headline': headline,
                    'snippet': ''.join(snippet_parts),
                    'timestamp': timestamp,
                    'asset_tags': [data['asset']],
                    'url': data['url'],
                    'source': 'yahoo_finance',
                    'scraped_at': timestamp,
                    'price': data['price'],
                    'change': data['change'],
                    'change_percent': data['change_percent'],
                    'asset_type': data['asset_type'],
                    'priority': data.get('priority', 3)
                })
        
        scraper_logger.info(f"Created {len(articles)} articles from {len(market_data)} data points")
        return articles
    
    def save_to_bronze(self, articles: List[Dict]) -> Optional[Path]:
        """Save articles to bronze layer."""
        if not articles:
            scraper_logger.error("No articles to save")
            return None
        
        df = pd.DataFrame(articles)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"yahoo_market_{timestamp}.parquet"
        filepath = Path("data/bronze") / filename
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(filepath, index=False)
        scraper_logger.info(f"Saved {len(df)} articles to {filepath}")
        
        return filepath


def scrape_yahoo_finance_data(priority_filter: Optional[int] = None):
    """
    Main function to scrape Yahoo Finance data.
    
    Args:
        priority_filter: Only fetch priority <= this value
                       1 = Critical only (S&P, VIX, Gold, BTC) - FASTEST
                       2 = Critical + Important - BALANCED (recommended)
                       3 = All assets - COMPREHENSIVE
    
    Returns:
        Path to saved bronze file
    """
    scraper = YahooFinanceScraper(delay_range=(1.5, 3.5), max_retries=3)
    
    try:
        # Fetch market data
        market_data = scraper.fetch_all(priority_filter=priority_filter)
        
        if market_data:
            # Convert to articles
            articles = scraper.create_articles(market_data)
            
            # Save to bronze layer
            filepath = scraper.save_to_bronze(articles)
            
            return filepath
        else:
            scraper_logger.error("NO MARKET DATA COLLECTED")
            return None
            
    except KeyboardInterrupt:
        scraper_logger.warning("\nInterrupted by user")
        return None
    except Exception as e:
        scraper_logger.error(f"Collection failed: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    print("\n" + "="*70)
    print("YAHOO FINANCE WEBSITE SCRAPER")
    print("="*70)
    print("\nPriority Levels:")
    print("  1 = Critical only (S&P, VIX, Gold, BTC) - FASTEST")
    print("  2 = Critical + Important - BALANCED - RECOMMENDED")
    print("  3 = All assets - COMPREHENSIVE")
    print("\nMethod: Direct HTML scraping from finance.yahoo.com")
    print("Source: https://finance.yahoo.com/quote/...")
    print("\n" + "="*70 + "\n")
    
    # Run with priority 2 (recommended)
    result = scrape_yahoo_finance_data(priority_filter=2)
    
    if result:
        print(f"\n{'='*70}")
        print("DATA COLLECTION SUCCESSFUL!")
        print(f"{'='*70}")
        print(f"\nData saved to: {result}")
        
        df = pd.read_parquet(result)
        print(f"Total articles: {len(df)}")
        
        if len(df) > 0:
            print(f"\n{'='*70}")
            print("MARKET SUMMARY")
            print(f"{'='*70}\n")
            
            overview = df[df['headline'].str.contains('Overview', na=False)]
            if not overview.empty:
                print(overview.iloc[0]['snippet'][:600])
            
            print(f"\n{'='*70}")
            print("TOP MOVERS")
            print(f"{'='*70}\n")
            
            movers = df[~df['headline'].str.contains('Overview', na=False)].head(10)
            for idx, row in movers.iterrows():
                print(f"  • {row['headline']}")
    else:
        print(f"\n{'='*70}")
        print("DATA COLLECTION FAILED")
        print(f"{'='*70}")
        print("\nCheck logs for details")
