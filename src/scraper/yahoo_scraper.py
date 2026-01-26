"""
Yahoo Finance Market Data Scraper - Real Website Scraping
"""

import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
import time
import random
import logging
from typing import Dict, List, Optional
import re
from bs4 import BeautifulSoup

try:
    from src.utils.logger import scraper_logger
except Exception:
    import logging
    scraper_logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

class YahooFinanceScraper:
    def __init__(self):
        self.session = requests.Session()
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0'
        ]
        
        # Real Yahoo Finance URLs
        self.markets = {
            # Indices
            'S&P 500': {'symbol': '^GSPC', 'type': 'index', 'url': 'https://finance.yahoo.com/quote/%5EGSPC', 'priority': 1},
            'DOW JONES': {'symbol': '^DJI', 'type': 'index', 'url': 'https://finance.yahoo.com/quote/%5EDJI', 'priority': 1},
            'NASDAQ': {'symbol': '^IXIC', 'type': 'index', 'url': 'https://finance.yahoo.com/quote/%5EIXIC', 'priority': 1},
            'VIX': {'symbol': '^VIX', 'type': 'volatility', 'url': 'https://finance.yahoo.com/quote/%5EVIX', 'priority': 1},
            'RUSSELL 2000': {'symbol': '^RUT', 'type': 'index', 'url': 'https://finance.yahoo.com/quote/%5ERUT', 'priority': 2},
            
            # Commodities
            'GOLD': {'symbol': 'GC=F', 'type': 'commodity', 'url': 'https://finance.yahoo.com/quote/GC=F', 'priority': 1},
            'CRUDE OIL': {'symbol': 'CL=F', 'type': 'commodity', 'url': 'https://finance.yahoo.com/quote/CL=F', 'priority': 1},
            'SILVER': {'symbol': 'SI=F', 'type': 'commodity', 'url': 'https://finance.yahoo.com/quote/SI=F', 'priority': 2},
            'NATURAL GAS': {'symbol': 'NG=F', 'type': 'commodity', 'url': 'https://finance.yahoo.com/quote/NG=F', 'priority': 2},
            
            # Forex
            'EUR/USD': {'symbol': 'EURUSD=X', 'type': 'forex', 'url': 'https://finance.yahoo.com/quote/EURUSD=X', 'priority': 1},
            'GBP/USD': {'symbol': 'GBPUSD=X', 'type': 'forex', 'url': 'https://finance.yahoo.com/quote/GBPUSD=X', 'priority': 2},
            'USD/JPY': {'symbol': 'USDJPY=X', 'type': 'forex', 'url': 'https://finance.yahoo.com/quote/USDJPY=X', 'priority': 2},
            
            # Crypto
            'BITCOIN': {'symbol': 'BTC-USD', 'type': 'crypto', 'url': 'https://finance.yahoo.com/quote/BTC-USD', 'priority': 1},
            'ETHEREUM': {'symbol': 'ETH-USD', 'type': 'crypto', 'url': 'https://finance.yahoo.com/quote/ETH-USD', 'priority': 2},
        }

    def _get_random_headers(self):
        """Generate random headers to mimic human behavior"""
        return {
            'User-Agent': random.choice(self.user_agents),
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

    def _human_delay(self):
        """Add random delay to mimic human browsing"""
        time.sleep(random.uniform(1, 3))

    def _extract_price_from_html(self, html_content, symbol):
        """Extract price data from Yahoo Finance HTML"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Look for the price element
            price_elements = soup.find_all('fin-streamer')
            price = None
            change = 0.0
            change_percent = 0.0
            
            for element in price_elements:
                if element.get('data-symbol') == symbol and element.get('data-field') == 'regularMarketPrice':
                    price_text = element.get_text().replace(',', '')
                    if price_text and re.match(r'^[\d\.]+$', price_text):
                        price = float(price_text)
                
                if element.get('data-symbol') == symbol and element.get('data-field') == 'regularMarketChange':
                    change_text = element.get_text().replace(',', '')
                    if change_text and re.match(r'^[\d\.\-\+]+$', change_text):
                        change = float(change_text)
                
                if element.get('data-symbol') == symbol and element.get('data-field') == 'regularMarketChangePercent':
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
            scraper_logger.error(f"Error extracting price from HTML for {symbol}: {e}")
            return None

    def _scrape_with_fallback(self, url, symbol):
        """Scrape data with multiple fallback methods"""
        try:
            # Method 1: Direct HTML scraping
            headers = self._get_random_headers()
            response = self.session.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                price_data = self._extract_price_from_html(response.text, symbol)
                if price_data:
                    return price_data
            
            # Method 2: Try simplified approach
            time.sleep(1)
            headers = self._get_random_headers()
            response = self.session.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                # Parse with regex as backup
                price_match = re.search(r'"regularMarketPrice":(\d+\.?\d*)', response.text)
                if price_match:
                    price = float(price_match.group(1))
                    change_match = re.search(r'"regularMarketChange":([\-]?\d+\.?\d*)', response.text)
                    change_percent_match = re.search(r'"regularMarketChangePercent":"([\-]?\d+\.?\d*)%', response.text)
                    
                    change = float(change_match.group(1)) if change_match else 0.0
                    change_percent = float(change_percent_match.group(1)) if change_percent_match else 0.0
                    
                    return {
                        'price': price,
                        'change': change,
                        'change_percent': change_percent
                    }
            
            return None
            
        except Exception as e:
            scraper_logger.error(f"Error scraping {url}: {e}")
            return None

    def get_market_data(self, name, market_info):
        """Get comprehensive market data for an instrument"""
        try:
            scraper_logger.info(f"Scraping {name} ({market_info['symbol']})")
            
            # Add human-like delay
            self._human_delay()
            
            # Scrape the data
            price_data = self._scrape_with_fallback(market_info['url'], market_info['symbol'])
            
            if not price_data:
                scraper_logger.warning(f"Failed to get price data for {name}")
                return None
            
            # Calculate derived values
            current_price = price_data['price']
            change = price_data['change']
            change_percent = price_data['change_percent']
            
            # Previous close calculation
            prev_close = current_price - change if change != 0 else current_price
            
            return {
                'asset': name,
                'symbol': market_info['symbol'],
                'type': market_info['type'],
                'priority': market_info['priority'],
                'price': round(float(current_price), 4),
                'prev_close': round(float(prev_close), 4),
                'change': round(float(change), 4),
                'change_percent': round(float(change_percent), 4),
                'timestamp': datetime.now().isoformat(),
                'source': 'yahoo_finance',
                'url': market_info['url']
            }
            
        except Exception as e:
            scraper_logger.error(f"Error getting market data for {name}: {e}")
            return None

    def scrape_all_markets(self, priority_filter=2):
        """Scrape all markets up to the specified priority level"""
        scraper_logger.info("Starting market data scraping")
        scraper_logger.info(f"Priority filter: {priority_filter}")
        
        market_data = []
        failed_count = 0
        
        # Filter markets by priority
        filtered_markets = {
            name: info for name, info in self.markets.items()
            if info['priority'] <= priority_filter
        }
        
        scraper_logger.info(f"Scraping {len(filtered_markets)} markets")
        
        for name, info in filtered_markets.items():
            data = self.get_market_data(name, info)
            if data:
                market_data.append(data)
                scraper_logger.info(f"✓ {name}: ${data['price']:.2f} ({data['change_percent']:+.2f}%)")
            else:
                failed_count += 1
                scraper_logger.warning(f"✗ Failed to scrape {name}")
            
            # Add delay between requests to avoid rate limiting
            if len(market_data) + failed_count < len(filtered_markets):
                time.sleep(random.uniform(2, 4))
        
        scraper_logger.info(f"Completed: {len(market_data)} successful, {failed_count} failed")
        return market_data

    def calculate_market_stress(self, market_data):
        """Calculate market stress score (0-10)"""
        if not market_data:
            return 0.0
        
        stress_factors = []
        
        for data in market_data:
            change_pct = data.get('change_percent', 0)
            priority = data.get('priority', 3)
            
            # VIX is direct fear indicator
            if 'VIX' in data['asset']:
                vix_stress = min(data['price'] / 10, 10)
                stress_factors.append(vix_stress * 3.0)
            
            # General market moves
            move_stress = min(abs(change_pct) / 2, 10)
            weight = (4 - priority)  # Higher priority = higher weight
            stress_factors.append(move_stress * weight)
            
            # Sharp negative moves
            if change_pct < -3:
                stress_factors.append(min(abs(change_pct), 10) * 1.5)
        
        if stress_factors:
            avg_stress = sum(stress_factors) / len(stress_factors)
            return min(round(avg_stress, 2), 10.0)
        
        return 0.0

    def create_articles(self, market_data):
        """Convert market data to article format"""
        if not market_data:
            scraper_logger.warning("No market data to convert to articles")
            return []
        
        articles = []
        timestamp = datetime.now().isoformat()
        stress_score = self.calculate_market_stress(market_data)
        
        # Create overview article
        overview_lines = [
            f"Market Stress Score: {stress_score:.1f}/10",
            f"Data Points: {len(market_data)}",
            f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
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
            for item in sorted(items, key=lambda x: abs(x.get('change_percent', 0)), reverse=True):
                overview_lines.append(
                    f"  {item['asset']}: ${item['price']:.2f} "
                    f"({item['change_percent']:+.2f}%, ${item['change']:+.2f})"
                )
            overview_lines.append("")
        
        overview_article = {
            'headline': f'Market Overview - Stress Level {stress_score:.1f}/10',
            'snippet': '\n'.join(overview_lines)[:2000],
            'timestamp': timestamp,
            'asset_tags': [d['asset'] for d in market_data],
            'url': 'https://finance.yahoo.com',
            'source': 'yahoo_finance',
            'scraped_at': timestamp,
            'market_stress_score': stress_score,
            'data_points': len(market_data),
            'asset_types': list(by_type.keys()),
        }
        
        articles.append(overview_article)
        
        # Create individual mover articles
        for data in market_data:
            change_pct = abs(data.get('change_percent', 0))
            
            # Create article for significant moves
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
                    f"{data['asset']} ({data['type']}) is trading at ${data['price']:.2f}, ",
                    f"{direction} {change_pct:.2f}% from previous close of ${data['prev_close']:.2f}. ",
                    f"Change: ${data['change']:+.2f}. "
                ]
                
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
                    'asset_type': data['type'],
                    'priority': data['priority']
                })
        
        scraper_logger.info(f"Created {len(articles)} articles")
        return articles

    def save_to_bronze(self, articles):
        """Save articles to bronze layer"""
        if not articles:
            scraper_logger.warning("No articles to save")
            return None
        
        df = pd.DataFrame(articles)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"market_data_{timestamp}.parquet"
        filepath = Path("data/bronze") / filename
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(filepath, index=False)
        scraper_logger.info(f"Saved {len(df)} articles to {filepath}")
        
        return filepath


def scrape_yahoo_finance_data(priority_filter=2):
    """Main function to scrape market data"""
    scraper = YahooFinanceScraper()
    
    try:
        # Scrape market data
        market_data = scraper.scrape_all_markets(priority_filter=priority_filter)
        
        if not market_data:
            scraper_logger.error("No market data collected")
            return None
        
        # Create articles
        articles = scraper.create_articles(market_data)
        
        # Save to bronze layer
        filepath = scraper.save_to_bronze(articles)
        
        return filepath
        
    except Exception as e:
        scraper_logger.error(f"Scraping failed: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    print("Yahoo Finance Market Data Scraper")
    print("=" * 50)
    
    result = scrape_yahoo_finance_data(priority_filter=2)
    
    if result:
        print(f"\nSuccess! Data saved to: {result}")
        
        # Show a quick preview
        df = pd.read_parquet(result)
        print(f"Articles created: {len(df)}")
        
        if len(df) > 0:
            overview = df[df['headline'].str.contains('Overview', na=False)]
            if not overview.empty:
                print("\nMarket Overview Preview:")
                print(overview.iloc[0]['snippet'][:300] + "...")
    else:
        print("\nScraping failed - check logs for details")
