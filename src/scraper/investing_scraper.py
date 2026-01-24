"""
Investing.com Real Market Data Scraper - FULL PRODUCTION VERSION
Updated with Cloudflare Bypass and JSON-LD Extraction.
"""
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
import json
import re
import cloudscraper  # The bypass engine

try:
    from src.utils.logger import scraper_logger
except ImportError:
    import logging
    scraper_logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

class SafeInvestingScraper:
    """
    Complete scraper with anti-ban measures and Cloudflare bypass.
    """
    
    def __init__(self, delay_range: Tuple[int, int] = (4, 8), max_retries: int = 3):
        self.base_url = "https://www.investing.com"
        self.delay_range = delay_range
        self.max_retries = max_retries
        self.request_count = 0
        self.success_count = 0
        self.failed_count = 0
        
        # Initialize Cloudflare Bypass Scraper
        self.scraper = cloudscraper.create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'windows',
                'desktop': True
            }
        )
        
        # Full Instrument List from your original project
        self.instruments = {
            'S&P 500': {'url': '/indices/us-spx-500', 'type': 'index', 'priority': 1},
            'Dow Jones': {'url': '/indices/us-30', 'type': 'index', 'priority': 1},
            'NASDAQ': {'url': '/indices/nasdaq-composite', 'type': 'index', 'priority': 1},
            'VIX': {'url': '/indices/volatility-s-p-500', 'type': 'volatility', 'priority': 1},
            'Russell 2000': {'url': '/indices/smallcap-2000', 'type': 'index', 'priority': 2},
            'FTSE 100': {'url': '/indices/uk-100', 'type': 'index', 'priority': 2},
            'DAX': {'url': '/indices/germany-30', 'type': 'index', 'priority': 2},
            'Nikkei 225': {'url': '/indices/japan-ni225', 'type': 'index', 'priority': 2},
            'Gold': {'url': '/commodities/gold', 'type': 'commodity', 'priority': 1},
            'Crude Oil': {'url': '/commodities/crude-oil', 'type': 'commodity', 'priority': 1},
            'Silver': {'url': '/commodities/silver', 'type': 'commodity', 'priority': 2},
            'Natural Gas': {'url': '/commodities/natural-gas', 'type': 'commodity', 'priority': 2},
            'EUR/USD': {'url': '/currencies/eur-usd', 'type': 'forex', 'priority': 1},
            'GBP/USD': {'url': '/currencies/gbp-usd', 'type': 'forex', 'priority': 2},
            'USD/JPY': {'url': '/currencies/usd-jpy', 'type': 'forex', 'priority': 2},
            'Bitcoin': {'url': '/crypto/bitcoin/usd', 'type': 'crypto', 'priority': 1},
            'Ethereum': {'url': '/crypto/ethereum/usd', 'type': 'crypto', 'priority': 1},
            'Market Overview': {'url': '/news/stock-market-news', 'type': 'news', 'priority': 1}
        }

    def _respectful_delay(self):
        time.sleep(random.uniform(self.delay_range[0], self.delay_range[1]))

    def _parse_price(self, text: str) -> float:
        if not text: return 0.0
        try:
            # Remove commas, currency symbols, and extra spaces
            cleaned = re.sub(r'[^\d.-]', '', str(text).strip())
            return float(cleaned) if cleaned else 0.0
        except Exception:
            return 0.0

    def _extract_price_data(self, soup: BeautifulSoup) -> Dict[str, float]:
        """
        Robust extraction using JSON-LD metadata and backup selectors.
        """
        data = {}
        
        # Strategy 1: JSON-LD (Search engine metadata - very stable)
        try:
            scripts = soup.find_all('script', type='application/ld+json')
            for script in scripts:
                if script.string and '"price"' in script.string:
                    js = json.loads(script.string)
                    info = js[0] if isinstance(js, list) else js
                    if 'price' in info:
                        data['price'] = float(info['price'])
                        break
        except: pass

        # Strategy 2: Modern CSS Selectors (Backup)
        if 'price' not in data:
            for selector in ['instrument-price-last', 'last-price']:
                elem = soup.find(attrs={"data-test": selector})
                if elem:
                    data['price'] = self._parse_price(elem.text)
                    break

        # Extract Change & Percent
        change_elem = soup.find(attrs={"data-test": "instrument-price-change"})
        if change_elem: data['change'] = self._parse_price(change_elem.text)

        pct_elem = soup.find(attrs={"data-test": "instrument-price-change-percent"})
        if pct_elem: data['change_percent'] = self._parse_price(pct_elem.text.replace('(', '').replace(')', ''))

        return data

    def scrape_instrument(self, name: str, info: Dict) -> Optional[Dict]:
        url = f"{self.base_url}{info['url']}"
        
        for attempt in range(self.max_retries):
            try:
                scraper_logger.info(f"Scraping: {name} ({info['type']})")
                response = self.scraper.get(url, timeout=20)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    if info['type'] == 'news':
                        # Market overview/news logic
                        return self._extract_market_news(soup, name, url)
                    
                    price_data = self._extract_price_data(soup)
                    if price_data.get('price', 0) > 0:
                        self.success_count += 1
                        return {
                            'asset': name,
                            'price': price_data['price'],
                            'change': price_data.get('change', 0.0),
                            'change_percent': price_data.get('change_percent', 0.0),
                            'timestamp': datetime.now().isoformat(),
                            'type': info['type'],
                            'url': url
                        }
                
                elif response.status_code == 403:
                    scraper_logger.warning(f"403 Forbidden for {name}. Retrying with delay...")
                
            except Exception as e:
                scraper_logger.error(f"Error scraping {name}: {str(e)}")
            
            time.sleep(random.uniform(5, 10)) # Delay between retries
            
        return None

    def _extract_market_news(self, soup: BeautifulSoup, name: str, url: str) -> Dict:
        # Simplified news extraction for market overview
        return {
            'asset': name,
            'headline': "Market Update",
            'snippet': "Market data summary and news overview.",
            'timestamp': datetime.now().isoformat(),
            'type': 'news',
            'url': url
        }

    def scrape_all(self, priority_filter: Optional[int] = None) -> List[Dict]:
        start_time = time.time()
        results = []
        
        target_instruments = {
            k: v for k, v in self.instruments.items() 
            if priority_filter is None or v['priority'] <= priority_filter
        }
        
        scraper_logger.info(f"Starting scraping for {len(target_instruments)} instruments")
        
        for i, (name, info) in enumerate(target_instruments.items(), 1):
            scraper_logger.info(f"\n[{i}/{len(target_instruments)}] Processing: {name}")
            data = self.scrape_instrument(name, info)
            if data:
                results.append(data)
            
            self._respectful_delay()

        elapsed = time.time() - start_time
        scraper_logger.info(f"\nSCRAPING COMPLETE: Success {self.success_count}/{len(target_instruments)}")
        scraper_logger.info(f"Time elapsed: {elapsed:.1f}s")
        
        return results

    def create_articles(self, market_data: List[Dict]) -> List[Dict]:
        articles = []
        for d in market_data:
            if d.get('type') == 'news':
                articles.append(d)
                continue
                
            sentiment = "positive" if d.get('change', 0) >= 0 else "negative"
            articles.append({
                'headline': f"{d['asset']} is trading at {d['price']}",
                'snippet': f"{d['asset']} ({d['type']}) moved {d['change']} ({d['change_percent']}%) today. Sentiment is {sentiment}.",
                'source': 'investing.com',
                'timestamp': d['timestamp'],
                'url': d['url'],
                'asset': d['asset'],
                'price': d['price'],
                'sentiment': sentiment
            })
        return articles

    def save_to_bronze(self, articles: List[Dict]) -> Optional[Path]:
        if not articles:
            scraper_logger.error("No market data collected")
            return None
            
        df = pd.DataFrame(articles)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/bronze/sample_{timestamp}.parquet"
        
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        scraper_logger.info(f"Scraping completed: {path}")
        return path

def scrape_investing_data(priority_filter: int = 2):
    scraper = SafeInvestingScraper()
    raw_data = scraper.scrape_all(priority_filter)
    articles = scraper.create_articles(raw_data)
    return scraper.save_to_bronze(articles)

if __name__ == "__main__":
    scrape_investing_data(priority_filter=2)
