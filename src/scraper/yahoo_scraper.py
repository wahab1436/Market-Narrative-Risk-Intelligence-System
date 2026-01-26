"""
Yahoo Finance Scraper - Reliable and Fast
"""

import json
import random
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
import pandas as pd
from bs4 import BeautifulSoup

try:
    from src.utils.logger import scraper_logger
except Exception:
    import logging
    scraper_logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s – %(message)s")


class YahooFinanceScraper:
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    ]

    def __init__(
        self,
        delay_range: Tuple[int, int] = (1, 3),
        max_retries: int = 2,
    ):
        self.base_url = "https://finance.yahoo.com"
        self.delay_range = delay_range
        self.max_retries = max_retries
        self.request_count = 0
        self.failed_count = 0
        
        # Yahoo Finance symbols for various instruments
        self.instruments: Dict[str, Dict] = {
            "S&P 500": {"symbol": "^GSPC", "type": "index", "priority": 1},
            "Dow Jones": {"symbol": "^DJI", "type": "index", "priority": 1},
            "NASDAQ": {"symbol": "^IXIC", "type": "index", "priority": 1},
            "VIX": {"symbol": "^VIX", "type": "volatility", "priority": 1},
            "Russell 2000": {"symbol": "^RUT", "type": "index", "priority": 2},
            "FTSE 100": {"symbol": "^FTSE", "type": "index", "priority": 2},
            "DAX": {"symbol": "^GDAXI", "type": "index", "priority": 2},
            "Nikkei 225": {"symbol": "^N225", "type": "index", "priority": 2},
            "Gold": {"symbol": "GC=F", "type": "commodity", "priority": 1},
            "Crude Oil": {"symbol": "CL=F", "type": "commodity", "priority": 1},
            "Silver": {"symbol": "SI=F", "type": "commodity", "priority": 2},
            "Natural Gas": {"symbol": "NG=F", "type": "commodity", "priority": 2},
            "Copper": {"symbol": "HG=F", "type": "commodity", "priority": 3},
            "EUR/USD": {"symbol": "EURUSD=X", "type": "forex", "priority": 1},
            "GBP/USD": {"symbol": "GBPUSD=X", "type": "forex", "priority": 2},
            "USD/JPY": {"symbol": "USDJPY=X", "type": "forex", "priority": 2},
            "USD/CHF": {"symbol": "USDCHF=X", "type": "forex", "priority": 3},
            "Bitcoin": {"symbol": "BTC-USD", "type": "crypto", "priority": 1},
            "Ethereum": {"symbol": "ETH-USD", "type": "crypto", "priority": 2},
        }
        
        self.session = self._create_session()
        scraper_logger.info(f"YahooFinanceScraper initialized – {len(self.instruments)} instruments")

    def _create_session(self):
        """Create a requests session with proper headers"""
        session = requests.Session()
        session.headers.update({
            "User-Agent": random.choice(self.USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        })
        return session

    def _human_delay(self):
        """Random delay between requests"""
        delay = random.uniform(*self.delay_range)
        time.sleep(delay)

    def _safe_request(self, url: str) -> Optional[requests.Response]:
        """Make a safe request with retries"""
        for attempt in range(self.max_retries + 1):
            try:
                self.request_count += 1
                response = self.session.get(url, timeout=15)
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:  # Rate limiting
                    wait_time = (2 ** attempt) * 5
                    scraper_logger.warning(f"Rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                else:
                    scraper_logger.warning(f"HTTP {response.status_code} for {url}")
                    
            except requests.RequestException as e:
                scraper_logger.warning(f"Request failed (attempt {attempt+1}): {e}")
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
            
        self.failed_count += 1
        return None

    def scrape_instrument(self, name: str, info: Dict) -> Optional[Dict]:
        """Scrape data for a single instrument"""
        symbol = info["symbol"]
        url = f"{self.base_url}/quote/{symbol}"
        
        scraper_logger.info(f"Scraping {name} ({symbol})")
        
        response = self._safe_request(url)
        if not response:
            return None

        try:
            soup = BeautifulSoup(response.content, 'html.parser')
            price_data = self._extract_price_data(soup, name)
            
            if not price_data.get('price'):
                scraper_logger.warning(f"No price data found for {name}")
                return None

            result = {
                "asset": name,
                "symbol": symbol,
                "type": info["type"],
                "priority": info.get("priority", 3),
                "url": url,
                "timestamp": datetime.now().isoformat(),
                "source": "yahoo finance",
                "price": round(float(price_data["price"]), 4),
                "change": round(float(price_data.get("change", 0)), 4),
                "change_percent": round(float(price_data.get("change_percent", 0)), 4),
                "prev_close": round(float(price_data.get("prev_close", price_data["price"])), 4),
                "day_low": round(float(price_data.get("day_low", price_data["price"])), 4),
                "day_high": round(float(price_data.get("day_high", price_data["price"])), 4),
            }
            
            scraper_logger.info(f"✓ {name}: ${result['price']} ({result['change_percent']:+.2f}%)")
            return result
            
        except Exception as e:
            scraper_logger.error(f"Error parsing {name}: {e}")
            return None

    def _extract_price_data(self, soup: BeautifulSoup, asset_name: str) -> Dict[str, float]:
        """Extract price data from Yahoo Finance page"""
        data = {}
        
        # Try to find the main price element
        price_element = soup.find('fin-streamer', {'data-field': 'regularMarketPrice'})
        if price_element and price_element.get('value'):
            data['price'] = float(price_element['value'])
        
        # Try alternative selectors for price
        if 'price' not in data:
            price_selectors = [
                '[data-test="qsp-price"]',
                '.Fw(b).Fz(36px)',
                '[data-symbol][data-field="regularMarketPrice"]',
            ]
            for selector in price_selectors:
                element = soup.select_one(selector)
                if element:
                    price_text = element.get_text().replace(',', '')
                    try:
                        data['price'] = float(price_text)
                        break
                    except ValueError:
                        continue
        
        # Extract change and change percentage
        change_element = soup.find('fin-streamer', {'data-field': 'regularMarketChange'})
        if change_element and change_element.get('value'):
            data['change'] = float(change_element['value'])
        
        change_percent_element = soup.find('fin-streamer', {'data-field': 'regularMarketChangePercent'})
        if change_percent_element and change_percent_element.get('value'):
            # Remove % sign and convert to float
            value = change_percent_element['value'].rstrip('%')
            data['change_percent'] = float(value)
        
        # Extract previous close
        prev_close_element = soup.find('td', string=re.compile('Previous Close', re.I))
        if prev_close_element:
            next_sibling = prev_close_element.find_next_sibling('td')
            if next_sibling:
                try:
                    data['prev_close'] = float(next_sibling.get_text().replace(',', ''))
                except ValueError:
                    pass
        
        # Extract day range
        day_range_element = soup.find('td', string=re.compile('Day\'s Range', re.I))
        if day_range_element:
            next_sibling = day_range_element.find_next_sibling('td')
            if next_sibling:
                range_text = next_sibling.get_text()
                if ' - ' in range_text:
                    low, high = range_text.split(' - ', 1)
                    try:
                        data['day_low'] = float(low.replace(',', ''))
                        data['day_high'] = float(high.replace(',', ''))
                    except ValueError:
                        pass
        
        # If we have price but not change, try to calculate it
        if 'price' in data and 'change' not in data and 'prev_close' in data:
            data['change'] = data['price'] - data['prev_close']
            if data['prev_close'] != 0:
                data['change_percent'] = (data['change'] / data['prev_close']) * 100
        
        return data

    def scrape_all(self, priority_filter: Optional[int] = None) -> List[Dict]:
        """Scrape all instruments"""
        scraper_logger.info("=" * 60)
        scraper_logger.info("STARTING YAHOO FINANCE SCRAPING")
        scraper_logger.info("=" * 60)
        
        if priority_filter:
            instruments = {
                n: i for n, i in self.instruments.items() 
                if i.get("priority", 3) <= priority_filter
            }
            scraper_logger.info(f"Filtering to priority ≤ {priority_filter} → {len(instruments)} instruments")
        else:
            instruments = self.instruments
        
        start_time = time.time()
        results = []
        
        for idx, (name, info) in enumerate(instruments.items(), 1):
            scraper_logger.info(f"[{idx}/{len(instruments)}] Processing {name}")
            
            data = self.scrape_instrument(name, info)
            if data:
                results.append(data)
            
            # Delay between requests
            if idx < len(instruments):
                self._human_delay()
        
        elapsed = time.time() - start_time
        success_rate = (len(results) / len(instruments)) * 100 if instruments else 0
        
        scraper_logger.info("=" * 60)
        scraper_logger.info("SCRAPING COMPLETE")
        scraper_logger.info(f"Success: {len(results)}/{len(instruments)} ({success_rate:.1f}%)")
        scraper_logger.info(f"Failed: {self.failed_count}")
        scraper_logger.info(f"Total requests: {self.request_count}")
        scraper_logger.info(f"Elapsed: {elapsed:.1f}s")
        scraper_logger.info("=" * 60)
        
        return results

    def calculate_market_stress(self, market_data: List[Dict]) -> float:
        """Calculate market stress score"""
        if not market_data:
            return 0.0
        
        factors = []
        for asset in market_data:
            change_pct = asset.get("change_percent", 0.0)
            priority = asset.get("priority", 3)
            
            # VIX has special weighting
            if "VIX" in asset["asset"]:
                factors.append(min(asset["price"] / 10, 10) * 3.0)
            
            # Movement factor based on percentage change and priority
            move_factor = min(abs(change_pct) / 2, 10) * (4 - priority)
            factors.append(move_factor)
            
            # Extra weight for large negative moves
            if change_pct < -3:
                factors.append(min(abs(change_pct), 10) * 1.5)
        
        avg_stress = sum(factors) / len(factors) if factors else 0.0
        return min(avg_stress, 10.0)

    def create_articles(self, market_data: List[Dict]) -> List[Dict]:
        """Create articles from market data"""
        if not market_data:
            scraper_logger.warning("No market data for articles")
            return []
        
        timestamp = datetime.now().isoformat()
        stress_score = self.calculate_market_stress(market_data)
        
        # Create overview article
        overview_lines = [
            f"Market Stress Score: {stress_score:.1f}/10",
            f"Data Points: {len(market_data)}",
            f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        
        # Group by asset type
        by_type = {}
        for asset in market_data:
            by_type.setdefault(asset["type"], []).append(asset)
        
        for asset_type, assets in sorted(by_type.items()):
            overview_lines.append(f"{asset_type.upper()} MARKETS:")
            for asset in sorted(assets, key=lambda x: abs(x.get("change_percent", 0)), reverse=True):
                overview_lines.append(
                    f"  {asset['asset']}: ${asset['price']:.2f} "
                    f"({asset['change_percent']:+.2f}%, ${asset['change']:+.2f})"
                )
            overview_lines.append("")
        
        overview_article = {
            "headline": f"Market Overview - Stress Score: {stress_score:.1f}/10",
            "snippet": "\n".join(overview_lines)[:2000],
            "timestamp": timestamp,
            "asset_tags": [asset["asset"] for asset in market_data],
            "url": "https://finance.yahoo.com",
            "source": "yahoo finance",
            "scraped_at": timestamp,
            "market_st
