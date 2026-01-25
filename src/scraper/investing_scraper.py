"""
Market Data Scraper - SAFE IMPORT VERSION
Handles missing yfinance gracefully
"""
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd
import json
import requests
from urllib.parse import quote

try:
    from src.utils.logger import scraper_logger
except ImportError:
    import logging
    scraper_logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

# Safe import for yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    scraper_logger.warning("yfinance not available - using alternative method")


class SafeInvestingScraper:
    """
    Multi-source API scraper for real market data.
    Falls back to direct Yahoo Finance API if yfinance unavailable.
    """
    
    # Yahoo Finance symbol mapping
    YAHOO_SYMBOLS = {
        # Major US Indices
        'S&P 500': {
            'symbol': '^GSPC',
            'type': 'index',
            'priority': 1,
        },
        'Dow Jones': {
            'symbol': '^DJI',
            'type': 'index',
            'priority': 1,
        },
        'NASDAQ': {
            'symbol': '^IXIC',
            'type': 'index',
            'priority': 1,
        },
        'VIX': {
            'symbol': '^VIX',
            'type': 'volatility',
            'priority': 1,
        },
        'Russell 2000': {
            'symbol': '^RUT',
            'type': 'index',
            'priority': 2,
        },
        
        # International Indices
        'FTSE 100': {
            'symbol': '^FTSE',
            'type': 'index',
            'priority': 2,
        },
        'DAX': {
            'symbol': '^GDAXI',
            'type': 'index',
            'priority': 2,
        },
        'Nikkei 225': {
            'symbol': '^N225',
            'type': 'index',
            'priority': 2,
        },
        
        # Commodities
        'Gold': {
            'symbol': 'GC=F',
            'type': 'commodity',
            'priority': 1,
        },
        'Crude Oil': {
            'symbol': 'CL=F',
            'type': 'commodity',
            'priority': 1,
        },
        'Silver': {
            'symbol': 'SI=F',
            'type': 'commodity',
            'priority': 2,
        },
        'Natural Gas': {
            'symbol': 'NG=F',
            'type': 'commodity',
            'priority': 2,
        },
        
        # Forex
        'EUR/USD': {
            'symbol': 'EURUSD=X',
            'type': 'forex',
            'priority': 1,
        },
        'GBP/USD': {
            'symbol': 'GBPUSD=X',
            'type': 'forex',
            'priority': 2,
        },
        'USD/JPY': {
            'symbol': 'JPY=X',
            'type': 'forex',
            'priority': 2,
        },
        
        # Cryptocurrencies
        'Bitcoin': {
            'symbol': 'BTC-USD',
            'type': 'crypto',
            'priority': 1,
        },
        'Ethereum': {
            'symbol': 'ETH-USD',
            'type': 'crypto',
            'priority': 2,
        },
    }
    
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    ]
    
    def __init__(self, delay_range: Tuple[float, float] = (0.5, 1.5), max_retries: int = 3):
        """Initialize scraper with fallback support."""
        self.delay_range = delay_range
        self.max_retries = max_retries
        self.request_count = 0
        self.failed_count = 0
        self.yahoo_success = 0
        
        scraper_logger.info(f"Initialized SafeInvestingScraper with {len(self.YAHOO_SYMBOLS)} instruments")
        
        if YFINANCE_AVAILABLE:
            scraper_logger.info("Using yfinance library (preferred)")
        else:
            scraper_logger.warning("Using Yahoo Finance Chart API (fallback)")
    
    def _respectful_delay(self):
        """Wait between requests."""
        delay = random.uniform(self.delay_range[0], self.delay_range[1])
        time.sleep(delay)
    
    def _fetch_yahoo_yfinance(self, name: str, info: Dict) -> Optional[Dict]:
        """
        Fetch using yfinance library (preferred method).
        """
        if not YFINANCE_AVAILABLE:
            return None
        
        if name not in self.YAHOO_SYMBOLS:
            return None
        
        symbol = self.YAHOO_SYMBOLS[name]['symbol']
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='5d')
            
            if hist.empty or len(hist) < 2:
                scraper_logger.warning(f"Insufficient data for {name}")
                return None
            
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2]
            change = current_price - prev_close
            change_percent = (change / prev_close * 100) if prev_close != 0 else 0
            
            self.yahoo_success += 1
            
            scraper_logger.info(
                f"✓ Yahoo (yfinance): {name} = ${current_price:.2f} "
                f"({change_percent:+.2f}%)"
            )
            
            return {
                'price': float(current_price),
                'change': float(change),
                'change_percent': float(change_percent),
                'prev_close': float(prev_close),
                'day_high': float(hist['High'].iloc[-1]),
                'day_low': float(hist['Low'].iloc[-1]),
                'source': 'yahoo_finance'
            }
            
        except Exception as e:
            scraper_logger.warning(f"yfinance error for {name}: {e}")
            return None
    
    def _fetch_yahoo_api(self, name: str, info: Dict) -> Optional[Dict]:
        """
        Fetch using Yahoo Finance Chart API directly (fallback).
        """
        if name not in self.YAHOO_SYMBOLS:
            return None
        
        symbol = self.YAHOO_SYMBOLS[name]['symbol']
        
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {
                'interval': '1d',
                'range': '5d',
            }
            
            headers = {
                'User-Agent': random.choice(self.USER_AGENTS),
                'Accept': 'application/json',
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=15)
            
            if response.status_code != 200:
                scraper_logger.warning(f"Yahoo API HTTP {response.status_code} for {name}")
                return None
            
            data = response.json()
            
            if 'chart' not in data or 'result' not in data['chart']:
                return None
            
            result = data['chart']['result'][0]
            meta = result.get('meta', {})
            
            current_price = meta.get('regularMarketPrice')
            prev_close = meta.get('chartPreviousClose') or meta.get('previousClose')
            
            if current_price is None:
                return None
            
            if prev_close is None:
                quotes = result.get('indicators', {}).get('quote', [{}])[0]
                closes = quotes.get('close', [])
                if closes and len(closes) >= 2:
                    prev_close = closes[-2] if closes[-2] is not None else closes[-1]
                else:
                    prev_close = current_price
            
            change = current_price - prev_close
            change_percent = (change / prev_close * 100) if prev_close != 0 else 0
            
            self.yahoo_success += 1
            
            scraper_logger.info(
                f"✓ Yahoo (API): {name} = ${current_price:.2f} "
                f"({change_percent:+.2f}%)"
            )
            
            return {
                'price': float(current_price),
                'change': float(change),
                'change_percent': float(change_percent),
                'prev_close': float(prev_close),
                'day_high': float(meta.get('regularMarketDayHigh', current_price)),
                'day_low': float(meta.get('regularMarketDayLow', current_price)),
                'source': 'yahoo_finance_api'
            }
            
        except Exception as e:
            scraper_logger.warning(f"Yahoo API error for {name}: {e}")
            return None
    
    def scrape_instrument(self, name: str, info: Dict) -> Optional[Dict]:
        """
        Scrape data for a single instrument.
        Tries yfinance first, falls back to direct API.
        """
        scraper_logger.info(f"Fetching: {name} ({info['type']})")
        
        # Try yfinance first if available
        if YFINANCE_AVAILABLE:
            price_data = self._fetch_yahoo_yfinance(name, info)
        else:
            price_data = None
        
        # Fallback to direct API
        if not price_data:
            price_data = self._fetch_yahoo_api(name, info)
        
        if not price_data:
            scraper_logger.error(f"❌ Failed to get data for {name}")
            self.failed_count += 1
            return None
        
        # Build result
        result = {
            'asset': name,
            'asset_type': info['type'],  # ✅ Correct column name
            'priority': info.get('priority', 3),
            'timestamp': datetime.now().isoformat(),
            'source': price_data['source'],
            'price': price_data['price'],
            'change': price_data['change'],
            'change_percent': price_data['change_percent'],
            'prev_close': price_data['prev_close'],
            'day_low': price_data.get('day_low'),
            'day_high': price_data.get('day_high'),
        }
        
        self.request_count += 1
        
        return result
    
    def scrape_all(self, priority_filter: Optional[int] = None) -> List[Dict]:
        """Fetch all instruments."""
        scraper_logger.info("="*70)
        scraper_logger.info("STARTING REAL MARKET DATA COLLECTION")
        scraper_logger.info("="*70)
        scraper_logger.info("Source: Yahoo Finance")
        scraper_logger.info("="*70)
        
        # Filter by priority
        instruments = {
            name: info for name, info in self.YAHOO_SYMBOLS.items()
        }
        
        if priority_filter:
            instruments = {
                name: info for name, info in instruments.items()
                if info.get('priority', 3) <= priority_filter
            }
            scraper_logger.info(f"Priority filter <= {priority_filter}: {len(instruments)} instruments")
        
        market_data = []
        start_time = time.time()
        
        for i, (name, info) in enumerate(instruments.items(), 1):
            scraper_logger.info(f"\n[{i}/{len(instruments)}] Processing: {name}")
            
            data = self.scrape_instrument(name, info)
            
            if data:
                market_data.append(data)
            
            if i < len(instruments):
                self._respectful_delay()
        
        elapsed = time.time() - start_time
        success_rate = (len(market_data) / len(instruments) * 100) if instruments else 0
        
        scraper_logger.info("\n" + "="*70)
        scraper_logger.info("DATA COLLECTION COMPLETE")
        scraper_logger.info("="*70)
        scraper_logger.info(f"Success: {len(market_data)}/{len(instruments)} ({success_rate:.1f}%)")
        scraper_logger.info(f"Yahoo Finance: {self.yahoo_success}")
        scraper_logger.info(f"Failed: {self.failed_count}")
        scraper_logger.info(f"Time elapsed: {elapsed:.1f}s")
        scraper_logger.info("="*70)
        
        return market_data
    
    def calculate_market_stress(self, market_data: List[Dict]) -> float:
        """Calculate market stress score."""
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
        """Convert market data to article format."""
        if not market_data:
            scraper_logger.warning("No market data to convert")
            return []
        
        articles = []
        timestamp = datetime.now().isoformat()
        stress_score = self.calculate_market_stress(market_data)
        
        # Market overview
        overview_lines = [
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
        
        # Main article
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
                'yahoo_success': self.yahoo_success,
                'failures': self.failed_count,
                'success_rate': f"{(len(market_data)/(len(market_data)+self.failed_count)*100):.1f}%" if market_data else "0%"
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
                    f"{data['asset']} ({data['asset_type']}) is trading at ${data['price']:.2f}, "
                    f"{direction[:-1]}ing {change_pct:.2f}% from ${data['prev_close']:.2f}. "
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
                    'url': f"https://finance.yahoo.com",
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
        """Save to bronze layer."""
        if not articles:
            scraper_logger.error("No articles to save")
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
    """Main scraping function."""
    scraper = SafeInvestingScraper(delay_range=(0.5, 1.5), max_retries=3)
    
    try:
        market_data = scraper.scrape_all(priority_filter=priority_filter)
        
        if market_data:
            articles = scraper.create_articles(market_data)
            filepath = scraper.save_to_bronze(articles)
            return filepath
        else:
            scraper_logger.error("NO REAL MARKET DATA COLLECTED")
            return None
            
    except KeyboardInterrupt:
        scraper_logger.warning("\nInterrupted by user")
        return None
    except Exception as e:
        scraper_logger.error(f"Collection failed: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    result = scrape_investing_data(priority_filter=2)
    
    if result:
        print(f"\n✓ Data saved to: {result}")
        df = pd.read_parquet(result)
        print(f"Data points: {len(df)}")
    else:
        print("\n❌ Collection failed")
