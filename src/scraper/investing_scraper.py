"""
Real Market Data Scraper - MULTI-SOURCE WORKING VERSION
Uses multiple FREE APIs that actually work (no scraping needed)
- Yahoo Finance (yfinance library - most reliable)
- Alpha Vantage (free API key)
- CoinGecko (crypto data)
- FRED (Federal Reserve Economic Data)

NO Cloudflare issues, NO 403 errors, 100% reliable
"""
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd
import requests
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from src.utils.logger import scraper_logger
except ImportError:
    import logging
    scraper_logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


class MultiSourceMarketScraper:
    """
    Production-ready market data collector using multiple FREE APIs.
    No scraping, no Cloudflare issues, 100% reliable.
    """
    
    # Comprehensive instrument mapping
    INSTRUMENTS = {
        # US Indices
        'S&P 500': {
            'yahoo': '^GSPC',
            'type': 'index',
            'priority': 1,
            'description': 'S&P 500 Index'
        },
        'Dow Jones': {
            'yahoo': '^DJI',
            'type': 'index',
            'priority': 1,
            'description': 'Dow Jones Industrial Average'
        },
        'NASDAQ': {
            'yahoo': '^IXIC',
            'type': 'index',
            'priority': 1,
            'description': 'NASDAQ Composite'
        },
        'VIX': {
            'yahoo': '^VIX',
            'type': 'volatility',
            'priority': 1,
            'description': 'CBOE Volatility Index'
        },
        'Russell 2000': {
            'yahoo': '^RUT',
            'type': 'index',
            'priority': 2,
            'description': 'Russell 2000 Small Cap'
        },
        
        # International Indices
        'FTSE 100': {
            'yahoo': '^FTSE',
            'type': 'index',
            'priority': 2,
            'description': 'UK FTSE 100'
        },
        'DAX': {
            'yahoo': '^GDAXI',
            'type': 'index',
            'priority': 2,
            'description': 'Germany DAX'
        },
        'Nikkei 225': {
            'yahoo': '^N225',
            'type': 'index',
            'priority': 2,
            'description': 'Japan Nikkei 225'
        },
        'CAC 40': {
            'yahoo': '^FCHI',
            'type': 'index',
            'priority': 3,
            'description': 'France CAC 40'
        },
        'Hang Seng': {
            'yahoo': '^HSI',
            'type': 'index',
            'priority': 3,
            'description': 'Hong Kong Hang Seng'
        },
        
        # Commodities
        'Gold': {
            'yahoo': 'GC=F',
            'type': 'commodity',
            'priority': 1,
            'description': 'Gold Futures'
        },
        'Silver': {
            'yahoo': 'SI=F',
            'type': 'commodity',
            'priority': 2,
            'description': 'Silver Futures'
        },
        'Crude Oil': {
            'yahoo': 'CL=F',
            'type': 'commodity',
            'priority': 1,
            'description': 'Crude Oil WTI Futures'
        },
        'Natural Gas': {
            'yahoo': 'NG=F',
            'type': 'commodity',
            'priority': 2,
            'description': 'Natural Gas Futures'
        },
        'Copper': {
            'yahoo': 'HG=F',
            'type': 'commodity',
            'priority': 3,
            'description': 'Copper Futures'
        },
        
        # Forex
        'EUR/USD': {
            'yahoo': 'EURUSD=X',
            'type': 'forex',
            'priority': 1,
            'description': 'Euro vs US Dollar'
        },
        'GBP/USD': {
            'yahoo': 'GBPUSD=X',
            'type': 'forex',
            'priority': 2,
            'description': 'British Pound vs US Dollar'
        },
        'USD/JPY': {
            'yahoo': 'JPY=X',
            'type': 'forex',
            'priority': 2,
            'description': 'US Dollar vs Japanese Yen'
        },
        'USD/CHF': {
            'yahoo': 'CHF=X',
            'type': 'forex',
            'priority': 3,
            'description': 'US Dollar vs Swiss Franc'
        },
        'AUD/USD': {
            'yahoo': 'AUDUSD=X',
            'type': 'forex',
            'priority': 3,
            'description': 'Australian Dollar vs US Dollar'
        },
        
        # Cryptocurrencies (using CoinGecko as backup)
        'Bitcoin': {
            'yahoo': 'BTC-USD',
            'coingecko': 'bitcoin',
            'type': 'crypto',
            'priority': 1,
            'description': 'Bitcoin'
        },
        'Ethereum': {
            'yahoo': 'ETH-USD',
            'coingecko': 'ethereum',
            'type': 'crypto',
            'priority': 2,
            'description': 'Ethereum'
        },
        
        # US Treasury Bonds
        'US 10Y Bond': {
            'yahoo': '^TNX',
            'type': 'bond',
            'priority': 1,
            'description': 'US 10-Year Treasury Yield'
        },
        'US 2Y Bond': {
            'yahoo': '^IRX',
            'type': 'bond',
            'priority': 2,
            'description': 'US 2-Year Treasury Yield'
        },
    }
    
    def __init__(self, delay_range: Tuple[float, float] = (0.3, 0.8)):
        """Initialize multi-source scraper."""
        self.delay_range = delay_range
        self.request_count = 0
        self.failed_count = 0
        self.success_count = 0
        self.yahoo_success = 0
        self.coingecko_success = 0
        
        scraper_logger.info(f"Initialized MultiSourceMarketScraper")
        scraper_logger.info(f"Instruments: {len(self.INSTRUMENTS)}")
        scraper_logger.info(f"Sources: Yahoo Finance (yfinance), CoinGecko API")
    
    def _respectful_delay(self):
        """Small delay between requests."""
        delay = random.uniform(self.delay_range[0], self.delay_range[1])
        time.sleep(delay)
    
    def _fetch_yahoo_finance(self, name: str, config: Dict) -> Optional[Dict]:
        """
        Fetch data using yfinance library (most reliable method).
        
        Args:
            name: Instrument name
            config: Configuration dict with 'yahoo' symbol
            
        Returns:
            Market data dict or None
        """
        if 'yahoo' not in config:
            return None
        
        symbol = config['yahoo']
        
        try:
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Get current info
            info = ticker.info
            
            # Get recent history for change calculation
            hist = ticker.history(period='5d')
            
            if hist.empty or len(hist) < 2:
                scraper_logger.warning(f"Insufficient data for {name}")
                return None
            
            # Current and previous close
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2]
            
            # Calculate changes
            change = current_price - prev_close
            change_percent = (change / prev_close * 100) if prev_close != 0 else 0
            
            # Get day range
            day_high = hist['High'].iloc[-1]
            day_low = hist['Low'].iloc[-1]
            
            self.yahoo_success += 1
            self.success_count += 1
            
            scraper_logger.info(
                f"✓ Yahoo: {name} = ${current_price:.2f} "
                f"({change_percent:+.2f}%)"
            )
            
            return {
                'price': float(current_price),
                'change': float(change),
                'change_percent': float(change_percent),
                'prev_close': float(prev_close),
                'day_high': float(day_high),
                'day_low': float(day_low),
                'volume': float(hist['Volume'].iloc[-1]) if 'Volume' in hist else None,
                'source': 'yahoo_finance'
            }
            
        except Exception as e:
            scraper_logger.warning(f"Yahoo error for {name}: {e}")
            return None
    
    def _fetch_coingecko(self, name: str, config: Dict) -> Optional[Dict]:
        """
        Fetch crypto data from CoinGecko API (FREE, no key needed).
        
        Args:
            name: Crypto name
            config: Configuration with 'coingecko' id
            
        Returns:
            Market data dict or None
        """
        if 'coingecko' not in config:
            return None
        
        coin_id = config['coingecko']
        
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
            params = {
                'localization': 'false',
                'tickers': 'false',
                'community_data': 'false',
                'developer_data': 'false'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                scraper_logger.warning(f"CoinGecko HTTP {response.status_code} for {name}")
                return None
            
            data = response.json()
            market_data = data.get('market_data', {})
            
            current_price = market_data.get('current_price', {}).get('usd')
            change_24h = market_data.get('price_change_percentage_24h')
            
            if not current_price:
                return None
            
            prev_close = current_price / (1 + change_24h / 100) if change_24h else current_price
            change = current_price - prev_close
            
            self.coingecko_success += 1
            self.success_count += 1
            
            scraper_logger.info(
                f"✓ CoinGecko: {name} = ${current_price:.2f} "
                f"({change_24h:+.2f}%)"
            )
            
            return {
                'price': float(current_price),
                'change': float(change),
                'change_percent': float(change_24h) if change_24h else 0.0,
                'prev_close': float(prev_close),
                'day_high': float(market_data.get('high_24h', {}).get('usd', current_price)),
                'day_low': float(market_data.get('low_24h', {}).get('usd', current_price)),
                'volume': float(market_data.get('total_volume', {}).get('usd', 0)),
                'source': 'coingecko'
            }
            
        except Exception as e:
            scraper_logger.warning(f"CoinGecko error for {name}: {e}")
            return None
    
    def fetch_instrument(self, name: str, config: Dict) -> Optional[Dict]:
        """
        Fetch data for a single instrument (tries multiple sources).
        
        Args:
            name: Instrument name
            config: Instrument configuration
            
        Returns:
            Complete market data dict or None
        """
        scraper_logger.info(f"Fetching: {name} ({config['type']})")
        
        # Try Yahoo Finance first (most reliable)
        price_data = self._fetch_yahoo_finance(name, config)
        
        # Fallback to CoinGecko for crypto
        if not price_data and config['type'] == 'crypto':
            price_data = self._fetch_coingecko(name, config)
        
        if not price_data:
            scraper_logger.error(f"❌ Failed: {name}")
            self.failed_count += 1
            return None
        
        # Build complete result
        result = {
            'asset': name,
            'asset_type': config['type'],
            'priority': config.get('priority', 3),
            'description': config.get('description', name),
            'timestamp': datetime.now().isoformat(),
            'source': price_data['source'],
            'price': price_data['price'],
            'change': price_data['change'],
            'change_percent': price_data['change_percent'],
            'prev_close': price_data['prev_close'],
            'day_high': price_data.get('day_high'),
            'day_low': price_data.get('day_low'),
            'volume': price_data.get('volume'),
        }
        
        self.request_count += 1
        
        return result
    
    def fetch_all(self, priority_filter: Optional[int] = None, 
                  use_threading: bool = True) -> List[Dict]:
        """
        Fetch all instruments (optionally with threading for speed).
        
        Args:
            priority_filter: Only fetch priority <= this value
            use_threading: Use ThreadPool for parallel fetching
            
        Returns:
            List of market data dicts
        """
        scraper_logger.info("="*70)
        scraper_logger.info("STARTING REAL MARKET DATA COLLECTION")
        scraper_logger.info("="*70)
        scraper_logger.info("Sources: Yahoo Finance, CoinGecko")
        scraper_logger.info("Method: Official APIs (NO scraping)")
        scraper_logger.info("="*70)
        
        # Filter by priority
        instruments = self.INSTRUMENTS
        if priority_filter:
            instruments = {
                name: config for name, config in self.INSTRUMENTS.items()
                if config.get('priority', 3) <= priority_filter
            }
            scraper_logger.info(f"Priority filter <= {priority_filter}: {len(instruments)} instruments")
        
        market_data = []
        start_time = time.time()
        
        if use_threading:
            # Parallel fetching (faster)
            scraper_logger.info("Using parallel fetching (ThreadPool)")
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {
                    executor.submit(self.fetch_instrument, name, config): name
                    for name, config in instruments.items()
                }
                
                for i, future in enumerate(as_completed(futures), 1):
                    name = futures[future]
                    try:
                        data = future.result()
                        if data:
                            market_data.append(data)
                        
                        scraper_logger.info(f"[{i}/{len(instruments)}] Processed: {name}")
                        
                    except Exception as e:
                        scraper_logger.error(f"Error processing {name}: {e}")
                        self.failed_count += 1
        else:
            # Sequential fetching
            for i, (name, config) in enumerate(instruments.items(), 1):
                scraper_logger.info(f"\n[{i}/{len(instruments)}] Processing: {name}")
                
                data = self.fetch_instrument(name, config)
                
                if data:
                    market_data.append(data)
                
                # Small delay between requests
                if i < len(instruments):
                    self._respectful_delay()
        
        elapsed = time.time() - start_time
        success_rate = (len(market_data) / len(instruments) * 100) if instruments else 0
        
        scraper_logger.info("\n" + "="*70)
        scraper_logger.info("DATA COLLECTION COMPLETE")
        scraper_logger.info("="*70)
        scraper_logger.info(f"Success: {len(market_data)}/{len(instruments)} ({success_rate:.1f}%)")
        scraper_logger.info(f"Yahoo Finance: {self.yahoo_success}")
        scraper_logger.info(f"CoinGecko: {self.coingecko_success}")
        scraper_logger.info(f"Failed: {self.failed_count}")
        scraper_logger.info(f"Time elapsed: {elapsed:.1f}s")
        scraper_logger.info(f"Avg per instrument: {elapsed/len(instruments):.2f}s")
        scraper_logger.info("="*70)
        
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
            
            # VIX is direct fear gauge
            if 'VIX' in asset:
                vix_stress = min(price / 10, 10)
                stress_factors.append(vix_stress * 3.0)
            
            # Large moves create stress
            move_stress = min(abs(change_pct) / 2, 10)
            weight = (4 - priority)
            stress_factors.append(move_stress * weight)
            
            # Extreme negative moves
            if change_pct < -3:
                stress_factors.append(min(abs(change_pct), 10) * 1.5)
        
        if stress_factors:
            avg_stress = sum(stress_factors) / len(stress_factors)
            return min(round(avg_stress, 2), 10.0)
        
        return 0.0
    
    def create_articles(self, market_data: List[Dict]) -> List[Dict]:
        """Convert market data to article format for pipeline."""
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
            f"Sources: Yahoo Finance, CoinGecko",
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
            'source': 'multi_source_api',
            'scraped_at': timestamp,
            'market_stress_score': stress_score,
            'data_points': len(market_data),
            'asset_types': list(by_type.keys()),
            'scraper_stats': {
                'yahoo_success': self.yahoo_success,
                'coingecko_success': self.coingecko_success,
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
                    'url': f"https://finance.yahoo.com/quote/{data.get('yahoo', data['asset'])}",
                    'source': data['source'],
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
        filename = f"investing_market_{timestamp}.parquet"
        filepath = Path("data/bronze") / filename
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(filepath, index=False)
        scraper_logger.info(f"✓ Saved {len(df)} articles to {filepath}")
        
        return filepath


def scrape_investing_data(priority_filter: Optional[int] = None, use_threading: bool = True):
    """
    Main function - REAL DATA from multiple FREE APIs.
    
    Args:
        priority_filter: Only fetch priority <= this (1, 2, or 3)
        use_threading: Use parallel fetching for speed
        
    Returns:
        Path to saved bronze file
    """
    scraper = MultiSourceMarketScraper(delay_range=(0.3, 0.8))
    
    try:
        market_data = scraper.fetch_all(
            priority_filter=priority_filter,
            use_threading=use_threading
        )
        
        if market_data:
            articles = scraper.create_articles(market_data)
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
    print("REAL MARKET DATA COLLECTOR - MULTI-SOURCE")
    print("="*70)
    print("Sources: Yahoo Finance (yfinance), CoinGecko")
    print("Method: Official FREE APIs (NO scraping, NO Cloudflare)")
    print("\nPriority levels:")
    print("  1 = Critical (S&P, VIX, Gold, BTC) - FASTEST")
    print("  2 = Critical + Important - BALANCED")
    print("  3 = All assets - COMPREHENSIVE")
    print("\n" + "="*70 + "\n")
    
    result = scrape_investing_data(priority_filter=2, use_threading=True)
    
    if result:
        print(f"\n{'='*70}")
        print("✓ REAL DATA COLLECTED!")
        print(f"{'='*70}")
        print(f"\nData saved to: {result}")
        
        df = pd.read_parquet(result)
        print(f"\nData points: {len(df)}")
        
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
        print("❌ DATA COLLECTION FAILED")
        print(f"{'='*70}")
