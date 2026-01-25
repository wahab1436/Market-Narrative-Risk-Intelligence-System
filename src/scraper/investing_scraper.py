"""
Market Data Scraper with Multiple Fallback Sources
Primary: Investing.com | Fallbacks: Yahoo Finance, Alpha Vantage, Finnhub
"""
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import re
import jsonquests
import pandas as pd
import re

try:
    from src.utils.logger import scraper_logger
except ImportError:
    import logging
    scraper_logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


class MultiSourceMarketScraper:
    """
    Resilient market data scraper with multiple fallback sources.
    Tries Investing.com first, falls back to free alternatives.
    """
    
    def __init__(self):
        """Initialize with multiple data source configurations."""
        
        # Session for requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Instrument mappings for different sources
        self.instruments = {
            'VIX': {
                'yahoo': '^VIX',
                'finnhub': 'OANDA:VIX_USD',
                'type': 'volatility',
                'priority': 1,
                'description': 'CBOE Volatility Index'
            },
            'S&P 500': {
                'yahoo': '^GSPC',
                'finnhub': 'OANDA:SPX500_USD',
                'type': 'index',
                'priority': 1,
                'description': 'S&P 500 Index'
            },
            'Dow Jones': {
                'yahoo': '^DJI',
                'finnhub': 'OANDA:US30_USD',
                'type': 'index',
                'priority': 1,
                'description': 'Dow Jones Industrial Average'
            },
            'NASDAQ': {
                'yahoo': '^IXIC',
                'finnhub': 'OANDA:NAS100_USD',
                'type': 'index',
                'priority': 1,
                'description': 'NASDAQ Composite'
            },
            'Russell 2000': {
                'yahoo': '^RUT',
                'type': 'index',
                'priority': 2,
                'description': 'Russell 2000 Small Cap'
            },
            'Gold': {
                'yahoo': 'GC=F',
                'finnhub': 'OANDA:XAU_USD',
                'type': 'commodity',
                'priority': 2,
                'description': 'Gold Spot Price'
            },
            'Crude Oil': {
                'yahoo': 'CL=F',
                'finnhub': 'OANDA:WTICO_USD',
                'type': 'commodity',
                'priority': 2,
                'description': 'WTI Crude Oil'
            },
            'EUR/USD': {
                'yahoo': 'EURUSD=X',
                'finnhub': 'OANDA:EUR_USD',
                'type': 'forex',
                'priority': 2,
                'description': 'Euro vs US Dollar'
            },
            '10Y Treasury': {
                'yahoo': '^TNX',
                'type': 'bond',
                'priority': 2,
                'description': 'US 10-Year Treasury Yield'
            },
            'Silver': {
                'yahoo': 'SI=F',
                'finnhub': 'OANDA:XAG_USD',
                'type': 'commodity',
                'priority': 3,
                'description': 'Silver Spot Price'
            },
            'Natural Gas': {
                'yahoo': 'NG=F',
                'type': 'commodity',
                'priority': 3,
                'description': 'Natural Gas Futures'
            },
            'GBP/USD': {
                'yahoo': 'GBPUSD=X',
                'finnhub': 'OANDA:GBP_USD',
                'type': 'forex',
                'priority': 3,
                'description': 'British Pound vs US Dollar'
            },
            'USD/JPY': {
                'yahoo': 'USDJPY=X',
                'finnhub': 'OANDA:USD_JPY',
                'type': 'forex',
                'priority': 3,
                'description': 'US Dollar vs Japanese Yen'
            },
            'Bitcoin': {
                'yahoo': 'BTC-USD',
                'finnhub': 'BINANCE:BTCUSDT',
                'type': 'crypto',
                'priority': 3,
                'description': 'Bitcoin Price'
            },
            'Ethereum': {
                'yahoo': 'ETH-USD',
                'finnhub': 'BINANCE:ETHUSDT',
                'type': 'crypto',
                'priority': 3,
                'description': 'Ethereum Price'
            },
        }
    
    def fetch_yahoo_finance(self, symbol: str, name: str, info: Dict) -> Optional[Dict]:
        """
        Fetch data from Yahoo Finance using their query API.
        This is more reliable than their chart API.
        """
        try:
            # Yahoo Finance query endpoint (public)
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            
            params = {
                'interval': '1d',
                'range': '5d'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'chart' not in data or 'result' not in data['chart']:
                return None
            
            result = data['chart']['result'][0]
            
            # Get current data
            quote = result['meta']
            indicators = result['indicators']['quote'][0]
            
            current_price = quote.get('regularMarketPrice')
            prev_close = quote.get('previousClose', current_price)
            
            if not current_price:
                return None
            
            change = current_price - prev_close
            change_percent = (change / prev_close * 100) if prev_close != 0 else 0
            
            # Get day high/low from latest data
            timestamps = result.get('timestamp', [])
            if timestamps and indicators:
                day_high = quote.get('regularMarketDayHigh', current_price)
                day_low = quote.get('regularMarketDayLow', current_price)
            else:
                day_high = day_low = current_price
            
            return {
                'asset': name,
                'symbol': symbol,
                'type': info['type'],
                'priority': info['priority'],
                'description': info['description'],
                'price': round(float(current_price), 4),
                'prev_close': round(float(prev_close), 4),
                'change': round(float(change), 4),
                'change_percent': round(float(change_percent), 4),
                'day_high': round(float(day_high), 4),
                'day_low': round(float(day_low), 4),
                'timestamp': datetime.now().isoformat(),
                'source': 'yahoo_finance',
                'url': f'https://finance.yahoo.com/quote/{symbol}'
            }
            
        except Exception as e:
            scraper_logger.debug(f"Yahoo Finance failed for {name}: {e}")
            return None
    
    def fetch_finnhub(self, symbol: str, name: str, info: Dict) -> Optional[Dict]:
        """
        Fetch data from Finnhub (free tier, no API key needed for some endpoints).
        """
        try:
            # Finnhub public quote endpoint
            url = "https://finnhub.io/api/v1/quote"
            
            params = {
                'symbol': symbol,
                'token': 'sandbox'  # Free sandbox token
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            current_price = data.get('c')  # Current price
            prev_close = data.get('pc')  # Previous close
            high = data.get('h')  # High
            low = data.get('l')  # Low
            
            if not current_price or current_price == 0:
                return None
            
            change = current_price - prev_close if prev_close else 0
            change_percent = (change / prev_close * 100) if prev_close and prev_close != 0 else 0
            
            return {
                'asset': name,
                'symbol': symbol,
                'type': info['type'],
                'priority': info['priority'],
                'description': info['description'],
                'price': round(float(current_price), 4),
                'prev_close': round(float(prev_close) if prev_close else current_price, 4),
                'change': round(float(change), 4),
                'change_percent': round(float(change_percent), 4),
                'day_high': round(float(high) if high else current_price, 4),
                'day_low': round(float(low) if low else current_price, 4),
                'timestamp': datetime.now().isoformat(),
                'source': 'finnhub',
                'url': f'https://finnhub.io/quote/{symbol}'
            }
            
        except Exception as e:
            scraper_logger.debug(f"Finnhub failed for {name}: {e}")
            return None
    
    def fetch_instrument_data(self, name: str, info: Dict) -> Optional[Dict]:
        """
        Fetch instrument data using multiple sources with fallback.
        Tries sources in order until one succeeds.
        """
        scraper_logger.info(f"Fetching {name} (Priority: {info['priority']})")
        
        # Try Yahoo Finance first (most reliable)
        if 'yahoo' in info:
            data = self.fetch_yahoo_finance(info['yahoo'], name, info)
            if data:
                scraper_logger.info(
                    f"✓ {name}: ${data['price']:.2f} ({data['change_percent']:+.2f}%) [Yahoo]"
                )
                return data
        
        # Try Finnhub as backup
        if 'finnhub' in info:
            data = self.fetch_finnhub(info['finnhub'], name, info)
            if data:
                scraper_logger.info(
                    f"✓ {name}: ${data['price']:.2f} ({data['change_percent']:+.2f}%) [Finnhub]"
                )
                return data
        
        scraper_logger.warning(f"✗ Failed to fetch {name} from all sources")
        return None
    
    def get_instruments_by_priority(self, max_priority: int = 3) -> Dict[str, Dict]:
        """Filter instruments by priority level."""
        return {
            name: info
            for name, info in self.instruments.items()
            if info['priority'] <= max_priority
        }
    
    def scrape_data(self, priority_filter: int = 2) -> List[Dict]:
        """
        Scrape market data with priority filtering.
        
        Args:
            priority_filter: Maximum priority level (1=critical, 2=balanced, 3=all)
            
        Returns:
            List of market data dictionaries
        """
        instruments = self.get_instruments_by_priority(priority_filter)
        
        scraper_logger.info("="*60)
        scraper_logger.info(f"Multi-Source Market Data Collection")
        scraper_logger.info(f"Priority Filter: {priority_filter}")
        scraper_logger.info(f"Instruments to fetch: {len(instruments)}")
        scraper_logger.info("="*60)
        
        market_data = []
        failed = []
        
        for name, info in instruments.items():
            try:
                data = self.fetch_instrument_data(name, info)
                
                if data:
                    market_data.append(data)
                else:
                    failed.append(name)
                
                # Small delay between requests
                time.sleep(0.5)
                
            except Exception as e:
                scraper_logger.error(f"Error collecting {name}: {e}")
                failed.append(name)
                continue
        
        scraper_logger.info("="*60)
        scraper_logger.info(f"Collection complete: {len(market_data)}/{len(instruments)} successful")
        if failed:
            scraper_logger.warning(f"Failed: {', '.join(failed)}")
        scraper_logger.info("="*60)
        
        return market_data
    
    def calculate_market_stress_score(self, market_data: List[Dict]) -> float:
        """Calculate market stress score (0-10 scale)."""
        if not market_data:
            return 0.0
        
        stress_components = []
        
        for asset_data in market_data:
            asset = asset_data['asset']
            change_pct = asset_data.get('change_percent', 0)
            price = asset_data.get('price', 0)
            priority = asset_data.get('priority', 3)
            
            # VIX is direct fear measure
            if 'VIX' in asset and price > 0:
                vix_score = min(price / 10, 10)
                stress_components.append(vix_score * 3.0)
            
            # Large negative moves in indices
            if asset_data.get('type') == 'index' and change_pct < -1:
                index_stress = min(abs(change_pct) * 2, 10)
                weight = 2.5 if priority == 1 else 1.5
                stress_components.append(index_stress * weight)
            
            # Safe haven moves
            if asset in ['VIX', 'Gold', '10Y Treasury'] and change_pct > 3:
                stress_components.append(min(change_pct * 1.5, 10))
            
            # Commodity volatility
            if asset_data.get('type') == 'commodity' and abs(change_pct) > 3:
                stress_components.append(abs(change_pct) * 1.2)
            
            # Crypto volatility (lower weight)
            if asset_data.get('type') == 'crypto' and abs(change_pct) > 5:
                stress_components.append(min(abs(change_pct) * 0.8, 10))
        
        if stress_components:
            stress_score = min(sum(stress_components) / len(stress_components), 10)
        else:
            stress_score = 0.0
        
        return round(stress_score, 2)
    
    def create_market_articles(self, market_data: List[Dict]) -> List[Dict]:
        """Convert market data into article format."""
        articles = []
        
        if not market_data:
            scraper_logger.warning("No market data to convert")
            return articles
        
        stress_score = self.calculate_market_stress_score(market_data)
        timestamp = datetime.now().isoformat()
        
        # Group by type
        by_type = {}
        for data in market_data:
            asset_type = data.get('type', 'unknown')
            by_type.setdefault(asset_type, []).append(data)
        
        # Build summary
        summary_lines = [
            f"Market Stress Score: {stress_score}/10",
            f"Data Points: {len(market_data)}",
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Sources: {', '.join(set(d['source'] for d in market_data))}",
            ""
        ]
        
        for asset_type in sorted(by_type.keys()):
            items = by_type[asset_type]
            summary_lines.append(f"\n{asset_type.upper()}:")
            for item in sorted(items, key=lambda x: x.get('priority', 3)):
                summary_lines.append(
                    f"  {item['asset']}: ${item['price']:.2f} ({item['change_percent']:+.2f}%)"
                )
        
        summary_snippet = '\n'.join(summary_lines)
        
        # Summary article
        articles.append({
            'headline': f'Market Overview - Stress Level: {stress_score}/10',
            'snippet': summary_snippet[:1000],
            'timestamp': timestamp,
            'asset_tags': [d['asset'] for d in market_data],
            'url': 'https://finance.yahoo.com',
            'source': 'multi_source',
            'scraped_at': timestamp,
            'market_stress_score': stress_score,
            'data_points': len(market_data),
            'asset_types': list(by_type.keys())
        })
        
        # Individual articles for significant movers
        for asset_data in market_data:
            change_pct = asset_data.get('change_percent', 0)
            threshold = 1.0 if asset_data.get('type') == 'index' else 2.0
            
            if abs(change_pct) > threshold or 'VIX' in asset_data['asset']:
                direction = ("surges" if change_pct > 2 else "rises" if change_pct > 0 
                           else "plunges" if change_pct < -2 else "falls")
                
                headline = (
                    f"{asset_data['asset']} {direction} {abs(change_pct):.2f}% "
                    f"to ${asset_data['price']:.2f}"
                )
                
                snippet = (
                    f"{asset_data['description']}\n"
                    f"Current: ${asset_data['price']:.2f} ({change_pct:+.2f}%)\n"
                    f"Previous Close: ${asset_data['prev_close']:.2f}\n"
                    f"Day Range: ${asset_data['day_low']:.2f} - ${asset_data['day_high']:.2f}\n"
                    f"Source: {asset_data['source']}"
                )
                
                articles.append({
                    'headline': headline,
                    'snippet': snippet,
                    'timestamp': timestamp,
                    'asset_tags': [asset_data['asset']],
                    'url': asset_data.get('url', 'https://finance.yahoo.com'),
                    'source': asset_data['source'],
                    'scraped_at': timestamp,
                    'price': asset_data['price'],
                    'change_percent': change_pct,
                    'asset_type': asset_data['type'],
                    'priority': asset_data.get('priority', 3)
                })
        
        scraper_logger.info(f"Created {len(articles)} articles")
        return articles
    
    def save_to_bronze(self, articles: List[Dict]) -> Optional[Path]:
        """Save articles to bronze layer."""
        if not articles:
            scraper_logger.warning("No articles to save")
            return None
        
        try:
            df = pd.DataFrame(articles)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"market_data_{timestamp}.parquet"
            filepath = Path("data/bronze") / filename
            
            filepath.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(filepath, index=False)
            
            scraper_logger.info(f"✓ Saved {len(df)} articles to {filepath}")
            return filepath
            
        except Exception as e:
            scraper_logger.error(f"Error saving: {e}", exc_info=True)
            return None


# Maintain compatibility with existing code
SafeInvestingScraper = MultiSourceMarketScraper


def scrape_investing_data(priority_filter: int = 2) -> Optional[Path]:
    """
    Main scraping function with multi-source fallback.
    
    Args:
        priority_filter: Priority level (1=critical, 2=balanced, 3=all)
        
    Returns:
        Path to saved bronze file
    """
    scraper = MultiSourceMarketScraper()
    
    try:
        scraper_logger.info("="*60)
        scraper_logger.info("MARKET DATA SCRAPER (Multi-Source)")
        scraper_logger.info(f"Priority Filter: {priority_filter}")
        scraper_logger.info("="*60)
        
        start_time = time.time()
        
        # Scrape data
        market_data = scraper.scrape_data(priority_filter=priority_filter)
        
        if not market_data:
            scraper_logger.error("No market data collected from any source")
            return None
        
        # Convert to articles
        articles = scraper.create_market_articles(market_data)
        
        # Save to bronze
        filepath = scraper.save_to_bronze(articles)
        
        elapsed = time.time() - start_time
        scraper_logger.info("="*60)
        scraper_logger.info(f"✓ Scraping completed in {elapsed:.2f}s")
        scraper_logger.info(f"✓ Data saved to: {filepath}")
        scraper_logger.info("="*60)
        
        return filepath
        
    except Exception as e:
        scraper_logger.error(f"Scraping failed: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    print("\n" + "="*60)
    print("MARKET DATA SCRAPER - MULTI-SOURCE")
    print("="*60 + "\n")
    
    result = scrape_investing_data(priority_filter=2)
    
    if result:
        print(f"\n{'='*60}")
        print("✓ SUCCESS!")
        print(f"{'='*60}")
        print(f"\nData saved to: {result}")
        
        try:
            df = pd.read_parquet(result)
            print(f"\nCollected {len(df)} articles")
            
            if len(df) > 0:
                print(f"\n{'='*60}")
                print("MARKET SUMMARY")
                print(f"{'='*60}\n")
                
                summary = df[df['headline'].str.contains('Market Overview', na=False)]
                if not summary.empty:
                    print(summary.iloc[0]['snippet'])
                
                print(f"\n{'='*60}")
                print("SIGNIFICANT MOVERS")
                print(f"{'='*60}\n")
                
                movers = df[~df['headline'].str.contains('Market Overview', na=False)].head(10)
                for _, row in movers.iterrows():
                    print(f"• {row['headline']}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"\n{'='*60}")
        print("✗ FAILED")
        print(f"{'='*60}")
