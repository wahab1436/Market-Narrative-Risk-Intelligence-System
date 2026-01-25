"""
Market Data Scraper using Yahoo Finance API
Real-time market data with 100% reliability
"""
import time
import json
import re
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import requests
import pandas as pd

try:
    from src.utils.logger import scraper_logger
except ImportError:
    import logging
    scraper_logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


class SafeInvestingScraper:
    """
    Market data scraper using Yahoo Finance API.
    Provides real, live market data with high reliability.
    """
    
    def __init__(self):
        """Initialize scraper with Yahoo Finance configuration."""
        
        self.base_url = "https://query1.finance.yahoo.com/v8/finance/chart"
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        # Yahoo Finance symbols with priority
        self.instruments = {
            # Priority 1: Critical market indicators
            'VIX': {
                'symbol': '^VIX',
                'type': 'volatility',
                'priority': 1,
                'description': 'CBOE Volatility Index (Fear Gauge)'
            },
            'S&P 500': {
                'symbol': '^GSPC',
                'type': 'index',
                'priority': 1,
                'description': 'S&P 500 Index'
            },
            'Dow Jones': {
                'symbol': '^DJI',
                'type': 'index',
                'priority': 1,
                'description': 'Dow Jones Industrial Average'
            },
            'NASDAQ': {
                'symbol': '^IXIC',
                'type': 'index',
                'priority': 1,
                'description': 'NASDAQ Composite'
            },
            
            # Priority 2: Important markets
            'Russell 2000': {
                'symbol': '^RUT',
                'type': 'index',
                'priority': 2,
                'description': 'Russell 2000 Small Cap'
            },
            'Gold': {
                'symbol': 'GC=F',
                'type': 'commodity',
                'priority': 2,
                'description': 'Gold Futures'
            },
            'Crude Oil': {
                'symbol': 'CL=F',
                'type': 'commodity',
                'priority': 2,
                'description': 'Crude Oil WTI Futures'
            },
            'EUR/USD': {
                'symbol': 'EURUSD=X',
                'type': 'forex',
                'priority': 2,
                'description': 'Euro vs US Dollar'
            },
            '10Y Treasury': {
                'symbol': '^TNX',
                'type': 'bond',
                'priority': 2,
                'description': 'US 10-Year Treasury Yield'
            },
            
            # Priority 3: Additional coverage
            'Silver': {
                'symbol': 'SI=F',
                'type': 'commodity',
                'priority': 3,
                'description': 'Silver Futures'
            },
            'Natural Gas': {
                'symbol': 'NG=F',
                'type': 'commodity',
                'priority': 3,
                'description': 'Natural Gas Futures'
            },
            'GBP/USD': {
                'symbol': 'GBPUSD=X',
                'type': 'forex',
                'priority': 3,
                'description': 'British Pound vs US Dollar'
            },
            'USD/JPY': {
                'symbol': 'USDJPY=X',
                'type': 'forex',
                'priority': 3,
                'description': 'US Dollar vs Japanese Yen'
            },
            'Bitcoin': {
                'symbol': 'BTC-USD',
                'type': 'crypto',
                'priority': 3,
                'description': 'Bitcoin USD'
            },
            'Ethereum': {
                'symbol': 'ETH-USD',
                'type': 'crypto',
                'priority': 3,
                'description': 'Ethereum USD'
            },
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def get_instruments_by_priority(self, max_priority: int = 3) -> Dict[str, Dict]:
        """
        Filter instruments by priority level.
        
        Args:
            max_priority: Maximum priority (1=critical, 2=balanced, 3=all)
        
        Returns:
            Filtered instruments dictionary
        """
        return {
            name: info
            for name, info in self.instruments.items()
            if info['priority'] <= max_priority
        }
    
    def fetch_instrument_data(self, name: str, info: Dict) -> Optional[Dict]:
        """
        Fetch real-time data from Yahoo Finance.
        
        Args:
            name: Instrument name
            info: Instrument configuration
        
        Returns:
            Market data dictionary or None
        """
        symbol = info['symbol']
        
        try:
            scraper_logger.info(f"Fetching {name} ({symbol})")
            
            url = f"{self.base_url}/{symbol}"
            params = {
                'interval': '1d',
                'range': '5d',
                'includePrePost': 'false'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Validate response structure
            if 'chart' not in data or 'result' not in data['chart']:
                scraper_logger.warning(f"Invalid response for {name}")
                return None
            
            if not data['chart']['result']:
                scraper_logger.warning(f"No data in response for {name}")
                return None
            
            result = data['chart']['result'][0]
            meta = result.get('meta', {})
            
            # Extract current price data
            current_price = meta.get('regularMarketPrice')
            prev_close = meta.get('previousClose')
            
            if current_price is None:
                scraper_logger.warning(f"No price data for {name}")
                return None
            
            # Get high/low
            day_high = meta.get('regularMarketDayHigh', current_price)
            day_low = meta.get('regularMarketDayLow', current_price)
            
            # Calculate change
            if prev_close and prev_close != 0:
                change = current_price - prev_close
                change_percent = (change / prev_close) * 100
            else:
                change = 0.0
                change_percent = 0.0
                prev_close = current_price
            
            # Build result
            market_data = {
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
            
            # Add volume if available
            volume = meta.get('regularMarketVolume')
            if volume:
                market_data['volume'] = int(volume)
            
            scraper_logger.info(
                f"✓ {name}: ${current_price:.2f} ({change_percent:+.2f}%)"
            )
            
            return market_data
            
        except requests.Timeout:
            scraper_logger.error(f"Timeout for {name}")
            return None
        except requests.RequestException as e:
            scraper_logger.error(f"Network error for {name}: {e}")
            return None
        except (KeyError, ValueError) as e:
            scraper_logger.error(f"Data parsing error for {name}: {e}")
            return None
        except Exception as e:
            scraper_logger.error(f"Unexpected error for {name}: {e}")
            return None
    
    def scrape_data(self, priority_filter: int = 2) -> List[Dict]:
        """
        Scrape market data with priority filtering.
        
        Args:
            priority_filter: Max priority level (1=critical, 2=balanced, 3=all)
        
        Returns:
            List of market data dictionaries
        """
        instruments = self.get_instruments_by_priority(priority_filter)
        
        scraper_logger.info("="*60)
        scraper_logger.info(f"Yahoo Finance Market Data Collection")
        scraper_logger.info(f"Priority Filter: {priority_filter}")
        scraper_logger.info(f"Instruments: {len(instruments)}")
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
                
                # Small delay
                time.sleep(0.3)
                
            except Exception as e:
                scraper_logger.error(f"Error collecting {name}: {e}")
                failed.append(name)
                continue
        
        scraper_logger.info("="*60)
        scraper_logger.info(f"Success: {len(market_data)}/{len(instruments)}")
        if failed:
            scraper_logger.warning(f"Failed: {', '.join(failed)}")
        scraper_logger.info("="*60)
        
        return market_data
    
    def calculate_market_stress_score(self, market_data: List[Dict]) -> float:
        """Calculate market stress score (0-10)."""
        if not market_data:
            return 0.0
        
        stress_components = []
        
        for asset in market_data:
            name = asset['asset']
            change_pct = asset.get('change_percent', 0)
            price = asset.get('price', 0)
            priority = asset.get('priority', 3)
            
            # VIX - direct fear measure
            if 'VIX' in name and price > 0:
                vix_score = min(price / 10, 10)
                stress_components.append(vix_score * 3.0)
            
            # Index drops
            if asset.get('type') == 'index' and change_pct < -1:
                stress = min(abs(change_pct) * 2, 10)
                weight = 2.5 if priority == 1 else 1.5
                stress_components.append(stress * weight)
            
            # Safe haven spikes
            if name in ['VIX', 'Gold', '10Y Treasury'] and change_pct > 3:
                stress_components.append(min(change_pct * 1.5, 10))
            
            # Commodity volatility
            if asset.get('type') == 'commodity' and abs(change_pct) > 3:
                stress_components.append(abs(change_pct) * 1.2)
        
        if stress_components:
            stress_score = min(sum(stress_components) / len(stress_components), 10)
        else:
            stress_score = 0.0
        
        return round(stress_score, 2)
    
    def create_market_articles(self, market_data: List[Dict]) -> List[Dict]:
        """Convert market data to article format."""
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
            f"Source: Yahoo Finance (Real-time)",
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        for asset_type in sorted(by_type.keys()):
            items = by_type[asset_type]
            summary_lines.append(f"\n{asset_type.upper()}:")
            for item in sorted(items, key=lambda x: x.get('priority', 3)):
                summary_lines.append(
                    f"  {item['asset']}: ${item['price']:.2f} "
                    f"({item['change_percent']:+.2f}%)"
                )
        
        summary_snippet = '\n'.join(summary_lines)
        
        # Summary article
        articles.append({
            'headline': f'Market Overview - Stress Level: {stress_score}/10',
            'snippet': summary_snippet[:1000],
            'timestamp': timestamp,
            'asset_tags': [d['asset'] for d in market_data],
            'url': 'https://finance.yahoo.com',
            'source': 'yahoo_finance',
            'scraped_at': timestamp,
            'market_stress_score': stress_score,
            'data_points': len(market_data),
            'asset_types': list(by_type.keys())
        })
        
        # Individual movers
        for asset in market_data:
            change_pct = asset.get('change_percent', 0)
            threshold = 1.0 if asset.get('type') == 'index' else 2.0
            
            if abs(change_pct) > threshold or 'VIX' in asset['asset']:
                direction = ("surges" if change_pct > 2 else "rises" if change_pct > 0 
                           else "plunges" if change_pct < -2 else "falls")
                
                headline = (
                    f"{asset['asset']} {direction} {abs(change_pct):.2f}% "
                    f"to ${asset['price']:.2f}"
                )
                
                snippet = (
                    f"{asset['description']}\n"
                    f"Current: ${asset['price']:.2f} ({change_pct:+.2f}%)\n"
                    f"Previous: ${asset['prev_close']:.2f}\n"
                    f"Range: ${asset['day_low']:.2f} - ${asset['day_high']:.2f}\n"
                    f"Source: Yahoo Finance"
                )
                
                articles.append({
                    'headline': headline,
                    'snippet': snippet,
                    'timestamp': timestamp,
                    'asset_tags': [asset['asset']],
                    'url': asset['url'],
                    'source': 'yahoo_finance',
                    'scraped_at': timestamp,
                    'price': asset['price'],
                    'change_percent': change_pct,
                    'asset_type': asset['type'],
                    'priority': asset['priority']
                })
        
        scraper_logger.info(f"Created {len(articles)} articles")
        return articles
    
    def save_to_bronze(self, articles: List[Dict]) -> Optional[Path]:
        """Save to bronze layer."""
        if not articles:
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
            scraper_logger.error(f"Save error: {e}")
            return None


def scrape_investing_data(priority_filter: int = 2) -> Optional[Path]:
    """
    Main scraping function.
    
    Args:
        priority_filter: 1=critical, 2=balanced, 3=all
    
    Returns:
        Path to saved file
    """
    scraper = SafeInvestingScraper()
    
    try:
        scraper_logger.info("="*60)
        scraper_logger.info("MARKET DATA SCRAPER - Yahoo Finance")
        scraper_logger.info(f"Priority: {priority_filter}")
        scraper_logger.info("="*60)
        
        start = time.time()
        
        market_data = scraper.scrape_data(priority_filter=priority_filter)
        
        if not market_data:
            scraper_logger.error("No data collected")
            return None
        
        articles = scraper.create_market_articles(market_data)
        filepath = scraper.save_to_bronze(articles)
        
        elapsed = time.time() - start
        scraper_logger.info("="*60)
        scraper_logger.info(f"✓ Completed in {elapsed:.2f}s")
        scraper_logger.info(f"✓ Saved to: {filepath}")
        scraper_logger.info("="*60)
        
        return filepath
        
    except Exception as e:
        scraper_logger.error(f"Failed: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    print("\n" + "="*60)
    print("MARKET DATA SCRAPER - YAHOO FINANCE")
    print("Real-time market data")
    print("="*60 + "\n")
    
    result = scrape_investing_data(priority_filter=2)
    
    if result:
        print(f"\n{'='*60}")
        print("✓ SUCCESS!")
        print(f"{'='*60}")
        print(f"\nSaved: {result}")
        
        try:
            df = pd.read_parquet(result)
            print(f"Articles: {len(df)}")
            
            if len(df) > 0:
                print(f"\n{'='*60}")
                print("SUMMARY")
                print(f"{'='*60}\n")
                
                summary = df[df['headline'].str.contains('Market Overview', na=False)]
                if not summary.empty:
                    print(summary.iloc[0]['snippet'])
                
                print(f"\n{'='*60}")
                print("MOVERS")
                print(f"{'='*60}\n")
                
                movers = df[~df['headline'].str.contains('Market Overview', na=False)].head(5)
                for _, row in movers.iterrows():
                    print(f"• {row['headline']}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"\n{'='*60}")
        print("✗ FAILED")
        print(f"{'='*60}")
