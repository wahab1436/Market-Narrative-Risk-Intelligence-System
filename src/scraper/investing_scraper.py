"""
Investing.com Market Data Scraper - WORKING VERSION
Uses actual Investing.com endpoints that are currently functional
Based on investiny approach using tvc6.investing.com
"""
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import requests
import pandas as pd
import json

try:
    from src.utils.logger import scraper_logger
except ImportError:
    import logging
    scraper_logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


class InvestingDataScraper:
    """
    Scrapes data from Investing.com using their TradingView-based chart API.
    This endpoint (tvc6.investing.com) is currently working and not Cloudflare protected.
    """
    
    def __init__(self):
        # Working Investing.com endpoint (TradingView charts)
        self.chart_api = "https://tvc6.investing.com/6898a2759cecc3a93d7b0e0ae14fe8a6/{}/1/1/8/history"
        self.search_api = "https://tvc6.investing.com/6898a2759cecc3a93d7b0e0ae14fe8a6/1/1/8/search"
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Origin': 'https://www.investing.com',
            'Referer': 'https://www.investing.com/',
        }
        
        # Investing.com instrument IDs (pairId)
        self.instruments = {
            'S&P 500': {'id': '166', 'symbol': 'SPX', 'type': 'index'},
            'Dow Jones': {'id': '169', 'symbol': 'DJI', 'type': 'index'},
            'NASDAQ': {'id': '14958', 'symbol': 'IXIC', 'type': 'index'},
            'VIX': {'id': '44336', 'symbol': 'VIX', 'type': 'volatility'},
            'Russell 2000': {'id': '8863', 'symbol': 'RUT', 'type': 'index'},
            'Gold': {'id': '8830', 'symbol': 'XAU/USD', 'type': 'commodity'},
            'Crude Oil': {'id': '8849', 'symbol': 'CL', 'type': 'commodity'},
            'Silver': {'id': '8836', 'symbol': 'XAG/USD', 'type': 'commodity'},
            'Natural Gas': {'id': '8862', 'symbol': 'NG', 'type': 'commodity'},
            'EUR/USD': {'id': '1', 'symbol': 'EUR/USD', 'type': 'forex'},
            'GBP/USD': {'id': '2', 'symbol': 'GBP/USD', 'type': 'forex'},
            'USD/JPY': {'id': '3', 'symbol': 'USD/JPY', 'type': 'forex'},
            'Bitcoin': {'id': '1057391', 'symbol': 'BTC/USD', 'type': 'crypto'},
            'Ethereum': {'id': '1061443', 'symbol': 'ETH/USD', 'type': 'crypto'},
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def get_historical_data(self, pair_id: str, name: str, asset_type: str) -> Optional[Dict]:
        """
        Get historical data from Investing.com's chart API.
        
        Args:
            pair_id: Investing.com pair ID
            name: Asset name
            asset_type: Type of asset
            
        Returns:
            Dictionary with current market data
        """
        try:
            scraper_logger.info(f"Fetching {name} (ID: {pair_id})")
            
            # Calculate timestamps (last 7 days to get recent data)
            end_time = int(time.time())
            start_time = end_time - (7 * 24 * 60 * 60)  # 7 days ago
            
            # Build URL with the pair_id
            url = self.chart_api.format(pair_id)
            
            # Parameters for daily data
            params = {
                'symbol': pair_id,
                'resolution': 'D',  # Daily resolution
                'from': start_time,
                'to': end_time,
            }
            
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse the response
            if data.get('s') == 'ok' and 't' in data and len(data['t']) > 0:
                # Get latest data point
                latest_idx = -1
                
                # Data structure: t=timestamps, o=open, h=high, l=low, c=close, v=volume
                current_price = float(data['c'][latest_idx])
                day_high = float(data['h'][latest_idx])
                day_low = float(data['l'][latest_idx])
                
                # Get previous close
                if len(data['c']) > 1:
                    prev_close = float(data['c'][-2])
                else:
                    prev_close = current_price
                
                # Calculate change
                change = current_price - prev_close
                change_percent = (change / prev_close * 100) if prev_close != 0 else 0
                
                result = {
                    'asset': name,
                    'pair_id': pair_id,
                    'type': asset_type,
                    'price': current_price,
                    'prev_close': prev_close,
                    'change': change,
                    'change_percent': change_percent,
                    'day_high': day_high,
                    'day_low': day_low,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'investing.com_tvc',
                    'url': f'https://www.investing.com/instruments/{pair_id}'
                }
                
                # Add volume if available
                if 'v' in data and len(data['v']) > 0:
                    result['volume'] = int(data['v'][latest_idx])
                
                scraper_logger.info(
                    f"✓ {name}: ${current_price:.2f} ({change_percent:+.2f}%)"
                )
                
                return result
            else:
                scraper_logger.warning(f"No data returned for {name}")
                return None
                
        except requests.RequestException as e:
            scraper_logger.error(f"Network error for {name}: {e}")
            return None
        except (KeyError, ValueError, IndexError) as e:
            scraper_logger.error(f"Data parsing error for {name}: {e}")
            return None
        except Exception as e:
            scraper_logger.error(f"Unexpected error for {name}: {e}")
            return None
    
    def search_asset(self, query: str) -> Optional[List[Dict]]:
        """
        Search for an asset on Investing.com to get its ID.
        Useful for finding new instruments.
        
        Args:
            query: Search query (e.g., "AAPL", "Gold")
            
        Returns:
            List of search results with IDs
        """
        try:
            params = {
                'query': query,
                'type': '',
                'exchange': '',
            }
            
            response = self.session.get(self.search_api, params=params, timeout=10)
            response.raise_for_status()
            
            results = response.json()
            
            if isinstance(results, list):
                return [
                    {
                        'name': r.get('description', ''),
                        'symbol': r.get('symbol', ''),
                        'pair_id': r.get('pairId', ''),
                        'type': r.get('type', ''),
                        'exchange': r.get('exchange', ''),
                    }
                    for r in results
                ]
            
            return None
            
        except Exception as e:
            scraper_logger.error(f"Search error for '{query}': {e}")
            return None
    
    def collect_all_data(self) -> List[Dict]:
        """Collect data for all configured instruments."""
        market_data = []
        failed = []
        
        scraper_logger.info(f"Starting data collection for {len(self.instruments)} instruments")
        
        for name, info in self.instruments.items():
            try:
                data = self.get_historical_data(info['id'], name, info['type'])
                
                if data:
                    market_data.append(data)
                else:
                    failed.append(name)
                
                # Polite delay
                time.sleep(1.5)
                
            except Exception as e:
                scraper_logger.error(f"Error collecting {name}: {e}")
                failed.append(name)
                continue
        
        scraper_logger.info(
            f"Successfully collected {len(market_data)}/{len(self.instruments)} instruments"
        )
        
        if failed:
            scraper_logger.warning(f"Failed: {', '.join(failed)}")
        
        return market_data
    
    def calculate_market_stress_score(self, market_data: List[Dict]) -> float:
        """Calculate overall market stress score from market data."""
        if not market_data:
            return 0.0
        
        stress_components = []
        
        for asset_data in market_data:
            asset = asset_data['asset']
            change_pct = asset_data.get('change_percent', 0)
            price = asset_data.get('price', 0)
            
            # VIX is direct fear measure
            if 'VIX' in asset and price > 0:
                vix_score = min(price / 10, 10)
                stress_components.append(vix_score * 2.5)
            
            # Large negative moves increase stress
            if change_pct < -2:
                stress_components.append(min(abs(change_pct), 10))
            
            # Major indices matter more
            if asset_data.get('type') == 'index' and abs(change_pct) > 1:
                stress_components.append(abs(change_pct) * 1.5)
            
            # Commodity volatility
            if asset_data.get('type') == 'commodity' and abs(change_pct) > 3:
                stress_components.append(abs(change_pct) * 1.2)
        
        if stress_components:
            stress_score = min(sum(stress_components) / len(stress_components), 10)
        else:
            stress_score = 0.0
        
        return round(stress_score, 2)
    
    def create_market_articles(self, market_data: List[Dict]) -> List[Dict]:
        """Convert market data into article format for pipeline compatibility."""
        articles = []
        
        if not market_data:
            scraper_logger.warning("No market data to convert")
            return articles
        
        stress_score = self.calculate_market_stress_score(market_data)
        timestamp = datetime.now().isoformat()
        
        # Build summary
        summary_lines = [f"Market Stress Score: {stress_score}/10\n"]
        
        by_type = {}
        for data in market_data:
            asset_type = data.get('type', 'unknown')
            by_type.setdefault(asset_type, []).append(data)
        
        for asset_type, items in sorted(by_type.items()):
            summary_lines.append(f"\n{asset_type.upper()}:")
            for item in items:
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
            'url': 'https://www.investing.com',
            'source': 'investing.com',
            'scraped_at': timestamp,
            'market_stress_score': stress_score,
            'data_points': len(market_data),
            'asset_types': list(by_type.keys())
        })
        
        # Individual articles for significant movers
        for asset_data in market_data:
            change_pct = asset_data.get('change_percent', 0)
            
            if abs(change_pct) > 1.5 or 'VIX' in asset_data['asset']:
                direction = ("surges" if change_pct > 2 else "rises" if change_pct > 0 
                           else "plunges" if change_pct < -2 else "falls")
                
                headline = (
                    f"{asset_data['asset']} {direction} {abs(change_pct):.1f}% "
                    f"to ${asset_data['price']:.2f}"
                )
                
                snippet = (
                    f"{asset_data['asset']} ({asset_data['type']}) is trading at "
                    f"${asset_data['price']:.2f}, {direction[:-1]}ing "
                    f"{abs(change_pct):.2f}% from previous close of "
                    f"${asset_data.get('prev_close', 0):.2f}. "
                    f"Day range: ${asset_data['day_low']:.2f} - ${asset_data['day_high']:.2f}."
                )
                
                articles.append({
                    'headline': headline,
                    'snippet': snippet,
                    'timestamp': timestamp,
                    'asset_tags': [asset_data['asset']],
                    'url': asset_data.get('url', 'https://www.investing.com'),
                    'source': 'investing.com',
                    'scraped_at': timestamp,
                    'price': asset_data['price'],
                    'change_percent': change_pct,
                    'asset_type': asset_data['type']
                })
        
        scraper_logger.info(f"Created {len(articles)} articles")
        return articles
    
    def save_to_bronze(self, articles: List[Dict]) -> Optional[Path]:
        """Save market data to bronze layer."""
        if not articles:
            return None
        
        try:
            df = pd.DataFrame(articles)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"investing_market_data_{timestamp}.parquet"
            filepath = Path("data/bronze") / filename
            
            filepath.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(filepath, index=False)
            
            scraper_logger.info(f"Saved {len(df)} articles to {filepath}")
            return filepath
        except Exception as e:
            scraper_logger.error(f"Error saving: {e}")
            return None


def scrape_investing_data():
    """Main function to scrape Investing.com data."""
    scraper = InvestingDataScraper()
    
    try:
        scraper_logger.info("="*60)
        scraper_logger.info("Investing.com Data Collection (TVC API)")
        scraper_logger.info("="*60)
        
        start_time = time.time()
        
        market_data = scraper.collect_all_data()
        
        if market_data:
            scraper_logger.info(f"\nCollected {len(market_data)} instruments")
            
            articles = scraper.create_market_articles(market_data)
            filepath = scraper.save_to_bronze(articles)
            
            elapsed = time.time() - start_time
            scraper_logger.info("="*60)
            scraper_logger.info(f"Completed in {elapsed:.2f}s")
            scraper_logger.info(f"Saved to: {filepath}")
            scraper_logger.info("="*60)
            
            return filepath
        else:
            scraper_logger.error("No data collected")
            return None
            
    except Exception as e:
        scraper_logger.error(f"Collection failed: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    print("\n" + "="*60)
    print("INVESTING.COM MARKET DATA SCRAPER")
    print("Using TVC Chart API (Working Method)")
    print("="*60 + "\n")
    
    result = scrape_investing_data()
    
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
            print(f"Error reading results: {e}")
    else:
        print(f"\n{'='*60}")
        print("✗ FAILED - Check logs")
        print(f"{'='*60}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. The TVC endpoint may have changed")
        print("3. Try the search_asset() function to find current IDs")
        print("\nExample:")
        print("  scraper = InvestingDataScraper()")
        print("  results = scraper.search_asset('AAPL')")
        print("  print(results)")
