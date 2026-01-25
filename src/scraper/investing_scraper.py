"""
Investing.com Market Data Scraper - Complete Implementation
Uses TVC Chart API with priority filtering for efficient data collection
"""
import time
from datetime import datetime
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


class SafeInvestingScraper:
    """
    Safe scraper for Investing.com using TVC Chart API.
    Includes priority filtering and robust error handling.
    """
    
    def __init__(self):
        """Initialize the scraper with working endpoints."""
        # Working TVC Chart API endpoint
        self.chart_api = "https://tvc6.investing.com/6898a2759cecc3a93d7b0e0ae14fe8a6/{}/1/1/8/history"
        self.search_api = "https://tvc6.investing.com/6898a2759cecc3a93d7b0e0ae14fe8a6/1/1/8/search"
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Origin': 'https://www.investing.com',
            'Referer': 'https://www.investing.com/',
        }
        
        # Instruments with priority levels
        # Priority 1 = Critical (VIX, major indices)
        # Priority 2 = Important (commodities, major forex)
        # Priority 3 = Additional (crypto, minor pairs)
        self.instruments = {
            # Priority 1: Critical - Market health indicators
            'VIX': {
                'id': '44336',
                'symbol': 'VIX',
                'type': 'volatility',
                'priority': 1,
                'description': 'CBOE Volatility Index - Fear gauge'
            },
            'S&P 500': {
                'id': '166',
                'symbol': 'SPX',
                'type': 'index',
                'priority': 1,
                'description': 'S&P 500 Index'
            },
            'Dow Jones': {
                'id': '169',
                'symbol': 'DJI',
                'type': 'index',
                'priority': 1,
                'description': 'Dow Jones Industrial Average'
            },
            'NASDAQ': {
                'id': '14958',
                'symbol': 'IXIC',
                'type': 'index',
                'priority': 1,
                'description': 'NASDAQ Composite'
            },
            
            # Priority 2: Important - Major markets
            'Russell 2000': {
                'id': '8863',
                'symbol': 'RUT',
                'type': 'index',
                'priority': 2,
                'description': 'Russell 2000 Small Cap Index'
            },
            'Gold': {
                'id': '8830',
                'symbol': 'XAU/USD',
                'type': 'commodity',
                'priority': 2,
                'description': 'Gold Spot Price'
            },
            'Crude Oil': {
                'id': '8849',
                'symbol': 'CL',
                'type': 'commodity',
                'priority': 2,
                'description': 'WTI Crude Oil'
            },
            'EUR/USD': {
                'id': '1',
                'symbol': 'EUR/USD',
                'type': 'forex',
                'priority': 2,
                'description': 'Euro vs US Dollar'
            },
            '10Y Treasury': {
                'id': '23705',
                'symbol': 'US10Y',
                'type': 'bond',
                'priority': 2,
                'description': 'US 10-Year Treasury Yield'
            },
            
            # Priority 3: Additional - Extended coverage
            'Silver': {
                'id': '8836',
                'symbol': 'XAG/USD',
                'type': 'commodity',
                'priority': 3,
                'description': 'Silver Spot Price'
            },
            'Natural Gas': {
                'id': '8862',
                'symbol': 'NG',
                'type': 'commodity',
                'priority': 3,
                'description': 'Natural Gas Futures'
            },
            'GBP/USD': {
                'id': '2',
                'symbol': 'GBP/USD',
                'type': 'forex',
                'priority': 3,
                'description': 'British Pound vs US Dollar'
            },
            'USD/JPY': {
                'id': '3',
                'symbol': 'USD/JPY',
                'type': 'forex',
                'priority': 3,
                'description': 'US Dollar vs Japanese Yen'
            },
            'Bitcoin': {
                'id': '1057391',
                'symbol': 'BTC/USD',
                'type': 'crypto',
                'priority': 3,
                'description': 'Bitcoin Price'
            },
            'Ethereum': {
                'id': '1061443',
                'symbol': 'ETH/USD',
                'type': 'crypto',
                'priority': 3,
                'description': 'Ethereum Price'
            },
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def get_instruments_by_priority(self, max_priority: int = 3) -> Dict[str, Dict]:
        """
        Filter instruments by priority level.
        
        Args:
            max_priority: Maximum priority to include (1=critical only, 2=critical+important, 3=all)
            
        Returns:
            Filtered dictionary of instruments
        """
        return {
            name: info
            for name, info in self.instruments.items()
            if info['priority'] <= max_priority
        }
    
    def fetch_instrument_data(self, pair_id: str, name: str, info: Dict) -> Optional[Dict]:
        """
        Fetch data for a single instrument from TVC Chart API.
        
        Args:
            pair_id: Investing.com pair ID
            name: Asset name
            info: Instrument info dictionary
            
        Returns:
            Dictionary with market data or None
        """
        try:
            scraper_logger.info(f"Fetching {name} (ID: {pair_id}, Priority: {info['priority']})")
            
            # Get data for last 7 days
            end_time = int(time.time())
            start_time = end_time - (7 * 24 * 60 * 60)
            
            url = self.chart_api.format(pair_id)
            
            params = {
                'symbol': pair_id,
                'resolution': 'D',  # Daily
                'from': start_time,
                'to': end_time,
            }
            
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            # Validate response
            if data.get('s') != 'ok' or 't' not in data or len(data['t']) == 0:
                scraper_logger.warning(f"No valid data for {name}")
                return None
            
            # Extract latest OHLCV data
            current_price = float(data['c'][-1])
            day_high = float(data['h'][-1])
            day_low = float(data['l'][-1])
            day_open = float(data['o'][-1])
            
            # Previous close
            prev_close = float(data['c'][-2]) if len(data['c']) > 1 else current_price
            
            # Calculate changes
            change = current_price - prev_close
            change_percent = (change / prev_close * 100) if prev_close != 0 else 0
            
            result = {
                'asset': name,
                'pair_id': pair_id,
                'symbol': info.get('symbol', ''),
                'type': info['type'],
                'priority': info['priority'],
                'description': info.get('description', ''),
                'price': round(current_price, 4),
                'prev_close': round(prev_close, 4),
                'change': round(change, 4),
                'change_percent': round(change_percent, 4),
                'day_open': round(day_open, 4),
                'day_high': round(day_high, 4),
                'day_low': round(day_low, 4),
                'timestamp': datetime.now().isoformat(),
                'source': 'investing.com_tvc',
                'url': f'https://www.investing.com/instruments/{pair_id}'
            }
            
            # Add volume if available
            if 'v' in data and len(data['v']) > 0:
                result['volume'] = int(data['v'][-1])
            
            scraper_logger.info(
                f"✓ {name}: ${current_price:.2f} ({change_percent:+.2f}%)"
            )
            
            return result
            
        except requests.Timeout:
            scraper_logger.error(f"Timeout fetching {name}")
            return None
        except requests.RequestException as e:
            scraper_logger.error(f"Network error for {name}: {e}")
            return None
        except (KeyError, ValueError, IndexError) as e:
            scraper_logger.error(f"Data parsing error for {name}: {e}")
            return None
        except Exception as e:
            scraper_logger.error(f"Unexpected error for {name}: {e}", exc_info=True)
            return None
    
    def scrape_data(self, priority_filter: int = 2) -> List[Dict]:
        """
        Scrape market data with priority filtering.
        
        Args:
            priority_filter: Maximum priority level to scrape
                1 = Only critical assets (fastest, ~5 instruments)
                2 = Critical + Important (balanced, ~9 instruments)
                3 = All assets (comprehensive, ~15 instruments)
                
        Returns:
            List of market data dictionaries
        """
        instruments = self.get_instruments_by_priority(priority_filter)
        
        scraper_logger.info("="*60)
        scraper_logger.info(f"Starting scrape with priority filter: {priority_filter}")
        scraper_logger.info(f"Instruments to fetch: {len(instruments)}")
        scraper_logger.info("="*60)
        
        market_data = []
        failed = []
        
        for name, info in instruments.items():
            try:
                data = self.fetch_instrument_data(info['id'], name, info)
                
                if data:
                    market_data.append(data)
                else:
                    failed.append(name)
                
                # Polite delay between requests
                time.sleep(1.5)
                
            except Exception as e:
                scraper_logger.error(f"Error collecting {name}: {e}")
                failed.append(name)
                continue
        
        scraper_logger.info("="*60)
        scraper_logger.info(f"Collection complete: {len(market_data)}/{len(instruments)} successful")
        if failed:
            scraper_logger.warning(f"Failed instruments: {', '.join(failed)}")
        scraper_logger.info("="*60)
        
        return market_data
    
    def calculate_market_stress_score(self, market_data: List[Dict]) -> float:
        """
        Calculate overall market stress score (0-10 scale).
        
        Args:
            market_data: List of market data dictionaries
            
        Returns:
            Stress score between 0 and 10
        """
        if not market_data:
            return 0.0
        
        stress_components = []
        
        for asset_data in market_data:
            asset = asset_data['asset']
            change_pct = asset_data.get('change_percent', 0)
            price = asset_data.get('price', 0)
            priority = asset_data.get('priority', 3)
            
            # VIX is direct fear measure (most important)
            if 'VIX' in asset and price > 0:
                # VIX > 20 = elevated fear, VIX > 30 = high fear
                vix_score = min(price / 10, 10)
                stress_components.append(vix_score * 3.0)  # Heavy weight
            
            # Large negative moves in major indices
            if asset_data.get('type') == 'index' and change_pct < -1:
                index_stress = min(abs(change_pct) * 2, 10)
                weight = 2.5 if priority == 1 else 1.5
                stress_components.append(index_stress * weight)
            
            # Significant positive moves in VIX or safe havens
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
        """
        Convert market data into article format for pipeline.
        
        Args:
            market_data: List of market data dictionaries
            
        Returns:
            List of article dictionaries
        """
        articles = []
        
        if not market_data:
            scraper_logger.warning("No market data to convert to articles")
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
        
        # Create summary article
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
        
        # Create articles for significant movers
        for asset_data in market_data:
            change_pct = asset_data.get('change_percent', 0)
            
            # Threshold varies by asset type
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
                    f"Type: {asset_data['type']}"
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
                    'asset_type': asset_data['type'],
                    'priority': asset_data.get('priority', 3)
                })
        
        scraper_logger.info(f"Created {len(articles)} articles from {len(market_data)} data points")
        return articles
    
    def save_to_bronze(self, articles: List[Dict]) -> Optional[Path]:
        """
        Save articles to bronze layer as parquet.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            Path to saved file or None
        """
        if not articles:
            scraper_logger.warning("No articles to save")
            return None
        
        try:
            df = pd.DataFrame(articles)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"investing_market_data_{timestamp}.parquet"
            filepath = Path("data/bronze") / filename
            
            filepath.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(filepath, index=False)
            
            scraper_logger.info(f"✓ Saved {len(df)} articles to {filepath}")
            return filepath
            
        except Exception as e:
            scraper_logger.error(f"Error saving to bronze: {e}", exc_info=True)
            return None


def scrape_investing_data(priority_filter: int = 2) -> Optional[Path]:
    """
    Main function to scrape Investing.com data and save to bronze.
    
    Args:
        priority_filter: Priority level (1=critical only, 2=balanced, 3=all)
        
    Returns:
        Path to saved bronze file or None
    """
    scraper = SafeInvestingScraper()
    
    try:
        scraper_logger.info("="*60)
        scraper_logger.info("INVESTING.COM MARKET DATA SCRAPER")
        scraper_logger.info(f"Priority Filter: {priority_filter}")
        scraper_logger.info("="*60)
        
        start_time = time.time()
        
        # Scrape data
        market_data = scraper.scrape_data(priority_filter=priority_filter)
        
        if not market_data:
            scraper_logger.error("No market data collected")
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
    print("INVESTING.COM MARKET DATA SCRAPER - TEST")
    print("="*60 + "\n")
    
    # Test with priority 2 (balanced)
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
            print(f"Error reading results: {e}")
    else:
        print(f"\n{'='*60}")
        print("✗ FAILED - Check logs for details")
        print(f"{'='*60}")
