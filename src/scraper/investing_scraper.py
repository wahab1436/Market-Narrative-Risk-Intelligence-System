"""
Investing.com Market Data Scraper
Scrapes actual market data: indices, stocks, commodities, currencies
This provides REAL market risk data instead of just news sentiment!
"""
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import requests
import pandas as pd
from bs4 import BeautifulSoup
import json
import re

try:
    from src.utils.logger import scraper_logger
except ImportError:
    import logging
    scraper_logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


class InvestingMarketScraper:
    """
    Scrapes real market data from Investing.com and other sources.
    Provides actual market metrics for risk analysis.
    """
    
    def __init__(self):
        """Initialize market data scraper."""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        # Major market indices to track
        self.indices = {
            'S&P 500': 'https://www.investing.com/indices/us-spx-500',
            'Dow Jones': 'https://www.investing.com/indices/us-30',
            'NASDAQ': 'https://www.investing.com/indices/nasdaq-composite',
            'VIX': 'https://www.investing.com/indices/volatility-s-p-500',  # Fear index!
        }
        
        # Alternative: Use Yahoo Finance API (more reliable)
        self.use_yahoo_fallback = True
    
    def scrape_yahoo_finance(self) -> List[Dict]:
        """
        Scrape market data from Yahoo Finance (more reliable alternative).
        
        Returns:
            List of market data dictionaries
        """
        market_data = []
        
        # Yahoo Finance tickers
        tickers = {
            'S&P 500': '^GSPC',
            'Dow Jones': '^DJI',
            'NASDAQ': '^IXIC',
            'VIX': '^VIX',
            'Gold': 'GC=F',
            'Oil': 'CL=F',
            'Bitcoin': 'BTC-USD',
            'EUR/USD': 'EURUSD=X'
        }
        
        scraper_logger.info("Fetching market data from Yahoo Finance")
        
        for name, ticker in tickers.items():
            try:
                # Yahoo Finance quote URL
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=5d"
                
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                
                if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                    result = data['chart']['result'][0]
                    
                    # Get latest quote
                    meta = result.get('meta', {})
                    current_price = meta.get('regularMarketPrice', 0)
                    prev_close = meta.get('previousClose', current_price)
                    
                    # Calculate change
                    change = current_price - prev_close
                    change_percent = (change / prev_close * 100) if prev_close > 0 else 0
                    
                    # Get historical data for volatility calculation
                    quotes = result.get('indicators', {}).get('quote', [{}])[0]
                    closes = quotes.get('close', [])
                    
                    # Calculate volatility (standard deviation of returns)
                    if len(closes) > 1:
                        returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes)) if closes[i-1] > 0]
                        volatility = pd.Series(returns).std() * 100 if returns else 0
                    else:
                        volatility = 0
                    
                    market_data.append({
                        'asset': name,
                        'ticker': ticker,
                        'price': current_price,
                        'change': change,
                        'change_percent': change_percent,
                        'prev_close': prev_close,
                        'volatility': volatility,
                        'timestamp': datetime.now().isoformat(),
                        'source': 'yahoo_finance'
                    })
                    
                    scraper_logger.info(f"{name}: ${current_price:.2f} ({change_percent:+.2f}%)")
                    
                    time.sleep(0.5)  # Be polite
                    
            except Exception as e:
                scraper_logger.error(f"Error fetching {name}: {e}")
                continue
        
        scraper_logger.info(f"Collected data for {len(market_data)} assets")
        return market_data
    
    def calculate_market_stress_score(self, market_data: List[Dict]) -> float:
        """
        Calculate overall market stress score from real market data.
        
        Args:
            market_data: List of market data dictionaries
            
        Returns:
            Stress score (0-10 scale)
        """
        if not market_data:
            return 0.0
        
        stress_components = []
        
        for asset_data in market_data:
            asset = asset_data['asset']
            change_pct = asset_data['change_percent']
            volatility = asset_data['volatility']
            
            # VIX is direct fear measure
            if 'VIX' in asset:
                vix_score = min(asset_data['price'] / 10, 10)  # VIX > 30 = high fear
                stress_components.append(vix_score * 2)  # Weight VIX heavily
            
            # Large negative moves increase stress
            if change_pct < -2:
                stress_components.append(abs(change_pct) / 2)
            
            # High volatility increases stress
            if volatility > 2:
                stress_components.append(volatility)
        
        # Calculate weighted average
        if stress_components:
            stress_score = min(sum(stress_components) / len(stress_components), 10)
        else:
            stress_score = 0.0
        
        return round(stress_score, 2)
    
    def create_market_articles(self, market_data: List[Dict]) -> List[Dict]:
        """
        Convert market data into article format for pipeline compatibility.
        
        Args:
            market_data: List of market data dictionaries
            
        Returns:
            List of article-like dictionaries
        """
        articles = []
        
        # Calculate overall stress
        stress_score = self.calculate_market_stress_score(market_data)
        
        # Create summary article
        summary_headline = f"Market Update: Stress Score {stress_score}/10"
        
        summary_snippet = "Market Data Summary:\n"
        for asset_data in market_data:
            summary_snippet += f"{asset_data['asset']}: ${asset_data['price']:.2f} ({asset_data['change_percent']:+.2f}%)\n"
        
        articles.append({
            'headline': summary_headline,
            'snippet': summary_snippet[:500],
            'timestamp': datetime.now().isoformat(),
            'asset_tags': [asset['asset'] for asset in market_data],
            'url': 'https://finance.yahoo.com',
            'source': 'market_data',
            'scraped_at': datetime.now().isoformat(),
            'market_stress_score': stress_score,
            'market_data': market_data  # Include raw data
        })
        
        # Create individual articles for each asset
        for asset_data in market_data:
            direction = "rises" if asset_data['change'] > 0 else "falls"
            
            headline = f"{asset_data['asset']} {direction} to ${asset_data['price']:.2f}"
            
            snippet = (
                f"{asset_data['asset']} is trading at ${asset_data['price']:.2f}, "
                f"{direction[:-1]}ing {abs(asset_data['change_percent']):.2f}% from previous close. "
                f"Volatility: {asset_data['volatility']:.2f}%"
            )
            
            articles.append({
                'headline': headline,
                'snippet': snippet,
                'timestamp': datetime.now().isoformat(),
                'asset_tags': [asset_data['asset']],
                'url': f"https://finance.yahoo.com/quote/{asset_data['ticker']}",
                'source': 'market_data',
                'scraped_at': datetime.now().isoformat(),
                'price': asset_data['price'],
                'change_percent': asset_data['change_percent'],
                'volatility': asset_data['volatility']
            })
        
        scraper_logger.info(f"Created {len(articles)} market data articles")
        return articles
    
    def save_to_bronze(self, articles: List[Dict]):
        """Save market data to bronze layer."""
        if not articles:
            scraper_logger.warning("No market data to save")
            return None
        
        df = pd.DataFrame(articles)
        
        # Create timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"market_data_{timestamp}.parquet"
        filepath = Path("data/bronze") / filename
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as Parquet
        df.to_parquet(filepath, index=False)
        scraper_logger.info(f"Saved {len(df)} market data records to {filepath}")
        
        return filepath


def scrape_market_data():
    """
    Main function to scrape real market data.
    
    Returns:
        Path to saved bronze file
    """
    scraper = InvestingMarketScraper()
    
    try:
        scraper_logger.info("Starting market data scraping")
        start_time = time.time()
        
        # Scrape market data
        market_data = scraper.scrape_yahoo_finance()
        
        if market_data:
            # Convert to article format
            articles = scraper.create_market_articles(market_data)
            
            # Save to bronze
            filepath = scraper.save_to_bronze(articles)
            
            elapsed_time = time.time() - start_time
            scraper_logger.info(f"Market data scraping completed in {elapsed_time:.2f}s: {filepath}")
            
            return filepath
        else:
            scraper_logger.warning("No market data collected")
            return None
        
    except Exception as e:
        scraper_logger.error(f"Market data scraping failed: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    # Test the scraper
    result = scrape_market_data()
    if result:
        print(f"\nSuccessfully scraped market data to: {result}")
        df = pd.read_parquet(result)
        print(f"\nCollected {len(df)} records")
        print(f"\nMarket Summary:")
        print(df[['headline', 'source']].to_string(index=False))
    else:
        print("Market data scraping failed")
