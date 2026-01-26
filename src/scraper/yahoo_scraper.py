"""
Yahoo Finance Scraper with Priority Filtering
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YahooFinanceScraper:
    """
    Yahoo Finance market data scraper with priority-based filtering
    """
    
    def __init__(self):
        self.instruments = {
            # Priority 1 - Critical market indicators
            'S&P 500': {
                'symbol': '^GSPC', 
                'type': 'index', 
                'url': 'https://finance.yahoo.com/quote/%5EGSPC',
                'description': 'S&P 500 Index',
                'priority': 1
            },
            'Dow Jones': {
                'symbol': '^DJI', 
                'type': 'index', 
                'url': 'https://finance.yahoo.com/quote/%5EDJI',
                'description': 'Dow Jones Industrial Average',
                'priority': 1
            },
            'NASDAQ': {
                'symbol': '^IXIC', 
                'type': 'index', 
                'url': 'https://finance.yahoo.com/quote/%5EIXIC',
                'description': 'NASDAQ Composite Index',
                'priority': 1
            },
            'VIX': {
                'symbol': '^VIX', 
                'type': 'volatility', 
                'url': 'https://finance.yahoo.com/quote/%5EVIX',
                'description': 'CBOE Volatility Index',
                'priority': 1
            },
            
            # Priority 2 - Important markets
            'Russell 2000': {
                'symbol': '^RUT', 
                'type': 'index', 
                'url': 'https://finance.yahoo.com/quote/%5ERUT',
                'description': 'Russell 2000 Index',
                'priority': 2
            },
            'FTSE 100': {
                'symbol': '^FTSE', 
                'type': 'index', 
                'url': 'https://finance.yahoo.com/quote/%5EFTSE',
                'description': 'UK 100 Index',
                'priority': 2
            },
            'DAX': {
                'symbol': '^GDAXI', 
                'type': 'index', 
                'url': 'https://finance.yahoo.com/quote/%5EGDAXI',
                'description': 'German DAX Index',
                'priority': 2
            },
            'Gold': {
                'symbol': 'GC=F', 
                'type': 'commodity', 
                'url': 'https://finance.yahoo.com/quote/GC=F',
                'description': 'Gold Futures',
                'priority': 2
            },
            'Crude Oil': {
                'symbol': 'CL=F', 
                'type': 'commodity', 
                'url': 'https://finance.yahoo.com/quote/CL=F',
                'description': 'Crude Oil Futures',
                'priority': 2
            },
            'EUR/USD': {
                'symbol': 'EURUSD=X', 
                'type': 'forex', 
                'url': 'https://finance.yahoo.com/quote/EURUSD=X',
                'description': 'Euro to US Dollar',
                'priority': 2
            },
            
            # Priority 3 - Additional coverage
            'Silver': {
                'symbol': 'SI=F', 
                'type': 'commodity', 
                'url': 'https://finance.yahoo.com/quote/SI=F',
                'description': 'Silver Futures',
                'priority': 3
            },
            'GBP/USD': {
                'symbol': 'GBPUSD=X', 
                'type': 'forex', 
                'url': 'https://finance.yahoo.com/quote/GBPUSD=X',
                'description': 'British Pound to US Dollar',
                'priority': 3
            },
            'USD/JPY': {
                'symbol': 'USDJPY=X', 
                'type': 'forex', 
                'url': 'https://finance.yahoo.com/quote/USDJPY=X',
                'description': 'US Dollar to Japanese Yen',
                'priority': 3
            },
            'Bitcoin': {
                'symbol': 'BTC-USD', 
                'type': 'crypto', 
                'url': 'https://finance.yahoo.com/quote/BTC-USD',
                'description': 'Bitcoin to US Dollar',
                'priority': 3
            },
            'Ethereum': {
                'symbol': 'ETH-USD', 
                'type': 'crypto', 
                'url': 'https://finance.yahoo.com/quote/ETH-USD',
                'description': 'Ethereum to US Dollar',
                'priority': 3
            },
        }
    
    def get_instruments_by_priority(self, max_priority: int = 3):
        """Filter instruments by priority level"""
        return {
            name: info for name, info in self.instruments.items()
            if info['priority'] <= max_priority
        }
    
    def scrape_market_data(self, priority_filter: int = 2):
        """
        Scrape market data with priority filtering
        Returns list of market data dictionaries
        """
        instruments = self.get_instruments_by_priority(priority_filter)
        articles = []
        timestamp = datetime.now().isoformat()
        successful_count = 0
        
        logger.info(f"Scraping {len(instruments)} instruments with priority <= {priority_filter}")
        
        for name, info in instruments.items():
            try:
                logger.info(f"Fetching {name} ({info['symbol']})")
                
                # Add small delay to be respectful
                time.sleep(0.1)
                
                # Get data from Yahoo Finance API
                ticker = yf.Ticker(info['symbol'])
                
                # Get comprehensive market data
                hist = ticker.history(period="5d")
                
                if len(hist) == 0:
                    logger.warning(f"No data available for {name} ({info['symbol']})")
                    continue
                
                # Get most recent values
                current_price = float(hist['Close'].iloc[-1])
                prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
                day_high = float(hist['High'].iloc[-1]) if 'High' in hist.columns else current_price
                day_low = float(hist['Low'].iloc[-1]) if 'Low' in hist.columns else current_price
                volume = int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0
                
                # Calculate changes
                change = current_price - prev_close
                change_percent = (change / prev_close * 100) if prev_close != 0 else 0
                
                # Create market data article
                article = {
                    'headline': f"{name} {change_percent:+.2f}% at ${current_price:.2f}",
                    'snippet': (f"{info['description']}\n"
                               f"Current: ${current_price:.2f} ({change_percent:+.2f}%)\n"
                               f"Previous Close: ${prev_close:.2f}\n"
                               f"Day Range: ${day_low:.2f} - ${day_high:.2f}\n"
                               f"Volume: {volume:,}"),
                    'timestamp': timestamp,
                    'asset_tags': [name],
                    'url': info['url'],
                    'source': 'yahoo_finance',
                    'scraped_at': timestamp,
                    'price': current_price,
                    'prev_close': prev_close,
                    'change': change,
                    'change_percent': change_percent,
                    'day_high': day_high,
                    'day_low': day_low,
                    'volume': volume,
                    'asset_type': info['type'],
                    'symbol': info['symbol'],
                    'priority': info['priority']
                }
                
                articles.append(article)
                successful_count += 1
                logger.info(f"Success {name}: ${current_price:.2f} ({change_percent:+.2f}%)")
                
            except Exception as e:
                logger.error(f"Error fetching {name}: {e}")
                continue
        
        return articles, successful_count


def scrape_yahoo_finance_data(priority_filter: int = 2):
    """
    Main function to scrape Yahoo Finance data with priority filtering
    Returns path to saved parquet file or None on failure
    """
    logger.info(f"Starting Yahoo Finance data collection (priority filter: {priority_filter})")
    
    scraper = YahooFinanceScraper()
    articles, successful_count = scraper.scrape_market_data(priority_filter)
    
    if not articles:
        logger.error("No market data collected")
        return None
    
    # Create market overview
    timestamp = datetime.now().isoformat()
    total_change = sum(abs(a['change_percent']) for a in articles)
    avg_abs_change = total_change / len(articles) if articles else 0
    market_stress_score = min(avg_abs_change * 3, 10)
    
    # Generate overview content
    overview_snippet = f"MARKET OVERVIEW REPORT\n"
    overview_snippet += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    overview_snippet += f"Data Points: {successful_count}\n"
    overview_snippet += f"Market Stress Score: {market_stress_score:.1f}/10\n\n"
    overview_snippet += "SIGNIFICANT MOVEMENTS:\n"
    
    # Sort by absolute change to highlight significant movements
    sorted_articles = sorted(articles, key=lambda x: abs(x['change_percent']), reverse=True)
    for article in sorted_articles[:6]:
        overview_snippet += f"  {article['asset_tags'][0]}: {article['change_percent']:+.2f}%\n"
    
    overview_article = {
        'headline': f'Market Overview - Stress Level {market_stress_score:.1f}/10',
        'snippet': overview_snippet,
        'timestamp': timestamp,
        'asset_tags': ['Market Overview'],
        'url': 'https://finance.yahoo.com',
        'source': 'yahoo_finance',
        'scraped_at': timestamp,
        'market_stress_score': market_stress_score,
        'data_points': successful_count,
        'asset_types': list(set(a['asset_type'] for a in articles))
    }
    
    # Insert overview as first article
    articles.insert(0, overview_article)
    
    # Save to bronze layer
    df = pd.DataFrame(articles)
    
    bronze_dir = Path("data/bronze")
    bronze_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"yahoo_finance_market_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    filepath = bronze_dir / filename
    
    try:
        df.to_parquet(filepath, index=False)
        logger.info(f"Saved {len(articles)} articles to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save parquet file: {e}")
        return None


def scrape_and_save():
    """
    Scrape real market data from Yahoo Finance.
    Uses priority 2 (critical + important assets) for balanced speed/coverage.
    
    Returns:
        Path to saved bronze file
    """
    # Use priority 2: Critical + Important assets (recommended)
    return scrape_yahoo_finance_data(priority_filter=2)


# For backward compatibility
__all__ = ['scrape_and_save', 'YahooFinanceScraper', 'scrape_yahoo_finance_data']
