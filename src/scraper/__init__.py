"""
Scraper module - Use existing scraper file
"""
# Import from the existing investing_scraper.py file
try:
    from src.scraper.investing_scraper import FastRSSNewsScraper, scrape_and_save as scrape_news
    
    def scrape_and_save():
        """Use existing RSS scraper."""
        return scrape_news()
        
except ImportError:
    # Fallback: Create minimal scraper
    import pandas as pd
    from pathlib import Path
    from datetime import datetime
    
    def scrape_and_save():
        """Minimal fallback scraper."""
        # Create sample data
        data = [{
            'headline': 'Market Update',
            'snippet': 'Sample market data',
            'timestamp': datetime.now().isoformat(),
            'asset_tags': [],
            'url': 'https://finance.yahoo.com',
            'source': 'sample',
            'scraped_at': datetime.now().isoformat()
        }]
        
        df = pd.DataFrame(data)
        filepath = Path("data/bronze") / f"sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(filepath, index=False)
        
        return filepath

__all__ = ['scrape_and_save']
