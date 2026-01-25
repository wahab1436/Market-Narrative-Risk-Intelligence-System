"""market_data_yahoo.py

A self‑contained scraper that:
* pulls real‑time prices from Yahoo Finance (chart endpoint, with quote fallback)
* builds a tiny “article” payload
* writes a proper parquet file in the bronze layer
* never creates a 1‑byte placeholder
"""

import time
import json
import random
import logging
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

import requests
import pandas as pd

# ----------------------------------------------------------------------
# Logging – use the project logger if it exists, otherwise fall back to a basic logger
# ----------------------------------------------------------------------
try:
    from src.utils.logger import scraper_logger  # type: ignore
except Exception:  # pragma: no cover
    scraper_logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s – %(message)s")

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def _rand_headers() -> Dict[str, str]:
    """Return a copy of the base headers with a randomised User‑Agent."""
    base = {
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    }
    agents = [
        # Chrome on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        # Chrome on macOS
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        # Firefox on Linux
        "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0",
    ]
    base["User-Agent"] = random.choice(agents)
    return base


# ----------------------------------------------------------------------
# The scraper class
# ----------------------------------------------------------------------
class SafeInvestingScraper:
    """
    Market‑data scraper that uses Yahoo Finance (chart endpoint + quote fallback).
    """

    # ------------------------------------------------------------------
    # 1️⃣  Configuration
    # ------------------------------------------------------------------
    def __init__(self):
        self.base_chart_url = "https://query1.finance.yahoo.com/v8/finance/chart"
        self.base_quote_url = "https://query1.finance.yahoo.com/v7/finance/quote"

        # Mapping of instrument → info (symbol, type, priority, description)
        self.instruments = {
            # ── Critical (priority 1) ────────────────────────────────────────────────────────
            "VIX": {"symbol": "^VIX", "type": "volatility", "priority": 1,
                    "description": "CBOE Volatility Index (Fear Gauge)"},
            "S&P 500": {"symbol": "^GSPC", "type": "index", "priority": 1,
                        "description": "S&P 500 Index"},
            "Dow Jones": {"symbol": "^DJI", "type": "index", "priority": 1,
                          "description": "Dow Jones Industrial Average"},
            "NASDAQ": {"symbol": "^IXIC", "type": "index", "priority": 1,
                       "description": "NASDAQ Composite"},
            # ── Important (priority 2) ───────────────────────────────────────────────────────
            "Russell 2000": {"symbol": "^RUT", "type": "index", "priority": 2,
                             "description": "Russell 2000 Small‑Cap Index"},
            "Gold": {"symbol": "GC=F", "type": "commodity", "priority": 2,
                     "description": "Gold Futures"},
            "Crude Oil": {"symbol": "CL=F", "type": "commodity", "priority": 2,
                         "description": "Crude Oil WTI Futures"},
            "EUR/USD": {"symbol": "EURUSD=X", "type": "forex", "priority": 2,
                        "description": "Euro vs US Dollar"},
            "10Y Treasury": {"symbol": "^TNX", "type": "bond", "priority": 2,
                            "description": "US 10‑Year Treasury Yield"},
            # ── Additional (priority 3) ───────────────────────────────────────────────────
            "Silver": {"symbol": "SI=F", "type": "commodity", "priority": 3,
                       "description": "Silver Futures"},
            "Natural Gas": {"symbol": "NG=F", "type": "commodity", "priority": 3,
                            "description": "Natural Gas Futures"},
            "GBP/USD": {"symbol": "GBPUSD=X", "type": "forex", "priority": 3,
                        "description": "British Pound vs US Dollar"},
            "USD/JPY": {"symbol": "USDJPY=X", "type": "forex", "priority": 3,
                        "description": "US Dollar vs Japanese Yen"},
            "Bitcoin": {"symbol": "BTC-USD", "type": "crypto", "priority": 3,
                        "description": "Bitcoin USD"},
            "Ethereum": {"symbol": "ETH-USD", "type": "crypto", "priority": 3,
                         "description": "Ethereum USD"},
        }

        # Persistent session – we only ever mutate headers per request (UA rotation)
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
        })

    # ------------------------------------------------------------------
    # 2️⃣  Helper – pick instruments by priority
    # ------------------------------------------------------------------
    def get_instruments_by_priority(self, max_priority: int = 3) -> Dict[str, Dict]:
        """Return a dict of instruments whose priority ≤ ``max_priority``."""
        return {
            name: info
            for name, info in self.instruments.items()
            if info["priority"] <= max_priority
        }

    # ------------------------------------------------------------------
    # 3️⃣  Helper – extract price fields from a chart payload
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_fields_from_chart(result: dict) -> dict:
        """
        Pull price‑related fields from a *chart* response.
        Falls back to the ``quote``‑style arrays that appear in the same payload.
        """
        meta = result.get("meta", {})
        quote = result.get("indicators", {}).get("quote", [{}])[0]

        def _val(key, default=None):
            # Prefer meta, then quote array, finally default
            if meta.get(key) is not None:
                return meta[key]
            return quote.get(key, default)

        return {
            "price": _val("regularMarketPrice"),
            "prev_close": _val("previousClose"),
            "day_high": _val("regularMarketDayHigh"),
            "day_low": _val("regularMarketDayLow"),
            "volume": _val("regularMarketVolume"),
        }

    # ------------------------------------------------------------------
    # 4️⃣  Helper – fallback to the "quote" endpoint if chart is empty
    # ------------------------------------------------------------------
    def _fallback_quote(self, symbol: str) -> Optional[dict]:
        """Call the simple quote endpoint; returns same dict shape as chart extraction."""
        try:
            url = f"{self.base_quote_url}"
            resp = self.session.get(url, params={"symbols": symbol}, timeout=8, headers=_rand_headers())
            resp.raise_for_status()
            data = resp.json()
            result = data.get("quoteResponse", {}).get("result", [])
            if not result:
                return None
            r = result[0]
            return {
                "price": r.get("regularMarketPrice"),
                "prev_close": r.get("regularMarketPreviousClose"),
                "day_high": r.get("regularMarketDayHigh"),
                "day_low": r.get("regularMarketDayLow"),
                "volume": r.get("regularMarketVolume"),
            }
        except Exception as exc:  # pragma: no cover
            scraper_logger.debug("Quote fallback failed for %s – %s", symbol, exc)
            return None

    # ------------------------------------------------------------------
    # 5️⃣  Core fetch for a single instrument
    # ------------------------------------------------------------------
    def fetch_instrument_data(self, name: str, info: dict) -> Optional[dict]:
        symbol = info["symbol"]
        try:
            scraper_logger.info("Fetching %s (%s)", name, symbol)

            # ------------------------------
            # 5.1 Call chart endpoint
            # ------------------------------
            chart_url = f"{self.base_chart_url}/{symbol}"
            params = {"interval": "1d", "range": "5d", "includePrePost": "false"}
            resp = self.session.get(
                chart_url,
                params=params,
                timeout=12,
                headers=_rand_headers(),
            )
            resp.raise_for_status()
            payload = resp.json()

            # ------------------------------
            # 5.2 Validate payload
            # ------------------------------
            if (
                "chart" not in payload
                or "result" not in payload["chart"]
                or not payload["chart"]["result"]
            ):
                scraper_logger.warning("Empty chart payload for %s – trying quote fallback", name)
                fields = self._fallback_quote(symbol)
                if not fields:
                    scraper_logger.error("Both chart & quote failed for %s", name)
                    return None
            else:
                result = payload["chart"]["result"][0]
                fields = self._extract_fields_from_chart(result)

            # ------------------------------------------------------------------
            # 5.3 Pull the individual numbers – guard against ``None``
            # ------------------------------------------------------------------
            price = fields["price"]
            if price is None:
                scraper_logger.warning("No price in payload for %s", name)
                return None

            prev_close = fields["prev_close"] if fields["prev_close"] is not None else price
            day_high = fields["day_high"] if fields["day_high"] is not None else price
            day_low = fields["day_low"] if fields["day_low"] is not None else price
            volume = fields["volume"]

            # ------------------------------
            # 5.4 Compute change & percentages
            # ------------------------------
            change = price - prev_close
            change_pct = (change / prev_close) * 100 if prev_close != 0 else 0.0

            market_data = {
                "asset": name,
                "symbol": symbol,
                "type": info["type"],
                "priority": info["priority"],
                "description": info["description"],
                "price": round(float(price), 4),
                "prev_close": round(float(prev_close), 4),
                "change": round(float(change), 4),
                "change_percent": round(float(change_pct), 4),
                "day_high": round(float(day_high), 4),
                "day_low": round(float(day_low), 4),
                "timestamp": datetime.now().isoformat(),
                "source": "yahoo_finance",
                "url": f"https://finance.yahoo.com/quote/{symbol}",
            }

            if volume is not None:
                market_data["volume"] = int(volume)

            scraper_logger.info(
                "✓ %s – $%s (%+.2f%%)", name, market_data["price"], market_data["change_percent"]
            )
            return market_data

        except requests.Timeout:
            scraper_logger.error("Timeout while fetching %s", name)
        except requests.RequestException as exc:
            scraper_logger.error("Network error for %s – %s", name, exc)
        except Exception as exc:  # pragma: no cover
            scraper_logger.exception("Unexpected error for %s – %s", name, exc)
        return None

    # ------------------------------------------------------------------
    # 6️⃣  Polite sleep (random jitter)
    # ------------------------------------------------------------------
    @staticmethod
    def _polite_sleep():
        time.sleep(random.uniform(0.8, 1.5))

    # ------------------------------------------------------------------
    # 7️⃣  The public scrape loop (priority filter)
    # ------------------------------------------------------------------
    def scrape_data(self, priority_filter: int = 2) -> List[Dict]:
        instruments = self.get_instruments_by_priority(priority_filter)

        scraper_logger.info("-" * 60)
        scraper_logger.info("YAHOO FINANCE – MARKET DATA COLLECTION")
        scraper_logger.info("Priority filter ≤ %s   →   %s instruments", priority_filter, len(instruments))
        scraper_logger.info("-" * 60)

        collected = []
        failed = []

        for name, info in instruments.items():
            data = self.fetch_instrument_data(name, info)
            if data:
                collected.append(data)
            else:
                failed.append(name)

            # Polite delay between calls
            self._polite_sleep()

        scraper_logger.info("-" * 60)
        scraper_logger.info("Collected %s / %s", len(collected), len(instruments))
        if failed:
            scraper_logger.warning("Failed: %s", ", ".join(failed))
        scraper_logger.info("-" * 60)

        return collected

    # ------------------------------------------------------------------
    # 8️⃣  Market‑stress scoring – the only line that needed syntax fixing
    # ------------------------------------------------------------------
    def calculate_market_stress_score(self, market_data: List[Dict]) -> float:
        """Return a 0 – 10 stress score (higher = more stressed)."""
        if not market_data:
            return 0.0

        components = []

        for asset in market_data:
            name = asset["asset"]
            change_pct = asset.get("change_percent", 0.0)
            price = asset.get("price", 0.0)
            priority = asset.get("priority", 3)

            # VIX is a direct fear gauge – scale it (price ≈ 10 → 10 stress)
            if "VIX" in name and price > 0:
                vix_score = min(price / 10, 10)
                components.append(vix_score * 3.0)               # give VIX extra weight

            # Sharp index drops (more weight for priority‑1)
            if asset.get("type") == "index" and change_pct < -1:
                weight = 2.5 if priority == 1 else 1.5
                components.append(min(abs(change_pct) * 2, 10) * weight)

            # Safe‑haven spikes: VIX, Gold, 10‑Y Treasury
            if name in {"VIX", "Gold", "10Y Treasury"} and change_pct > 3:
                components.append(min(change_pct * 1.5, 10))   # ← fixed extra “)”

            # Commodity volatility
            if asset.get("type") == "commodity" and abs(change_pct) > 3:
                components.append(min(abs(change_pct) * 1.2, 10))

        if not components:
            return 0.0

        score = min(sum(components) / len(components), 10)
        return round(score, 2)

    # ------------------------------------------------------------------
    # 9️⃣  Turn market rows into “articles”
    # ------------------------------------------------------------------
    def create_market_articles(self, market_data: List[Dict]) -> List[Dict]:
        if not market_data:
            scraper_logger.warning("No market data received → no articles")
            return []

        stress_score = self.calculate_market_stress_score(market_data)
        timestamp = datetime.now().isoformat()

        # ------------------------------------------------------------------
        # 9.1  Build the “overview” snippet
        # ------------------------------------------------------------------
        by_type: dict[str, List[Dict]] = {}
        for item in market_data:
            by_type.setdefault(item.get("type", "unknown"), []).append(item)

        lines = [
            f"Market Stress Score: {stress_score}/10",
            f"Data points collected: {len(market_data)}",
            f"Source: Yahoo Finance (real‑time)",
            f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        for t, assets in sorted(by_type.items()):
            lines.append(f"{t.upper()}:")
            for a in sorted(assets, key=lambda x: x["priority"]):
                lines.append(
                    f"  {a['asset']}: ${a['price']:.2f} ({a['change_percent']:+.2f}%)"
                )
            lines.append("")  # blank line between groups

        overview_article = {
            "headline": f"Market Overview – Stress {stress_score}/10",
            "snippet": "\n".join(lines)[:1000],
            "timestamp": timestamp,
            "asset_tags": [d["asset"] for d in market_data],
            "url": "https://finance.yahoo.com",
            "source": "yahoo_finance",
            "scraped_at": timestamp,
            "market_stress_score": stress_score,
            "data_points": len(market_data),
            "asset_types": list(by_type.keys()),
        }

        # ------------------------------------------------------------------
        # 9.2  Build “movers” articles (individual assets)
        # ------------------------------------------------------------------
        movers = []
        for a in market_data:
            pct = a.get("change_percent", 0.0)
            # Show anything that moved >1 % (indexes) or >2 % (others), plus VIX always
            thr = 1.0 if a["type"] == "index" else 2.0
            if abs(pct) > thr or "VIX" in a["asset"]:
                direction = (
                    "surges"
                    if pct > 2
                    else "rises"
                    if pct > 0
                    else "plunges"
                    if pct < -2
                    else "falls"
                )
                headline = f"{a['asset']} {direction} {abs(pct):.2f}% to ${a['price']:.2f}"
                snippet = (
                    f"{a['description']}\n"
                    f"Current price: ${a['price']:.2f} ({pct:+.2f}%)\n"
                    f"Previous close: ${a['prev_close']:.2f}\n"
                    f"Day range: ${a['day_low']:.2f} – ${a['day_high']:.2f}\n"
                    f"Source: Yahoo Finance"
                )
                movers.append(
                    {
                        "headline": headline,
                        "snippet": snippet,
                        "timestamp": timestamp,
                        "asset_tags": [a["asset"]],
                        "url": a["url"],
                        "source": "yahoo_finance",
                        "scraped_at": timestamp,
                        "price": a["price"],
                        "change_percent": pct,
                        "asset_type": a["type"],
                        "priority": a["priority"],
                    }
                )

        return [overview_article] + movers

    # ------------------------------------------------------------------
    # 🔟  Persist to the bronze layer (parquet)
    # ------------------------------------------------------------------
    def save_to_bronze(self, articles: List[Dict]) -> Optional[Path]:
        if not articles:
            scraper_logger.warning("No articles to persist → skipping bronze write")
            return None

        try:
            df = pd.DataFrame(articles)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"market_data_{ts}.parquet"
            out_path = Path("data/bronze") / filename
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Use pyarrow (fast) and write an index‑free table
            df.to_parquet(out_path, index=False, engine="pyarrow")
            scraper_logger.info("✓ Saved %d articles to %s", len(df), out_path)
            return out_path
        except Exception as exc:
            scraper_logger.error("Failed to write parquet – %s", exc)
            return None


# ----------------------------------------------------------------------
# 🎬  Convenience entry‑point (mirrors your original “if __name__ == '__main__'”)
# ----------------------------------------------------------------------
def scrape_yahoo_data(priority_filter: int = 2) -> Optional[Path]:
    """
    Orchestrates the whole pipeline:
    1️⃣ scrape → 2️⃣ turn into articles → 3️⃣ write parquet.
    Returns the parquet Path on success, ``None`` on total failure.
    """
    scraper = SafeInvestingScraper()
    scraper_logger.info("=" * 60)
    scraper_logger.info("MARKET DATA SCRAPER – YAHOO FINANCE")
    scraper_logger.info("Priority filter: %s", priority_filter)
    scraper_logger.info("=" * 60)

    start = time.time()

    market = scraper.scrape_data(priority_filter)
    if not market:
        scraper_logger.error("No market data collected – aborting")
        return None

    articles = scraper.create_market_articles(market)
    parquet_path = scraper.save_to_bronze(articles)

    elapsed = time.time() - start
    scraper_logger.info("-" * 60)
    scraper_logger.info("✅ Finished in %.2f s → %s", elapsed, parquet_path)
    scraper_logger.info("-" * 60)

    return parquet_path


# ----------------------------------------------------------------------
# 🧪  Small demo when you run the file directly
# ----------------------------------------------------------------------
if __name__ == "__main__":
    result = scrape_yahoo_data(priority_filter=2)

    if result:
        print("\n" + "=" * 60)
        print("✅ SUCCESS – parquet written to:", result)
        try:
            df = pd.read_parquet(result)
            print(f"Rows: {len(df)}  |  Columns: {list(df.columns)}")
            print("\n--- Overview snippet ---")
            print(df.loc[df["headline"].str.contains("Overview", na=False), "snippet"].iloc[0][:500])
        except Exception as e:  # pragma: no cover
            print("Error loading the parquet:", e)
    else:
        print("\n❌ SCRAPER FAILED – see logs for details")
