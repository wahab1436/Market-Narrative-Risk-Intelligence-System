"""
Investing.com Human‑like Scraper
"""

import json
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cloudscraper
import pandas as pd
from bs4 import BeautifulSoup

try:
    from src.utils.logger import scraper_logger  # type: ignore
except Exception:
    import logging

    scraper_logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s – %(message)s",
    )


def _rand_headers(user_agents: List[str]) -> Dict[str, str]:
    return {
        "User-Agent": random.choice(user_agents),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Cache-Control": "max-age=0",
        "Referer": "https://www.google.com/",
    }


class SafeInvestingScraper:
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    ]

    def __init__(
        self,
        delay_range: Tuple[int, int] = (3, 7),
        max_retries: int = 3,
    ):
        self.base_url = "https://www.investing.com"
        self.delay_range = delay_range
        self.max_retries = max_retries
        self.request_count = 0
        self.failed_count = 0
        self.session = self._create_safe_session()
        self.instruments: Dict[str, Dict] = {
            "S&P 500": {"url": "/indices/us-spx-500", "type": "index", "priority": 1},
            "Dow Jones": {"url": "/indices/us-30", "type": "index", "priority": 1},
            "NASDAQ": {"url": "/indices/nasdaq-composite", "type": "index", "priority": 1},
            "VIX": {"url": "/indices/volatility-s-p-500", "type": "volatility", "priority": 1},
            "Russell 2000": {"url": "/indices/smallcap-2000", "type": "index", "priority": 2},
            "FTSE 100": {"url": "/indices/uk-100", "type": "index", "priority": 2},
            "DAX": {"url": "/indices/germany-30", "type": "index", "priority": 2},
            "Nikkei 225": {"url": "/indices/japan-ni225", "type": "index", "priority": 2},
            "Gold": {"url": "/commodities/gold", "type": "commodity", "priority": 1},
            "Crude Oil": {"url": "/commodities/crude-oil", "type": "commodity", "priority": 1},
            "Silver": {"url": "/commodities/silver", "type": "commodity", "priority": 2},
            "Natural Gas": {"url": "/commodities/natural-gas", "type": "commodity", "priority": 2},
            "Copper": {"url": "/commodities/copper", "type": "commodity", "priority": 3},
            "EUR/USD": {"url": "/currencies/eur-usd", "type": "forex", "priority": 1},
            "GBP/USD": {"url": "/currencies/gbp-usd", "type": "forex", "priority": 2},
            "USD/JPY": {"url": "/currencies/usd-jpy", "type": "forex", "priority": 2},
            "USD/CHF": {"url": "/currencies/usd-chf", "type": "forex", "priority": 3},
            "Bitcoin": {"url": "/crypto/bitcoin/usd", "type": "crypto", "priority": 1},
            "Ethereum": {"url": "/crypto/ethereum/usd", "type": "crypto", "priority": 2},
        }
        scraper_logger.info(
            "SafeInvestingScraper initialised – %d instruments (delay %ds‑%ds, retries %d)",
            len(self.instruments),
            self.delay_range[0],
            self.delay_range[1],
            self.max_retries,
        )
        self._warm_up_session()

    def _create_safe_session(self) -> cloudscraper.CloudScraper:
        return cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "mobile": False},
            delay=10,
            interpreter="native",
        )

    def _human_delay(self) -> None:
        delay = random.uniform(*self.delay_range) + random.uniform(-0.5, 0.5)
        time.sleep(max(0, delay))

    def _random_navigation(self) -> None:
        pages = [
            "/equities",
            "/commodities",
            "/currencies",
            "/crypto",
            "/news",
        ]
        if random.random() < 0.25:
            url = f"{self.base_url}{random.choice(pages)}"
            try:
                self.session.get(url, headers=_rand_headers(self.USER_AGENTS), timeout=15)
                time.sleep(random.uniform(1, 3))
            except Exception:
                pass

    def _warm_up_session(self) -> None:
        try:
            self.session.get(self.base_url, headers=_rand_headers(self.USER_AGENTS), timeout=15)
            time.sleep(random.uniform(1, 2))
        except Exception:
            pass

    def _safe_request(self, url: str) -> Optional[cloudscraper.CloudScraper]:
        for attempt in range(1, self.max_retries + 1):
            try:
                headers = _rand_headers(self.USER_AGENTS)
                resp = self.session.get(url, headers=headers, timeout=30, allow_redirects=True)
                self.request_count += 1
                if resp.status_code == 429:
                    wait = 2 ** attempt * 5
                    time.sleep(wait)
                    continue
                if resp.status_code == 403:
                    self.session = self._create_safe_session()
                    time.sleep(2 ** attempt * 3)
                    continue
                resp.raise_for_status()
                return resp
            except Exception as exc:
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
        self.failed_count += 1
        return None

    @staticmethod
    def _parse_price(text: str) -> float:
        if not text:
            return 0.0
        cleaned = (
            str(text)
            .replace(",", "")
            .replace("$", "")
            .replace("€", "")
            .replace("£", "")
            .replace("%", "")
            .replace("+", "")
            .replace(" ", "")
        )
        if "(" in cleaned and ")" in cleaned:
            cleaned = "-" + cleaned.replace("(", "").replace(")", "")
        m = re.search(r"-?\d+\.?\d*", cleaned)
        return float(m.group()) if m else 0.0

    def _extract_price_data(self, soup: BeautifulSoup) -> Dict[str, float]:
        data: Dict[str, float] = {}
        for script in soup.find_all("script", {"type": "application/ld+json"}):
            try:
                if not script.string:
                    continue
                payload = json.loads(script.string)
                if isinstance(payload, list):
                    for item in payload:
                        if not isinstance(item, dict) or "@type" not in item:
                            continue
                        if item["@type"] not in {"Product", "FinancialProduct"}:
                            continue
                        offers = item.get("offers")
                        if isinstance(offers, dict):
                            price = offers.get("price")
                            if price and "price" not in data:
                                data["price"] = float(price)
                        if "lowPrice" in item:
                            data["day_low"] = float(item["lowPrice"])
                        if "highPrice" in item:
                            data["day_high"] = float(item["highPrice"])
                elif isinstance(payload, dict):
                    if "price" in payload and "price" not in data:
                        data["price"] = float(payload["price"])
                    if "lowPrice" in payload:
                        data["day_low"] = float(payload["lowPrice"])
                    if "highPrice" in payload:
                        data["day_high"] = float(payload["highPrice"])
            except (json.JSONDecodeError, ValueError, TypeError, KeyError):
                continue
        if "price" not in data:
            price_tag = soup.find("div", {"data-test": "instrument-price-last"})
            if price_tag:
                data["price"] = self._parse_price(price_tag.get_text())
        change_tag = soup.find("span", {"data-test": "instrument-price-change"})
        if change_tag:
            data["change"] = self._parse_price(change_tag.get_text())
        change_pct_tag = soup.find("span", {"data-test": "instrument-price-change-percent"})
        if change_pct_tag:
            data["change_percent"] = self._parse_price(change_pct_tag.get_text())
        if "price" not in data:
            meta_price = soup.find("meta", {"property": "og:price:amount"})
            if meta_price and meta_price.get("content"):
                data["price"] = self._parse_price(meta_price["content"])
        if "price" not in data:
            for cls in ["text-5xl", "text-4xl", "instrument-price", "last-price"]:
                elem = soup.find(class_=re.compile(cls, re.I))
                if elem:
                    price = self._parse_price(elem.get_text())
                    if price > 0:
                        data["price"] = price
                        break
        if "price" in data and "change" in data and data["change"] != 0:
            data["prev_close"] = data["price"] - data["change"]
        if (
            "price" in data
            and "change_percent" in data
            and data["change_percent"] != 0
        ):
            if "prev_close" not in data:
                data["prev_close"] = data["price"] / (1 + data["change_percent"] / 100)
            if "change" not in data:
                data["change"] = data["price"] - data["prev_close"]
        return data

    def scrape_instrument(self, name: str, info: Dict) -> Optional[Dict]:
        url = f"{self.base_url}{info['url']}"
        scraper_logger.info("Scraping %s (%s)", name, info["type"])
        self._random_navigation()
        resp = self._safe_request(url)
        if resp is None:
            return None
        try:
            soup = BeautifulSoup(resp.content, "html.parser")
            price_data = self._extract_price_data(soup)
            if "price" not in price_data or price_data["price"] == 0:
                scraper_logger.warning("No valid price extracted for %s", name)
                return None
            result = {
                "asset": name,
                "symbol": info["url"].split("/")[-1].upper(),
                "type": info["type"],
                "priority": info.get("priority", 3),
                "url": url,
                "timestamp": datetime.now().isoformat(),
                "source": "investing.com",
                "price": round(float(price_data["price"]), 4),
                "change": round(float(price_data.get("change", 0.0))),
                "change_percent": round(float(price_data.get("change_percent", 0.0))),
                "prev_close": round(float(price_data.get("prev_close", price_data["price"]))),
                "day_low": price_data.get("day_low"),
                "day_high": price_data.get("day_high"),
            }
            scraper_logger.info(
                "✓ %s: $%s (%+.2f%%) change $%+.2f",
                name,
                result["price"],
                result["change_percent"],
                result["change"],
            )
            return result
        except Exception as exc:
            scraper_logger.error("Parsing error for %s – %s", name, exc, exc_info=True)
            return None

    def scrape_all(self, priority_filter: Optional[int] = None) -> List[Dict]:
        scraper_logger.info("=" * 70)
        scraper_logger.info("STARTING SAFE INVESTING.COM SCRAPING")
        scraper_logger.info("=" * 70)
        if priority_filter:
            instruments = {
                n: i
                for n, i in self.instruments.items()
                if i.get("priority", 3) <= priority_filter
            }
            scraper_logger.info(
                "Filtering to priority ≤ %d → %d instruments", priority_filter, len(instruments)
            )
        else:
            instruments = self.instruments
        start = time.time()
        collected: List[Dict] = []
        for idx, (name, info) in enumerate(instruments.items(), start=1):
            scraper_logger.info("\n[%d/%d] %s", idx, len(instruments), name)
            data = self.scrape_instrument(name, info)
            if data:
                collected.append(data)
            if idx < len(instruments):
                self._human_delay()
        elapsed = time.time() - start
        success_pct = len(collected) / len(instruments) * 100 if instruments else 0
        scraper_logger.info("\n" + "=" * 70)
        scraper_logger.info("SCRAPING COMPLETE")
        scraper_logger.info("=" * 70)
        scraper_logger.info(
            "Success: %d / %d (%.1f%%)", len(collected), len(instruments), success_pct
        )
        scraper_logger.info("Failed requests: %d", self.failed_count)
        scraper_logger.info("Total HTTP calls: %d", self.request_count)
        scraper_logger.info("Elapsed: %.1f s, avg %.2f s per call", elapsed, elapsed / max(self.request_count, 1))
        scraper_logger.info("=" * 70)
        return collected

    def calculate_market_stress(self, market_data: List[Dict]) -> float:
        if not market_data:
            return 0.0
        factors: List[float] = []
        for d in market_data:
            change = d.get("change_percent", 0.0)
            pri = d.get("priority", 3)
            if "VIX" in d["asset"]:
                factors.append(min(d["price"] / 10, 10) * 3.0)
            move_factor = min(abs(change) / 2, 10) * (4 - pri)
            factors.append(move_factor)
            if change < -3:
                factors.append(min(abs(change), 10) * 1.5)
        avg = sum(factors) / len(factors) if factors else 0.0
        return round(min(avg, 10.0), 2)

    def create_articles(self, market_data: List[Dict]) -> List[Dict]:
        if not market_data:
            scraper_logger.warning("No market data – no articles to create")
            return []
        timestamp = datetime.now().isoformat()
        stress = self.calculate_market_stress(market_data)
        overview_lines = [
            f"Market Stress Score: {stress:.1f}/10",
            f"Data points: {len(market_data)}",
            f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        by_type: Dict[str, List[Dict]] = {}
        for d in market_data:
            by_type.setdefault(d["type"], []).append(d)
        for typ, items in sorted(by_type.items()):
            overview_lines.append(f"{typ.upper()}:")
            for i in sorted(items, key=lambda x: abs(x.get("change_percent", 0)), reverse=True):
                overview_lines.append(
                    f"  {i['asset']}: ${i['price']:.2f} ({i['change_percent']:+.2f}%, ${i['change']:+.2f})"
                )
            overview_lines.append("")
        overview_article = {
            "headline": f"Market Overview – Stress {stress:.1f}/10",
            "snippet": "\n".join(overview_lines)[:2000],
            "timestamp": timestamp,
            "asset_tags": [d["asset"] for d in market_data],
            "url": "https://www.investing.com",
            "source": "investing.com",
            "scraped_at": timestamp,
            "market_stress_score": stress,
            "data_points": len(market_data),
            "asset_types": list(by_type.keys()),
            "scraper_stats": {
                "requests": self.request_count,
                "failures": self.failed_count,
                "success_rate": f"{(len(market_data)/self.request_count*100):.1f}%" if self.request_count else "0%",
            },
        }
        articles = [overview_article]
        for d in market_data:
            pct = abs(d.get("change_percent", 0))
            make_article = (
                pct > 2.0
                or (d.get("priority") == 1 and pct > 1.0)
                or ("VIX" in d["asset"])
            )
            if not make_article:
                continue
            direction = (
                "surges"
                if d["change_percent"] > 3
                else "rises"
                if d["change_percent"] > 0
                else "plunges"
                if d["change_percent"] < -3
                else "falls"
            )
            headline = f"{d['asset']} {direction} {pct:.1f}% to ${d['price']:.2f}"
            snippet_parts = [
                f"{d['asset']} ({d['type']}) is at ${d['price']:.2f}, ",
                f"{direction} {pct:.1f}% from the previous close of ${d['prev_close']:.2f}. ",
                f"Absolute change: ${abs(d['change']):.2f}. ",
            ]
            if d.get("day_low") and d.get("day_high"):
                snippet_parts.append(
                    f"Today's range: ${d['day_low']:.2f} – ${d['day_high']:.2f}. "
                )
            if "VIX" in d["asset"]:
                if d["price"] > 30:
                    snippet_parts.append("Fear index is high – markets volatile. ")
                elif d["price"] < 15:
                    snippet_parts.append("Fear index low – complacent markets. ")
            articles.append(
                {
                    "headline": headline,
                    "snippet": "".join(snippet_parts),
                    "timestamp": timestamp,
                    "asset_tags": [d["asset"]],
                    "url": d["url"],
                    "source": "investing.com",
                    "scraped_at": timestamp,
                    "price": d["price"],
                    "change": d["change"],
                    "change_percent": d["change_percent"],
                    "asset_type": d["type"],
                    "priority": d.get("priority", 3),
                }
            )
        scraper_logger.info("Created %d articles (incl. overview)", len(articles))
        return articles

    def save_to_bronze(self, articles: List[Dict]) -> Optional[Path]:
        if not articles:
            scraper_logger.warning("No articles to persist – skipping bronze write")
            return None
        df = pd.DataFrame(articles)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"investing_market_{ts}.parquet"
        out_path = Path("data/bronze") / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False, engine="pyarrow")
        scraper_logger.info("✓ Saved %d articles to %s", len(df), out_path)
        return out_path


def scrape_investing_data(priority_filter: Optional[int] = None) -> Optional[Path]:
    scraper = SafeInvestingScraper(delay_range=(3, 7), max_retries=3)
    try:
        market = scraper.scrape_all(priority_filter=priority_filter)
        if not market:
            scraper_logger.error("No market data collected – aborting")
            return None
        articles = scraper.create_articles(market)
        return scraper.save_to_bronze(articles)
    except KeyboardInterrupt:
        scraper_logger.warning("Scraping interrupted by user")
        return None
    except Exception as exc:
        scraper_logger.error("Unexpected error during scraping: %s", exc, exc_info=True)
        return None


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("INVESTING.COM SAFE SCRAPER – Cloudflare Bypass")
    print("=" * 70)
    print("\nPriority levels:")
    print("  1 = Critical only (VIX, S&P 500, Gold …)")
    print("  2 = Critical + Important")
    print("  3 = All assets")
    print("\n" + "=" * 70 + "\n")
    result_path = scrape_investing_data(priority_filter=2)
    if result_path:
        print("\n" + "=" * 70)
        print("SCRAPING SUCCESSFUL")
        print("=" * 70)
        print(f"\nSaved to: {result_path}")
        df = pd.read_parquet(result_path)
        print(f"\nCollected {len(df)} articles")
        if not df.empty:
            print("\n--- Overview snippet ---")
            overview = df[df["headline"].str.contains("Overview", na=False)].iloc[0]["snippet"]
            print(overview[:500] + "…")
    else:
        print("\n" + "=" * 70)
        print("SCRAPING FAILED – see log output")
        print("=" * 70)
