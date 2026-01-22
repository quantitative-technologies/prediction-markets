"""
Count single-condition markets using py-clob-client only.

Per paper Section 4.1: "We retrieved market metadata directly from the
Polymarket API using the official Python client [24]."
[24] references py-clob-client.

No Gamma API usage - all data comes from CLOB.
"""
import json
import os
import sys
import time

from py_clob_client.client import ClobClient
from datetime import datetime

client = ClobClient("https://clob.polymarket.com")

START_DATE = datetime(2024, 4, 1)
END_DATE = datetime(2025, 4, 1)

ESTIMATED_CLOB = 350000

CACHE_DIR = "cache"
CLOB_CACHE = os.path.join(CACHE_DIR, "clob_markets.json")
RESULTS_FILE = "single_condition_markets.json"


def draw_progress_bar(current, total, start_time, label=""):
    """Draw a progress bar to the terminal."""
    bar_length = 30
    percent = min(1.0, current / total)
    filled = int(bar_length * percent)
    bar = "=" * filled + ">" + " " * (bar_length - filled - 1)

    elapsed = time.time() - start_time
    rate = current / elapsed if elapsed > 0 else 0
    remaining = total - current
    eta = remaining / rate if rate > 0 else 0

    prefix = f"{label}: " if label else ""
    sys.stdout.write(
        f"\r{prefix}[{bar}] {percent*100:5.1f}% | {current:,}/{total:,} | "
        f"{rate:,.0f}/s | ETA: {eta:.0f}s"
    )
    sys.stdout.flush()


def load_or_fetch_clob_markets():
    """Load CLOB markets from cache or fetch from API."""
    if os.path.exists(CLOB_CACHE):
        print(f"Loading from cache: {CLOB_CACHE}")
        with open(CLOB_CACHE) as f:
            return json.load(f)

    all_markets = []
    cursor = "MA=="
    start_time = time.time()

    while True:
        response = client.get_markets(next_cursor=cursor)
        data = response['data']
        all_markets.extend(data)

        draw_progress_bar(len(all_markets), ESTIMATED_CLOB, start_time, "CLOB")

        if len(data) < response['limit']:
            break
        cursor = response['next_cursor']

    sys.stdout.write("\n")

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CLOB_CACHE, "w") as f:
        json.dump(all_markets, f)
    print(f"Saved to cache: {CLOB_CACHE}")

    return all_markets


def parse_end_date(m):
    """Parse end_date_iso from market, handling timezone."""
    end_date_str = m.get('end_date_iso')
    if not end_date_str:
        return None

    try:
        # Handle ISO format with Z suffix
        if end_date_str.endswith('Z'):
            end_date_str = end_date_str[:-1]
        # Parse datetime (ignore timezone for comparison)
        return datetime.fromisoformat(end_date_str.split('+')[0])
    except (ValueError, TypeError):
        return None


def filter_single_condition_markets(markets):
    """Filter single-condition markets with end_date within study period.

    Uses end_date_iso from CLOB data (no Gamma API).
    """
    results = []
    no_end_date = 0
    outside_range = 0

    for m in markets:
        # Single-condition = neg_risk is False
        if m['neg_risk']:
            continue

        # Must have exactly 2 tokens (binary market)
        if len(m['tokens']) != 2:
            continue

        # Parse end date from CLOB data
        end_date = parse_end_date(m)
        if end_date is None:
            no_end_date += 1
            continue

        # Filter by study period
        if not (START_DATE <= end_date <= END_DATE):
            outside_range += 1
            continue

        result = {
            "condition_id": m['condition_id'],
            "question": m['question'],
            "end_date_iso": m['end_date_iso'],
            "tokens": m['tokens'],
            "market_slug": m['market_slug'],
        }
        results.append(result)

    print(f"  Filtered out: {no_end_date} (no end date), {outside_range} (outside study period)")
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("SINGLE-CONDITION MARKET COUNTER")
    print("Using py-clob-client only (no Gamma API)")
    print("=" * 60)
    print(f"Study period: {START_DATE.date()} to {END_DATE.date()}")
    print()

    print("Loading CLOB markets...")
    markets = load_or_fetch_clob_markets()
    print(f"Total CLOB markets: {len(markets):,}")

    print("\nFiltering single-condition markets...")
    results = filter_single_condition_markets(markets)

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results):,} markets to {RESULTS_FILE}")

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Single-condition markets (end_date in study period): {len(results):,}")
    print(f"Paper Section 4.1 reference: 8,659")
    print()
