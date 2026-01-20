import json
import os
import sys
import time

import requests
from py_clob_client.client import ClobClient
from datetime import datetime

client = ClobClient("https://clob.polymarket.com")

START_DATE = datetime(2024, 4, 1)
END_DATE = datetime(2025, 4, 1)

GAMMA_API_URL = "https://gamma-api.polymarket.com/markets"
ESTIMATED_GAMMA = 50000
ESTIMATED_CLOB = 350000

CACHE_DIR = "cache"
CLOB_CACHE = os.path.join(CACHE_DIR, "clob_markets.json")
GAMMA_CACHE = os.path.join(CACHE_DIR, "gamma_closing_dates.json")
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


def load_or_fetch_closing_dates():
    """Load closing dates from cache or fetch from Gamma API."""
    if os.path.exists(GAMMA_CACHE):
        print(f"Loading from cache: {GAMMA_CACHE}")
        with open(GAMMA_CACHE) as f:
            return json.load(f)

    closing_dates = {}
    offset = 0
    limit = 500
    start_time = time.time()

    while True:
        response = requests.get(
            GAMMA_API_URL,
            params={
                "closed": "true",
                "limit": limit,
                "offset": offset,
                "end_date_min": "2024-01-01",
                "end_date_max": "2025-07-01",
            },
            timeout=30,
        )
        response.raise_for_status()
        markets = response.json()

        if not markets:
            break

        for m in markets:
            closed_time = m.get('closedTime')
            if closed_time:
                closing_dates[m['conditionId']] = closed_time

        offset += len(markets)
        draw_progress_bar(offset, ESTIMATED_GAMMA, start_time, "Gamma")

        if len(markets) < limit:
            break

        time.sleep(0.05)

    sys.stdout.write("\n")

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(GAMMA_CACHE, "w") as f:
        json.dump(closing_dates, f)
    print(f"Saved to cache: {GAMMA_CACHE}")

    return closing_dates


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


def filter_single_condition_markets(markets, closing_dates):
    """Filter and return single-condition markets resolved within the date range."""
    results = []
    missing = 0

    for m in markets:
        if m['neg_risk']:
            continue

        assert len(m['tokens']) == 2

        condition_id = m['condition_id']
        if condition_id not in closing_dates:
            missing += 1
            continue

        closed_time_str = closing_dates[condition_id]
        closed_time = datetime.strptime(closed_time_str[:19], "%Y-%m-%d %H:%M:%S")

        if not (START_DATE <= closed_time <= END_DATE):
            continue

        # Combine CLOB market data with closing time from Gamma
        result = {
            "condition_id": condition_id,
            "question": m['question'],
            "closed_time": closed_time_str,
            "end_date_iso": m['end_date_iso'],
            "tokens": m['tokens'],
            "market_slug": m['market_slug'],
        }
        results.append(result)

    print(f"  (missing closing dates: {missing})")
    return results


if __name__ == "__main__":
    print("Loading closing dates...")
    closing_dates = load_or_fetch_closing_dates()
    print(f"Closing dates: {len(closing_dates)}")

    print("\nLoading CLOB markets...")
    markets = load_or_fetch_clob_markets()
    print(f"CLOB markets: {len(markets)}")

    print("\nFiltering single-condition markets...")
    results = filter_single_condition_markets(markets, closing_dates)

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} markets to {RESULTS_FILE}")

    print(f"\nSingle-condition markets resolved Apr 2024 - Apr 2025: {len(results)}")
    print(f"Paper reference: 8659")
