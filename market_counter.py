import sys
import time

from py_clob_client.client import ClobClient
from datetime import datetime

client = ClobClient("https://clob.polymarket.com")

START_DATE = datetime(2024, 4, 1)
END_DATE = datetime(2025, 4, 1)

ESTIMATED_TOTAL = 350000  # Based on previous runs


def draw_progress_bar(current, total, start_time):
    """Draw a progress bar to the terminal."""
    bar_length = 30
    percent = min(1.0, current / total)
    filled = int(bar_length * percent)
    bar = "=" * filled + ">" + " " * (bar_length - filled - 1)

    elapsed = time.time() - start_time
    rate = current / elapsed if elapsed > 0 else 0
    remaining = total - current
    eta = remaining / rate if rate > 0 else 0

    sys.stdout.write(
        f"\r[{bar}] {percent*100:5.1f}% | {current:,}/{total:,} | "
        f"{rate:,.0f}/s | ETA: {eta:.0f}s"
    )
    sys.stdout.flush()


def fetch_all_markets():
    """Fetch all markets from CLOB API with pagination."""
    all_markets = []
    cursor = "MA=="  # Initial cursor (base64 for "0")
    start_time = time.time()

    while True:
        response = client.get_markets(next_cursor=cursor)
        data = response['data']
        all_markets.extend(data)

        draw_progress_bar(len(all_markets), ESTIMATED_TOTAL, start_time)

        if len(data) < response['limit']:
            break
        cursor = response['next_cursor']

    sys.stdout.write("\n")
    return all_markets


def count_single_condition_markets(markets):
    """Count single-condition markets resolved within the date range."""
    count = 0

    for m in markets:
        # Skip NegRisk (multi-outcome) markets
        if m['neg_risk']:
            continue

        # Single-condition markets always have exactly 2 tokens (YES/NO)
        assert len(m['tokens']) == 2

        # Filter by end date within range
        end_date_str = m['end_date_iso']
        if not end_date_str:
            continue
        market_end = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
        market_end_naive = market_end.replace(tzinfo=None)

        if not (START_DATE <= market_end_naive <= END_DATE):
            continue

        count += 1

    return count


if __name__ == "__main__":
    print("Fetching markets via CLOB Client...")
    markets = fetch_all_markets()
    print(f"\nTotal markets retrieved: {len(markets)}")

    count = count_single_condition_markets(markets)
    print(f"\nSingle-condition markets (Apr 2024 - Apr 2025): {count}")
    print(f"Paper reference: 8659")