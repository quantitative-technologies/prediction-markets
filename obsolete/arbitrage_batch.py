"""
Batch arbitrage detection for the full study period.

Processes one day at a time, caches results, and can be resumed if interrupted.
Validates findings from Section 6.1 of arXiv:2508.03474.
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta
from dataclasses import asdict

from arbitrage_detector import (
    find_arbitrage_for_day,
    get_rpc_url,
    RPC_PRESETS,
    PROFIT_THRESHOLD,
)

# Study period from the paper
START_DATE = datetime(2024, 4, 1)
END_DATE = datetime(2025, 4, 1)

# Cache directory for daily results
CACHE_DIR = "cache"
DAILY_CACHE_DIR = os.path.join(CACHE_DIR, "arbitrage_daily")
RESULTS_FILE = "arbitrage_results.json"


def get_daily_cache_path(date: datetime) -> str:
    """Get cache file path for a specific date."""
    return os.path.join(DAILY_CACHE_DIR, f"{date.strftime('%Y-%m-%d')}.json")


def load_daily_results(date: datetime) -> list[dict] | None:
    """Load cached results for a date, or None if not cached."""
    path = get_daily_cache_path(date)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_daily_results(date: datetime, results: list[dict]) -> None:
    """Save results for a date to cache."""
    os.makedirs(DAILY_CACHE_DIR, exist_ok=True)
    path = get_daily_cache_path(date)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def process_date(
    date: datetime,
    rpc_url: str,
    block_limit: int,
    market_type: str = "both",
    use_cache: bool = True,
) -> list[dict]:
    """Process a single date and return arbitrage opportunities."""
    # Check cache first
    if use_cache:
        cached = load_daily_results(date)
        if cached is not None:
            return cached

    # Run arbitrage detection
    opportunities = find_arbitrage_for_day(
        date=date,
        rpc_url=rpc_url,
        profit_threshold=PROFIT_THRESHOLD,
        hours=24,
        max_tokens=None,  # Process all tokens
        market_type=market_type,
        use_forward_carry=True,  # Paper's methodology
        block_chunk=block_limit,
    )

    # Convert to dicts for JSON serialization
    results = [asdict(opp) for opp in opportunities]

    # Cache results
    save_daily_results(date, results)

    return results


def count_days_cached() -> tuple[int, int]:
    """Count how many days are already cached."""
    total_days = (END_DATE - START_DATE).days
    cached_days = 0

    if os.path.exists(DAILY_CACHE_DIR):
        current = START_DATE
        while current < END_DATE:
            if os.path.exists(get_daily_cache_path(current)):
                cached_days += 1
            current += timedelta(days=1)

    return cached_days, total_days


def run_batch(
    rpc_preset: str | None = None,
    rpc_url: str | None = None,
    market_type: str = "both",
    use_cache: bool = True,
    start_from: datetime | None = None,
) -> dict:
    """
    Run batch arbitrage detection for the full study period.

    Returns aggregated statistics.
    """
    # Get RPC configuration
    url, block_limit = get_rpc_url(preset=rpc_preset, custom_url=rpc_url)

    # Check existing progress
    cached_days, total_days = count_days_cached()
    print(f"\nStudy period: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
    print(f"Total days: {total_days}")
    print(f"Already cached: {cached_days} days")
    print(f"RPC: {url[:50]}... (block limit: {block_limit})")
    print(f"Market type: {market_type}")
    print()

    # Aggregate results
    all_conditions = {}  # condition_id -> best opportunity
    daily_stats = []

    current = start_from or START_DATE
    day_num = (current - START_DATE).days
    start_time = time.time()

    while current < END_DATE:
        day_num += 1
        date_str = current.strftime("%Y-%m-%d")

        # Check if cached
        is_cached = os.path.exists(get_daily_cache_path(current))
        cache_status = "[cached]" if is_cached else "[fetching]"

        print(f"\n{'='*60}")
        print(f"Day {day_num}/{total_days}: {date_str} {cache_status}")
        print(f"{'='*60}")

        try:
            results = process_date(
                date=current,
                rpc_url=url,
                block_limit=block_limit,
                market_type=market_type,
                use_cache=use_cache,
            )

            # Track daily stats
            daily_stats.append({
                "date": date_str,
                "opportunities": len(results),
                "cached": is_cached,
            })

            # Aggregate by condition (keep best profit per condition)
            for opp in results:
                cid = opp["condition_id"]
                if cid not in all_conditions or opp["profit_per_dollar"] > all_conditions[cid]["profit_per_dollar"]:
                    all_conditions[cid] = opp

            # Progress update
            elapsed = time.time() - start_time
            days_done = day_num
            rate = days_done / elapsed if elapsed > 0 else 0
            remaining = total_days - days_done
            eta = remaining / rate if rate > 0 else 0

            print(f"\nDay summary: {len(results)} opportunities found")
            print(f"Running total: {len(all_conditions)} unique conditions with arbitrage")
            print(f"Progress: {days_done}/{total_days} days ({days_done/total_days*100:.1f}%)")
            print(f"Rate: {rate*60:.1f} days/min | ETA: {eta/60:.1f} min")

        except Exception as e:
            print(f"ERROR processing {date_str}: {e}")
            daily_stats.append({
                "date": date_str,
                "opportunities": 0,
                "error": str(e),
            })

        current += timedelta(days=1)

    # Final aggregation
    print("\n" + "="*60)
    print("BATCH PROCESSING COMPLETE")
    print("="*60)

    # Calculate statistics
    total_opportunities = sum(d["opportunities"] for d in daily_stats if "error" not in d)
    unique_conditions = len(all_conditions)
    days_with_arbitrage = sum(1 for d in daily_stats if d["opportunities"] > 0)
    errors = sum(1 for d in daily_stats if "error" in d)

    print(f"\nTotal days processed: {len(daily_stats)}")
    print(f"Days with errors: {errors}")
    print(f"Days with arbitrage: {days_with_arbitrage}")
    print(f"Total arbitrage instances: {total_opportunities}")
    print(f"Unique conditions with arbitrage: {unique_conditions}")

    if all_conditions:
        profits = [opp["profit_per_dollar"] for opp in all_conditions.values()]
        print(f"\nProfit statistics (per condition):")
        print(f"  Average: ${sum(profits)/len(profits):.4f}")
        print(f"  Max: ${max(profits):.4f}")
        print(f"  Min: ${min(profits):.4f}")
        print(f"  Median: ${sorted(profits)[len(profits)//2]:.4f}")

    # Save final results
    final_results = {
        "study_period": {
            "start": START_DATE.strftime("%Y-%m-%d"),
            "end": END_DATE.strftime("%Y-%m-%d"),
        },
        "market_type": market_type,
        "summary": {
            "total_days": len(daily_stats),
            "days_with_arbitrage": days_with_arbitrage,
            "total_opportunities": total_opportunities,
            "unique_conditions": unique_conditions,
        },
        "daily_stats": daily_stats,
        "conditions": list(all_conditions.values()),
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    # Paper comparison
    print("\n" + "="*60)
    print("COMPARISON WITH PAPER (Section 6.1)")
    print("="*60)
    print(f"Paper: 7,051 conditions with arbitrage (all markets)")
    print(f"Paper: 4,423 conditions with arbitrage (single-condition only)")
    print(f"Found: {unique_conditions} conditions with arbitrage ({market_type} markets)")

    return final_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch arbitrage detection for full study period"
    )
    parser.add_argument(
        "--rpc-preset",
        type=str,
        choices=list(RPC_PRESETS.keys()),
        default="drpc",
        help="RPC provider preset (default: drpc)"
    )
    parser.add_argument(
        "--rpc-url",
        type=str,
        default=None,
        help="Custom RPC URL (overrides --rpc-preset)"
    )
    parser.add_argument(
        "--market-type",
        type=str,
        choices=["single", "negrisk", "both"],
        default="single",
        help="Market type to analyze (default: single for paper comparison)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't use cached results, re-fetch everything"
    )
    parser.add_argument(
        "--start-from",
        type=str,
        default=None,
        help="Start from a specific date (YYYY-MM-DD), useful for resuming"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show cache status and exit"
    )

    args = parser.parse_args()

    if args.status:
        cached, total = count_days_cached()
        print(f"Cache status: {cached}/{total} days cached ({cached/total*100:.1f}%)")
        if os.path.exists(DAILY_CACHE_DIR):
            files = os.listdir(DAILY_CACHE_DIR)
            if files:
                files.sort()
                print(f"First cached: {files[0].replace('.json', '')}")
                print(f"Last cached: {files[-1].replace('.json', '')}")
        sys.exit(0)

    start_from = None
    if args.start_from:
        start_from = datetime.strptime(args.start_from, "%Y-%m-%d")

    run_batch(
        rpc_preset=args.rpc_preset,
        rpc_url=args.rpc_url,
        market_type=args.market_type,
        use_cache=not args.no_cache,
        start_from=start_from,
    )
