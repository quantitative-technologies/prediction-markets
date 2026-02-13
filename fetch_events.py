"""
Fetch and cache blockchain events from Polygon.

This script downloads raw blockchain events once and caches them locally,
enabling fast repeated analysis without re-fetching from RPC providers.

Supported event types:
  - OrderFilled from CTF Exchange and NegRisk CTF Exchange
  - PositionSplit and PositionsMerge from the Conditional Token contract

Data structure:
    cache/events/{source}/{YYYY-MM-DD}.json

Each file contains:
    {
        "date": "2024-04-01",
        "exchange": "ctf" | "negrisk" | "ct",
        "from_block": 55304250,
        "to_block": 55340808,
        "events": [...raw event logs...]
    }
"""

import json
import os
import sys
import time
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

# Ensure unbuffered output for background execution
sys.stdout.reconfigure(line_buffering=True)

# Polymarket contracts on Polygon
CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
NEG_RISK_CTF_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
CONDITIONAL_TOKEN = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

# Event signatures
ORDER_FILLED_TOPIC = "0xd0a08e8c493f9c94f29311604c9de1b4e8c8d4c06bd0c789af57f2d65bfec0f6"
POSITION_SPLIT_TOPIC = "0x2e6bb91f8cbcda0c93623c54d0403a43514fabc40084ec96b6d5379a74786298"
POSITIONS_MERGE_TOPIC = "0x6f13ca62553fcc2bcd2372180a43949c1e4cebba603901ede2f4e14f36b282ca"

# Study period from the paper
START_DATE = datetime(2024, 4, 1)
END_DATE = datetime(2025, 4, 1)

# Cache directories
CACHE_DIR = "cache"
EVENTS_CACHE_DIR = os.path.join(CACHE_DIR, "events")
CTF_CACHE_DIR = os.path.join(EVENTS_CACHE_DIR, "ctf")
NEGRISK_CACHE_DIR = os.path.join(EVENTS_CACHE_DIR, "negrisk")
CT_CACHE_DIR = os.path.join(EVENTS_CACHE_DIR, "ct")

# Source configs: (contract_address, [event_topics])
SOURCE_CONFIGS = {
    "ctf": (CTF_EXCHANGE, [ORDER_FILLED_TOPIC]),
    "negrisk": (NEG_RISK_CTF_EXCHANGE, [ORDER_FILLED_TOPIC]),
    "ct": (CONDITIONAL_TOKEN, [POSITION_SPLIT_TOPIC, POSITIONS_MERGE_TOPIC]),
}

CACHE_DIRS = {
    "ctf": CTF_CACHE_DIR,
    "negrisk": NEGRISK_CACHE_DIR,
    "ct": CT_CACHE_DIR,
}

# RPC configuration
RPC_PRESETS = {
    "drpc": ("https://polygon.drpc.org", 3500),
    "alchemy": (None, 2000),  # Needs ALCHEMY_API_KEY
}
RPC_TIMEOUT = 120  # seconds per eth_getLogs request


def get_rpc_url(preset: str = "drpc") -> tuple[str, int]:
    """Get RPC URL and block limit for a preset."""
    if preset == "alchemy":
        api_key = os.environ.get("ALCHEMY_API_KEY")
        if not api_key:
            raise ValueError("ALCHEMY_API_KEY not set")
        return f"https://polygon-mainnet.g.alchemy.com/v2/{api_key}", 2000

    url, limit = RPC_PRESETS.get(preset, RPC_PRESETS["drpc"])
    return url, limit


def get_block_for_timestamp(rpc_url: str, target_ts: int) -> int:
    """Get block number for a Unix timestamp using binary search."""

    def get_block_timestamp(block_num: int) -> int | None:
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_getBlockByNumber",
            "params": [hex(block_num), False],
            "id": 1
        }
        try:
            resp = requests.post(rpc_url, json=payload, timeout=10)
            data = resp.json()
            if "error" in data:
                return None
            result = data.get("result")
            if result:
                return int(result["timestamp"], 16)
        except Exception:
            pass
        return None

    # Get current block as upper bound
    payload = {"jsonrpc": "2.0", "method": "eth_blockNumber", "params": [], "id": 1}
    resp = requests.post(rpc_url, json=payload, timeout=10)
    data = resp.json()
    if "error" in data:
        raise ValueError(f"Failed to get current block: {data['error']}")
    current_block = int(data["result"], 16)
    current_ts = get_block_timestamp(current_block)

    if current_ts is None:
        raise ValueError("Failed to get current block timestamp")

    # Quick estimate for starting point
    seconds_diff = current_ts - target_ts
    blocks_diff = seconds_diff // 2  # Polygon ~2 sec blocks
    estimated = max(1, current_block - blocks_diff)

    # Binary search to refine
    low = max(1, estimated - 2000000)
    high = min(current_block, estimated + 2000000)

    while low < high:
        mid = (low + high) // 2
        mid_ts = get_block_timestamp(mid)
        if mid_ts is None:
            low = mid + 1
            continue
        if mid_ts < target_ts:
            low = mid + 1
        else:
            high = mid
        time.sleep(0.02)

    return low


def fetch_events_for_range(
    rpc_url: str,
    from_block: int,
    to_block: int,
    contract_address: str,
    event_topic: str,
    block_chunk: int = 3500,
) -> list[dict] | None:
    """Fetch events matching a single topic for a block range.

    Returns None on failure (partial data is discarded).
    Retries individual chunks with smaller sizes and backoff on errors.
    """
    all_logs = []
    total_blocks = to_block - from_block
    start_time = time.time()

    current_from = from_block
    while current_from <= to_block:
        chunk = block_chunk
        current_to = min(current_from + chunk - 1, to_block)
        max_retries = 5
        success = False

        for attempt in range(max_retries):
            current_to = min(current_from + chunk - 1, to_block)
            payload = {
                "jsonrpc": "2.0",
                "method": "eth_getLogs",
                "params": [{
                    "address": contract_address,
                    "topics": [event_topic],
                    "fromBlock": hex(current_from),
                    "toBlock": hex(current_to),
                }],
                "id": 1
            }

            try:
                resp = requests.post(rpc_url, json=payload, timeout=RPC_TIMEOUT)
                result = resp.json()
            except Exception as e:
                print(f"    FETCH ERROR (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"    Waiting {wait}s before retry...")
                    time.sleep(wait)
                    continue
                return None

            if "error" in result:
                error = result["error"]
                error_msg = str(error.get("message", ""))

                retriable = any(x in error_msg.lower() for x in [
                    "block range", "too large", "exceed", "limit",
                    "timed out", "timeout", "internal error",
                    "server error", "rate limit", "too many",
                    "temporary", "please retry",
                ])
                if retriable and attempt < max_retries - 1:
                    new_chunk = chunk // 2
                    if new_chunk >= 10:
                        chunk = new_chunk
                    wait = 2 ** attempt
                    print(f"    {error_msg} (attempt {attempt+1}), chunk={chunk}, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                else:
                    print(f"    ERROR: {error}")
                    return None

            # Success
            logs = result.get("result", [])
            all_logs.extend(logs)
            success = True
            break

        if not success:
            return None

        # Progress indicator
        progress = (current_from - from_block) / total_blocks * 100
        elapsed = time.time() - start_time
        blocks_done = current_from - from_block
        rate = blocks_done / elapsed if elapsed > 0 else 0

        if int(progress) % 20 == 0 and progress > 0:
            print(f"    {progress:.0f}% | {len(all_logs)} events | {rate:.0f} blocks/s")

        current_from = current_to + 1
        time.sleep(0.1)  # Rate limiting

    return all_logs


def get_cache_path(date: datetime, exchange: str) -> str:
    """Get cache file path for a date and source."""
    cache_dir = CACHE_DIRS[exchange]
    return os.path.join(cache_dir, f"{date.strftime('%Y-%m-%d')}.json")


def is_cached(date: datetime, exchange: str) -> bool:
    """Check if events for a date/exchange are cached."""
    return os.path.exists(get_cache_path(date, exchange))


def load_cached_events(date: datetime, exchange: str) -> dict | None:
    """Load cached events for a date/exchange."""
    path = get_cache_path(date, exchange)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_events(date: datetime, exchange: str, from_block: int, to_block: int, events: list) -> None:
    """Save events to cache."""
    cache_dir = CACHE_DIRS[exchange]
    os.makedirs(cache_dir, exist_ok=True)

    data = {
        "date": date.strftime("%Y-%m-%d"),
        "exchange": exchange,
        "from_block": from_block,
        "to_block": to_block,
        "event_count": len(events),
        "events": events,
    }

    path = get_cache_path(date, exchange)
    with open(path, "w") as f:
        json.dump(data, f)


def fetch_day(
    date: datetime,
    rpc_url: str,
    block_limit: int,
    sources: list[str] = ["ctf", "negrisk"],
    use_cache: bool = True,
) -> dict:
    """Fetch events for a single day."""
    results = {}

    # Check if all sources are cached — skip block range lookup entirely
    if use_cache and all(is_cached(date, s) for s in sources):
        for source in sources:
            cached = load_cached_events(date, source)
            results[source] = {
                "from_block": cached["from_block"],
                "to_block": cached["to_block"],
                "event_count": cached["event_count"],
                "cached": True,
            }
            print(f"  {source.upper()}: {cached['event_count']} events [cached]")
        return results

    # Get block range for this day (only needed for uncached sources)
    day_start = datetime(date.year, date.month, date.day, tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)

    from_block = get_block_for_timestamp(rpc_url, int(day_start.timestamp()))
    to_block = get_block_for_timestamp(rpc_url, int(day_end.timestamp()))

    print(f"  Block range: {from_block} to {to_block} ({to_block - from_block} blocks)")

    for source in sources:
        # Check cache first
        if use_cache and is_cached(date, source):
            cached = load_cached_events(date, source)
            results[source] = {
                "from_block": cached["from_block"],
                "to_block": cached["to_block"],
                "event_count": cached["event_count"],
                "cached": True,
            }
            print(f"  {source.upper()}: {cached['event_count']} events [cached]")
            continue

        # Fetch from blockchain
        contract, event_topics = SOURCE_CONFIGS[source]
        print(f"  {source.upper()}: Fetching...")

        # Fetch each event topic and combine
        all_events = []
        fetch_failed = False
        for topic in event_topics:
            events = fetch_events_for_range(
                rpc_url, from_block, to_block, contract, topic, block_limit,
            )
            if events is None:
                fetch_failed = True
                break
            all_events.extend(events)

        if fetch_failed:
            print(f"  {source.upper()}: SKIPPED (fetch failed, not caching partial data)")
            continue

        # Sort combined events by block number
        all_events.sort(key=lambda e: int(e.get("blockNumber", "0x0"), 16))

        # Save to cache
        save_events(date, source, from_block, to_block, all_events)

        results[source] = {
            "from_block": from_block,
            "to_block": to_block,
            "event_count": len(all_events),
            "cached": False,
        }
        print(f"  {source.upper()}: {len(all_events)} events [fetched]")

    return results


def get_cache_status() -> dict:
    """Get status of cached data."""
    all_sources = list(SOURCE_CONFIGS.keys())
    status = {s: [] for s in all_sources}

    for source in all_sources:
        current = START_DATE
        while current < END_DATE:
            if is_cached(current, source):
                status[source].append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)

    total_days = (END_DATE - START_DATE).days
    result = {"total_days": total_days}
    for source in all_sources:
        result[f"{source}_cached"] = len(status[source])
        result[f"{source}_dates"] = status[source]
    return result


def run_fetch(
    rpc_preset: str = "drpc",
    sources: list[str] = ["ctf", "negrisk"],
    use_cache: bool = True,
    start_from: datetime | None = None,
    block_chunk_override: int | None = None,
) -> None:
    """Fetch events for the full study period."""
    rpc_url, block_limit = get_rpc_url(rpc_preset)
    if block_chunk_override is not None:
        block_limit = block_chunk_override

    total_days = (END_DATE - START_DATE).days
    status = get_cache_status()

    print(f"\n{'='*60}")
    print("BLOCKCHAIN EVENT FETCHER")
    print(f"{'='*60}")
    print(f"Study period: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
    print(f"Total days: {total_days}")
    for source in sources:
        cached = status.get(f"{source}_cached", 0)
        print(f"{source.upper()} cached: {cached} days")
    print(f"RPC: {rpc_url}")
    print(f"Block limit: {block_limit}")
    print()

    current = start_from or START_DATE
    day_num = (current - START_DATE).days
    start_time = time.time()
    total_events = {s: 0 for s in sources}

    while current < END_DATE:
        day_num += 1
        date_str = current.strftime("%Y-%m-%d")

        print(f"\n[{day_num}/{total_days}] {date_str}")

        try:
            results = fetch_day(current, rpc_url, block_limit, sources, use_cache)

            for source, data in results.items():
                total_events[source] += data["event_count"]

            # Progress
            elapsed = time.time() - start_time
            rate = day_num / elapsed if elapsed > 0 else 0
            remaining = total_days - day_num
            eta = remaining / rate if rate > 0 else 0

            print(f"  Progress: {day_num/total_days*100:.1f}% | ETA: {eta/60:.1f} min")

        except Exception as e:
            print(f"  ERROR: {e}")

        current += timedelta(days=1)

    print(f"\n{'='*60}")
    print("FETCH COMPLETE")
    print(f"{'='*60}")
    for source in sources:
        print(f"Total {source.upper()} events: {total_events[source]:,}")
    print(f"Total events: {sum(total_events.values()):,}")

    # Calculate storage size
    total_size = 0
    for source in sources:
        cache_dir = CACHE_DIRS[source]
        if os.path.exists(cache_dir):
            size = sum(
                os.path.getsize(os.path.join(cache_dir, f))
                for f in os.listdir(cache_dir)
                if f.endswith(".json")
            )
            total_size += size
    print(f"Cache size: {total_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    import argparse

    all_sources = list(SOURCE_CONFIGS.keys())

    parser = argparse.ArgumentParser(
        description="Fetch and cache blockchain events from Polygon",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rpc-preset",
        type=str,
        choices=list(RPC_PRESETS.keys()),
        default="drpc",
        help="RPC provider preset"
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=all_sources + ["orders", "all"],
        default="orders",
        help="Event source to fetch: ctf, negrisk, ct (split/merge), "
             "orders (ctf+negrisk), or all"
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
        help="Start from a specific date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--block-chunk",
        type=int,
        default=None,
        help="Override block chunk size (default: from RPC preset)"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show cache status and exit"
    )

    args = parser.parse_args()

    if args.status:
        status = get_cache_status()
        print(f"Cache status:")
        print(f"  Total days in study period: {status['total_days']}")
        for source in all_sources:
            cached = status.get(f"{source}_cached", 0)
            label = {"ctf": "CTF Exchange", "negrisk": "NegRisk Exchange",
                     "ct": "Conditional Token (split/merge)"}[source]
            print(f"  {label}: {cached}/{status['total_days']} days cached")

        for source in all_sources:
            cache_dir = CACHE_DIRS[source]
            if os.path.exists(cache_dir):
                size = sum(
                    os.path.getsize(os.path.join(cache_dir, f))
                    for f in os.listdir(cache_dir)
                    if f.endswith(".json")
                )
                print(f"  {source.upper()} cache size: {size / 1024 / 1024:.1f} MB")

        sys.exit(0)

    source_map = {
        "ctf": ["ctf"],
        "negrisk": ["negrisk"],
        "ct": ["ct"],
        "orders": ["ctf", "negrisk"],
        "all": all_sources,
    }
    sources = source_map[args.source]

    start_from = None
    if args.start_from:
        start_from = datetime.strptime(args.start_from, "%Y-%m-%d")

    run_fetch(
        rpc_preset=args.rpc_preset,
        sources=sources,
        use_cache=not args.no_cache,
        start_from=start_from,
        block_chunk_override=args.block_chunk,
    )
