"""
Fetch and cache OrderFilled events from Polygon blockchain.

This script downloads raw blockchain events once and caches them locally,
enabling fast repeated analysis without re-fetching from RPC providers.

Data structure:
    cache/events/{exchange}/{YYYY-MM-DD}.json

Each file contains:
    {
        "date": "2024-04-01",
        "exchange": "ctf" or "negrisk",
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

# Polymarket contracts on Polygon
CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
NEG_RISK_CTF_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a"

# OrderFilled event signature
ORDER_FILLED_TOPIC = "0xd0a08e8c493f9c94f29311604c9de1b4e8c8d4c06bd0c789af57f2d65bfec0f6"

# Study period from the paper
START_DATE = datetime(2024, 4, 1)
END_DATE = datetime(2025, 4, 1)

# Cache directories
CACHE_DIR = "cache"
EVENTS_CACHE_DIR = os.path.join(CACHE_DIR, "events")
CTF_CACHE_DIR = os.path.join(EVENTS_CACHE_DIR, "ctf")
NEGRISK_CACHE_DIR = os.path.join(EVENTS_CACHE_DIR, "negrisk")

# RPC configuration
RPC_PRESETS = {
    "drpc": ("https://polygon.drpc.org", 3500),
    "alchemy": (None, 2000),  # Needs ALCHEMY_API_KEY
}


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
    block_chunk: int = 3500,
) -> list[dict]:
    """Fetch OrderFilled events for a block range."""
    all_logs = []
    total_blocks = to_block - from_block
    start_time = time.time()

    current_from = from_block
    while current_from <= to_block:
        current_to = min(current_from + block_chunk - 1, to_block)

        payload = {
            "jsonrpc": "2.0",
            "method": "eth_getLogs",
            "params": [{
                "address": contract_address,
                "topics": [ORDER_FILLED_TOPIC],
                "fromBlock": hex(current_from),
                "toBlock": hex(current_to),
            }],
            "id": 1
        }

        resp = requests.post(rpc_url, json=payload, timeout=30)
        result = resp.json()

        if "error" in result:
            error = result["error"]
            error_msg = str(error.get("message", ""))

            # Handle block range limits - try smaller chunks
            if any(x in error_msg.lower() for x in ["block range", "too large", "exceed", "limit"]):
                new_chunk = block_chunk // 2
                if new_chunk >= 10:
                    print(f"  Reducing chunk size to {new_chunk}...")
                    return fetch_events_for_range(
                        rpc_url, from_block, to_block, contract_address, new_chunk
                    )
                else:
                    print(f"  ERROR: Block range too restrictive: {error}")
                    break

            print(f"  ERROR: {error}")
            break

        logs = result.get("result", [])
        all_logs.extend(logs)

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
    """Get cache file path for a date and exchange."""
    cache_dir = CTF_CACHE_DIR if exchange == "ctf" else NEGRISK_CACHE_DIR
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
    cache_dir = CTF_CACHE_DIR if exchange == "ctf" else NEGRISK_CACHE_DIR
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
    exchanges: list[str] = ["ctf", "negrisk"],
    use_cache: bool = True,
) -> dict:
    """Fetch events for a single day."""
    results = {}

    # Get block range for this day
    day_start = datetime(date.year, date.month, date.day, tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)

    from_block = get_block_for_timestamp(rpc_url, int(day_start.timestamp()))
    to_block = get_block_for_timestamp(rpc_url, int(day_end.timestamp()))

    print(f"  Block range: {from_block} to {to_block} ({to_block - from_block} blocks)")

    for exchange in exchanges:
        # Check cache first
        if use_cache and is_cached(date, exchange):
            cached = load_cached_events(date, exchange)
            results[exchange] = {
                "from_block": cached["from_block"],
                "to_block": cached["to_block"],
                "event_count": cached["event_count"],
                "cached": True,
            }
            print(f"  {exchange.upper()}: {cached['event_count']} events [cached]")
            continue

        # Fetch from blockchain
        contract = CTF_EXCHANGE if exchange == "ctf" else NEG_RISK_CTF_EXCHANGE
        print(f"  {exchange.upper()}: Fetching...")

        events = fetch_events_for_range(rpc_url, from_block, to_block, contract, block_limit)

        # Save to cache
        save_events(date, exchange, from_block, to_block, events)

        results[exchange] = {
            "from_block": from_block,
            "to_block": to_block,
            "event_count": len(events),
            "cached": False,
        }
        print(f"  {exchange.upper()}: {len(events)} events [fetched]")

    return results


def get_cache_status() -> dict:
    """Get status of cached data."""
    status = {"ctf": [], "negrisk": []}

    for exchange in ["ctf", "negrisk"]:
        current = START_DATE
        while current < END_DATE:
            if is_cached(current, exchange):
                status[exchange].append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)

    total_days = (END_DATE - START_DATE).days
    return {
        "total_days": total_days,
        "ctf_cached": len(status["ctf"]),
        "negrisk_cached": len(status["negrisk"]),
        "ctf_dates": status["ctf"],
        "negrisk_dates": status["negrisk"],
    }


def run_fetch(
    rpc_preset: str = "drpc",
    exchanges: list[str] = ["ctf", "negrisk"],
    use_cache: bool = True,
    start_from: datetime | None = None,
) -> None:
    """Fetch events for the full study period."""
    rpc_url, block_limit = get_rpc_url(rpc_preset)

    total_days = (END_DATE - START_DATE).days
    status = get_cache_status()

    print(f"\n{'='*60}")
    print("BLOCKCHAIN EVENT FETCHER")
    print(f"{'='*60}")
    print(f"Study period: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
    print(f"Total days: {total_days}")
    print(f"CTF cached: {status['ctf_cached']} days")
    print(f"NegRisk cached: {status['negrisk_cached']} days")
    print(f"RPC: {rpc_url}")
    print(f"Block limit: {block_limit}")
    print()

    current = start_from or START_DATE
    day_num = (current - START_DATE).days
    start_time = time.time()
    total_events = {"ctf": 0, "negrisk": 0}

    while current < END_DATE:
        day_num += 1
        date_str = current.strftime("%Y-%m-%d")

        print(f"\n[{day_num}/{total_days}] {date_str}")

        try:
            results = fetch_day(current, rpc_url, block_limit, exchanges, use_cache)

            for exchange, data in results.items():
                total_events[exchange] += data["event_count"]

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
    print(f"Total CTF events: {total_events['ctf']:,}")
    print(f"Total NegRisk events: {total_events['negrisk']:,}")
    print(f"Total events: {sum(total_events.values()):,}")

    # Calculate storage size
    ctf_size = sum(
        os.path.getsize(os.path.join(CTF_CACHE_DIR, f))
        for f in os.listdir(CTF_CACHE_DIR)
        if f.endswith(".json")
    ) if os.path.exists(CTF_CACHE_DIR) else 0

    negrisk_size = sum(
        os.path.getsize(os.path.join(NEGRISK_CACHE_DIR, f))
        for f in os.listdir(NEGRISK_CACHE_DIR)
        if f.endswith(".json")
    ) if os.path.exists(NEGRISK_CACHE_DIR) else 0

    print(f"Cache size: {(ctf_size + negrisk_size) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch and cache OrderFilled events from Polygon"
    )
    parser.add_argument(
        "--rpc-preset",
        type=str,
        choices=list(RPC_PRESETS.keys()),
        default="drpc",
        help="RPC provider preset (default: drpc)"
    )
    parser.add_argument(
        "--exchange",
        type=str,
        choices=["ctf", "negrisk", "both"],
        default="both",
        help="Which exchange to fetch (default: both)"
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
        "--status",
        action="store_true",
        help="Show cache status and exit"
    )

    args = parser.parse_args()

    if args.status:
        status = get_cache_status()
        print(f"Cache status:")
        print(f"  Total days in study period: {status['total_days']}")
        print(f"  CTF Exchange: {status['ctf_cached']}/{status['total_days']} days cached")
        print(f"  NegRisk Exchange: {status['negrisk_cached']}/{status['total_days']} days cached")

        if os.path.exists(CTF_CACHE_DIR):
            ctf_size = sum(os.path.getsize(os.path.join(CTF_CACHE_DIR, f)) for f in os.listdir(CTF_CACHE_DIR))
            print(f"  CTF cache size: {ctf_size / 1024 / 1024:.1f} MB")
        if os.path.exists(NEGRISK_CACHE_DIR):
            negrisk_size = sum(os.path.getsize(os.path.join(NEGRISK_CACHE_DIR, f)) for f in os.listdir(NEGRISK_CACHE_DIR))
            print(f"  NegRisk cache size: {negrisk_size / 1024 / 1024:.1f} MB")

        sys.exit(0)

    exchanges = ["ctf"] if args.exchange == "ctf" else ["negrisk"] if args.exchange == "negrisk" else ["ctf", "negrisk"]

    start_from = None
    if args.start_from:
        start_from = datetime.strptime(args.start_from, "%Y-%m-%d")

    run_fetch(
        rpc_preset=args.rpc_preset,
        exchanges=exchanges,
        use_cache=not args.no_cache,
        start_from=start_from,
    )
