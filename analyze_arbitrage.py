"""
Analyze arbitrage opportunities using cached blockchain events.

This script reads locally cached OrderFilled events and detects arbitrage
without needing to re-fetch from RPC providers.
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional
from collections import defaultdict

# Cache directories
CACHE_DIR = "cache"
EVENTS_CACHE_DIR = os.path.join(CACHE_DIR, "events")
CTF_CACHE_DIR = os.path.join(EVENTS_CACHE_DIR, "ctf")
NEGRISK_CACHE_DIR = os.path.join(EVENTS_CACHE_DIR, "negrisk")
RESULTS_DIR = os.path.join(CACHE_DIR, "arbitrage_results")

# CLOB markets cache (from py-clob-client via market_counter.py)
CLOB_MARKETS_PATH = os.path.join(CACHE_DIR, "clob_markets.json")

# Paper's threshold: $0.05 minimum profit per $1 invested
PROFIT_THRESHOLD = 0.05

# Forward-carry window: 5000 blocks (~2.5 hours at 2s/block)
FORWARD_CARRY_BLOCKS = 5000

# Study period
START_DATE = datetime(2024, 4, 1)
END_DATE = datetime(2025, 4, 1)


@dataclass
class Trade:
    """A single OrderFilled trade."""
    block_number: int
    tx_hash: str
    token_id: str
    side: str  # 'buy' or 'sell'
    token_amount: float
    usdc_amount: float
    price: float


@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity."""
    condition_id: str
    question: str
    yes_price: float
    no_price: float
    combined_price: float
    profit_per_dollar: float
    yes_trade_count: int
    no_trade_count: int
    block_number: int
    date: str


def load_events_for_date(date: datetime, exchange: str) -> list[dict] | None:
    """Load cached events for a date/exchange."""
    cache_dir = CTF_CACHE_DIR if exchange == "ctf" else NEGRISK_CACHE_DIR
    path = os.path.join(cache_dir, f"{date.strftime('%Y-%m-%d')}.json")

    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
            return data.get("events", [])
    return None


def decode_order_filled_event(log: dict) -> Optional[dict]:
    """Decode OrderFilled event data."""
    try:
        topics = log.get("topics", [])
        data = log.get("data", "0x")

        if len(topics) < 4:
            return None

        data_hex = data[2:] if data.startswith("0x") else data
        if len(data_hex) < 320:
            return None

        maker_asset_id = int(data_hex[0:64], 16)
        taker_asset_id = int(data_hex[64:128], 16)
        maker_amount = int(data_hex[128:192], 16)
        taker_amount = int(data_hex[192:256], 16)

        return {
            "block_number": int(log["blockNumber"], 16),
            "tx_hash": log["transactionHash"],
            "maker_asset_id": str(maker_asset_id),
            "taker_asset_id": str(taker_asset_id),
            "maker_amount": maker_amount,
            "taker_amount": taker_amount,
        }
    except Exception:
        return None


def build_trades_index(events: list[dict]) -> dict[str, list[Trade]]:
    """Build token_id -> trades index in a single pass through events.

    This is O(events) instead of O(tokens × events) for the old approach.
    """
    trades_by_token: dict[str, list[Trade]] = {}

    for log in events:
        decoded = decode_order_filled_event(log)
        if not decoded:
            continue

        maker_asset = decoded["maker_asset_id"]
        taker_asset = decoded["taker_asset_id"]
        maker_amount = decoded["maker_amount"]
        taker_amount = decoded["taker_amount"]
        block_number = decoded["block_number"]
        tx_hash = decoded["tx_hash"]

        # BUY: maker_asset is the token, taker_asset is USDC (0)
        if maker_asset != "0" and taker_asset == "0":
            token_id = maker_asset
            token_amount = maker_amount / 1e6
            usdc_amount = taker_amount / 1e6
            if token_amount > 0:
                price = usdc_amount / token_amount
                if token_id not in trades_by_token:
                    trades_by_token[token_id] = []
                trades_by_token[token_id].append(Trade(
                    block_number=block_number,
                    tx_hash=tx_hash,
                    token_id=token_id,
                    side="buy",
                    token_amount=token_amount,
                    usdc_amount=usdc_amount,
                    price=price,
                ))

        # SELL: taker_asset is the token, maker_asset is USDC (0)
        if taker_asset != "0" and maker_asset == "0":
            token_id = taker_asset
            token_amount = taker_amount / 1e6
            usdc_amount = maker_amount / 1e6
            if token_amount > 0:
                price = usdc_amount / token_amount
                if token_id not in trades_by_token:
                    trades_by_token[token_id] = []
                trades_by_token[token_id].append(Trade(
                    block_number=block_number,
                    tx_hash=tx_hash,
                    token_id=token_id,
                    side="sell",
                    token_amount=token_amount,
                    usdc_amount=usdc_amount,
                    price=price,
                ))

    return trades_by_token


def get_price_at_block_with_forward_carry(trades: list[Trade], block: int) -> Optional[float]:
    """Get price at block using forward-carry."""
    min_block = block - FORWARD_CARRY_BLOCKS
    valid_trades = [t for t in trades if min_block <= t.block_number <= block]

    if not valid_trades:
        return None

    most_recent = max(valid_trades, key=lambda t: t.block_number)
    return most_recent.price


def find_arbitrage_for_market(
    yes_trades: list[Trade],
    no_trades: list[Trade],
    from_block: int,
    to_block: int,
    profit_threshold: float,
) -> Optional[dict]:
    """Find best arbitrage opportunity for a market."""
    all_trade_blocks = sorted(set(
        t.block_number for t in yes_trades + no_trades
        if from_block <= t.block_number <= to_block
    ))

    if not all_trade_blocks:
        return None

    best_result = None
    best_profit = profit_threshold

    for block in all_trade_blocks:
        yes_price = get_price_at_block_with_forward_carry(yes_trades, block)
        no_price = get_price_at_block_with_forward_carry(no_trades, block)

        if yes_price is None or no_price is None:
            continue

        if yes_price > 0.95 or no_price > 0.95:
            continue

        combined = yes_price + no_price
        profit = 1.0 - combined

        if profit > best_profit:
            best_profit = profit
            best_result = {
                "block": block,
                "yes_price": yes_price,
                "no_price": no_price,
                "combined": combined,
                "profit": profit,
            }

    return best_result


# Token ID -> market lookup (built from CLOB cache)
_token_to_market: dict[str, dict] = {}


def load_clob_markets():
    """Load market metadata from CLOB cache (created by market_counter.py using py-clob-client).

    Builds a token_id -> market lookup for fast access.
    """
    global _token_to_market

    if not os.path.exists(CLOB_MARKETS_PATH):
        print(f"ERROR: CLOB markets cache not found at {CLOB_MARKETS_PATH}")
        print("Run market_counter.py first to fetch market metadata from py-clob-client.")
        sys.exit(1)

    print(f"Loading market metadata from {CLOB_MARKETS_PATH}...")

    with open(CLOB_MARKETS_PATH) as f:
        markets = json.load(f)

    # Build token_id -> market lookup
    for m in markets:
        tokens = m.get("tokens", [])
        if len(tokens) != 2:
            continue  # Only binary markets

        # Normalize market structure for our use
        market = {
            "condition_id": m["condition_id"],
            "question": m["question"],
            "yes_token_id": tokens[0]["token_id"],
            "no_token_id": tokens[1]["token_id"],
            "neg_risk": m["neg_risk"],
        }

        # Index by both token IDs
        _token_to_market[tokens[0]["token_id"]] = market
        _token_to_market[tokens[1]["token_id"]] = market

    print(f"Loaded {len(markets)} markets, indexed {len(_token_to_market)} token IDs")


def get_market_by_token_id(token_id: str) -> Optional[dict]:
    """Look up market info by token ID from pre-loaded CLOB cache."""
    return _token_to_market.get(token_id)


def analyze_date(
    date: datetime,
    market_type: str = "single",
    profit_threshold: float = PROFIT_THRESHOLD,
    max_tokens: int | None = None,
) -> list[ArbitrageOpportunity]:
    """Analyze a single date for arbitrage opportunities."""
    date_str = date.strftime("%Y-%m-%d")

    # Load events based on market type
    if market_type == "single":
        events = load_events_for_date(date, "ctf")
        if events is None:
            return []
    elif market_type == "negrisk":
        events = load_events_for_date(date, "negrisk")
        if events is None:
            return []
    else:  # both
        ctf_events = load_events_for_date(date, "ctf") or []
        negrisk_events = load_events_for_date(date, "negrisk") or []
        events = ctf_events + negrisk_events

    if not events:
        return []

    # Build trades index in single pass - O(events) instead of O(tokens × events)
    trades_by_token = build_trades_index(events)

    # Get block range
    blocks = [int(e["blockNumber"], 16) for e in events if "blockNumber" in e]
    if not blocks:
        return []
    from_block = min(blocks)
    to_block = max(blocks)

    # Look up markets from pre-loaded CLOB cache
    markets = {}
    for token_id in trades_by_token.keys():
        market = get_market_by_token_id(token_id)
        if market:
            cid = market["condition_id"]
            if cid not in markets:
                # Filter by market type
                if market_type == "single" and market.get("neg_risk"):
                    continue
                if market_type == "negrisk" and not market.get("neg_risk"):
                    continue
                markets[cid] = market

    # Find arbitrage - O(1) lookup per market now
    opportunities = []

    for condition_id, market in markets.items():
        yes_trades = trades_by_token.get(market["yes_token_id"], [])
        no_trades = trades_by_token.get(market["no_token_id"], [])

        if not yes_trades and not no_trades:
            continue

        result = find_arbitrage_for_market(
            yes_trades, no_trades, from_block, to_block, profit_threshold
        )

        if result:
            opportunities.append(ArbitrageOpportunity(
                condition_id=condition_id,
                question=market["question"],
                yes_price=result["yes_price"],
                no_price=result["no_price"],
                combined_price=result["combined"],
                profit_per_dollar=result["profit"],
                yes_trade_count=len(yes_trades),
                no_trade_count=len(no_trades),
                block_number=result["block"],
                date=date_str,
            ))

    return opportunities


def get_cached_dates(exchange: str) -> list[datetime]:
    """Get list of dates that have cached events."""
    cache_dir = CTF_CACHE_DIR if exchange == "ctf" else NEGRISK_CACHE_DIR
    dates = []

    if os.path.exists(cache_dir):
        for filename in os.listdir(cache_dir):
            if filename.endswith(".json"):
                date_str = filename.replace(".json", "")
                try:
                    dates.append(datetime.strptime(date_str, "%Y-%m-%d"))
                except ValueError:
                    pass

    return sorted(dates)


def run_analysis(
    market_type: str = "single",
    profit_threshold: float = PROFIT_THRESHOLD,
    max_tokens: int | None = None,
) -> dict:
    """Run full analysis on all cached data."""
    load_clob_markets()

    # Get available dates
    if market_type == "single":
        available_dates = get_cached_dates("ctf")
    elif market_type == "negrisk":
        available_dates = get_cached_dates("negrisk")
    else:
        ctf_dates = set(get_cached_dates("ctf"))
        negrisk_dates = set(get_cached_dates("negrisk"))
        available_dates = sorted(ctf_dates | negrisk_dates)

    print(f"\n{'='*60}")
    print("ARBITRAGE ANALYSIS")
    print(f"{'='*60}")
    print(f"Market type: {market_type}")
    print(f"Profit threshold: {profit_threshold:.0%}")
    print(f"Available dates: {len(available_dates)}")
    print()

    if not available_dates:
        print("No cached data available. Run fetch_events.py first.")
        return {}

    # Analyze each date
    all_conditions = {}
    daily_stats = []
    start_time = time.time()

    for i, date in enumerate(available_dates):
        date_str = date.strftime("%Y-%m-%d")
        print(f"[{i+1}/{len(available_dates)}] {date_str}...")

        opportunities = analyze_date(date, market_type, profit_threshold, max_tokens)

        daily_stats.append({
            "date": date_str,
            "opportunities": len(opportunities),
        })

        for opp in opportunities:
            cid = opp.condition_id
            if cid not in all_conditions or opp.profit_per_dollar > all_conditions[cid].profit_per_dollar:
                all_conditions[cid] = opp

        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        remaining = len(available_dates) - i - 1
        eta = remaining / rate if rate > 0 else 0

        print(f"  Found {len(opportunities)} opportunities | Total unique: {len(all_conditions)} | ETA: {eta/60:.1f} min")

    # Final results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Days analyzed: {len(available_dates)}")
    print(f"Unique conditions with arbitrage: {len(all_conditions)}")

    if all_conditions:
        profits = [opp.profit_per_dollar for opp in all_conditions.values()]
        print(f"\nProfit statistics:")
        print(f"  Average: {sum(profits)/len(profits):.2%}")
        print(f"  Max: {max(profits):.2%}")
        print(f"  Min: {min(profits):.2%}")
        print(f"  Median: {sorted(profits)[len(profits)//2]:.2%}")

    # Paper comparison
    print(f"\n{'='*60}")
    print("COMPARISON WITH PAPER (Section 6.1)")
    print(f"{'='*60}")
    if market_type == "single":
        print(f"Paper: 4,423 single-condition markets with arbitrage")
    elif market_type == "negrisk":
        print(f"Paper: 2,628 NegRisk conditions with arbitrage")
    else:
        print(f"Paper: 7,051 total conditions with arbitrage")
    print(f"Found: {len(all_conditions)} conditions with arbitrage")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, f"arbitrage_{market_type}.json")

    results = {
        "market_type": market_type,
        "profit_threshold": profit_threshold,
        "days_analyzed": len(available_dates),
        "unique_conditions": len(all_conditions),
        "daily_stats": daily_stats,
        "opportunities": [asdict(opp) for opp in all_conditions.values()],
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze arbitrage from cached events")
    parser.add_argument(
        "--market-type",
        choices=["single", "negrisk", "both"],
        default="single",
        help="Market type to analyze (default: single)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=PROFIT_THRESHOLD,
        help=f"Minimum profit threshold (default: {PROFIT_THRESHOLD})"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max tokens to look up per day (for testing)"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show cache status and exit"
    )

    args = parser.parse_args()

    if args.status:
        ctf_dates = get_cached_dates("ctf")
        negrisk_dates = get_cached_dates("negrisk")
        print(f"Cached data status:")
        print(f"  CTF Exchange: {len(ctf_dates)} days")
        print(f"  NegRisk Exchange: {len(negrisk_dates)} days")
        if ctf_dates:
            print(f"  CTF range: {ctf_dates[0].strftime('%Y-%m-%d')} to {ctf_dates[-1].strftime('%Y-%m-%d')}")
        if negrisk_dates:
            print(f"  NegRisk range: {negrisk_dates[0].strftime('%Y-%m-%d')} to {negrisk_dates[-1].strftime('%Y-%m-%d')}")

        if os.path.exists(CLOB_MARKETS_PATH):
            load_clob_markets()
            print(f"  CLOB market metadata: {len(_token_to_market)} token IDs indexed")
        else:
            print(f"  CLOB market cache not found (run market_counter.py first)")
        sys.exit(0)

    run_analysis(
        market_type=args.market_type,
        profit_threshold=args.threshold,
        max_tokens=args.max_tokens,
    )
