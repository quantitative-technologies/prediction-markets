"""
Analyze arbitrage trade execution (Section 7 of arXiv:2508.03474).

This script identifies actual arbitrage trades by finding users who
acquired both YES and NO tokens for the same condition within a time window.

Methodology:
1. Group trades by user address within 950-block window (~1 hour)
2. For each group, check if user acquired both YES and NO tokens
3. If combined cost != $1, classify as arbitrage trade
4. Calculate profit = min(YES_tokens, NO_tokens) - cost
"""

import json
import os
import sys
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional
from collections import defaultdict

import polars as pl

# Cache directories
CACHE_DIR = "cache"
EVENTS_CACHE_DIR = os.path.join(CACHE_DIR, "events")
CTF_CACHE_DIR = os.path.join(EVENTS_CACHE_DIR, "ctf")
NEGRISK_CACHE_DIR = os.path.join(EVENTS_CACHE_DIR, "negrisk")
RESULTS_DIR = os.path.join(CACHE_DIR, "arbitrage_results")

# CLOB markets cache
CLOB_MARKETS_PATH = os.path.join(CACHE_DIR, "clob_markets.json")

# Parameters from Section 7
TRADE_WINDOW_BLOCKS = 950  # ~1 hour, groups related trades
MIN_TRADE_VALUE = 2.0  # Filter trades below $2


@dataclass
class UserTrade:
    """A single trade by a user."""
    block_number: int
    tx_hash: str
    user_address: str  # taker address
    token_id: str
    side: str  # 'buy' or 'sell'
    token_amount: float
    usdc_amount: float
    price: float


@dataclass
class ArbitrageTrade:
    """An identified arbitrage trade execution."""
    user_address: str
    condition_id: str
    question: str
    start_block: int
    end_block: int
    yes_tokens: float
    no_tokens: float
    yes_cost: float
    no_cost: float
    total_cost: float
    profit: float
    trade_count: int
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
    """Decode OrderFilled event data including user addresses."""
    try:
        topics = log.get("topics", [])
        data = log.get("data", "0x")

        if len(topics) < 4:
            return None

        # Extract addresses from topics
        # topic[2] = maker address (padded to 32 bytes)
        # topic[3] = taker address (padded to 32 bytes)
        maker_address = "0x" + topics[2][-40:]
        taker_address = "0x" + topics[3][-40:]

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
            "maker_address": maker_address,
            "taker_address": taker_address,
            "maker_asset_id": str(maker_asset_id),
            "taker_asset_id": str(taker_asset_id),
            "maker_amount": maker_amount,
            "taker_amount": taker_amount,
        }
    except Exception:
        return None


# Exchange contract addresses to filter out
CTF_EXCHANGE = "0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e"
NEG_RISK_CTF_EXCHANGE = "0xc5d563a36ae78145c45a50134d48a1215220f80a"
EXCHANGE_CONTRACTS = {CTF_EXCHANGE.lower(), NEG_RISK_CTF_EXCHANGE.lower()}


def build_user_trades_index(events: list[dict]) -> dict[str, list[UserTrade]]:
    """Build user_address -> trades index.

    We track the MAKER as the "user" since the taker is often the exchange
    contract matching orders. Exchange contracts are filtered out.
    """
    trades_by_user: dict[str, list[UserTrade]] = defaultdict(list)

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
        maker_address = decoded["maker_address"]

        # Skip if maker is an exchange contract
        if maker_address.lower() in EXCHANGE_CONTRACTS:
            continue

        # BUY: maker provides tokens, receives USDC
        # (maker_asset is the token, taker_asset is USDC)
        if maker_asset != "0" and taker_asset == "0":
            token_id = maker_asset
            token_amount = maker_amount / 1e6
            usdc_amount = taker_amount / 1e6
            if token_amount > 0 and usdc_amount >= MIN_TRADE_VALUE:
                price = usdc_amount / token_amount
                # From maker's perspective: they SOLD tokens
                trades_by_user[maker_address].append(UserTrade(
                    block_number=block_number,
                    tx_hash=tx_hash,
                    user_address=maker_address,
                    token_id=token_id,
                    side="sell",
                    token_amount=token_amount,
                    usdc_amount=usdc_amount,
                    price=price,
                ))

        # SELL: maker provides USDC, receives tokens
        # (maker_asset is USDC, taker_asset is the token)
        if maker_asset == "0" and taker_asset != "0":
            token_id = taker_asset
            token_amount = taker_amount / 1e6
            usdc_amount = maker_amount / 1e6
            if token_amount > 0 and usdc_amount >= MIN_TRADE_VALUE:
                price = usdc_amount / token_amount
                # From maker's perspective: they BOUGHT tokens
                trades_by_user[maker_address].append(UserTrade(
                    block_number=block_number,
                    tx_hash=tx_hash,
                    user_address=maker_address,
                    token_id=token_id,
                    side="buy",
                    token_amount=token_amount,
                    usdc_amount=usdc_amount,
                    price=price,
                ))

    return trades_by_user


# Token ID -> market lookup
_token_to_market: dict[str, dict] = {}


def load_clob_markets():
    """Load market metadata from CLOB cache."""
    global _token_to_market

    if not os.path.exists(CLOB_MARKETS_PATH):
        print(f"ERROR: CLOB markets cache not found at {CLOB_MARKETS_PATH}")
        sys.exit(1)

    print(f"Loading market metadata from {CLOB_MARKETS_PATH}...")

    with open(CLOB_MARKETS_PATH) as f:
        markets = json.load(f)

    for m in markets:
        tokens = m.get("tokens", [])
        if len(tokens) != 2:
            continue

        market = {
            "condition_id": m["condition_id"],
            "question": m["question"],
            "yes_token_id": tokens[0]["token_id"],
            "no_token_id": tokens[1]["token_id"],
            "neg_risk": m["neg_risk"],
        }

        _token_to_market[tokens[0]["token_id"]] = market
        _token_to_market[tokens[1]["token_id"]] = market

    print(f"Loaded {len(markets)} markets, indexed {len(_token_to_market)} token IDs")


def get_market_by_token_id(token_id: str) -> Optional[dict]:
    """Look up market info by token ID."""
    return _token_to_market.get(token_id)


def group_trades_by_window(trades: list[UserTrade], window_blocks: int) -> list[list[UserTrade]]:
    """Group trades into time windows."""
    if not trades:
        return []

    sorted_trades = sorted(trades, key=lambda t: t.block_number)
    groups = []
    current_group = [sorted_trades[0]]

    for trade in sorted_trades[1:]:
        if trade.block_number - current_group[0].block_number <= window_blocks:
            current_group.append(trade)
        else:
            groups.append(current_group)
            current_group = [trade]

    groups.append(current_group)
    return groups


def find_arbitrage_in_trade_group(
    trades: list[UserTrade],
    user_address: str,
    date_str: str,
) -> list[ArbitrageTrade]:
    """Find arbitrage trades within a group of trades by the same user.

    Arbitrage occurs when:
    - User buys both YES and NO tokens for the same condition
    - Combined cost < $1 (long arbitrage) or > $1 (short arbitrage)
    """
    # Group trades by condition
    trades_by_condition: dict[str, dict] = defaultdict(lambda: {
        "yes_buys": [],
        "no_buys": [],
        "yes_sells": [],
        "no_sells": [],
    })

    for trade in trades:
        market = get_market_by_token_id(trade.token_id)
        if not market:
            continue

        cid = market["condition_id"]
        is_yes = trade.token_id == market["yes_token_id"]

        if trade.side == "buy":
            if is_yes:
                trades_by_condition[cid]["yes_buys"].append(trade)
            else:
                trades_by_condition[cid]["no_buys"].append(trade)
        else:  # sell
            if is_yes:
                trades_by_condition[cid]["yes_sells"].append(trade)
            else:
                trades_by_condition[cid]["no_sells"].append(trade)

    arbitrage_trades = []

    for cid, condition_trades in trades_by_condition.items():
        yes_buys = condition_trades["yes_buys"]
        no_buys = condition_trades["no_buys"]

        # Long arbitrage: buy both YES and NO
        if yes_buys and no_buys:
            yes_tokens = sum(t.token_amount for t in yes_buys)
            no_tokens = sum(t.token_amount for t in no_buys)
            yes_cost = sum(t.usdc_amount for t in yes_buys)
            no_cost = sum(t.usdc_amount for t in no_buys)
            total_cost = yes_cost + no_cost

            # Profit = min tokens held (guaranteed payout) - cost
            min_tokens = min(yes_tokens, no_tokens)
            profit = min_tokens - total_cost

            # Only count as arbitrage if profitable (cost < tokens)
            if profit > 0:
                market = get_market_by_token_id(yes_buys[0].token_id)
                all_trades = yes_buys + no_buys

                arbitrage_trades.append(ArbitrageTrade(
                    user_address=user_address,
                    condition_id=cid,
                    question=market["question"] if market else "",
                    start_block=min(t.block_number for t in all_trades),
                    end_block=max(t.block_number for t in all_trades),
                    yes_tokens=yes_tokens,
                    no_tokens=no_tokens,
                    yes_cost=yes_cost,
                    no_cost=no_cost,
                    total_cost=total_cost,
                    profit=profit,
                    trade_count=len(all_trades),
                    date=date_str,
                ))

    return arbitrage_trades


def analyze_date_for_arbitrage_trades(date: datetime, market_type: str = "single") -> list[ArbitrageTrade]:
    """Analyze a single date for arbitrage trade executions."""
    date_str = date.strftime("%Y-%m-%d")

    # Load events
    if market_type == "single":
        events = load_events_for_date(date, "ctf")
        if events is None:
            return []
    elif market_type == "negrisk":
        events = load_events_for_date(date, "negrisk")
        if events is None:
            return []
    else:
        ctf_events = load_events_for_date(date, "ctf") or []
        negrisk_events = load_events_for_date(date, "negrisk") or []
        events = ctf_events + negrisk_events

    if not events:
        return []

    # Build user trades index
    trades_by_user = build_user_trades_index(events)

    all_arbitrage_trades = []

    for user_address, user_trades in trades_by_user.items():
        # Filter by market type
        filtered_trades = []
        for trade in user_trades:
            market = get_market_by_token_id(trade.token_id)
            if not market:
                continue
            if market_type == "single" and market.get("neg_risk"):
                continue
            if market_type == "negrisk" and not market.get("neg_risk"):
                continue
            filtered_trades.append(trade)

        if not filtered_trades:
            continue

        # Group trades by time window
        trade_groups = group_trades_by_window(filtered_trades, TRADE_WINDOW_BLOCKS)

        # Find arbitrage in each group
        for group in trade_groups:
            arb_trades = find_arbitrage_in_trade_group(group, user_address, date_str)
            all_arbitrage_trades.extend(arb_trades)

    return all_arbitrage_trades


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


def run_analysis(market_type: str = "single") -> dict:
    """Run Section 7 arbitrage trade analysis."""
    import time

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
    print("ARBITRAGE TRADE ANALYSIS (Section 7 Methodology)")
    print(f"{'='*60}")
    print(f"Market type: {market_type}")
    print(f"Trade window: {TRADE_WINDOW_BLOCKS} blocks (~1 hour)")
    print(f"Min trade value: ${MIN_TRADE_VALUE}")
    print(f"Available dates: {len(available_dates)}")
    print()

    if not available_dates:
        print("No cached data available.")
        return {}

    all_trades = []
    start_time = time.time()

    for i, date in enumerate(available_dates):
        date_str = date.strftime("%Y-%m-%d")

        trades = analyze_date_for_arbitrage_trades(date, market_type)
        all_trades.extend(trades)

        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        remaining = len(available_dates) - i - 1
        eta = remaining / rate if rate > 0 else 0

        if trades or (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(available_dates)}] {date_str}: {len(trades)} arb trades | Total: {len(all_trades)} | ETA: {eta/60:.1f} min", flush=True)

    # Aggregate results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Total arbitrage trades: {len(all_trades)}")

    if all_trades:
        total_profit = sum(t.profit for t in all_trades)
        print(f"Total profit extracted: ${total_profit:,.2f}")

        # By user
        profit_by_user = defaultdict(float)
        trades_by_user = defaultdict(int)
        for t in all_trades:
            profit_by_user[t.user_address] += t.profit
            trades_by_user[t.user_address] += 1

        print(f"\nUnique arbitrageurs: {len(profit_by_user)}")

        # Top arbitrageurs
        top_users = sorted(profit_by_user.items(), key=lambda x: -x[1])[:10]
        print(f"\nTop 10 arbitrageurs:")
        for addr, profit in top_users:
            print(f"  {addr[:10]}...{addr[-6:]}: ${profit:,.2f} ({trades_by_user[addr]} trades)")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save as parquet
    if all_trades:
        trades_df = pl.DataFrame([asdict(t) for t in all_trades])
        parquet_path = os.path.join(RESULTS_DIR, f"arbitrage_trades_{market_type}.parquet")
        trades_df.write_parquet(parquet_path)
        print(f"\nSaved to {parquet_path}")

    return {
        "total_trades": len(all_trades),
        "total_profit": sum(t.profit for t in all_trades) if all_trades else 0,
        "unique_users": len(set(t.user_address for t in all_trades)) if all_trades else 0,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze arbitrage trade executions (Section 7)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--market-type",
        choices=["single", "negrisk", "both"],
        default="single",
        help="Market type to analyze"
    )

    args = parser.parse_args()
    run_analysis(market_type=args.market_type)
