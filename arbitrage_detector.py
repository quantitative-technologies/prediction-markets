"""
Polymarket Single-Market Arbitrage Detector

Validates findings from Section 6.1 of "Unravelling the Probabilistic Forest"
(arXiv:2508.03474) by detecting long arbitrage opportunities where
YES + NO prices sum to less than $1.00.

Uses on-chain OrderFilled events from Polygon via Alchemy API (per Section 4.2).
"""

import os
import json
import requests
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from dotenv import load_dotenv
from typing import Optional
from collections import defaultdict
import time

load_dotenv()
# Polymarket contracts on Polygon
CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
NEG_RISK_CTF_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a"

# OrderFilled event signature
# OrderFilled(bytes32,address,address,uint256,uint256,uint256,uint256,uint256)
ORDER_FILLED_TOPIC = "0xd0a08e8c493f9c94f29311604c9de1b4e8c8d4c06bd0c789af57f2d65bfec0f6"

# USDC on Polygon (6 decimals)
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
USDC_DECIMALS = 6

# Gamma API for market metadata
GAMMA_API = "https://gamma-api.polymarket.com"

# Paper's threshold: $0.05 minimum profit per $1 invested
PROFIT_THRESHOLD = 0.05

# Polygon block time ~2 seconds, so ~43200 blocks per day
BLOCKS_PER_DAY = 43200


@dataclass
class Market:
    """Represents a Polymarket condition with YES/NO tokens."""
    id: str
    question: str
    condition_id: str
    yes_token_id: str
    no_token_id: str
    slug: str
    active: bool
    closed: bool


@dataclass
class Trade:
    """A single OrderFilled trade."""
    block_number: int
    tx_hash: str
    token_id: str
    side: str  # 'buy' or 'sell'
    token_amount: float
    usdc_amount: float
    price: float  # USDC per token


@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity based on VWAP."""
    condition_id: str
    question: str
    yes_vwap: float
    no_vwap: float
    combined_price: float
    profit_per_dollar: float
    yes_trade_count: int
    no_trade_count: int
    yes_volume: float
    no_volume: float
    block_number: Optional[int] = None  # Block where opportunity was detected (None = aggregated)


def get_alchemy_url() -> str:
    """Get Alchemy API URL from environment."""
    api_key = os.environ.get("ALCHEMY_API_KEY")
    if not api_key:
        raise ValueError(
            "ALCHEMY_API_KEY environment variable not set.\n"
            "Get a free key at https://www.alchemy.com/ and run:\n"
            "  export ALCHEMY_API_KEY=your_key_here"
        )
    return f"https://polygon-mainnet.g.alchemy.com/v2/{api_key}"


def fetch_market_by_token_id(
    token_id: str,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> tuple[Optional[Market], Optional[str]]:
    """
    Fetch a market by its token ID using Gamma API.

    This is the key fix: instead of fetching all markets and hoping they match,
    we look up markets by the actual token IDs found in on-chain trades.

    Returns:
        Tuple of (Market or None, error_message or None)
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            resp = requests.get(
                f"{GAMMA_API}/markets",
                params={"clob_token_ids": token_id},
                timeout=10
            )

            # Handle rate limiting
            if resp.status_code == 429:
                last_error = "rate_limited"
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                time.sleep(wait_time)
                continue

            if not resp.ok:
                last_error = f"http_{resp.status_code}"
                return None, last_error

            markets = resp.json()
            if not markets:
                return None, None  # No market found (not an error)

            m = markets[0]
            token_ids_raw = m.get("clobTokenIds", "")
            if not token_ids_raw:
                return None, None

            try:
                token_ids = json.loads(token_ids_raw)
            except (json.JSONDecodeError, TypeError):
                return None, "json_parse_error"

            if len(token_ids) != 2:
                return None, None  # Not a binary market

            return Market(
                id=m.get("id", ""),
                question=m.get("question", ""),
                condition_id=m.get("conditionId", ""),
                yes_token_id=token_ids[0],
                no_token_id=token_ids[1],
                slug=m.get("slug", ""),
                active=m.get("active", False),
                closed=m.get("closed", False),
            ), None

        except requests.exceptions.Timeout:
            last_error = "timeout"
            time.sleep(retry_delay)
        except requests.exceptions.RequestException as e:
            last_error = f"request_error:{type(e).__name__}"
            time.sleep(retry_delay)

    return None, last_error


def fetch_markets_for_tokens(
    token_ids: list[str],
    max_lookups: Optional[int] = None,
) -> tuple[dict[str, Market], dict[str, int]]:
    """
    Fetch markets for a list of traded token IDs.

    Returns:
        Tuple of (token_to_market dict, error_counts dict)
    """
    token_to_market: dict[str, Market] = {}
    error_counts: dict[str, int] = defaultdict(int)
    tokens_to_lookup = token_ids[:max_lookups] if max_lookups else token_ids

    for i, token_id in enumerate(tokens_to_lookup):
        if token_id in token_to_market:
            continue

        market, error = fetch_market_by_token_id(token_id)
        if market:
            # Map both YES and NO tokens to this market
            token_to_market[market.yes_token_id] = market
            token_to_market[market.no_token_id] = market
        elif error:
            error_counts[error] += 1

        # Progress indicator
        if (i + 1) % 20 == 0:
            print(f"  Looked up {i + 1}/{len(tokens_to_lookup)} tokens...")

        time.sleep(0.05)  # Rate limiting

    # Report errors if any
    if error_counts:
        print(f"  Errors during lookup: {dict(error_counts)}")

    return token_to_market, dict(error_counts)


def get_block_for_timestamp(alchemy_url: str, target_ts: int) -> int:
    """Get block number for a Unix timestamp using binary search."""

    def get_block_timestamp(block_num: int) -> Optional[int]:
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_getBlockByNumber",
            "params": [hex(block_num), False],
            "id": 1
        }
        resp = requests.post(alchemy_url, json=payload)
        result = resp.json().get("result")
        if result:
            return int(result["timestamp"], 16)
        return None

    # Get current block as upper bound
    payload = {"jsonrpc": "2.0", "method": "eth_blockNumber", "params": [], "id": 1}
    resp = requests.post(alchemy_url, json=payload)
    current_block = int(resp.json()["result"], 16)
    current_ts = get_block_timestamp(current_block)

    # Quick estimate for starting point
    seconds_diff = current_ts - target_ts
    blocks_diff = seconds_diff // 2  # Polygon ~2 sec blocks
    estimated = max(1, current_block - blocks_diff)

    # Binary search to refine (wide range to handle block time variance)
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


def fetch_order_filled_events(
    alchemy_url: str,
    from_block: int,
    to_block: int,
    contract_address: str = CTF_EXCHANGE,
    block_chunk: int = 2000,
) -> list[dict]:
    """Fetch OrderFilled events from Alchemy."""
    all_logs = []
    total_blocks = to_block - from_block

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

        resp = requests.post(alchemy_url, json=payload)
        result = resp.json()

        if "error" in result:
            error = result["error"]
            # Handle Free tier limit (10 blocks)
            if "Free tier" in str(error.get("message", "")):
                if block_chunk > 10:
                    print(f"  Free tier detected, reducing to 10 blocks/request...")
                    return fetch_order_filled_events(
                        alchemy_url, from_block, to_block, contract_address, block_chunk=10
                    )
            print(f"  Error fetching logs: {error}")
            break

        logs = result.get("result", [])
        all_logs.extend(logs)

        # Progress indicator for slow free-tier fetching
        progress = (current_from - from_block) / total_blocks * 100
        if block_chunk <= 10 and int(progress) % 10 == 0 and progress > 0:
            print(f"    Progress: {progress:.0f}% ({len(all_logs)} events so far)")

        current_from = current_to + 1
        time.sleep(0.05 if block_chunk > 10 else 0.02)  # Rate limiting

    return all_logs


def decode_order_filled_event(log: dict) -> Optional[dict]:
    """
    Decode OrderFilled event data.

    Event: OrderFilled(bytes32 orderHash, address maker, address taker,
                       uint256 makerAssetId, uint256 takerAssetId,
                       uint256 makerAmountFilled, uint256 takerAmountFilled, uint256 fee)

    Topics: [event_sig, orderHash, maker, taker]
    Data: makerAssetId, takerAssetId, makerAmountFilled, takerAmountFilled, fee
    """
    try:
        topics = log.get("topics", [])
        data = log.get("data", "0x")

        if len(topics) < 4:
            return None

        # Remove '0x' prefix and split into 32-byte (64 char) chunks
        data_hex = data[2:] if data.startswith("0x") else data
        if len(data_hex) < 320:  # 5 uint256 values * 64 chars
            return None

        # Decode data fields (each uint256 is 32 bytes = 64 hex chars)
        maker_asset_id = int(data_hex[0:64], 16)
        taker_asset_id = int(data_hex[64:128], 16)
        maker_amount = int(data_hex[128:192], 16)
        taker_amount = int(data_hex[192:256], 16)
        fee = int(data_hex[256:320], 16)

        return {
            "block_number": int(log["blockNumber"], 16),
            "tx_hash": log["transactionHash"],
            "order_hash": topics[1],
            "maker": topics[2],
            "taker": topics[3],
            "maker_asset_id": str(maker_asset_id),
            "taker_asset_id": str(taker_asset_id),
            "maker_amount": maker_amount,
            "taker_amount": taker_amount,
            "fee": fee,
        }
    except Exception as e:
        return None


def parse_trades_for_token(
    events: list[dict],
    token_id: str,
) -> list[Trade]:
    """
    Parse OrderFilled events into trades for a specific token.

    In Polymarket:
    - Asset ID 0 = USDC (collateral)
    - Other asset IDs = conditional tokens (YES/NO)

    When buying tokens: taker gives USDC, receives tokens
    When selling tokens: taker gives tokens, receives USDC
    """
    trades = []

    for event in events:
        decoded = decode_order_filled_event(event)
        if not decoded:
            continue

        maker_asset = decoded["maker_asset_id"]
        taker_asset = decoded["taker_asset_id"]
        maker_amount = decoded["maker_amount"]
        taker_amount = decoded["taker_amount"]

        # Check if this trade involves our token
        # USDC is represented as asset ID "0" in the exchange
        if maker_asset == token_id and taker_asset == "0":
            # Maker provided tokens, taker provided USDC -> BUY
            token_amount = maker_amount / 1e6  # Tokens have 6 decimals like USDC
            usdc_amount = taker_amount / 1e6
            if token_amount > 0:
                price = usdc_amount / token_amount
                trades.append(Trade(
                    block_number=decoded["block_number"],
                    tx_hash=decoded["tx_hash"],
                    token_id=token_id,
                    side="buy",
                    token_amount=token_amount,
                    usdc_amount=usdc_amount,
                    price=price,
                ))
        elif taker_asset == token_id and maker_asset == "0":
            # Taker provided tokens, maker provided USDC -> SELL
            token_amount = taker_amount / 1e6
            usdc_amount = maker_amount / 1e6
            if token_amount > 0:
                price = usdc_amount / token_amount
                trades.append(Trade(
                    block_number=decoded["block_number"],
                    tx_hash=decoded["tx_hash"],
                    token_id=token_id,
                    side="sell",
                    token_amount=token_amount,
                    usdc_amount=usdc_amount,
                    price=price,
                ))

    return trades


def calculate_vwap(trades: list[Trade]) -> Optional[float]:
    """Calculate Volume-Weighted Average Price from trades."""
    if not trades:
        return None

    total_value = sum(t.usdc_amount for t in trades)
    total_volume = sum(t.token_amount for t in trades)

    if total_volume == 0:
        return None

    return total_value / total_volume


def calculate_vwap_for_block_window(
    trades: list[Trade],
    end_block: int,
    window_size: int,
) -> tuple[Optional[float], list[Trade]]:
    """
    Calculate VWAP for trades within a block window.

    Args:
        trades: List of trades (should be sorted by block_number)
        end_block: The ending block of the window
        window_size: Number of blocks to include (T parameter from paper)

    Returns:
        Tuple of (VWAP or None, trades in window)
    """
    start_block = end_block - window_size + 1
    window_trades = [t for t in trades if start_block <= t.block_number <= end_block]

    if not window_trades:
        return None, []

    total_value = sum(t.usdc_amount for t in window_trades)
    total_volume = sum(t.token_amount for t in window_trades)

    if total_volume == 0:
        return None, window_trades

    return total_value / total_volume, window_trades


def find_arbitrage_at_block(
    block: int,
    yes_trades: list[Trade],
    no_trades: list[Trade],
    window_size: int,
    profit_threshold: float,
) -> Optional[dict]:
    """
    Check for arbitrage opportunity at a specific block using windowed VWAP.

    Returns dict with VWAP info if arbitrage exists, None otherwise.
    """
    yes_vwap, yes_window = calculate_vwap_for_block_window(yes_trades, block, window_size)
    no_vwap, no_window = calculate_vwap_for_block_window(no_trades, block, window_size)

    if yes_vwap is None or no_vwap is None:
        return None

    # Paper's filter: neither exceeds 0.95
    if yes_vwap > 0.95 or no_vwap > 0.95:
        return None

    combined = yes_vwap + no_vwap
    profit = 1.0 - combined

    if profit >= profit_threshold:
        return {
            "yes_vwap": yes_vwap,
            "no_vwap": no_vwap,
            "combined": combined,
            "profit": profit,
            "yes_trades": yes_window,
            "no_trades": no_window,
        }

    return None


def extract_traded_token_ids(events: list[dict]) -> list[str]:
    """Extract unique token IDs from OrderFilled events."""
    token_ids = set()

    for log in events:
        data_hex = log.get("data", "0x")[2:]
        if len(data_hex) < 128:
            continue

        maker_asset = str(int(data_hex[0:64], 16))
        taker_asset = str(int(data_hex[64:128], 16))

        # Token IDs are non-zero asset IDs (0 = USDC)
        if maker_asset != "0":
            token_ids.add(maker_asset)
        if taker_asset != "0":
            token_ids.add(taker_asset)

    return list(token_ids)


def find_arbitrage_for_day(
    date: datetime,
    alchemy_url: str,
    profit_threshold: float = PROFIT_THRESHOLD,
    hours: int = 24,
    max_tokens: Optional[int] = None,
    market_type: str = "both",
    vwap_blocks: Optional[int] = None,  # None = "all", integer = T blocks
) -> list[ArbitrageOpportunity]:
    """
    Find arbitrage opportunities for a given day using on-chain data.

    This uses the correct approach:
    1. Fetch on-chain OrderFilled events first
    2. Extract the token IDs that were actually traded
    3. Look up markets by those token IDs (using clob_token_ids parameter)
    4. Calculate VWAP and detect arbitrage

    Per paper methodology (Section 6.1):
    - Calculate VWAP for YES and NO tokens from OrderFilled events
    - Long arbitrage: VWAP_yes + VWAP_no < 1.00
    - Filter: Neither VWAP > 0.95 (ensures liquidity/uncertainty)
    - Threshold: Profit >= $0.05 per $1 invested

    Args:
        market_type: Which markets to analyze (per Section 4.1):
            - "single": Single-condition markets only (8659 markets in paper)
            - "negrisk": NegRisk multi-condition markets only (1578 markets, 8559 conditions)
            - "both": All markets (17218 total conditions)
        vwap_blocks: VWAP window size in blocks (T parameter from paper):
            - None/"all": Use all trades in the time window (aggregated VWAP)
            - Integer T: Calculate VWAP using sliding window of T blocks
    """
    # Get block range for the time period
    day_start = datetime(date.year, date.month, date.day, tzinfo=timezone.utc)
    day_end = day_start + timedelta(hours=hours)

    print(f"Getting block range for {day_start.strftime('%Y-%m-%d')}...")
    from_block = get_block_for_timestamp(alchemy_url, int(day_start.timestamp()))
    to_block = get_block_for_timestamp(alchemy_url, int(day_end.timestamp()))
    print(f"Block range: {from_block} to {to_block} ({to_block - from_block} blocks)")

    # Step 1: Fetch OrderFilled events based on market type selection
    all_events = []

    if market_type in ("single", "both"):
        print("\nFetching OrderFilled events from CTF Exchange (single-condition markets)...")
        ctf_events = fetch_order_filled_events(alchemy_url, from_block, to_block, CTF_EXCHANGE)
        print(f"  Found {len(ctf_events)} events")
        all_events.extend(ctf_events)

    if market_type in ("negrisk", "both"):
        print("\nFetching OrderFilled events from NegRisk CTF Exchange (multi-condition markets)...")
        neg_risk_events = fetch_order_filled_events(alchemy_url, from_block, to_block, NEG_RISK_CTF_EXCHANGE)
        print(f"  Found {len(neg_risk_events)} events")
        all_events.extend(neg_risk_events)

    print(f"Total events: {len(all_events)}")

    if not all_events:
        print("No events found in this time period.")
        return []

    # Step 2: Extract token IDs from events
    print("\nExtracting traded token IDs...")
    traded_token_ids = extract_traded_token_ids(all_events)
    print(f"  Found {len(traded_token_ids)} unique tokens traded")

    # Step 3: Look up markets by token ID
    print("\nLooking up markets for traded tokens...")
    token_to_market, lookup_errors = fetch_markets_for_tokens(traded_token_ids, max_lookups=max_tokens)
    unique_markets = {m.condition_id: m for m in token_to_market.values()}
    print(f"  Found {len(unique_markets)} matching markets")

    if lookup_errors:
        total_errors = sum(lookup_errors.values())
        print(f"  WARNING: {total_errors} token lookups failed - some arbitrage may be missed")

    # Step 4: Calculate VWAP and find arbitrage
    total_blocks = to_block - from_block
    if vwap_blocks is None:
        print(f"\nAnalyzing markets for arbitrage (VWAP over all {total_blocks} blocks)...")
    else:
        print(f"\nAnalyzing markets for arbitrage (VWAP window = {vwap_blocks} blocks)...")

    opportunities = []

    for i, (condition_id, market) in enumerate(unique_markets.items()):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(unique_markets)} markets...")

        # Parse trades for YES and NO tokens
        yes_trades = parse_trades_for_token(all_events, market.yes_token_id)
        no_trades = parse_trades_for_token(all_events, market.no_token_id)

        if not yes_trades or not no_trades:
            continue

        if vwap_blocks is None:
            # Mode: "all" - aggregate VWAP over entire window
            yes_vwap = calculate_vwap(yes_trades)
            no_vwap = calculate_vwap(no_trades)

            if yes_vwap is None or no_vwap is None:
                continue

            # Paper's filter: neither exceeds 0.95
            if yes_vwap > 0.95 or no_vwap > 0.95:
                continue

            combined = yes_vwap + no_vwap
            profit = 1.0 - combined

            # Long arbitrage: can buy both for less than $1
            if profit >= profit_threshold:
                opportunities.append(ArbitrageOpportunity(
                    condition_id=market.condition_id,
                    question=market.question,
                    yes_vwap=yes_vwap,
                    no_vwap=no_vwap,
                    combined_price=combined,
                    profit_per_dollar=profit,
                    yes_trade_count=len(yes_trades),
                    no_trade_count=len(no_trades),
                    yes_volume=sum(t.token_amount for t in yes_trades),
                    no_volume=sum(t.token_amount for t in no_trades),
                    block_number=None,
                ))
        else:
            # Mode: sliding window - check each block
            # Get all unique blocks where trades occurred for this market
            all_trade_blocks = sorted(set(
                t.block_number for t in yes_trades + no_trades
            ))

            # Track best opportunity for this market to avoid duplicates
            best_opportunity = None
            best_profit = profit_threshold

            for block in all_trade_blocks:
                result = find_arbitrage_at_block(
                    block, yes_trades, no_trades, vwap_blocks, profit_threshold
                )
                if result and result["profit"] > best_profit:
                    best_profit = result["profit"]
                    best_opportunity = ArbitrageOpportunity(
                        condition_id=market.condition_id,
                        question=market.question,
                        yes_vwap=result["yes_vwap"],
                        no_vwap=result["no_vwap"],
                        combined_price=result["combined"],
                        profit_per_dollar=result["profit"],
                        yes_trade_count=len(result["yes_trades"]),
                        no_trade_count=len(result["no_trades"]),
                        yes_volume=sum(t.token_amount for t in result["yes_trades"]),
                        no_volume=sum(t.token_amount for t in result["no_trades"]),
                        block_number=block,
                    )

            if best_opportunity:
                opportunities.append(best_opportunity)

    return opportunities


def print_opportunities(opportunities: list[ArbitrageOpportunity], date: datetime) -> None:
    """Print summary of detected opportunities."""
    print(f"\n{'='*80}")
    print(f"ARBITRAGE ANALYSIS FOR {date.strftime('%Y-%m-%d')}")
    print(f"{'='*80}")

    if not opportunities:
        print("\nNo arbitrage opportunities found meeting the criteria.")
        print("(VWAP_yes + VWAP_no < 0.95, neither > 0.95)")
        return

    print(f"\nFound {len(opportunities)} conditions with arbitrage opportunities\n")

    # Sort by profit
    sorted_opps = sorted(opportunities, key=lambda x: x.profit_per_dollar, reverse=True)

    for i, opp in enumerate(sorted_opps[:20], 1):
        print(f"{i}. {opp.question[:70]}...")
        print(f"   Condition: {opp.condition_id[:20]}...")
        if opp.block_number is not None:
            print(f"   Block: {opp.block_number}")
        print(f"   YES VWAP: ${opp.yes_vwap:.4f} ({opp.yes_trade_count} trades, {opp.yes_volume:.2f} tokens)")
        print(f"   NO VWAP:  ${opp.no_vwap:.4f} ({opp.no_trade_count} trades, {opp.no_volume:.2f} tokens)")
        print(f"   Combined: ${opp.combined_price:.4f}")
        print(f"   PROFIT:   ${opp.profit_per_dollar:.4f} per $1 ({opp.profit_per_dollar*100:.1f}%)")
        print()

    # Summary stats
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Total conditions with arbitrage: {len(opportunities)}")
    print(f"Average profit per $1: ${sum(o.profit_per_dollar for o in opportunities)/len(opportunities):.4f}")
    print(f"Max profit per $1: ${max(o.profit_per_dollar for o in opportunities):.4f}")
    print(f"Median profit per $1: ${sorted([o.profit_per_dollar for o in opportunities])[len(opportunities)//2]:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect Polymarket arbitrage opportunities using on-chain data"
    )
    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="Date to analyze (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Number of hours to analyze (default: 24, use smaller for free tier)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum number of token IDs to look up (default: all)"
    )
    parser.add_argument(
        "--market-type",
        type=str,
        choices=["single", "negrisk", "both"],
        default="both",
        help=(
            "Which market type to analyze (per Section 4.1): "
            "'single' = single-condition markets only, "
            "'negrisk' = NegRisk multi-condition markets only, "
            "'both' = all markets (default)"
        )
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=PROFIT_THRESHOLD,
        help="Minimum profit threshold (default: 0.05)"
    )
    parser.add_argument(
        "--vwap-blocks",
        type=str,
        default="all",
        help=(
            "VWAP window size in blocks (T parameter from paper): "
            "'all' = aggregate over entire time window (default), "
            "or integer T = sliding window of T blocks"
        )
    )

    args = parser.parse_args()

    # Parse vwap_blocks argument
    if args.vwap_blocks.lower() == "all":
        vwap_blocks = None
    else:
        try:
            vwap_blocks = int(args.vwap_blocks)
            if vwap_blocks < 1:
                print("Error: --vwap-blocks must be a positive integer or 'all'")
                exit(1)
        except ValueError:
            print(f"Error: --vwap-blocks must be a positive integer or 'all', got '{args.vwap_blocks}'")
            exit(1)

    # Validate Alchemy API key
    try:
        alchemy_url = get_alchemy_url()
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)

    target_date = datetime.strptime(args.date, "%Y-%m-%d")

    market_type_labels = {
        "single": "Single-condition markets only",
        "negrisk": "NegRisk multi-condition markets only",
        "both": "All markets",
    }

    vwap_label = "all blocks" if vwap_blocks is None else f"{vwap_blocks} blocks"

    print(f"Polymarket Arbitrage Detector")
    print(f"Date: {target_date.strftime('%Y-%m-%d')}")
    print(f"Market type: {market_type_labels[args.market_type]}")
    print(f"VWAP window: {vwap_label}")
    print(f"Profit threshold: ${args.threshold:.2f} per $1 invested")
    print()

    # Find arbitrage (markets are now looked up by traded token IDs)
    opportunities = find_arbitrage_for_day(
        date=target_date,
        alchemy_url=alchemy_url,
        profit_threshold=args.threshold,
        hours=args.hours,
        max_tokens=args.max_tokens,
        market_type=args.market_type,
        vwap_blocks=vwap_blocks,
    )

    print_opportunities(opportunities, target_date)
