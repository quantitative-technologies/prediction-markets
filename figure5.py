"""
Reproduce Figure 5 from arXiv:2508.03474.

Shows the VWAP evolution for the Assad market ("Will Assad remain President
of Syria through 2024?") with arbitrage opportunity markers from Section 7
trade detection.

Reads from analysis output:
    cache/arbitrage_results/observations_single.parquet   (Section 6)
    cache/arbitrage_results/arbitrage_trades_single.parquet (Section 7)

Usage:
    uv run python figure5.py [--output figure5_assad.png]
"""

import argparse

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import polars as pl
from datetime import datetime

ASSAD_CONDITION_ID = "0x9ce4ce441b98560634b67589135dac0034a45edaacf13455c9add1ddbb6b7dec"
OBSERVATIONS_PATH = "cache/arbitrage_results/observations_single.parquet"
TRADES_PATH = "cache/arbitrage_results/arbitrage_trades_single.parquet"



def load_assad_observations() -> pl.DataFrame:
    """Load VWAP observations for the Assad market."""
    df = pl.read_parquet(OBSERVATIONS_PATH)
    return df.filter(pl.col("condition_id") == ASSAD_CONDITION_ID).sort("block")


def load_assad_trades() -> pl.DataFrame:
    """Load Section 7 arbitrage trades for the Assad market."""
    df = pl.read_parquet(TRADES_PATH)
    return df.filter(pl.col("condition_id") == ASSAD_CONDITION_ID).sort("start_block")


def build_block_to_datetime(obs: pl.DataFrame) -> dict[int, datetime]:
    """Map block numbers to datetimes using the actual date column.

    Within each date, observations are spread across the day proportional
    to their block position.
    """
    from datetime import timedelta

    # Get block range per date
    day_ranges = {}
    for row in obs.iter_rows(named=True):
        d = row["date"]
        b = row["block"]
        if d not in day_ranges:
            day_ranges[d] = [b, b]
        day_ranges[d][0] = min(day_ranges[d][0], b)
        day_ranges[d][1] = max(day_ranges[d][1], b)

    mapping = {}
    for row in obs.iter_rows(named=True):
        b = row["block"]
        d = row["date"]
        base = datetime.strptime(d, "%Y-%m-%d")
        lo, hi = day_ranges[d]
        frac = (b - lo) / (hi - lo) if hi > lo else 0.5
        mapping[b] = base + timedelta(hours=frac * 24)
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Reproduce Figure 5 from arXiv:2508.03474")
    parser.add_argument("--output", default="figure5_assad.png", help="Output file path")
    args = parser.parse_args()

    # Load observations
    obs = load_assad_observations()
    if len(obs) == 0:
        print("ERROR: No observations found for Assad market.")
        print("Run: uv run python analyze_arbitrage.py --market-type single --parquet")
        return

    print(f"Assad market observations: {len(obs):,}")
    print(f"Date range: {obs['date'].min()} to {obs['date'].max()}")

    block_dt = build_block_to_datetime(obs)
    blocks = obs["block"].to_list()
    dates = [block_dt[b] for b in blocks]
    yes_vwap = obs["yes_vwap"].to_list()
    no_vwap = obs["no_vwap"].to_list()
    deviation = obs["deviation"].to_list()

    # Arbitrage opportunity 'x' markers (Section 6: deviation > θ)
    arb_dates = []
    arb_yes = []
    arb_no = []
    for i, dev in enumerate(deviation):
        if dev > 0.02:
            arb_dates.append(dates[i])
            arb_yes.append(yes_vwap[i])
            arb_no.append(no_vwap[i])
    print(f"Arbitrage opportunity observations (θ > 0.02): {len(arb_dates):,}")

    # Load Section 7 arbitrage trades for "Opportunity Taken" markers
    trades = load_assad_trades()
    print(f"Assad arbitrage trades (Section 7): {len(trades):,}")

    # Look up VWAP values at each trade's start_block
    block_to_idx = {b: i for i, b in enumerate(blocks)}
    trade_dates = []
    trade_yes = []
    trade_no = []
    for row in trades.iter_rows(named=True):
        b = row["start_block"]
        if b in block_to_idx:
            idx = block_to_idx[b]
            trade_dates.append(block_dt[b])
            trade_yes.append(yes_vwap[idx])
            trade_no.append(no_vwap[idx])

    print(f"Trade markers plotted: {len(trade_dates)}")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))

    # VWAP as continuous thin lines
    ax.plot(dates, yes_vwap, "-", color="blue", linewidth=0.5, alpha=0.7,
            label=r"VWAP$_Y$ (Yes)")
    ax.plot(dates, no_vwap, "-", color="red", linewidth=0.5, alpha=0.7,
            label=r"VWAP$_N$ (No)")

    # Arbitrage opportunity 'x' markers (only where deviation > θ)
    ax.plot(arb_dates, arb_yes, "x", color="blue", markersize=5, alpha=0.5)
    ax.plot(arb_dates, arb_no, "x", color="red", markersize=5, alpha=0.5)

    # Arbitrage trade markers (Section 7 "Opportunity Taken")
    ax.scatter(trade_dates, trade_yes, s=60, color="darkblue", zorder=5,
               label="Opportunity Taken")
    ax.scatter(trade_dates, trade_no, s=60, color="darkred", zorder=5)

    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.set_title("Will Assad remain President of Syria through 2024?")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower left")

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    fig.autofmt_xdate(rotation=45)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
