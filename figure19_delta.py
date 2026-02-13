"""
Reproduce the right side of Figure 19 (Appendix I.1) from arXiv:2508.03474.

Delta distribution boxplot: for each (user, condition) pair, compute the
number of blocks between consecutive trades. Show boxplots of the global
distribution and the per-pair median distribution.

A delta is "the number of blocks between the placement and execution of an
order for a given user u and condition c in a market."
"""

import json
import glob
import os
import sys
import time
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# CLOB markets for token -> condition mapping
CLOB_MARKETS_PATH = "cache/clob_markets.json"

# Exchange contracts to filter out
EXCHANGE_CONTRACTS = {
    "0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e",
    "0xc5d563a36ae78145c45a50134d48a1215220f80a",
}


def load_token_to_condition() -> dict[str, str]:
    """Load mapping from token_id -> condition_id."""
    print(f"Loading market metadata from {CLOB_MARKETS_PATH}...")
    with open(CLOB_MARKETS_PATH) as f:
        markets = json.load(f)

    mapping = {}
    for m in markets:
        tokens = m.get("tokens", [])
        cond = m["condition_id"]
        for t in tokens:
            mapping[t["token_id"]] = cond

    print(f"Loaded {len(markets)} markets, {len(mapping)} token IDs")
    return mapping


def process_events_file(
    filepath: str,
    token_to_condition: dict[str, str],
    user_condition_blocks: dict[tuple, list],
):
    """Process a single day's events file.

    For each OrderFilled event, record (user, condition, block_number)
    for the maker address. Filters bids below $2 and exchange contracts.
    """
    with open(filepath) as f:
        data = json.load(f)

    events = data.get("events", [])

    for log in events:
        topics = log.get("topics", [])
        raw_data = log.get("data", "0x")

        if len(topics) < 4:
            continue

        data_hex = raw_data[2:] if raw_data.startswith("0x") else raw_data
        if len(data_hex) < 320:
            continue

        try:
            maker_asset_id = int(data_hex[0:64], 16)
            taker_asset_id = int(data_hex[64:128], 16)
            maker_amount = int(data_hex[128:192], 16)
            taker_amount = int(data_hex[192:256], 16)
        except (ValueError, IndexError):
            continue

        # Determine USDC amount and token_id
        if maker_asset_id == 0 and taker_asset_id != 0:
            usdc = maker_amount / 1e6
            token_id = str(taker_asset_id)
        elif taker_asset_id == 0 and maker_asset_id != 0:
            usdc = taker_amount / 1e6
            token_id = str(maker_asset_id)
        else:
            continue

        # Filter bids below $2
        if usdc < 2.0:
            continue

        # Get condition_id
        condition_id = token_to_condition.get(token_id)
        if not condition_id:
            continue

        block_number = int(log["blockNumber"], 16)

        # Extract maker address (main user)
        maker_address = "0x" + topics[2][-40:]
        if maker_address.lower() not in EXCHANGE_CONTRACTS:
            key = (maker_address, condition_id)
            user_condition_blocks[key].append(block_number)

        # Also extract taker address
        taker_address = "0x" + topics[3][-40:]
        if taker_address.lower() not in EXCHANGE_CONTRACTS:
            key = (taker_address, condition_id)
            user_condition_blocks[key].append(block_number)


def main():
    token_to_condition = load_token_to_condition()
    sys.stdout.flush()

    # Accumulate block numbers per (user, condition)
    user_condition_blocks: dict[tuple, list] = defaultdict(list)

    sources = ["ctf", "negrisk"]
    total_files = 0

    for source in sources:
        cache_dir = os.path.join("cache", "events", source)
        files = sorted(glob.glob(os.path.join(cache_dir, "*.json")))
        print(f"\nProcessing {source}: {len(files)} files", flush=True)
        start = time.time()

        for i, filepath in enumerate(files):
            process_events_file(filepath, token_to_condition, user_condition_blocks)
            total_files += 1

            if (i + 1) % 30 == 0 or i == len(files) - 1:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                eta = (len(files) - i - 1) / rate if rate > 0 else 0
                n_pairs = len(user_condition_blocks)
                print(f"  [{i+1}/{len(files)}] {n_pairs:,} (user,cond) pairs | "
                      f"{rate:.1f} files/s | ETA: {eta:.0f}s", flush=True)

    # Compute deltas
    print(f"\nComputing deltas for {len(user_condition_blocks):,} (user, condition) pairs...")

    all_deltas = []       # Every individual delta
    median_deltas = []    # Median delta per (user, condition) pair

    for key, blocks in user_condition_blocks.items():
        if len(blocks) < 2:
            continue

        blocks_sorted = sorted(blocks)
        deltas = np.diff(blocks_sorted)
        # Filter out zero deltas (same block)
        deltas = deltas[deltas > 0]

        if len(deltas) == 0:
            continue

        all_deltas.append(deltas)
        median_deltas.append(float(np.median(deltas)))

    all_deltas_arr = np.concatenate(all_deltas)
    median_deltas_arr = np.array(median_deltas)

    print(f"Total individual deltas: {len(all_deltas_arr):,}")
    print(f"(User, condition) pairs with deltas: {len(median_deltas_arr):,}")

    # Print statistics
    print(f"\nGlobal delta stats:")
    print(f"  Mean:   {np.mean(all_deltas_arr):,.1f} blocks")
    print(f"  Median: {np.median(all_deltas_arr):,.1f} blocks")
    print(f"  25th:   {np.percentile(all_deltas_arr, 25):,.1f} blocks")
    print(f"  75th:   {np.percentile(all_deltas_arr, 75):,.1f} blocks")

    print(f"\nMedian Global delta stats:")
    print(f"  Mean:   {np.mean(median_deltas_arr):,.1f} blocks")
    print(f"  Median: {np.median(median_deltas_arr):,.1f} blocks")
    print(f"  25th:   {np.percentile(median_deltas_arr, 25):,.1f} blocks")
    print(f"  75th:   {np.percentile(median_deltas_arr, 75):,.1f} blocks")

    # Create boxplot matching paper's Figure 19 style
    #
    # Boxplot elements:
    #   Box:      Interquartile range (IQR) -- 25th to 75th percentile
    #   Line:     Median (50th percentile)
    #   Whiskers: Extend to furthest data point within 1.5 * IQR from box edges
    #   Circles:  Outliers (extreme values beyond the whiskers)
    #
    fig, ax = plt.subplots(figsize=(6, 6))

    # With millions of data points, plotting every outlier is impractical.
    # Draw the boxplot without fliers, then manually add a subsampled set.
    bp = ax.boxplot(
        [all_deltas_arr, median_deltas_arr],
        tick_labels=["Global", "Median Global"],
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(color="black", linewidth=1),
        capprops=dict(color="black", linewidth=1),
    )

    # Manually plot subsampled outliers as black circles
    max_outliers = 500
    for i, data in enumerate([all_deltas_arr, median_deltas_arr]):
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = data[(data < lower) | (data > upper)]
        if len(outliers) > max_outliers:
            outliers = np.random.default_rng(42).choice(
                outliers, max_outliers, replace=False
            )
        ax.scatter(
            np.full(len(outliers), i + 1),
            outliers,
            marker="o", s=6, c="black", alpha=0.4, zorder=3,
        )

    bp["boxes"][0].set_facecolor("#4C72B0")
    bp["boxes"][1].set_facecolor("#DD8452")

    ax.set_yscale("log")
    ax.set_ylabel("Value (Log Scale)")
    ax.set_title("Distribution of Deltas")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("figure19_delta.png", dpi=150)
    print("\nSaved figure19_delta.png")

    # Print boxplot element guide
    print("\nBoxplot elements:")
    print("  Box:      IQR (25th to 75th percentile)")
    print("  Line:     Median (50th percentile)")
    print("  Whiskers: Furthest point within 1.5 * IQR from box edges")
    print("  Circles:  Outliers (extreme values beyond whiskers)")


if __name__ == "__main__":
    main()
