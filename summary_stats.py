"""
Reproduce the left table of Figure 19 (Appendix I.1) from arXiv:2508.03474.

Summary statistics of all bids (OrderFilled events) across the study period.
Also reports the count of bids <= $2.
"""

import json
import glob
import os
import sys
import time

import numpy as np


def extract_usdc_values_from_file(filepath: str) -> np.ndarray:
    """Extract USDC dollar values from all OrderFilled events in a cache file.

    Returns array of dollar values (float64).
    """
    with open(filepath) as f:
        data = json.load(f)

    events = data.get("events", [])
    values = []

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

        # USDC has asset_id = 0, amounts are in 1e6 (6 decimals)
        if maker_asset_id == 0:
            usdc = maker_amount / 1e6
        elif taker_asset_id == 0:
            usdc = taker_amount / 1e6
        else:
            # Token-for-token trade (rare), skip
            continue

        values.append(usdc)

    return np.array(values, dtype=np.float64)


def main():
    sources = ["ctf", "negrisk"]
    all_values = []
    total_events = 0

    for source in sources:
        cache_dir = os.path.join("cache", "events", source)
        files = sorted(glob.glob(os.path.join(cache_dir, "*.json")))

        print(f"\nProcessing {source}: {len(files)} files")
        start = time.time()

        for i, filepath in enumerate(files):
            vals = extract_usdc_values_from_file(filepath)
            all_values.append(vals)
            total_events += len(vals)

            if (i + 1) % 30 == 0 or i == len(files) - 1:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                eta = (len(files) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(files)}] {total_events:,} events | "
                      f"{rate:.1f} files/s | ETA: {eta:.0f}s")

    print(f"\nConcatenating {total_events:,} values...")
    values = np.concatenate(all_values)

    print("Computing statistics...\n")

    # Compute statistics matching Figure 19
    count = len(values)
    mean = np.mean(values)
    median = np.median(values)
    minimum = np.min(values)
    maximum = np.max(values)
    std = np.std(values)
    p25 = np.percentile(values, 25)
    p75 = np.percentile(values, 75)

    # Count bids <= $2
    bids_lte_2 = np.sum(values <= 2.0)
    pct_lte_2 = bids_lte_2 / count * 100

    # Print table
    print("=" * 50)
    print("Figure 19 (Left): Summary Statistics of All Bids")
    print("=" * 50)
    print(f"{'Statistic':<25} {'Ours':>15} {'Paper':>15}")
    print("-" * 55)
    print(f"{'# of txs':<25} {count:>15,} {86620143:>15,}")
    print(f"{'Mean':<25} {mean:>15,.3f} $ {135.616:>12,.3f} $")
    print(f"{'Median':<25} {median:>15,.6f} $ {8.289:>12,.6f} $")
    print(f"{'Minimum':<25} {minimum:>15,.6f} $ {0.000001:>12,.6f} $")
    print(f"{'Maximum':<25} {maximum:>15,.3f} $ {2478476.448:>12,.3f} $")
    print(f"{'Standard Deviation':<25} {std:>15,.3f} $ {1831.994:>12,.3f} $")
    print(f"{'25th Percentile':<25} {p25:>15,.6f} $ {0.999999:>12,.6f} $")
    print(f"{'75th Percentile':<25} {p75:>15,.6f} $ {46.437:>12,.6f} $")
    print("-" * 55)
    print(f"{'Bids <= $2':<25} {bids_lte_2:>15,} ({pct_lte_2:.1f}%)")
    print("=" * 50)


if __name__ == "__main__":
    main()
