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

# Cache for computed delta arrays
DELTA_CACHE_DIR = "cache/deltas"

# Exchange contracts to filter out
EXCHANGE_CONTRACTS = {
    "0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e",
    "0xc5d563a36ae78145c45a50134d48a1215220f80a",
}

# Window size from paper (Section 7)
T_BLOCKS = 950


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
    blocks_all: dict[tuple, list],
    blocks_gt2: dict[tuple, list],
):
    """Process a single day's events file.

    For each OrderFilled event, record (user, condition, block_number)
    for both maker and taker addresses. Populates two dicts:
      blocks_all: no dollar filter
      blocks_gt2: only bids > $2
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

        # Get condition_id
        condition_id = token_to_condition.get(token_id)
        if not condition_id:
            continue

        block_number = int(log["blockNumber"], 16)
        gt2 = usdc > 2.0

        # Extract maker address
        maker_address = "0x" + topics[2][-40:]
        if maker_address.lower() not in EXCHANGE_CONTRACTS:
            key = (maker_address, condition_id)
            blocks_all[key].append(block_number)
            if gt2:
                blocks_gt2[key].append(block_number)

        # Extract taker address
        taker_address = "0x" + topics[3][-40:]
        if taker_address.lower() not in EXCHANGE_CONTRACTS:
            key = (taker_address, condition_id)
            blocks_all[key].append(block_number)
            if gt2:
                blocks_gt2[key].append(block_number)


def compute_deltas(user_condition_blocks: dict[tuple, list]):
    """Compute all deltas and per-pair median deltas."""
    all_deltas = []
    median_deltas = []

    for key, blocks in user_condition_blocks.items():
        if len(blocks) < 2:
            continue

        blocks_sorted = sorted(blocks)
        deltas = np.diff(blocks_sorted)
        deltas = deltas[deltas > 0]

        if len(deltas) == 0:
            continue

        all_deltas.append(deltas)
        median_deltas.append(float(np.median(deltas)))

    all_arr = np.concatenate(all_deltas) if all_deltas else np.array([])
    med_arr = np.array(median_deltas)
    return all_arr, med_arr


def print_delta_stats(label: str, all_deltas: np.ndarray, median_deltas: np.ndarray):
    """Print delta statistics including T=950 match rate."""
    n_total = len(all_deltas)
    if n_total == 0:
        print(f"\n{label}: no deltas")
        return

    within_T = np.sum(all_deltas <= T_BLOCKS)
    pct_within = within_T / n_total * 100

    med_within_T = np.sum(median_deltas <= T_BLOCKS)
    med_pct = med_within_T / len(median_deltas) * 100 if len(median_deltas) > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"{label}")
    print(f"{'=' * 60}")
    print(f"Total individual deltas:       {n_total:>14,}")
    print(f"(User, condition) pairs:       {len(median_deltas):>14,}")
    print(f"")
    print(f"Global delta stats:")
    print(f"  Mean:                        {np.mean(all_deltas):>14,.1f} blocks")
    print(f"  Median:                      {np.median(all_deltas):>14,.1f} blocks")
    print(f"  25th percentile:             {np.percentile(all_deltas, 25):>14,.1f} blocks")
    print(f"  75th percentile:             {np.percentile(all_deltas, 75):>14,.1f} blocks")
    print(f"")
    print(f"Deltas within T={T_BLOCKS} blocks:")
    print(f"  Global:  {within_T:>12,} / {n_total:,} ({pct_within:.1f}%)")
    print(f"  Median:  {med_within_T:>12,} / {len(median_deltas):,} ({med_pct:.1f}%)")
    print(f"{'=' * 60}")


def save_delta_cache(all_deltas: np.ndarray, median_deltas: np.ndarray, suffix: str):
    """Save computed delta arrays to disk."""
    os.makedirs(DELTA_CACHE_DIR, exist_ok=True)
    np.save(os.path.join(DELTA_CACHE_DIR, f"all_deltas_{suffix}.npy"), all_deltas)
    np.save(os.path.join(DELTA_CACHE_DIR, f"median_deltas_{suffix}.npy"), median_deltas)
    print(f"Cached {suffix} deltas to {DELTA_CACHE_DIR}/", flush=True)


def load_delta_cache(suffix: str):
    """Load cached delta arrays if available."""
    all_path = os.path.join(DELTA_CACHE_DIR, f"all_deltas_{suffix}.npy")
    med_path = os.path.join(DELTA_CACHE_DIR, f"median_deltas_{suffix}.npy")
    if os.path.exists(all_path) and os.path.exists(med_path):
        return np.load(all_path), np.load(med_path)
    return None, None


def main():
    # Try loading from cache first
    all_deltas_all, med_deltas_all = load_delta_cache("all_bids")
    all_deltas_gt2, med_deltas_gt2 = load_delta_cache("gt2")

    if all_deltas_all is not None and all_deltas_gt2 is not None:
        print("Loaded deltas from cache", flush=True)
    else:
        print("No cache found, processing events...", flush=True)
        token_to_condition = load_token_to_condition()
        sys.stdout.flush()

        blocks_all: dict[tuple, list] = defaultdict(list)
        blocks_gt2: dict[tuple, list] = defaultdict(list)

        sources = ["ctf", "negrisk"]

        for source in sources:
            cache_dir = os.path.join("cache", "events", source)
            files = sorted(glob.glob(os.path.join(cache_dir, "*.json")))
            print(f"\nProcessing {source}: {len(files)} files", flush=True)
            start = time.time()

            for i, filepath in enumerate(files):
                process_events_file(filepath, token_to_condition, blocks_all, blocks_gt2)

                if (i + 1) % 30 == 0 or i == len(files) - 1:
                    elapsed = time.time() - start
                    rate = (i + 1) / elapsed
                    eta = (len(files) - i - 1) / rate if rate > 0 else 0
                    n_all = len(blocks_all)
                    n_gt2 = len(blocks_gt2)
                    print(f"  [{i+1}/{len(files)}] {n_all:,} pairs (all) | "
                          f"{n_gt2:,} pairs (>$2) | "
                          f"{rate:.1f} files/s | ETA: {eta:.0f}s", flush=True)

        print(f"\nComputing deltas (all bids)...", flush=True)
        all_deltas_all, med_deltas_all = compute_deltas(blocks_all)
        save_delta_cache(all_deltas_all, med_deltas_all, "all_bids")

        del blocks_all  # free memory

        print(f"Computing deltas (bids > $2)...", flush=True)
        all_deltas_gt2, med_deltas_gt2 = compute_deltas(blocks_gt2)
        save_delta_cache(all_deltas_gt2, med_deltas_gt2, "gt2")

        del blocks_gt2

    # Print stats for both
    print_delta_stats("ALL BIDS (no filter)", all_deltas_all, med_deltas_all)
    print_delta_stats("BIDS > $2 (paper filter)", all_deltas_gt2, med_deltas_gt2)

    # Create boxplot using >$2 data (matching paper)
    # Boxplot elements:
    #   Box:      Interquartile range (IQR) -- 25th to 75th percentile
    #   Line:     Median (50th percentile)
    #   Whiskers: Extend to furthest data point within 1.5 * IQR from box edges
    #   Circles:  Outliers (extreme values beyond the whiskers)
    fig, ax = plt.subplots(figsize=(6, 6))

    # With millions of data points, plotting every outlier is impractical.
    # Draw the boxplot without fliers, then manually add a subsampled set.
    bp = ax.boxplot(
        [all_deltas_gt2, med_deltas_gt2],
        tick_labels=["Global", "Median Global"],
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(color="black", linewidth=1),
        capprops=dict(color="black", linewidth=1),
    )

    # Manually plot subsampled outliers as black circles
    max_outliers = 500
    for i, data in enumerate([all_deltas_gt2, med_deltas_gt2]):
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
