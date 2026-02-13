# Scripts

Reproducing the analysis from [arXiv:2508.03474](https://arxiv.org/abs/2508.03474) ("Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets").

## Data Collection

| Script | Description | Run |
|---|---|---|
| `market_counter.py` | Fetches and caches CLOB market metadata from Polymarket API. Outputs `cache/clob_markets.json`. | `uv run python market_counter.py` |
| `fetch_events.py` | Downloads OrderFilled and PositionSplit/Merge events from Polygon via dRPC. Cached as daily JSON files in `cache/events/{source}/`. | `uv run python fetch_events.py --source all` |

### fetch_events.py options

- `--source {ctf|negrisk|ct|orders|all}` -- which events to fetch (default: `orders` = ctf+negrisk)
- `--rpc-preset {drpc|alchemy}` -- RPC provider (default: drpc)
- `--start-from YYYY-MM-DD` -- resume from a specific date
- `--block-chunk INT` -- override block chunk size for RPC calls
- `--no-cache` -- re-fetch everything
- `--status` -- show cache status and exit

## Analysis (Paper Sections 6 & 7)

| Script | Description | Run |
|---|---|---|
| `analyze_arbitrage.py` | **Section 6**: Detects arbitrage *opportunities* via VWAP deviations from $1. | `uv run python analyze_arbitrage.py --parquet` |
| `analyze_arbitrage_trades.py` | **Section 7**: Identifies executed arbitrage *trades* -- users buying both YES+NO within 950 blocks (~1 hour). | `uv run python analyze_arbitrage_trades.py --market-type both` |

### analyze_arbitrage.py options

- `--market-type {single|negrisk|both}` -- which markets (default: single)
- `--threshold FLOAT` -- arbitrage threshold (default: 0.02 = 2%)
- `--vwap-window INT` -- VWAP window size T in blocks (default: 1)
- `--forward-carry INT` -- forward-carry lookback W in blocks (default: 5000)
- `--parquet` -- save observations to Parquet (required for figure5.py)

### analyze_arbitrage_trades.py options

- `--market-type {single|negrisk|both}` -- which markets (default: single)

## Figures & Tables (Paper Reproduction)

| Script | Description | Run |
|---|---|---|
| `figure5.py` | **Figure 5**: VWAP evolution for the Assad market with arbitrage opportunity and trade markers. | `uv run python figure5.py` |
| `summary_stats.py` | **Figure 19 (left)**: Summary statistics table of all 86M bids (count, mean, median, percentiles). | `uv run python summary_stats.py` |
| `figure19_delta.py` | **Figure 19 (right)**: Delta distribution boxplot -- blocks between consecutive trades per (user, condition) pair. | `uv run python figure19_delta.py` |

### figure5.py options

- `--output FILE` -- output PNG filename (default: `figure5_assad.png`)

## Typical Workflow

```bash
# 1. One-time: fetch market metadata
uv run python market_counter.py

# 2. Cache all blockchain events (~86M events, ~85 GB)
uv run python fetch_events.py --source all

# 3. Section 6: detect arbitrage opportunities
uv run python analyze_arbitrage.py --parquet

# 4. Section 7: identify executed arbitrage trades
uv run python analyze_arbitrage_trades.py --market-type single
uv run python analyze_arbitrage_trades.py --market-type negrisk

# 5. Generate figures and tables
uv run python figure5.py
uv run python summary_stats.py
uv run python figure19_delta.py
```
