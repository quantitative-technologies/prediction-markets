# Arbitrage Detection: Mathematical Formulation

This document describes the mathematical methodology used for detecting arbitrage opportunities in Polymarket prediction markets, following Section 6 of [arXiv:2508.03474](https://arxiv.org/abs/2508.03474).

## 1. Trade Price

For an executed bid (OrderFilled event) with token amount $q$ and USDC amount $u$:

$$P = \frac{u}{q}$$

where $P \in [0, 1]$ represents the price per outcome token in USDC.

---

## 2. Volume-Weighted Average Price (VWAP)

For position $i$ (e.g., YES or NO token), let $\{(u_j, q_j)\}$ be the set of executed bids within the window $[b - T + 1, b]$.

The VWAP at block $b$ is:

$$VWAP_i(b) = \frac{\sum_j u_j}{\sum_j q_j}$$

This weights each trade's price by its token amount, giving more influence to larger trades.

For a binary market:
- $VWAP_Y(b)$ = volume-weighted average price of the YES token at block $b$
- $VWAP_N(b)$ = volume-weighted average price of the NO token at block $b$

---

## 3. Forward-Carry

If no trades exist for position $i$ in the window $[b - T + 1, b]$, we carry forward the VWAP from the most recent window with trades:

$$VWAP_i(b) = \begin{cases} \frac{\sum_j u_j}{\sum_j q_j} & \text{if trades exist in } [b - T + 1, b] \\ VWAP_i(b') & \text{if no trades in window, where } b' = \max\{b'' < b : \text{trades exist in } [b'' - T + 1, b'']\} \\ \text{undefined} & \text{if no trades exist in } [b - W, b] \end{cases}$$

The VWAP is carried forward for at most $W$ blocks. If no trades occurred within the lookback window, the VWAP is undefined.

---

## 4. Price Filter

To ensure sufficient market uncertainty and liquidity, we only consider blocks where:

$$VWAP_Y(b) \leq VWAP_{max} \quad \land \quad VWAP_N(b) \leq VWAP_{max}$$

This excludes near-resolved markets where one outcome has high implied probability.

---

## 5. Arbitrage Condition

Arbitrage exists at block $b$ when:

$$|1 - (VWAP_Y(b) + VWAP_N(b))| > \theta$$

In practice, all observed arbitrage is **long** (paper Section 6.1):

$$VWAP_Y(b) + VWAP_N(b) < 1 - \theta$$

The **profit per dollar invested** is:

$$\pi(b) = 1 - VWAP_Y(b) - VWAP_N(b)$$

An arbitrageur buying both outcomes for $VWAP_Y + VWAP_N$ is guaranteed a payout of \$1, yielding profit $\pi$.

---

## 6. Summary Statistics

Given $n$ markets with arbitrage opportunities $\{\pi_1, \ldots, \pi_n\}$:

| Statistic | Formula |
|-----------|---------|
| Count | $n$ |
| Average | $\bar{\pi} = \frac{1}{n}\sum_{i=1}^{n} \pi_i$ |
| Median | $\tilde{\pi} = \pi_{(\lceil n/2 \rceil)}$ (order statistic) |
| Range | $[\min_i \pi_i, \max_i \pi_i]$ |

---

## Parameters (Section 6)

| Symbol | Value | Description |
|--------|-------|-------------|
| $T$ | 1 block | VWAP window size |
| $W$ | 5,000 blocks | Forward-carry lookback window (~2.5 hours) |
| $\theta$ | 0.02 | Arbitrage threshold (2%) |
| $VWAP_{max}$ | 0.95 | Maximum VWAP for any position |

---

## References

- Paper: "Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets" (arXiv:2508.03474), Section 6
- Data source: Polymarket CLOB API via py-clob-client
- Blockchain events: Polygon OrderFilled events from CTF Exchange contracts
