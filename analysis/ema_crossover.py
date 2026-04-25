"""
EMA crossover strategy optimisation with trade cost penalty.
Signal + execution: VEV_5200 mid price.

Position:
  fast_ema > slow_ema  ->  +1 (long)
  fast_ema < slow_ema  ->  -1 (short)

Objective: adj_pnl = raw_pnl - cost * n_crossovers
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
PRICE_FILES = sorted(DATA_DIR.glob("prices_round_3_day_*.csv"))


def load(product: str) -> np.ndarray:
    dfs = []
    for f in PRICE_FILES:
        df = pd.read_csv(f, sep=";")
        dfs.append(df[df["product"] == product])
    data = pd.concat(dfs, ignore_index=True).sort_values(["day", "timestamp"]).reset_index(drop=True)
    return data["mid_price"].values.astype(np.float64)


def compute_ema(prices: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1)
    ema = np.empty_like(prices)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    return ema


def main():
    prices = load("VEV_5200")
    dprice = np.diff(prices).astype(np.float32)
    n = len(prices)

    fast_spans = np.arange(2, 51, 2)        # 2,4,6,...,50
    slow_spans = np.arange(10, 201, 5)       # 10,15,...,200

    # precompute all needed EMAs
    all_spans = sorted(set(fast_spans) | set(slow_spans))
    print(f"Precomputing {len(all_spans)} EMAs over {n} ticks...")
    ema_cache = {s: compute_ema(prices, s).astype(np.float32) for s in all_spans}

    # build all valid (fast < slow) pairs
    pairs = [(f, s) for f in fast_spans for s in slow_spans if f < s]
    M = len(pairs)
    print(f"Evaluating {M} (fast, slow) pairs...")

    # position matrix: (M, n)  values in {-1, 0, +1}
    pos = np.stack([
        np.sign(ema_cache[f] - ema_cache[s]).astype(np.int8)
        for f, s in pairs
    ])  # (M, n)

    # raw PnL:  sum over t of pos[t] * dprice[t]
    pnl = (pos[:, :-1].astype(np.float32) @ dprice)  # (M,)

    # n_crossovers: number of ticks where position changes sign
    n_cross = np.sum(np.abs(np.diff(pos.astype(np.int16), axis=1)) > 0, axis=1)  # (M,)

    print(f"PnL range: [{pnl.min():.1f}, {pnl.max():.1f}]")
    print(f"Crossovers range: [{n_cross.min()}, {n_cross.max()}]")

    # reshape to grid (n_fast, n_slow)
    NF, NS = len(fast_spans), len(slow_spans)
    valid_mask = np.array([f < s for f in fast_spans for s in slow_spans])
    pnl_grid    = np.full((NF, NS), np.nan)
    ncross_grid = np.full((NF, NS), np.nan)
    idx = 0
    for i, f in enumerate(fast_spans):
        for j, s in enumerate(slow_spans):
            if f < s:
                pnl_grid[i, j]    = pnl[idx]
                ncross_grid[i, j] = n_cross[idx]
                idx += 1

    FG, SG = np.meshgrid(fast_spans, slow_spans, indexing="ij")

    cost_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

    fig = plt.figure(figsize=(22, 10))
    fig.suptitle(
        "VEV_5200 EMA crossover  |  adj_pnl = raw_pnl − cost × n_crossovers",
        fontsize=12
    )
    gs = gridspec.GridSpec(2, len(cost_values), figure=fig, hspace=0.45, wspace=0.35)

    summary = []
    for col, cost in enumerate(cost_values):
        adj = pnl_grid - cost * ncross_grid

        flat_idx = np.nanargmax(adj)
        fi, si   = np.unravel_index(flat_idx, adj.shape)
        best_f, best_s = fast_spans[fi], slow_spans[si]
        best_adj = adj[fi, si]
        best_raw = pnl_grid[fi, si]
        best_nc  = int(ncross_grid[fi, si])
        summary.append((cost, best_f, best_s, best_adj, best_raw, best_nc))

        vmin = np.nanpercentile(adj, 5)
        vmax = np.nanpercentile(adj, 95)

        ax = fig.add_subplot(gs[0, col])
        cf = ax.contourf(FG, SG, np.clip(adj, vmin, vmax), levels=25, cmap="RdYlGn")
        plt.colorbar(cf, ax=ax, pad=0.02)
        ax.scatter(best_f, best_s, color="blue", s=80, marker="*", zorder=5)
        ax.set_title(f"cost={cost:.1f}\nbest: fast={best_f} slow={best_s}", fontsize=8)
        ax.set_xlabel("fast span", fontsize=7)
        ax.set_ylabel("slow span", fontsize=7)
        ax.tick_params(labelsize=6)

        # slice through best fast span
        ax2 = fig.add_subplot(gs[1, col])
        slice_adj = adj[fi, :]
        ax2.plot(slow_spans, slice_adj, color="steelblue", linewidth=1)
        ax2.axvline(best_s, color="red",  linewidth=0.8, linestyle="--", label=f"best slow={best_s}")
        ax2.axhline(0,      color="gray", linewidth=0.5, linestyle=":")
        ax2.fill_between(slow_spans, slice_adj, 0,
                         where=slice_adj > 0, alpha=0.2, color="green")
        ax2.fill_between(slow_spans, slice_adj, 0,
                         where=slice_adj < 0, alpha=0.2, color="red")
        ax2.set_title(f"fast={best_f} fixed\nadj PnL vs slow span", fontsize=8)
        ax2.set_xlabel("slow span", fontsize=7)
        ax2.tick_params(labelsize=6)
        ax2.legend(fontsize=6)

    plt.savefig(Path(__file__).parent / "ema_crossover.png", dpi=150)

    print(f"\n{'cost':>6}  {'fast':>6}  {'slow':>6}  {'adj_pnl':>10}  {'raw_pnl':>10}  {'n_cross':>8}")
    for row in summary:
        print(f"{row[0]:6.1f}  {row[1]:6d}  {row[2]:6d}  {row[3]:10.2f}  {row[4]:10.2f}  {row[5]:8d}")

    plt.show()


if __name__ == "__main__":
    main()
