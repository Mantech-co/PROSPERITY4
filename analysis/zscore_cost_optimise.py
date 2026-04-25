"""
Z-score parameter optimisation with trade cost penalty.

Signal:    VELVETFRUIT_EXTRACT z-score (window=10)
Execution: VEV_5200 mid-price fills
Objective: adjusted_pnl = raw_pnl - cost_per_trade * n_trades
           where n_trades = number of entry + exit events
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
PRICE_FILES = sorted(DATA_DIR.glob("prices_round_3_day_*.csv"))
WINDOW = 10


def load(product: str) -> np.ndarray:
    dfs = []
    for f in PRICE_FILES:
        df = pd.read_csv(f, sep=";")
        dfs.append(df[df["product"] == product])
    data = pd.concat(dfs, ignore_index=True).sort_values(["day", "timestamp"]).reset_index(drop=True)
    return data["mid_price"].values.astype(np.float32)


def simulate_vectorized(signal_prices, exec_prices, window, entries, exits):
    """
    For every (entry_z, exit_z) pair simultaneously:
      - compute signal from signal_prices z-score
      - execute on exec_prices at mid
    Returns:
      pnl    (E, X)  raw pnl
      ntrade (E, X)  number of trade events (entries + exits)
    """
    n   = len(signal_prices)
    E, X = len(entries), len(exits)
    M   = E * X

    en = np.repeat(entries, X).astype(np.float32)
    ex = np.tile(exits,    E).astype(np.float32)
    valid = ex < en

    pnl      = np.zeros(M, dtype=np.float32)
    ntrade   = np.zeros(M, dtype=np.int32)
    position = np.zeros(M, dtype=np.int8)    # +1 long, -1 short, 0 flat
    entry_px = np.zeros(M, dtype=np.float32)

    for i in range(window, n):
        wp    = signal_prices[i - window:i]
        mu    = wp.mean()
        sigma = wp.std()
        if sigma < 1e-9:
            continue
        z  = np.float32((signal_prices[i] - mu) / sigma)
        ep = exec_prices[i]   # execution price (mid of VEV_5200)

        flat  = (position == 0) & valid
        long_ = position == 1
        short = position == -1

        go_long  = flat & (z < -en)
        go_short = flat & (z >  en)

        exit_long  = long_  & (z >= -ex)
        exit_short = short  & (z <=  ex)

        # exits
        pnl    += exit_long  * (ep - entry_px)
        pnl    += exit_short * (entry_px - ep)
        ntrade += (exit_long | exit_short).astype(np.int32)
        position[exit_long | exit_short] = 0

        # entries
        ntrade += (go_long | go_short).astype(np.int32)
        position[go_long]  = 1
        position[go_short] = -1
        entry_px[go_long | go_short] = ep

    # close residuals
    ep = exec_prices[-1]
    pnl += (position == 1)  * (ep - entry_px)
    pnl += (position == -1) * (entry_px - ep)
    ntrade += (position != 0).astype(np.int32)

    return pnl.reshape(E, X), ntrade.reshape(E, X)


def main():
    sig  = load("VELVETFRUIT_EXTRACT")
    exec_ = load("VEV_5200")

    entries = np.linspace(0.1, 4.0, 80)
    exits   = np.linspace(0.0, 3.9, 80)

    print(f"Grid: {len(entries)}x{len(exits)} = {len(entries)*len(exits)} combos")

    import time
    t0 = time.time()
    pnl_grid, ntrade_grid = simulate_vectorized(sig, exec_, WINDOW, entries, exits)
    print(f"Done in {time.time()-t0:.2f}s")

    ENG, EXG = np.meshgrid(entries, exits, indexing="ij")
    mask = EXG < ENG
    pnl_grid[~mask]    = np.nan
    ntrade_grid[~mask] = 0

    # VEV_5200 spread ≈ 3  -> cost per trade event = 1.5 (half spread per leg)
    cost_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(
        f"VEV_5200 z-score strategy (signal=VELVETFRUIT, window={WINDOW})\n"
        "Adjusted PnL = raw PnL − cost × n_trades   (cost = spread cost per trade event)",
        fontsize=11
    )
    gs = gridspec.GridSpec(2, len(cost_values), figure=fig, hspace=0.4, wspace=0.35)

    best_rows = []
    for col, cost in enumerate(cost_values):
        adj = pnl_grid - cost * ntrade_grid
        adj[~mask] = np.nan

        flat_idx = np.nanargmax(adj)
        ei, xi   = np.unravel_index(flat_idx, adj.shape)
        best_en, best_ex, best_adj = entries[ei], exits[xi], adj[ei, xi]
        best_raw = pnl_grid[ei, xi]
        best_nt  = ntrade_grid[ei, xi]
        best_rows.append((cost, best_en, best_ex, best_adj, best_raw, best_nt))

        vmin, vmax = np.nanpercentile(adj, 5), np.nanpercentile(adj, 95)

        # contour
        ax_top = fig.add_subplot(gs[0, col])
        cf = ax_top.contourf(ENG, EXG, np.clip(adj, vmin, vmax), levels=25, cmap="RdYlGn")
        plt.colorbar(cf, ax=ax_top)
        ax_top.scatter(best_en, best_ex, color="blue", s=60, marker="*", zorder=5)
        ax_top.set_title(f"cost={cost:.1f}\nbest: en={best_en:.2f} ex={best_ex:.2f}", fontsize=8)
        ax_top.set_xlabel("entry z", fontsize=7)
        ax_top.set_ylabel("exit z",  fontsize=7)
        ax_top.tick_params(labelsize=6)

        # best-entry slice: adj pnl vs exit_z
        ax_bot = fig.add_subplot(gs[1, col])
        slice_adj = adj[ei, :]
        ax_bot.plot(exits, slice_adj, color="steelblue", linewidth=1)
        ax_bot.axvline(best_ex, color="red",  linewidth=0.8, linestyle="--")
        ax_bot.axhline(0,       color="gray", linewidth=0.5, linestyle=":")
        ax_bot.set_title(f"entry={best_en:.2f} fixed\nAdj PnL vs exit z", fontsize=8)
        ax_bot.set_xlabel("exit z", fontsize=7)
        ax_bot.tick_params(labelsize=6)

    plt.savefig(Path(__file__).parent / "zscore_cost_optimise.png", dpi=150)

    print(f"\n{'cost':>6}  {'entry_z':>8}  {'exit_z':>7}  {'adj_pnl':>9}  {'raw_pnl':>9}  {'n_trades':>9}")
    for row in best_rows:
        print(f"{row[0]:6.1f}  {row[1]:8.3f}  {row[2]:7.3f}  {row[3]:9.2f}  {row[4]:9.2f}  {row[5]:9d}")

    plt.show()


if __name__ == "__main__":
    main()
