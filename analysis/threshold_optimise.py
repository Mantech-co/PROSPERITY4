"""
Threshold strategy: buy when price < mean - buy_thresh, sell when price > mean + sell_thresh.
Exit when deviation crosses back through 0.
Vectorised over all (buy_thresh, sell_thresh) combos.
Objective: adj_pnl = raw_pnl - cost * n_trades
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


def simulate_vectorized(prices, window, buy_thresholds, sell_thresholds):
    """
    For all (buy_thresh, sell_thresh) combos simultaneously.
    buy_thresh:  enter long  when deviation < -buy_thresh
    sell_thresh: enter short when deviation > +sell_thresh
    exit: when deviation crosses 0

    Returns pnl (B, S), ntrades (B, S)
    """
    n  = len(prices)
    B  = len(buy_thresholds)
    S  = len(sell_thresholds)
    M  = B * S

    bt = np.repeat(buy_thresholds,  S).astype(np.float32)   # (M,)
    st = np.tile(sell_thresholds,   B).astype(np.float32)   # (M,)

    pnl      = np.zeros(M, dtype=np.float32)
    ntrades  = np.zeros(M, dtype=np.int32)
    position = np.zeros(M, dtype=np.int8)
    entry_px = np.zeros(M, dtype=np.float32)

    for i in range(window, n):
        mu  = prices[i - window:i].mean()
        dev = prices[i] - mu
        p   = prices[i]

        flat  = position == 0
        long_ = position == 1
        short = position == -1

        go_long   = flat  & (dev < -bt)
        go_short  = flat  & (dev >  st)
        exit_long = long_ & (dev >= 0)
        exit_sht  = short & (dev <= 0)

        pnl    += exit_long * (p - entry_px)
        pnl    += exit_sht  * (entry_px - p)
        ntrades += (exit_long | exit_sht).astype(np.int32)
        position[exit_long | exit_sht] = 0

        ntrades += (go_long | go_short).astype(np.int32)
        position[go_long]  = 1
        position[go_short] = -1
        entry_px[go_long | go_short] = p

    # close residuals at last price
    p = prices[-1]
    pnl    += (position == 1)  * (p - entry_px)
    pnl    += (position == -1) * (entry_px - p)
    ntrades += (position != 0).astype(np.int32)

    return pnl.reshape(B, S), ntrades.reshape(B, S)


def poly2d_fit(bx, sy, z, deg=4):
    """Fit a 2D polynomial surface to scattered (bx, sy, z) data."""
    feats = []
    for i in range(deg + 1):
        for j in range(deg + 1 - i):
            feats.append((bx ** i) * (sy ** j))
    X = np.column_stack(feats)
    valid = ~np.isnan(z)
    coeffs, _, _, _ = np.linalg.lstsq(X[valid], z[valid], rcond=None)
    z_pred = X @ coeffs
    ss_res = np.nansum((z - z_pred) ** 2)
    ss_tot = np.nansum((z - np.nanmean(z)) ** 2)
    return coeffs, 1 - ss_res / ss_tot, feats


def main():
    prices = load("VEV_5200")
    print(f"Loaded {len(prices)} ticks  |  mid mean={prices.mean():.2f}  spread≈3")

    buy_thresholds  = np.linspace(0.1, 5.0, 80)
    sell_thresholds = np.linspace(0.1, 5.0, 80)

    print(f"Grid: {len(buy_thresholds)}×{len(sell_thresholds)} = {len(buy_thresholds)*len(sell_thresholds)} combos")

    import time
    t0 = time.time()
    pnl_grid, ntrade_grid = simulate_vectorized(prices, WINDOW, buy_thresholds, sell_thresholds)
    print(f"Simulation done in {time.time()-t0:.2f}s")
    print(f"raw PnL range: [{np.nanmin(pnl_grid):.1f}, {np.nanmax(pnl_grid):.1f}]")
    print(f"n_trades range: [{ntrade_grid.min()}, {ntrade_grid.max()}]")

    BG, SG = np.meshgrid(buy_thresholds, sell_thresholds, indexing="ij")

    # VEV_5200 spread ≈ 3
    cost_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

    fig = plt.figure(figsize=(22, 11))
    fig.suptitle(
        f"VEV_5200 threshold strategy  |  window={WINDOW}\n"
        "Buy when price < mean − buy_thresh  |  Sell when price > mean + sell_thresh  |  Exit at mean\n"
        "adj_pnl = raw_pnl − cost × n_trades",
        fontsize=10
    )
    gs = gridspec.GridSpec(2, len(cost_values), figure=fig, hspace=0.5, wspace=0.35)

    summary = []
    for col, cost in enumerate(cost_values):
        adj = pnl_grid - cost * ntrade_grid

        flat_idx          = np.nanargmax(adj)
        bi, si            = np.unravel_index(flat_idx, adj.shape)
        best_bt, best_st  = buy_thresholds[bi], sell_thresholds[si]
        best_adj          = adj[bi, si]
        best_raw          = pnl_grid[bi, si]
        best_nt           = int(ntrade_grid[bi, si])
        summary.append((cost, best_bt, best_st, best_adj, best_raw, best_nt))

        vmin = np.nanpercentile(adj, 5)
        vmax = np.nanpercentile(adj, 95)

        ax = fig.add_subplot(gs[0, col])
        cf = ax.contourf(BG, SG, np.clip(adj, vmin, vmax), levels=30, cmap="RdYlGn")
        plt.colorbar(cf, ax=ax, pad=0.02)
        ax.scatter(best_bt, best_st, color="blue", s=80, marker="*", zorder=5)
        ax.plot([0, 5], [0, 5], color="white", linewidth=0.4, linestyle="--", alpha=0.4)
        ax.set_title(f"cost={cost:.1f}  adj_pnl={best_adj:.0f}\nbuy_t={best_bt:.2f}  sell_t={best_st:.2f}", fontsize=8)
        ax.set_xlabel("buy threshold", fontsize=7)
        ax.set_ylabel("sell threshold", fontsize=7)
        ax.tick_params(labelsize=6)

        # diagonal slice (symmetric thresh) for intuition
        ax2 = fig.add_subplot(gs[1, col])
        diag_idx = np.arange(len(buy_thresholds))
        diag_adj = adj[diag_idx, diag_idx]
        ax2.plot(buy_thresholds, diag_adj, color="steelblue", linewidth=1, label="symmetric")
        ax2.axhline(0,           color="gray", linewidth=0.5, linestyle=":")
        ax2.fill_between(buy_thresholds, diag_adj, 0,
                         where=diag_adj > 0, alpha=0.25, color="green")
        ax2.fill_between(buy_thresholds, diag_adj, 0,
                         where=diag_adj < 0, alpha=0.25, color="red")
        # best fixed-buy slice
        ax2.plot(sell_thresholds, adj[bi, :], color="orange", linewidth=0.8,
                 linestyle="--", label=f"buy={best_bt:.2f} fixed")
        ax2.axvline(best_st, color="red", linewidth=0.8, linestyle=":")
        ax2.set_title(f"adj PnL slice  (cost={cost:.1f})", fontsize=8)
        ax2.set_xlabel("threshold", fontsize=7)
        ax2.tick_params(labelsize=6)
        ax2.legend(fontsize=6)

    plt.savefig(Path(__file__).parent / "threshold_optimise.png", dpi=150)

    print(f"\n{'cost':>6}  {'buy_t':>7}  {'sell_t':>7}  {'adj_pnl':>10}  {'raw_pnl':>10}  {'n_trades':>9}")
    for row in summary:
        print(f"{row[0]:6.1f}  {row[1]:7.3f}  {row[2]:7.3f}  {row[3]:10.2f}  {row[4]:10.2f}  {row[5]:9d}")

    plt.show()


if __name__ == "__main__":
    main()
