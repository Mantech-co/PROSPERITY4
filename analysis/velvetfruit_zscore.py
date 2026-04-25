import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product as iproduct

DATA_DIR = Path(__file__).parent.parent / "data"
PRICE_FILES = sorted(DATA_DIR.glob("prices_round_3_day_*.csv"))


def load() -> np.ndarray:
    dfs = []
    for f in PRICE_FILES:
        df = pd.read_csv(f, sep=";")
        dfs.append(df[df["product"] == "VELVETFRUIT_EXTRACT"])
    data = pd.concat(dfs, ignore_index=True)
    data = data.sort_values(["day", "timestamp"]).reset_index(drop=True)
    return data["mid_price"].values


def simulate(prices, window, entry, exit_z, max_pos=1):
    """
    Z-score mean reversion sim.
    Buy when z < -entry, sell short when z > entry.
    Close when z crosses exit_z toward 0.
    Returns total PnL.
    """
    n = len(prices)
    pnl = 0.0
    position = 0   # +1 long, -1 short, 0 flat
    entry_price = 0.0

    for i in range(window, n):
        window_prices = prices[i - window:i]
        mu = window_prices.mean()
        sigma = window_prices.std()
        if sigma < 1e-9:
            continue
        z = (prices[i] - mu) / sigma

        if position == 0:
            if z < -entry:
                position = 1
                entry_price = prices[i]
            elif z > entry:
                position = -1
                entry_price = prices[i]
        elif position == 1:
            if z >= -exit_z:
                pnl += prices[i] - entry_price
                position = 0
        elif position == -1:
            if z <= exit_z:
                pnl += entry_price - prices[i]
                position = 0

    # close any open position at end
    if position == 1:
        pnl += prices[-1] - entry_price
    elif position == -1:
        pnl += entry_price - prices[-1]

    return pnl


def grid_search(prices):
    windows = [20, 50, 100, 200, 400]
    entries = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    exits   = [0.0, 0.1, 0.25, 0.5]

    results = []
    for w, en, ex in iproduct(windows, entries, exits):
        if ex >= en:
            continue
        p = simulate(prices, w, en, ex)
        results.append((p, w, en, ex))

    results.sort(reverse=True)
    return results


def equity_curve(prices, window, entry, exit_z):
    n = len(prices)
    equity = []
    pnl = 0.0
    position = 0
    entry_price = 0.0

    for i in range(n):
        if i >= window:
            window_prices = prices[i - window:i]
            mu = window_prices.mean()
            sigma = window_prices.std()
            if sigma > 1e-9:
                z = (prices[i] - mu) / sigma
                if position == 0:
                    if z < -entry:
                        position = 1
                        entry_price = prices[i]
                    elif z > entry:
                        position = -1
                        entry_price = prices[i]
                elif position == 1:
                    if z >= -exit_z:
                        pnl += prices[i] - entry_price
                        position = 0
                elif position == -1:
                    if z <= exit_z:
                        pnl += entry_price - prices[i]
                        position = 0
        mark = 0.0
        if position == 1:
            mark = prices[i] - entry_price
        elif position == -1:
            mark = entry_price - prices[i]
        equity.append(pnl + mark)

    return np.array(equity)


def main():
    prices = load()
    print(f"Loaded {len(prices)} ticks")

    results = grid_search(prices)

    print("\nTop 10 parameter sets:")
    print(f"{'PnL':>10}  {'window':>8}  {'entry_z':>8}  {'exit_z':>8}")
    for pnl, w, en, ex in results[:10]:
        print(f"{pnl:10.2f}  {w:8d}  {en:8.2f}  {ex:8.2f}")

    best_pnl, best_w, best_en, best_ex = results[0]
    print(f"\nBest: window={best_w}, entry={best_en}, exit={best_ex}, PnL={best_pnl:.2f}")

    eq = equity_curve(prices, best_w, best_en, best_ex)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(
        f"VELVETFRUIT_EXTRACT Z-score reversion  |  window={best_w}, entry={best_en}, exit={best_ex}  |  PnL={best_pnl:.2f}",
        fontsize=11
    )

    axes[0].plot(prices, linewidth=0.5, color="steelblue")
    axes[0].set_ylabel("mid price")
    axes[0].set_title("Mid price")

    axes[1].plot(eq, linewidth=0.8, color="green")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_ylabel("cumulative PnL")
    axes[1].set_xlabel("tick")
    axes[1].set_title("Equity curve (best params)")

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "velvetfruit_zscore.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
