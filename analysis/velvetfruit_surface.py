import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product as iproduct
from numpy.polynomial import polynomial as P

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


def simulate(prices, window, entry, exit_z):
    n = len(prices)
    pnl = 0.0
    position = 0
    entry_price = 0.0
    for i in range(window, n):
        wp = prices[i - window:i]
        mu = wp.mean()
        sigma = wp.std()
        if sigma < 1e-9:
            continue
        z = (prices[i] - mu) / sigma
        if position == 0:
            if z < -entry:
                position = 1; entry_price = prices[i]
            elif z > entry:
                position = -1; entry_price = prices[i]
        elif position == 1:
            if z >= -exit_z:
                pnl += prices[i] - entry_price; position = 0
        elif position == -1:
            if z <= exit_z:
                pnl += entry_price - prices[i]; position = 0
    if position == 1:
        pnl += prices[-1] - entry_price
    elif position == -1:
        pnl += entry_price - prices[-1]
    return pnl


def poly2d_features(en, ex, deg=3):
    """Build polynomial feature matrix for 2D poly surface up to given degree."""
    feats = []
    for i in range(deg + 1):
        for j in range(deg + 1 - i):
            feats.append((en ** i) * (ex ** j))
    return np.column_stack(feats)


def main():
    prices = load()
    WINDOW = 10

    entries = np.linspace(0.1, 3.5, 35)
    exits   = np.linspace(0.0, 1.5, 20)

    rows = []
    for en, ex in iproduct(entries, exits):
        if ex >= en:
            continue
        p = simulate(prices, WINDOW, en, ex)
        rows.append({"entry": en, "exit": ex, "pnl": p})
    df = pd.DataFrame(rows)

    best = df.loc[df["pnl"].idxmax()]
    print(f"Best: entry={best.entry:.3f}, exit={best.exit:.3f}, PnL={best.pnl:.2f}")

    # --- polynomial surface fit ---
    en_arr = df["entry"].values
    ex_arr = df["exit"].values
    pnl_arr = df["pnl"].values

    DEG = 4
    X = poly2d_features(en_arr, ex_arr, deg=DEG)
    coeffs, res, rank, sv = np.linalg.lstsq(X, pnl_arr, rcond=None)
    pnl_pred = X @ coeffs
    ss_res = np.sum((pnl_arr - pnl_pred) ** 2)
    ss_tot = np.sum((pnl_arr - pnl_arr.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"Poly deg={DEG} surface fit R² = {r2:.4f}")

    # grid for surface plot
    en_grid = np.linspace(entries.min(), entries.max(), 200)
    ex_grid = np.linspace(exits.min(), exits.max(), 200)
    ENG, EXG = np.meshgrid(en_grid, ex_grid)
    mask = EXG < ENG
    en_flat = ENG[mask]
    ex_flat = EXG[mask]
    Xg = poly2d_features(en_flat, ex_flat, deg=DEG)
    Z_flat = Xg @ coeffs
    Z = np.full(ENG.shape, np.nan)
    Z[mask] = Z_flat

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(f"VELVETFRUIT_EXTRACT  window=10  |  2D poly surface (deg={DEG}, R²={r2:.3f})  |  best PnL={best.pnl:.2f}", fontsize=11)

    # --- scatter actual ---
    ax1 = fig.add_subplot(131)
    sc = ax1.scatter(df["entry"], df["exit"], c=df["pnl"], cmap="RdYlGn", s=25)
    plt.colorbar(sc, ax=ax1, label="PnL")
    ax1.scatter(best.entry, best.exit, color="blue", s=80, marker="*", zorder=5, label=f"best ({best.entry:.2f},{best.exit:.2f})")
    ax1.set_xlabel("entry z"); ax1.set_ylabel("exit z")
    ax1.set_title("Actual PnL (grid search)")
    ax1.legend(fontsize=8)

    # --- fitted surface contour ---
    ax2 = fig.add_subplot(132)
    cf = ax2.contourf(ENG, EXG, Z, levels=30, cmap="RdYlGn")
    plt.colorbar(cf, ax=ax2, label="PnL (fitted)")
    ax2.scatter(best.entry, best.exit, color="blue", s=80, marker="*", zorder=5)
    ax2.set_xlabel("entry z"); ax2.set_ylabel("exit z")
    ax2.set_title("Polynomial surface fit")

    # --- 3D surface ---
    ax3 = fig.add_subplot(133, projection="3d")
    ax3.plot_surface(ENG, EXG, Z, cmap="RdYlGn", alpha=0.85, linewidth=0)
    ax3.scatter(df["entry"], df["exit"], df["pnl"], color="steelblue", s=4, alpha=0.4)
    ax3.set_xlabel("entry z"); ax3.set_ylabel("exit z"); ax3.set_zlabel("PnL")
    ax3.set_title("3D surface")

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "velvetfruit_surface.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
