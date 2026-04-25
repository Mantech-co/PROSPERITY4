import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product as iproduct
from scipy.optimize import curve_fit

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


def main():
    prices = load()

    windows = [10, 20, 30, 50, 75, 100, 150, 200, 300, 400]
    entries = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    exits   = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5]

    # full grid
    rows = []
    for w, en, ex in iproduct(windows, entries, exits):
        if ex >= en:
            continue
        p = simulate(prices, w, en, ex)
        rows.append({"window": w, "entry": en, "exit": ex, "pnl": p})
    df = pd.DataFrame(rows)

    best = df.loc[df["pnl"].idxmax()]
    print(f"Best: window={best.window:.0f}, entry={best.entry}, exit={best.exit}, PnL={best.pnl:.2f}")

    # marginal: max PnL for each value of one param
    w_vals = sorted(df["window"].unique())
    en_vals = sorted(df["entry"].unique())
    ex_vals = sorted(df["exit"].unique())

    w_pnl  = [df[df["window"] == v]["pnl"].max() for v in w_vals]
    en_pnl = [df[df["entry"]  == v]["pnl"].max() for v in en_vals]
    ex_pnl = [df[df["exit"]   == v]["pnl"].max() for v in ex_vals]

    # fits
    def exp_decay(x, a, b, c):
        return a * np.exp(-b * np.array(x, dtype=float)) + c

    def poly2(x, a, b, c):
        x = np.array(x, dtype=float)
        return a * x**2 + b * x + c

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("VELVETFRUIT_EXTRACT — PnL vs parameters (max over other params)", fontsize=12)

    # --- window ---
    ax = axes[0]
    xf = np.linspace(min(w_vals), max(w_vals), 300)
    try:
        p0, _ = curve_fit(exp_decay, w_vals, w_pnl, p0=[1000, 0.01, 200], maxfev=5000)
        ax.plot(xf, exp_decay(xf, *p0), color="red", linewidth=1.5, label=f"exp fit")
    except Exception:
        pass
    ax.scatter(w_vals, w_pnl, color="steelblue", zorder=3)
    ax.plot(w_vals, w_pnl, color="steelblue", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("window")
    ax.set_ylabel("max PnL")
    ax.set_title("PnL vs window")
    ax.legend()
    ax.axvline(best.window, color="orange", linestyle="--", linewidth=1, label=f"best={best.window:.0f}")
    ax.legend()

    # --- entry_z ---
    ax = axes[1]
    xf = np.linspace(min(en_vals), max(en_vals), 300)
    try:
        p0, _ = curve_fit(exp_decay, en_vals, en_pnl, p0=[1000, 1.0, 100], maxfev=5000)
        ax.plot(xf, exp_decay(xf, *p0), color="red", linewidth=1.5, label="exp fit")
    except Exception:
        pass
    ax.scatter(en_vals, en_pnl, color="steelblue", zorder=3)
    ax.plot(en_vals, en_pnl, color="steelblue", linewidth=0.8, alpha=0.5)
    ax.axvline(best.entry, color="orange", linestyle="--", linewidth=1, label=f"best={best.entry}")
    ax.set_xlabel("entry z-score")
    ax.set_title("PnL vs entry threshold")
    ax.legend()

    # --- exit_z ---
    ax = axes[2]
    xf = np.linspace(min(ex_vals), max(ex_vals), 300)
    try:
        p0, _ = curve_fit(poly2, ex_vals, ex_pnl, maxfev=5000)
        ax.plot(xf, poly2(xf, *p0), color="red", linewidth=1.5, label="poly2 fit")
    except Exception:
        pass
    ax.scatter(ex_vals, ex_pnl, color="steelblue", zorder=3)
    ax.plot(ex_vals, ex_pnl, color="steelblue", linewidth=0.8, alpha=0.5)
    ax.axvline(best.exit, color="orange", linestyle="--", linewidth=1, label=f"best={best.exit}")
    ax.set_xlabel("exit z-score")
    ax.set_title("PnL vs exit threshold")
    ax.legend()

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "velvetfruit_param_sweep.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
