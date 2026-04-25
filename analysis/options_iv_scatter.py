import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm
from scipy.optimize import brentq

DATA_DIR = Path(__file__).parent.parent / "data"
DAYS = [0, 1, 2]
# timestamps: 0, 100, 200, ..., 999900  -> 10000 steps, span = 1_000_000 units
TS_SPAN = 1_000_000
TRADING_DAYS_PER_YEAR = 252


def tte_years(day: int, timestamp: int) -> float:
    """TTE in years. 8 days at (day=0, ts=0), 7 at (day=1, ts=0), 6 at (day=2, ts=0)."""
    tte_days = (8 - day) - timestamp / TS_SPAN
    return max(tte_days, 1e-9) / TRADING_DAYS_PER_YEAR


def bs_call(S, K, T, sigma, r=0.0):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_vol(price, S, K, T, r=0.0):
    if T <= 1e-9 or price <= 0:
        return np.nan
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + 1e-6:
        return np.nan
    try:
        return brentq(lambda s: bs_call(S, K, T, s, r) - price, 1e-4, 20.0, xtol=1e-6)
    except (ValueError, RuntimeError):
        return np.nan


def load_data():
    dfs = [pd.read_csv(DATA_DIR / f"prices_round_3_day_{d}.csv", sep=";") for d in DAYS]
    return pd.concat(dfs, ignore_index=True)


def main():
    df = load_data()

    strikes = sorted(
        int(p.split("_")[1]) for p in df["product"].unique() if p.startswith("VEV_")
    )

    underlying = (
        df[df["product"] == "VELVETFRUIT_EXTRACT"][
            ["day", "timestamp", "bid_price_1", "ask_price_1"]
        ].rename(columns={"bid_price_1": "S_bid", "ask_price_1": "S_ask"})
    )

    records = []
    for K in strikes:
        opt = df[df["product"] == f"VEV_{K}"][
            ["day", "timestamp", "bid_price_1", "ask_price_1"]
        ].copy()
        merged = opt.merge(underlying, on=["day", "timestamp"], how="inner")
        merged = merged.sample(frac=0.05, random_state=42).reset_index(drop=True)

        for _, row in merged.iterrows():
            T = tte_years(int(row["day"]), int(row["timestamp"]))
            global_ts = row["day"] * TS_SPAN + row["timestamp"]
            # for bid IV: hedge by buying underlying at ask → use S_ask
            # for ask IV: hedge by selling underlying at bid → use S_bid
            for side, opt_col, S_col in [
                ("bid", "bid_price_1", "S_ask"),
                ("ask", "ask_price_1", "S_bid"),
            ]:
                price = row[opt_col]
                S = row[S_col]
                iv = implied_vol(price, S, K, T)
                if not np.isnan(iv):
                    records.append({"strike": K, "side": side, "iv": iv,
                                    "global_ts": global_ts})

    ivdf = pd.DataFrame(records)
    if ivdf.empty:
        print("no valid IVs")
        return

    BID_COLOR = "#39ff6e"   # green
    ASK_COLOR = "#ff3d5a"   # red

    n = len(strikes)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes_flat = axes.flatten()

    for i, K in enumerate(strikes):
        ax = axes_flat[i]
        sub = ivdf[ivdf["strike"] == K].sort_values("global_ts")

        for side, color, label in [("bid", BID_COLOR, "bid IV"), ("ask", ASK_COLOR, "ask IV")]:
            ss = sub[sub["side"] == side]
            x, y = ss["global_ts"].values, ss["iv"].values
            if len(x) == 0:
                continue
            ax.scatter(x, y, s=6, alpha=0.35, color=color, rasterized=True, label=label)
            if len(x) >= 2:
                m, b = np.polyfit(x, y, 1)
                xs = np.array([x.min(), x.max()])
                ax.plot(xs, m * xs + b, color=color, linewidth=1.8, linestyle="--", zorder=5)
            if len(y) >= 20:
                roll_med = pd.Series(y).rolling(20, center=True, min_periods=1).median().values
                ax.plot(x, roll_med, color="#ff9f43", linewidth=1.8, zorder=6,
                        label=f"median(20) {side}")

        for d in [1, 2]:
            ax.axvline(d * TS_SPAN, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_title(f"K = {K}", fontsize=11)
        ax.set_xlabel("Global timestamp")
        ax.set_ylabel("Implied vol")
        ax.legend(fontsize=7, framealpha=0.8)
        ax.grid(True, alpha=0.2)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("VEV Options — Implied Volatility over time (per strike)", fontsize=13, y=1.01)
    plt.tight_layout()
    out = Path(__file__).parent / "options_iv_scatter.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"saved {out}")

    # --- MSE histogram for VEV_5200 bid and ask separately ---
    K = 5200
    sub5200 = ivdf[ivdf["strike"] == K].sort_values("global_ts")

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle("VEV_5200 — Squared Error from rolling median IV (window=20)", fontsize=12)

    for ax, side, color in [(axes2[0], "bid", "#39ff6e"), (axes2[1], "ask", "#ff3d5a")]:
        ss = sub5200[sub5200["side"] == side]
        y  = ss["iv"].values
        if len(y) >= 20:
            roll_med = pd.Series(y).rolling(20, center=True, min_periods=1).median().values
            se = (y - roll_med) ** 2
        else:
            se = np.array([])
        p80 = np.percentile(se, 80)
        tail = se[se >= p80]
        ax.hist(tail, bins=50, color=color, edgecolor="none", alpha=0.8)
        ax.set_title(f"{side} IV squared error (top 20%, p80={p80:.6f})")
        ax.set_xlabel("(IV − median IV)²")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out2 = Path(__file__).parent / "options_iv_mse_hist.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"saved {out2}")


if __name__ == "__main__":
    main()
