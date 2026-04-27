import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm

DATA_DIR = Path(__file__).parent.parent / "data" / "iv"
OUT_DIR = Path(__file__).parent
K = 5200
ROLL_WIN = 20


def bs_call(S, K, T, sigma, r=0.0):
    if T <= 1e-9 or sigma <= 1e-6:
        return max(S - K, 0.0)
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def main():
    df = pd.read_csv(DATA_DIR / f"iv_underlying_mid_{K}.csv")
    df["mid_price"] = (df["ask_price"] + df["bid_price"]) / 2
    df["mid_iv"] = (df["ask_iv"] + df["bid_iv"]) / 2
    df = df.dropna(subset=["mid_iv"]).reset_index(drop=True)

    # rolling median — same approach as options_iv_scatter.py (window=20, centered)
    df["roll_med_iv"] = (
        pd.Series(df["mid_iv"].values)
        .rolling(ROLL_WIN, center=True, min_periods=1)
        .median()
        .values
    )
    df["residual"] = df["mid_iv"] - df["roll_med_iv"]

    # IQR fence on residuals to flag outliers
    q1 = df["residual"].quantile(0.25)
    q3 = df["residual"].quantile(0.75)
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr

    df["outlier"] = (df["residual"] < lo) | (df["residual"] > hi)

    outliers = df[df["outlier"]].copy()
    outliers["fair_price"] = outliers.apply(
        lambda r: bs_call(r["S_mid"], K, r["tte_years"], r["roll_med_iv"]), axis=1
    )
    outliers["misprice"] = outliers["mid_price"] - outliers["fair_price"]
    outliers["misprice_pct"] = outliers["misprice"] / outliers["fair_price"] * 100

    print(f"K={K}  roll_window={ROLL_WIN}  residual IQR fence=[{lo:.6f}, {hi:.6f}]")
    print(f"outliers: {len(outliers)} / {len(df)} ({len(outliers)/len(df)*100:.1f}%)\n")
    print(outliers[[
        "global_ts", "S_mid", "mid_iv", "roll_med_iv",
        "residual", "mid_price", "fair_price", "misprice", "misprice_pct"
    ]].to_string(index=False))

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"IV outlier misprice analysis — K={K}  (rolling median window={ROLL_WIN})", fontsize=13)

    ts = df["global_ts"].values

    # 1. IV + rolling median + outliers
    ax = axes[0]
    ax.plot(ts, df["mid_iv"], linewidth=0.6, color="steelblue", label="mid IV")
    ax.plot(ts, df["roll_med_iv"], linewidth=1.2, color="#ff9f43", zorder=4, label=f"rolling median ({ROLL_WIN})")
    ax.fill_between(ts, df["roll_med_iv"] + hi, df["roll_med_iv"] + lo,
                    alpha=0.15, color="red", label="IQR fence band")
    ax.scatter(outliers["global_ts"], outliers["mid_iv"], color="red", s=10, zorder=5, label="outliers")
    ax.set_ylabel("mid IV")
    ax.legend(fontsize=8)

    # 2. actual vs fair price
    ax = axes[1]
    ax.plot(ts, df["mid_price"], linewidth=0.6, color="steelblue", label="actual mid price")
    fair_all = df.apply(lambda r: bs_call(r["S_mid"], K, r["tte_years"], r["roll_med_iv"]), axis=1)
    ax.plot(ts, fair_all, linewidth=0.9, color="#ff9f43", linestyle="--", label="fair price @ rolling median IV")
    ax.scatter(outliers["global_ts"], outliers["mid_price"], color="red", s=12, zorder=5, label="outlier actual")
    ax.scatter(outliers["global_ts"], outliers["fair_price"], color="green", s=12, zorder=5, label="outlier fair")
    ax.set_ylabel("option price")
    ax.legend(fontsize=8)

    # 3. misprice magnitude at outliers
    ax = axes[2]
    colors = np.where(outliers["misprice"] > 0, "green", "red")
    ax.bar(outliers["global_ts"], outliers["misprice"], width=800, color=colors, alpha=0.75)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("misprice (actual − fair)")
    ax.set_xlabel("global timestamp")

    plt.tight_layout()
    out = OUT_DIR / f"iv_misprice_K{K}.png"
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out.name}")


if __name__ == "__main__":
    main()
