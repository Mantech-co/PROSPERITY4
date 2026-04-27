import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm

DATA_DIR = Path(__file__).parent.parent / "data" / "iv"
OUT_DIR = Path(__file__).parent / "greek_plots"

STRIKES = sorted(
    int(f.stem.split("_")[-1]) for f in DATA_DIR.glob("iv_underlying_mid_*.csv")
)

GREEK_NAMES = ["delta", "gamma", "vega", "theta", "rho"]


def bs_greeks(S, K, T, sigma, r=0.0):
    if T <= 1e-9 or sigma <= 1e-6:
        return dict(delta=np.nan, gamma=np.nan, vega=np.nan, theta=np.nan, rho=np.nan)
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    npdf_d1 = norm.pdf(d1)
    delta = norm.cdf(d1)
    gamma = npdf_d1 / (S * sigma * sqrtT)
    vega = S * npdf_d1 * sqrtT
    theta = (-(S * npdf_d1 * sigma) / (2 * sqrtT) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 252
    rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    return dict(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def load_strike(K):
    df = pd.read_csv(DATA_DIR / f"iv_underlying_mid_{K}.csv")
    df["mid_iv"] = (df["ask_iv"] + df["bid_iv"]) / 2
    greeks = df.apply(
        lambda r: bs_greeks(r["S_mid"], r["strike"], r["tte_years"], r["mid_iv"]),
        axis=1,
        result_type="expand",
    )
    return pd.concat([df[["global_ts"]], greeks], axis=1).sort_values("global_ts")


def plot_strike(K, df):
    fig, axes = plt.subplots(len(GREEK_NAMES), 1, figsize=(14, 14), sharex=True)
    fig.suptitle(f"Option Greeks — Strike K={K}", fontsize=13)

    for ax, gname in zip(axes, GREEK_NAMES):
        x = df["global_ts"].values
        y = df[gname].values
        ax.plot(x, y, linewidth=0.9, color="steelblue")
        ax.set_ylabel(gname)
        valid = y[~np.isnan(y)]
        if len(valid):
            pad = (valid.max() - valid.min()) * 0.05 or 0.01
            ax.set_ylim(valid.min() - pad, valid.max() + pad)

    axes[-1].set_xlabel("global timestamp")
    plt.tight_layout()
    out = OUT_DIR / f"greeks_K{K}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out.name}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for K in STRIKES:
        df = load_strike(K)
        plot_strike(K, df)


if __name__ == "__main__":
    main()
