import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

DATA_DIR = "../data"
DAYS = [0, 1, 2]
TOTAL_STEPS = 3 * 10000  # 3 days × 10000 timestamps/day


def load_round3():
    dfs = []
    for d in DAYS:
        df = pd.read_csv(f"{DATA_DIR}/prices_round_3_day_{d}.csv", sep=";")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def bs_call(S, K, T, sigma, r=0):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_vol(price, S, K, T, r=0):
    if T <= 1e-6 or price <= 0:
        return np.nan
    intrinsic = max(S - K, 0)
    if price <= intrinsic + 1e-6:
        return np.nan
    try:
        iv = brentq(lambda s: bs_call(S, K, T, s, r) - price, 1e-4, 10.0, xtol=1e-6)
        return iv
    except (ValueError, RuntimeError):
        return np.nan


def time_to_expiry(day, timestamp):
    elapsed = day * 10000 + timestamp / 100
    return max((TOTAL_STEPS - elapsed) / TOTAL_STEPS, 1e-6)


def main():
    df = load_round3()

    strikes = sorted([
        int(p.split("_")[1])
        for p in df["product"].unique()
        if p.startswith("VEV_")
    ])

    records = []
    underlying = df[df["product"] == "VELVETFRUIT_EXTRACT"][["day", "timestamp", "mid_price"]].copy()
    underlying = underlying.rename(columns={"mid_price": "S"})

    iv_vec = np.vectorize(implied_vol)

    for K in strikes:
        opt = df[df["product"] == f"VEV_{K}"][["day", "timestamp", "mid_price"]].copy()
        opt = opt.rename(columns={"mid_price": "opt_price"})
        merged = opt.merge(underlying, on=["day", "timestamp"], how="inner")
        merged = merged.sample(frac=0.05, random_state=42).reset_index(drop=True)
        T_arr = merged.apply(lambda r: time_to_expiry(r["day"], r["timestamp"]), axis=1).values
        iv_arr = iv_vec(merged["opt_price"].values, merged["S"].values, K, T_arr)
        mask = ~np.isnan(iv_arr)
        for iv, S in zip(iv_arr[mask], merged["S"].values[mask]):
            records.append({"strike": K, "iv": iv, "S": S, "m_t": np.log(S / K)})

    ivdf = pd.DataFrame(records)
    if ivdf.empty:
        print("No valid IVs computed")
        return

    # Quadratic fit over all points
    coeffs = np.polyfit(ivdf["m_t"], ivdf["iv"], 2)
    m_range = np.linspace(ivdf["m_t"].min(), ivdf["m_t"].max(), 300)
    iv_fit = np.polyval(coeffs, m_range)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#e8edf2")
    ax.set_facecolor("#e8edf2")

    colors = plt.cm.tab10(np.linspace(0, 0.6, len(strikes)))
    for i, K in enumerate(strikes):
        sub = ivdf[ivdf["strike"] == K]
        ax.scatter(sub["m_t"], sub["iv"], s=4, alpha=0.25,
                   color=colors[i], label=f"strike={K}", rasterized=True)

    ax.plot(m_range, iv_fit, color="black", linewidth=2.5, label="fitted Parabola", zorder=5)

    ax.set_xlabel("m_t  =  log(S / K)")
    ax.set_ylabel("v_t  (implied vol)")
    ax.set_title("VEV Options — Volatility Smile (VELVETFRUIT_EXTRACT)")
    ax.legend(markerscale=2, framealpha=0.85)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("vol_smile_vev.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved vol_smile_vev.png")


if __name__ == "__main__":
    main()
