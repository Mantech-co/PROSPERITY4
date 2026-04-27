import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from scipy.optimize import brentq
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent.parent / "data" / "iv"
DAYS = [0, 1, 2]
TS_SPAN = 1_000_000
TRADING_DAYS_PER_YEAR = 252


def tte_years(day: int, timestamp: int) -> float:
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    strikes = sorted(
        int(p.split("_")[1]) for p in df["product"].unique() if p.startswith("VEV_")
    )[2:8]

    underlying = df[df["product"] == "VELVETFRUIT_EXTRACT"][
        ["day", "timestamp", "bid_price_1", "ask_price_1"]
    ].copy()
    underlying["S_mid"] = (underlying["bid_price_1"] + underlying["ask_price_1"]) / 2

    for K in tqdm(strikes, desc="strikes"):
        opt = df[df["product"] == f"VEV_{K}"][
            ["day", "timestamp", "bid_price_1", "ask_price_1"]
        ].copy()
        merged = opt.merge(underlying[["day", "timestamp", "S_mid"]], on=["day", "timestamp"], how="inner")

        records = []
        for _, row in tqdm(merged.iterrows(), desc=f"K={K}", total=len(merged), leave=False):
            day, ts = int(row["day"]), int(row["timestamp"])
            T = tte_years(day, ts)
            global_ts = day * TS_SPAN + ts
            S_mid = row["S_mid"]
            moneyness = (K - S_mid)

            ask_iv = implied_vol(row["ask_price_1"], S_mid, K, T)
            bid_iv = implied_vol(row["bid_price_1"], S_mid, K, T)

            records.append({
                "day": day,
                "timestamp": ts,
                "global_ts": global_ts,
                "S_mid": S_mid,
                "strike": K,
                "moneyness": moneyness,
                "tte_years": T,
                "ask_price": row["ask_price_1"],
                "bid_price": row["bid_price_1"],
                "ask_iv": ask_iv,
                "bid_iv": bid_iv,
            })

        out = pd.DataFrame(records)
        out.to_csv(OUT_DIR / f"iv_underlying_mid_{K}.csv", index=False)
        tqdm.write(f"saved iv_underlying_mid_{K}.csv  ({out['ask_iv'].notna().sum()} valid ask IVs)")


if __name__ == "__main__":
    main()
