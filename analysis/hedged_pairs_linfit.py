"""
For all pairs with return correlation > 0.8:
  1. Compute delta-neutral hedge ratio: beta = OLS(R_A ~ R_B)
  2. Build hedged spread: S = P_A - beta * P_B  (delta ≈ 0)
  3. Fit straight line S ~ t, export slope and R²
"""
import pandas as pd
import numpy as np
from scipy.stats import linregress
from itertools import combinations
from pathlib import Path

DATA_DIR = Path("data/round5")
OUT_CSV  = Path("analysis_output/hedged_pairs_linfit.csv")

# ── load ──────────────────────────────────────────────────────────────────────
dfs = []
for f in sorted(DATA_DIR.glob("prices_round_5_day_*.csv")):
    dfs.append(pd.read_csv(f, sep=";"))
raw = pd.concat(dfs, ignore_index=True)

raw["gts"] = (raw["day"] - raw["day"].min()) * 1_000_000 + raw["timestamp"]
pivot = raw.pivot_table(index="gts", columns="product", values="mid_price")
pivot.sort_index(inplace=True)
pivot.ffill(inplace=True)

returns = pivot.pct_change().dropna(how="all")
price_corr = pivot.corr()

t = pivot.index.values.astype(float)

# ── find high-corr pairs (price-level correlation) ────────────────────────────
products = list(pivot.columns)
rows = []

for a, b in combinations(products, 2):
    corr = price_corr.loc[a, b]
    if abs(corr) <= 0.8:
        continue

    # delta-neutral hedge ratio: beta = OLS(R_a ~ R_b) on returns
    ra = returns[a].dropna()
    rb = returns[b].dropna()
    idx = ra.index.intersection(rb.index)
    ra, rb = ra[idx].values, rb[idx].values

    if len(ra) < 10:
        continue

    beta, intercept, r_ret, *_ = linregress(rb, ra)

    # hedged spread in price space
    pa = pivot[a].values
    pb = pivot[b].values
    spread = pa - beta * pb

    mask = ~np.isnan(spread)
    t_valid, s_valid = t[mask], spread[mask]

    if len(t_valid) < 2:
        continue

    slope, _, r_spread, *_ = linregress(t_valid, s_valid)

    rows.append({
        "product_a":    a,
        "product_b":    b,
        "price_corr":   round(corr, 4),
        "hedge_ratio":  round(beta, 6),
        "spread_slope": slope,
        "spread_r2":    r_spread ** 2,
    })

out = pd.DataFrame(rows).sort_values("spread_r2", ascending=False)
out.to_csv(OUT_CSV, index=False)
print(f"{len(out)} pairs with |corr| > 0.8\n")
print(out.to_string(index=False))
