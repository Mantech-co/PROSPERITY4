"""
For every pair/triple combination in each product group, fit a best-fit line
(combination value vs global timestamp), compute R² and slope, save to CSV.
"""
import itertools
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import linregress

DATA_DIR = Path(__file__).parent.parent / "data" / "round5"
OUT_CSV  = Path(__file__).parent.parent / "analysis_output" / "combo_regression.csv"

# ── load data ──────────────────────────────────────────────────────────────────
dfs = []
for f in sorted(DATA_DIR.glob("prices_round_5_day_*.csv")):
    dfs.append(pd.read_csv(f, sep=";"))
raw = pd.concat(dfs, ignore_index=True)

max_ts = raw["timestamp"].max() + 100
raw["gts"] = raw["day"] * max_ts + raw["timestamp"]

wide = raw.pivot_table(index="gts", columns="product", values="mid_price")
wide.sort_index(inplace=True)
ts = wide.index.values.astype(float)

# ── group products ─────────────────────────────────────────────────────────────
groups: dict[str, list[str]] = defaultdict(list)
for p in wide.columns:
    groups[p.split("_")[0]].append(p)

def short(name: str) -> str:
    parts = name.split("_")
    return "_".join(parts[1:]) if len(parts) > 1 else name

rows = []

def fit(series_vals):
    mask = ~np.isnan(series_vals)
    x, y = ts[mask], series_vals[mask]
    if len(x) < 2:
        return np.nan, np.nan
    slope, intercept, r, p, se = linregress(x, y)
    return slope, r**2

# ── pairs ──────────────────────────────────────────────────────────────────────
for group_key, members in sorted(groups.items()):
    for a, b in itertools.combinations(members, 2):
        la, lb = short(a), short(b)
        sa, sb = wide[a].values, wide[b].values

        for combo_type, vals in [("sum", sa + sb), ("diff", sa - sb)]:
            slope, r2 = fit(vals)
            rows.append({
                "group":      group_key,
                "combo_type": f"pair_{combo_type}",
                "combination": f"{la}+{lb}" if combo_type == "sum" else f"{la}-{lb}",
                "products":   f"{a} | {b}",
                "slope":      slope,
                "r2":         r2,
            })

# ── triples ────────────────────────────────────────────────────────────────────
for group_key, members in sorted(groups.items()):
    for a, b, c in itertools.combinations(members, 3):
        la, lb, lc = short(a), short(b), short(c)
        sa, sb, sc = wide[a].values, wide[b].values, wide[c].values

        combos = [
            (sa + sb + sc, f"{la}+{lb}+{lc}",   "sum"),
            (sa + sb - sc, f"{la}+{lb}-{lc}",   "A+B-C"),
            (sa + sc - sb, f"{la}+{lc}-{lb}",   "A+C-B"),
            (sb + sc - sa, f"{lb}+{lc}-{la}",   "B+C-A"),
        ]

        for vals, label, kind in combos:
            slope, r2 = fit(vals)
            rows.append({
                "group":      group_key,
                "combo_type": f"triple_{kind}",
                "combination": label,
                "products":   f"{a} | {b} | {c}",
                "slope":      slope,
                "r2":         r2,
            })

df_out = pd.DataFrame(rows)
df_out.to_csv(OUT_CSV, index=False)
print(f"saved {OUT_CSV}  ({len(df_out)} rows)")
print(df_out.to_string(max_rows=20))
