import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

data_dir = Path("data/round4")

dfs = []
for f in sorted(data_dir.glob("prices_round_4_day_*.csv")):
    dfs.append(pd.read_csv(f, sep=";"))

df = pd.concat(dfs, ignore_index=True)

# global timestamp: day offset + timestamp within day
day_offsets = {d: i * 1_000_000 for i, d in enumerate(sorted(df["day"].unique()))}
df["t"] = df["day"].map(day_offsets) + df["timestamp"]

results = []
for product, grp in df.groupby("product"):
    grp = grp.dropna(subset=["mid_price"]).sort_values("t")
    if len(grp) < 2:
        continue
    slope, intercept, r, p, se = stats.linregress(grp["t"], grp["mid_price"])
    results.append({"product": product, "slope": slope, "r_squared": r**2})

out = pd.DataFrame(results).sort_values("product")
out.to_csv("analysis_output/linfit_round4.csv", index=False)
print(out.to_string(index=False))
