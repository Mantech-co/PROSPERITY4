import pandas as pd
import numpy as np
from scipy import stats
import os

data_dir = "/media/manukrishnan/Mk/prosperity_4/data/round5"
results = []

for fname in sorted(os.listdir(data_dir)):
    if not fname.startswith("prices_"):
        continue
    day = int(fname.split("_day_")[1].replace(".csv", ""))
    df = pd.read_csv(os.path.join(data_dir, fname), sep=";")
    for product, grp in df.groupby("product"):
        grp = grp.sort_values("timestamp")
        x = grp["timestamp"].values
        y = grp["mid_price"].values
        if len(x) < 2:
            r2 = float("nan")
        else:
            slope, intercept, r, p, se = stats.linregress(x, y)
            r2 = r ** 2
        results.append({"day": day, "product": product, "r2": r2})

out = pd.DataFrame(results).sort_values(["product", "day"]).reset_index(drop=True)
out.to_csv("/media/manukrishnan/Mk/prosperity_4/analysis/linfit_r2.csv", index=False)
print(out.to_string())
