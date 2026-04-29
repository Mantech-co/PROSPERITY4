from pathlib import Path
import pandas as pd, numpy as np, itertools, sys
from collections import defaultdict
from statsmodels.tsa.stattools import adfuller

DATA_DIR = Path(__file__).parent.parent / "data" / "round5"
dfs = [pd.read_csv(f, sep=";") for f in sorted(DATA_DIR.glob("prices_round_5_day_*.csv"))]
raw = pd.concat(dfs, ignore_index=True)
max_ts = raw["timestamp"].max() + 100
raw["gts"] = raw["day"] * max_ts + raw["timestamp"]
wide = raw.pivot_table(index="gts", columns="product", values="mid_price").sort_index()

groups = defaultdict(list)
for p in wide.columns:
    groups[p.split("_")[0]].append(p)

def half_life(y):
    y = y.dropna()
    lag = y.shift(1); delta = y.diff()
    mask = lag.notna() & delta.notna()
    b = np.polyfit(lag[mask].values, delta[mask].values, 1)[0]
    return round(-np.log(2)/b, 1) if b < 0 else 9999

rows = []
for gk, members in sorted(groups.items()):
    for a, b in itertools.combinations(members, 2):
        s = (wide[a] - wide[b]).dropna()
        p = adfuller(s, autolag="AIC")[1]
        hl = half_life(s)
        rows.append((gk,"pair",f"{a.split('_')[-1]}-{b.split('_')[-1]}",round(p,4),hl,round(float(s.std()),2)))
    for a,b,c in itertools.combinations(members, 3):
        la,lb,lc = a.split("_")[-1],b.split("_")[-1],c.split("_")[-1]
        for s,lbl in [((wide[a]+wide[b]-wide[c]).dropna(),f"{la}+{lb}-{lc}"),
                      ((wide[a]+wide[c]-wide[b]).dropna(),f"{la}+{lc}-{lb}"),
                      ((wide[b]+wide[c]-wide[a]).dropna(),f"{lb}+{lc}-{la}")]:
            p = adfuller(s, autolag="AIC")[1]
            hl = half_life(s)
            rows.append((gk,"triple",lbl,round(p,4),hl,round(float(s.std()),2)))

df = pd.DataFrame(rows, columns=["group","kind","combo","adf_p","hl","std"])
stat = df[df["adf_p"]<0.05].sort_values("hl")
pd.set_option("display.max_rows",500,"display.width",200,"display.max_colwidth",60)
print(stat.to_string(index=False))
print(f"\nTotal stationary: {len(stat)}/{len(df)}")
