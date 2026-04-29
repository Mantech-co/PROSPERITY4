import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

DATA_DIR = Path("/media/manukrishnan/Mk/prosperity_4/data")
OUT_DIR = Path("/media/manukrishnan/Mk/prosperity_4/analysis_output/deep")
OUT_DIR.mkdir(parents=True, exist_ok=True)

price_files = sorted(DATA_DIR.glob("prices_round_5_day_*.csv"))
dfs = []
for f in price_files:
    dfs.append(pd.read_csv(f, sep=";"))
df = pd.concat(dfs, ignore_index=True)
df = df.sort_values(["product", "day", "timestamp"]).reset_index(drop=True)

products = sorted(df["product"].unique())
assert len(products) == 50

NCOLS = 5
NROWS = 10
MAX_DEGREE = 5
DEGREE_LABELS = ["Δ", "Δ²", "Δ³", "Δ⁴", "Δ⁵"]

figs = []
axes_list = []
for deg in range(1, MAX_DEGREE + 1):
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(25, 40))
    fig.suptitle(f"Degree-{deg} diff of mid_price ({DEGREE_LABELS[deg-1]}) scatter per product", fontsize=16, y=1.002)
    figs.append(fig)
    axes_list.append(axes)

for idx, product in enumerate(products):
    r, c = divmod(idx, NCOLS)
    sub = df[df["product"] == product].copy()
    # compute per-day to avoid cross-day boundary artifacts
    day_mids = [grp["mid_price"].values for _, grp in sub.groupby("day")]

    for deg in range(1, MAX_DEGREE + 1):
        ax = axes_list[deg - 1][r, c]
        series_parts = []
        for mid in day_mids:
            d = mid.copy()
            for _ in range(deg):
                d = np.diff(d)
            series_parts.append(d)
        series = np.concatenate(series_parts)
        t = np.arange(len(series))
        ax.scatter(t, series, s=1, alpha=0.4, rasterized=True)
        ax.axhline(0, color="red", linewidth=0.5)
        ax.set_title(product, fontsize=6)
        ax.tick_params(labelsize=5)
        ax.set_xlabel("tick", fontsize=5)
        ax.set_ylabel(f"{DEGREE_LABELS[deg-1]}price", fontsize=5)

for deg, fig in enumerate(figs, start=1):
    fig.tight_layout()
    out = OUT_DIR / f"midprice_delta{deg}_scatter.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)
