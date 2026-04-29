import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import os

data_dir = "/media/manukrishnan/Mk/prosperity_4/data/round5"
out_dir  = "/media/manukrishnan/Mk/prosperity_4/analysis_output/latenightplots/vol_analysis"
os.makedirs(out_dir, exist_ok=True)

dfs = []
for fname in sorted(os.listdir(data_dir)):
    if not fname.startswith("prices_"):
        continue
    day = int(fname.split("_day_")[1].replace(".csv", ""))
    df  = pd.read_csv(os.path.join(data_dir, fname), sep=";")
    df["day"] = day
    dfs.append(df)
data = pd.concat(dfs).sort_values(["product", "day", "timestamp"]).reset_index(drop=True)

products = sorted(data["product"].unique())

WINDOWS  = [20, 50, 100, 200]
COLS, ROWS = 5, 10

def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print("saved", os.path.basename(path))

def make_axes():
    fig, axes = plt.subplots(ROWS, COLS, figsize=(30, 40))
    return fig, axes.flatten()

def rolling_linfit_r2(price, window):
    r2 = np.full(len(price), np.nan)
    arr = price.values
    for end in range(window, len(arr) + 1):
        y = arr[end - window: end]
        x = np.arange(window, dtype=float)
        _, _, r, _, _ = stats.linregress(x, y)
        r2[end - 1] = r * r
    return pd.Series(r2, index=price.index)

# ── rolling linfit R²: one image per window ───────────────────────────────────
for w in WINDOWS:
    fig, axs = make_axes()
    for i, prod in enumerate(products):
        grp   = data[data["product"] == prod].sort_values(["day", "timestamp"]).reset_index(drop=True)
        r2    = rolling_linfit_r2(grp["mid_price"], w)
        axs[i].plot(r2, linewidth=0.7, color="mediumpurple")
        axs[i].axhline(0, color="k", linewidth=0.4, linestyle="--")
        axs[i].set_ylim(-0.05, 1.05)
        axs[i].set_title(prod, fontsize=6)
        axs[i].tick_params(labelsize=5)
    fig.suptitle(f"Rolling Linear Fit R²  window={w}", fontsize=14)
    save_fig(fig, os.path.join(out_dir, f"rolling_linfit_r2_w{w}.png"))

# ── vol of vol: one image per window ─────────────────────────────────────────
for w in WINDOWS:
    fig, axs = make_axes()
    for i, prod in enumerate(products):
        grp  = data[data["product"] == prod].sort_values(["day", "timestamp"]).reset_index(drop=True)
        ret  = grp["mid_price"].pct_change()
        rvol = ret.rolling(w).std()
        axs[i].plot(rvol.rolling(w).std(), linewidth=0.7, color="darkorange")
        axs[i].set_title(prod, fontsize=6)
        axs[i].tick_params(labelsize=5)
    fig.suptitle(f"Vol of Vol  window={w}", fontsize=14)
    save_fig(fig, os.path.join(out_dir, f"vol_of_vol_w{w}.png"))
