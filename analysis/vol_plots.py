import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

products = sorted(data["product"].unique())  # 50

WINDOWS  = [20, 50, 100, 200]
R2_WINS  = [20, 50, 100, 200]
COLS, ROWS = 5, 10   # 50 subplots

def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print("saved", os.path.basename(path))

def make_axes():
    fig, axes = plt.subplots(ROWS, COLS, figsize=(30, 40))
    return fig, axes.flatten()

# ── rolling volatility: one image per window ─────────────────────────────────
for w in WINDOWS:
    fig, axs = make_axes()
    for i, prod in enumerate(products):
        grp = data[data["product"] == prod].sort_values(["day", "timestamp"]).reset_index(drop=True)
        ret = grp["mid_price"].pct_change()
        axs[i].plot(ret.rolling(w).std(), linewidth=0.7, color="steelblue")
        axs[i].set_title(prod, fontsize=6)
        axs[i].tick_params(labelsize=5)
    fig.suptitle(f"Rolling Volatility  window={w}", fontsize=14)
    save_fig(fig, os.path.join(out_dir, f"rolling_vol_w{w}.png"))

# ── vol of vol: one image per window ─────────────────────────────────────────
for w in WINDOWS:
    fig, axs = make_axes()
    for i, prod in enumerate(products):
        grp = data[data["product"] == prod].sort_values(["day", "timestamp"]).reset_index(drop=True)
        ret  = grp["mid_price"].pct_change()
        rvol = ret.rolling(w).std()
        axs[i].plot(rvol.rolling(w).std(), linewidth=0.7, color="darkorange")
        axs[i].set_title(prod, fontsize=6)
        axs[i].tick_params(labelsize=5)
    fig.suptitle(f"Vol of Vol  window={w}", fontsize=14)
    save_fig(fig, os.path.join(out_dir, f"vol_of_vol_w{w}.png"))

# ── rolling R² vs EMA: one image per EMA span ────────────────────────────────
R2_ROLL = 50

def rolling_r2_vs_ema(price, span, window):
    ema     = price.ewm(span=span, adjust=False).mean()
    ss_res  = (price - ema).rolling(window).var()
    ss_tot  = price.rolling(window).var()
    return 1 - ss_res / ss_tot

for span in R2_WINS:
    fig, axs = make_axes()
    for i, prod in enumerate(products):
        grp   = data[data["product"] == prod].sort_values(["day", "timestamp"]).reset_index(drop=True)
        r2    = rolling_r2_vs_ema(grp["mid_price"], span=span, window=R2_ROLL)
        axs[i].plot(r2, linewidth=0.7, color="seagreen")
        axs[i].axhline(0, color="k", linewidth=0.4, linestyle="--")
        axs[i].set_title(prod, fontsize=6)
        axs[i].tick_params(labelsize=5)
    fig.suptitle(f"Rolling R² vs EMA  ema_span={span}  roll_window={R2_ROLL}", fontsize=14)
    save_fig(fig, os.path.join(out_dir, f"rolling_r2_ema{span}.png"))
