"""
Deep market analysis:
  - Return pair scatter for ALL groups
  - Multi-lag (1,2,3,5) return ACF
  - Trade volume, IAT, size distributions (product + class)
  - Random walk / trend detection (ADF, Hurst, Variance Ratio)
  - 50x50 return & delta correlation matrices
"""

import glob, os, warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import expon, kstest
from statsmodels.tsa.stattools import adfuller, acf
from itertools import combinations

OUT = "/media/manukrishnan/Mk/prosperity_4/analysis_output/deep/"
os.makedirs(OUT, exist_ok=True)

DARK = "#1a1a2e"
MID  = "#16213e"
ACCENT = "#f39c12"
CI_COLOR = "#e74c3c"

plt.rcParams.update({"figure.facecolor": DARK, "axes.facecolor": MID,
                     "text.color": "white", "axes.labelcolor": "#aaa",
                     "xtick.color": "#aaa", "ytick.color": "#aaa",
                     "axes.edgecolor": "#444", "grid.color": "#333",
                     "font.size": 8})

GROUPS = {
    "Galaxy Sounds":  ["GALAXY_SOUNDS_DARK_MATTER","GALAXY_SOUNDS_BLACK_HOLES",
                       "GALAXY_SOUNDS_PLANETARY_RINGS","GALAXY_SOUNDS_SOLAR_WINDS",
                       "GALAXY_SOUNDS_SOLAR_FLAMES"],
    "Sleep Pods":     ["SLEEP_POD_SUEDE","SLEEP_POD_LAMB_WOOL","SLEEP_POD_POLYESTER",
                       "SLEEP_POD_NYLON","SLEEP_POD_COTTON"],
    "Microchips":     ["MICROCHIP_CIRCLE","MICROCHIP_OVAL","MICROCHIP_SQUARE",
                       "MICROCHIP_RECTANGLE","MICROCHIP_TRIANGLE"],
    "Pebbles":        ["PEBBLES_XS","PEBBLES_S","PEBBLES_M","PEBBLES_L","PEBBLES_XL"],
    "Robots":         ["ROBOT_VACUUMING","ROBOT_MOPPING","ROBOT_DISHES",
                       "ROBOT_LAUNDRY","ROBOT_IRONING"],
    "UV Visors":      ["UV_VISOR_YELLOW","UV_VISOR_AMBER","UV_VISOR_ORANGE",
                       "UV_VISOR_RED","UV_VISOR_MAGENTA"],
    "Translators":    ["TRANSLATOR_SPACE_GRAY","TRANSLATOR_ASTRO_BLACK",
                       "TRANSLATOR_ECLIPSE_CHARCOAL","TRANSLATOR_GRAPHITE_MIST",
                       "TRANSLATOR_VOID_BLUE"],
    "Panels":         ["PANEL_1X2","PANEL_2X2","PANEL_1X4","PANEL_2X4","PANEL_4X4"],
    "Oxygen Shakes":  ["OXYGEN_SHAKE_MORNING_BREATH","OXYGEN_SHAKE_EVENING_BREATH",
                       "OXYGEN_SHAKE_MINT","OXYGEN_SHAKE_CHOCOLATE","OXYGEN_SHAKE_GARLIC"],
    "Snack Packs":    ["SNACKPACK_CHOCOLATE","SNACKPACK_VANILLA","SNACKPACK_PISTACHIO",
                       "SNACKPACK_STRAWBERRY","SNACKPACK_RASPBERRY"],
}
P2G = {p: g for g, ps in GROUPS.items() for p in ps}
GCOLORS = dict(zip(GROUPS, plt.cm.tab10(np.linspace(0,1,10))))

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
price_frames, trade_frames = [], []
for d in [2, 3, 4]:
    pf = pd.read_csv(f"/media/manukrishnan/Mk/prosperity_4/data/prices_round_5_day_{d}.csv", sep=";")
    pf["day"] = d
    price_frames.append(pf)
    tf = pd.read_csv(f"/media/manukrishnan/Mk/prosperity_4/data/trades_round_5_day_{d}.csv", sep=";")
    tf["day"] = d
    trade_frames.append(tf)

prices = pd.concat(price_frames, ignore_index=True)
prices["gts"] = (prices["day"] - 2) * 1_000_000 + prices["timestamp"]
prices.sort_values(["product","gts"], inplace=True)

trades = pd.concat(trade_frames, ignore_index=True)
trades["gts"] = (trades["day"] - 2) * 1_000_000 + trades["timestamp"]
trades.rename(columns={"symbol":"product"}, inplace=True)
trades["group"] = trades["product"].map(P2G)
trades.sort_values(["product","gts"], inplace=True)

ALL_PRODUCTS = sorted(p for p in prices["product"].unique() if not p.startswith("VEV_"))

pivot = prices[prices["product"].isin(ALL_PRODUCTS)].pivot_table(
    index="gts", columns="product", values="mid_price")
pivot.sort_index(inplace=True)
pivot.ffill(inplace=True)

returns = pivot.pct_change().dropna(how="all")
deltas  = pivot.diff().dropna(how="all")

print(f"Products: {len(ALL_PRODUCTS)}  |  Price rows: {len(pivot)}  |  Trade rows: {len(trades)}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Return pair scatter — all groups (one file per group)
# ─────────────────────────────────────────────────────────────────────────────
print("1. Return pair scatter (all groups)...")
for gname, gprods in GROUPS.items():
    avail = [p for p in gprods if p in returns.columns]
    if len(avail) < 2:
        continue
    n = len(avail)
    labels = [p.split("_")[-1] for p in avail]
    gc = GCOLORS[gname]

    fig, axes = plt.subplots(n, n, figsize=(n*3, n*3), facecolor=DARK)
    fig.suptitle(f"{gname} — Return Pair Scatter Matrix", color="white", fontsize=12, y=1.01)
    for i, p1 in enumerate(avail):
        for j, p2 in enumerate(avail):
            ax = axes[i][j]
            ax.set_facecolor(MID)
            ax.tick_params(labelsize=6, colors="#aaa")
            ax.spines[:].set_color("#333")
            if i == j:
                ret = returns[p1].dropna().values
                ax.hist(ret, bins=50, color=gc, alpha=0.8, edgecolor="none")
                ax.set_xlabel(labels[i], fontsize=7, color="white")
            else:
                x = returns[p2].dropna()
                y = returns[p1].reindex(x.index)
                valid = x.notna() & y.notna()
                xv, yv = x[valid].values, y[valid].values
                ax.scatter(xv, yv, s=0.5, alpha=0.3, color=gc, rasterized=True)
                if len(xv) > 5:
                    m, b, r, *_ = stats.linregress(xv, yv)
                    xx = np.array([xv.min(), xv.max()])
                    ax.plot(xx, m*xx + b, color=ACCENT, lw=1)
                    ax.text(0.05, 0.92, f"r={r:.2f}", transform=ax.transAxes,
                            fontsize=6.5, color=ACCENT)
            if j == 0:
                ax.set_ylabel(labels[i], fontsize=7, color="white")
            if i == n-1:
                ax.set_xlabel(labels[j], fontsize=7, color="white")
    plt.tight_layout()
    slug = gname.lower().replace(" ","_")
    plt.savefig(OUT + f"ret_scatter_{slug}.png", dpi=110, bbox_inches="tight", facecolor=DARK)
    plt.close()
print("   done")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Multi-lag return ACF bars: lags 1, 2, 3, 5
# ─────────────────────────────────────────────────────────────────────────────
print("2. Multi-lag return ACF bars...")
lag_targets = [1, 2, 3, 5]
n_obs = len(returns)
ci95 = 1.96 / np.sqrt(n_obs)

lag_ac = {lag: {} for lag in lag_targets}
for p in ALL_PRODUCTS:
    if p not in returns.columns:
        continue
    r = returns[p].dropna()
    for lag in lag_targets:
        lag_ac[lag][p] = r.autocorr(lag=lag)

fig, axes = plt.subplots(2, 2, figsize=(20, 14), facecolor=DARK)
axes = axes.flatten()
for ax_idx, lag in enumerate(lag_targets):
    ax = axes[ax_idx]
    ax.set_facecolor(MID)
    ax.spines[:].set_color("#333")
    lag_s = pd.Series(lag_ac[lag]).dropna().sort_values()
    bar_c = [GCOLORS.get(P2G.get(p,""), "#888") for p in lag_s.index]
    ax.bar(range(len(lag_s)), lag_s.values, color=bar_c, alpha=0.85, width=0.85)
    ax.axhline(ci95,  color=ACCENT, lw=1, ls="--", label="±95% CI")
    ax.axhline(-ci95, color=ACCENT, lw=1, ls="--")
    ax.axhline(0, color="#666", lw=0.5)
    ax.set_xticks(range(len(lag_s)))
    ax.set_xticklabels(lag_s.index, rotation=90, fontsize=6, color="white")
    ax.set_ylabel(f"Lag-{lag} ACF", color="white")
    ax.set_title(f"Lag-{lag} Return Autocorrelation", color="white", fontsize=11)
    ax.tick_params(colors="#aaa")
    if ax_idx == 0:
        legend_els = [Patch(color=GCOLORS[g], label=g) for g in GROUPS]
        ax.legend(handles=legend_els, fontsize=6.5, ncol=2,
                  facecolor="#0f3460", edgecolor="#444", labelcolor="white")

fig.suptitle("Return Autocorrelation — Lags 1, 2, 3, 5 (All Products)", color="white", fontsize=13)
plt.tight_layout()
plt.savefig(OUT + "multi_lag_return_acf.png", dpi=120, bbox_inches="tight", facecolor=DARK)
plt.close()
print("   done")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Trade volume per product (bar) + per group (stacked)
# ─────────────────────────────────────────────────────────────────────────────
print("3. Trade volume per product & group...")
vol_prod = trades.groupby("product")["quantity"].agg(["sum","count","mean"])
vol_prod.columns = ["total_vol","n_trades","avg_size"]
vol_prod["group"] = vol_prod.index.map(P2G)
vol_prod.sort_values("total_vol", ascending=False, inplace=True)

fig, axes = plt.subplots(1, 3, figsize=(22, 8), facecolor=DARK)
for ax in axes:
    ax.set_facecolor(MID)
    ax.spines[:].set_color("#333")

# total volume
clr = [GCOLORS.get(P2G.get(p,""),"#888") for p in vol_prod.index]
axes[0].barh(vol_prod.index, vol_prod["total_vol"], color=clr, alpha=0.85)
axes[0].set_xlabel("Total Traded Volume", color="white")
axes[0].set_title("Total Volume per Product", color="white", fontsize=10)
axes[0].tick_params(labelsize=6.5, colors="white")

# trade count
axes[1].barh(vol_prod.index, vol_prod["n_trades"], color=clr, alpha=0.85)
axes[1].set_xlabel("Number of Trades", color="white")
axes[1].set_title("Trade Count per Product", color="white", fontsize=10)
axes[1].tick_params(labelsize=6.5, colors="white")

# avg trade size
axes[2].barh(vol_prod.index, vol_prod["avg_size"], color=clr, alpha=0.85)
axes[2].set_xlabel("Avg Trade Size", color="white")
axes[2].set_title("Avg Trade Size per Product", color="white", fontsize=10)
axes[2].tick_params(labelsize=6.5, colors="white")

legend_els = [Patch(color=GCOLORS[g], label=g) for g in GROUPS]
axes[0].legend(handles=legend_els, fontsize=7, facecolor="#0f3460",
               edgecolor="#444", labelcolor="white")
plt.tight_layout()
plt.savefig(OUT + "trade_volume_per_product.png", dpi=120, bbox_inches="tight", facecolor=DARK)
plt.close()

# group-level stacked bar
vol_group = trades.groupby(["group","product"])["quantity"].sum().unstack(fill_value=0)
vol_group = vol_group.loc[[g for g in GROUPS if g in vol_group.index]]
fig, ax = plt.subplots(figsize=(14, 7), facecolor=DARK)
ax.set_facecolor(MID); ax.spines[:].set_color("#333")
bottom = np.zeros(len(vol_group))
colors_p = plt.cm.Set3(np.linspace(0, 1, vol_group.shape[1]))
for (col, c) in zip(vol_group.columns, colors_p):
    ax.bar(vol_group.index, vol_group[col], bottom=bottom, color=c, alpha=0.9,
           label=col.split("_")[-1], width=0.7)
    bottom += vol_group[col].values
ax.set_ylabel("Total Volume", color="white"); ax.set_xlabel("Group", color="white")
ax.set_title("Total Volume per Group (stacked by product)", color="white", fontsize=11)
ax.tick_params(colors="white", rotation=30)
ax.legend(fontsize=5.5, ncol=4, facecolor="#0f3460", edgecolor="#444", labelcolor="white")
plt.tight_layout()
plt.savefig(OUT + "trade_volume_per_group.png", dpi=120, bbox_inches="tight", facecolor=DARK)
plt.close()
print("   done")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Trade size distribution — violin per group
# ─────────────────────────────────────────────────────────────────────────────
print("4. Trade size distributions...")
fig, ax = plt.subplots(figsize=(16, 7), facecolor=DARK)
ax.set_facecolor(MID); ax.spines[:].set_color("#333")
gnames = list(GROUPS.keys())
positions = range(len(gnames))
data_violin = []
for gname in gnames:
    gdata = trades[trades["group"] == gname]["quantity"].dropna().values
    data_violin.append(gdata if len(gdata) > 0 else np.array([0]))

parts = ax.violinplot(data_violin, positions=list(positions), showmedians=True,
                      showextrema=True, widths=0.7)
for i, (pc, gname) in enumerate(zip(parts["bodies"], gnames)):
    pc.set_facecolor(GCOLORS[gname])
    pc.set_alpha(0.75)
parts["cmedians"].set_color(ACCENT)
parts["cmins"].set_color("#888")
parts["cmaxes"].set_color("#888")
parts["cbars"].set_color("#888")
ax.set_xticks(list(positions))
ax.set_xticklabels(gnames, rotation=25, color="white", fontsize=8)
ax.set_ylabel("Trade Size (quantity)", color="white")
ax.set_title("Trade Size Distribution per Group", color="white", fontsize=12)
ax.tick_params(colors="#aaa")
plt.tight_layout()
plt.savefig(OUT + "trade_size_violin.png", dpi=120, bbox_inches="tight", facecolor=DARK)
plt.close()

# per product box (sorted by median)
fig, ax = plt.subplots(figsize=(20, 8), facecolor=DARK)
ax.set_facecolor(MID); ax.spines[:].set_color("#333")
prod_order = (trades.groupby("product")["quantity"].median().sort_values(ascending=False).index.tolist())
prod_order = [p for p in prod_order if p in ALL_PRODUCTS]
box_data = [trades[trades["product"]==p]["quantity"].dropna().values for p in prod_order]
bp = ax.boxplot(box_data, patch_artist=True, showfliers=False, widths=0.6,
                medianprops=dict(color=ACCENT, lw=1.5))
for patch, prod in zip(bp["boxes"], prod_order):
    patch.set_facecolor(GCOLORS.get(P2G.get(prod,""),"#888"))
    patch.set_alpha(0.75)
ax.set_xticks(range(1, len(prod_order)+1))
ax.set_xticklabels(prod_order, rotation=90, fontsize=6, color="white")
ax.set_ylabel("Trade Size", color="white")
ax.set_title("Trade Size Boxplot per Product (sorted by median)", color="white", fontsize=11)
ax.tick_params(colors="#aaa")
plt.tight_layout()
plt.savefig(OUT + "trade_size_boxplot.png", dpi=120, bbox_inches="tight", facecolor=DARK)
plt.close()
print("   done")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Inter-Arrival Time (IAT) of trades — product & group
# ─────────────────────────────────────────────────────────────────────────────
print("5. Trade inter-arrival times...")
iat_stats = {}
for prod in ALL_PRODUCTS:
    t = trades[trades["product"]==prod]["gts"].sort_values().values
    if len(t) < 2:
        continue
    iat = np.diff(t).astype(float)
    iat = iat[iat > 0]
    if len(iat) == 0:
        continue
    iat_stats[prod] = {"median": np.median(iat), "mean": iat.mean(),
                       "p95": np.percentile(iat, 95), "data": iat}

# IAT summary bar
iat_df = pd.DataFrame({p: {"median_iat": v["median"], "mean_iat": v["mean"], "p95_iat": v["p95"]}
                        for p,v in iat_stats.items()}).T
iat_df["group"] = iat_df.index.map(P2G)
iat_df.sort_values("median_iat", inplace=True)

fig, axes = plt.subplots(1, 2, figsize=(22, 8), facecolor=DARK)
for ax in axes:
    ax.set_facecolor(MID); ax.spines[:].set_color("#333")

clr_iat = [GCOLORS.get(P2G.get(p,""),"#888") for p in iat_df.index]
axes[0].barh(iat_df.index, iat_df["median_iat"], color=clr_iat, alpha=0.85)
axes[0].set_xlabel("Median IAT (ticks)", color="white")
axes[0].set_title("Median Trade Inter-Arrival Time per Product", color="white", fontsize=10)
axes[0].tick_params(labelsize=6.5, colors="white")
legend_els = [Patch(color=GCOLORS[g], label=g) for g in GROUPS]
axes[0].legend(handles=legend_els, fontsize=7, facecolor="#0f3460", edgecolor="#444", labelcolor="white")

axes[1].barh(iat_df.index, iat_df["p95_iat"], color=clr_iat, alpha=0.85)
axes[1].set_xlabel("95th Percentile IAT (ticks)", color="white")
axes[1].set_title("P95 IAT per Product", color="white", fontsize=10)
axes[1].tick_params(labelsize=6.5, colors="white")
plt.tight_layout()
plt.savefig(OUT + "iat_bars.png", dpi=120, bbox_inches="tight", facecolor=DARK)
plt.close()

# IAT distributions: one KDE per group (all products in group overlaid)
fig, axes = plt.subplots(2, 5, figsize=(22, 10), facecolor=DARK)
axes = axes.flatten()
for idx, (gname, gprods) in enumerate(GROUPS.items()):
    ax = axes[idx]
    ax.set_facecolor(MID); ax.spines[:].set_color("#333")
    avail = [p for p in gprods if p in iat_stats]
    for prod in avail:
        iat = iat_stats[prod]["data"]
        iat_clip = iat[iat < np.percentile(iat, 99)]
        ax.hist(iat_clip, bins=50, alpha=0.5, density=True,
                label=prod.split("_")[-1], color=GCOLORS[gname])
        # exponential fit
        loc, scale = expon.fit(iat_clip, floc=0)
        xs = np.linspace(0, iat_clip.max(), 200)
        ax.plot(xs, expon.pdf(xs, loc=loc, scale=scale), lw=1.2, color=ACCENT, alpha=0.6)
    ax.set_title(gname, color="white", fontsize=8)
    ax.set_xlabel("IAT (ticks)", color="#aaa", fontsize=7)
    ax.tick_params(colors="#aaa", labelsize=6)
    ax.legend(fontsize=5.5, facecolor="#0f3460", edgecolor="#444", labelcolor="white")
fig.suptitle("IAT Distributions per Group (orange = exponential fit)", color="white", fontsize=12)
plt.tight_layout()
plt.savefig(OUT + "iat_distributions.png", dpi=120, bbox_inches="tight", facecolor=DARK)
plt.close()

# IAT violin per group
fig, ax = plt.subplots(figsize=(16, 7), facecolor=DARK)
ax.set_facecolor(MID); ax.spines[:].set_color("#333")
iat_vdata = []
for gname in gnames:
    gprods = GROUPS[gname]
    all_iat = np.concatenate([iat_stats[p]["data"] for p in gprods if p in iat_stats] or [np.array([0])])
    clip = all_iat[all_iat < np.percentile(all_iat, 99)] if len(all_iat) > 1 else all_iat
    iat_vdata.append(clip)

parts2 = ax.violinplot(iat_vdata, positions=list(range(len(gnames))),
                       showmedians=True, showextrema=True, widths=0.7)
for pc, gname in zip(parts2["bodies"], gnames):
    pc.set_facecolor(GCOLORS[gname]); pc.set_alpha(0.75)
parts2["cmedians"].set_color(ACCENT)
for key in ["cmins","cmaxes","cbars"]:
    parts2[key].set_color("#888")
ax.set_xticks(range(len(gnames)))
ax.set_xticklabels(gnames, rotation=25, color="white", fontsize=8)
ax.set_ylabel("IAT (ticks, <p99)", color="white")
ax.set_title("Trade IAT Distribution per Group", color="white", fontsize=12)
ax.tick_params(colors="#aaa")
plt.tight_layout()
plt.savefig(OUT + "iat_violin.png", dpi=120, bbox_inches="tight", facecolor=DARK)
plt.close()
print("   done")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Volume over time (rolling window) per group
# ─────────────────────────────────────────────────────────────────────────────
print("6. Volume over time per group...")
BIN = 50_000  # tick bins
trades["bin"] = (trades["gts"] // BIN) * BIN

fig, axes = plt.subplots(5, 2, figsize=(18, 22), facecolor=DARK)
axes = axes.flatten()
for idx, (gname, gprods) in enumerate(GROUPS.items()):
    ax = axes[idx]
    ax.set_facecolor(MID); ax.spines[:].set_color("#333")
    avail_t = [p for p in gprods if p in trades["product"].values]
    for prod in avail_t:
        sub = trades[trades["product"]==prod].groupby("bin")["quantity"].sum()
        ax.plot(sub.index/1e6, sub.values, lw=0.8, label=prod.split("_")[-1], alpha=0.8)
    ax.set_title(gname, color="white", fontsize=9)
    ax.set_xlabel("Time (M ticks)", color="#aaa", fontsize=7)
    ax.set_ylabel("Volume / bin", color="#aaa", fontsize=7)
    ax.tick_params(labelsize=6, colors="#aaa")
    ax.legend(fontsize=5.5, facecolor="#0f3460", edgecolor="#444", labelcolor="white")
fig.suptitle(f"Binned Trade Volume Over Time (bin={BIN//1000}k ticks)", color="white", fontsize=13)
plt.tight_layout()
plt.savefig(OUT + "volume_over_time.png", dpi=110, bbox_inches="tight", facecolor=DARK)
plt.close()
print("   done")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Buyer/Seller breakdown heatmap
# ─────────────────────────────────────────────────────────────────────────────
print("7. Buyer/seller heatmap...")
bs = trades.groupby(["buyer","product"])["quantity"].sum().unstack(fill_value=0)
bs = bs[[c for c in ALL_PRODUCTS if c in bs.columns]]
# keep top 15 buyers by total volume
top_buyers = bs.sum(axis=1).sort_values(ascending=False).head(15).index
bs = bs.loc[top_buyers]

fig, ax = plt.subplots(figsize=(18, 7), facecolor=DARK)
ax.set_facecolor(MID)
im = ax.imshow(bs.values, aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(len(bs.columns)))
ax.set_xticklabels(bs.columns, rotation=90, fontsize=6, color="white")
ax.set_yticks(range(len(bs.index)))
ax.set_yticklabels(bs.index, fontsize=7.5, color="white")
ax.set_title("Volume by Buyer × Product (top 15 buyers)", color="white", fontsize=11)
cb = plt.colorbar(im, ax=ax, fraction=0.02)
cb.ax.tick_params(colors="white"); cb.set_label("Volume", color="white")
plt.tight_layout()
plt.savefig(OUT + "buyer_product_heatmap.png", dpi=120, bbox_inches="tight", facecolor=DARK)
plt.close()
print("   done")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Random walk detection: ADF + Hurst + Variance Ratio
# ─────────────────────────────────────────────────────────────────────────────
print("8. Random walk / trend tests...")

def hurst(ts, lags=range(2, 100)):
    """Hurst exponent via R/S analysis."""
    ts = np.array(ts, dtype=float)
    ts = ts[~np.isnan(ts)]
    if len(ts) < 100:
        return np.nan
    rs = []
    for lag in lags:
        sub = ts[:lag]
        mean = sub.mean()
        dev = np.cumsum(sub - mean)
        r = dev.max() - dev.min()
        s = sub.std() or 1e-10
        rs.append(r / s)
    lags_arr = np.array(list(lags), dtype=float)
    rs_arr   = np.array(rs)
    valid    = rs_arr > 0
    if valid.sum() < 3:
        return np.nan
    poly = np.polyfit(np.log(lags_arr[valid]), np.log(rs_arr[valid]), 1)
    return poly[0]

def variance_ratio(ts, q=5):
    """Lo-MacKinlay variance ratio test statistic."""
    ts = np.array(ts, dtype=float)
    ts = ts[~np.isnan(ts)]
    if len(ts) < q * 10:
        return np.nan, np.nan
    mu = np.diff(ts).mean()
    n  = len(ts) - 1
    sig1 = ((np.diff(ts) - mu)**2).sum() / (n - 1)
    nq   = len(ts) // q
    sub  = ts[:nq*q+1]
    returns_q = sub[q::q] - sub[:-q:q] if q <= nq else np.array([])
    if len(returns_q) < 2:
        return np.nan, np.nan
    sigq = ((returns_q - q*mu)**2).sum() / (len(returns_q) - 1)
    vr   = sigq / (q * sig1) if sig1 > 0 else np.nan
    # z-stat (simplified)
    z    = (vr - 1) / np.sqrt(2*(2*q-1)*(q-1)/(3*q*n)) if sig1 > 0 else np.nan
    return vr, z

rw_rows = []
for prod in ALL_PRODUCTS:
    if prod not in pivot.columns:
        continue
    ts = pivot[prod].dropna().values
    if len(ts) < 200:
        continue
    # ADF on price level
    try:
        adf_stat, adf_p, *_ = adfuller(ts, maxlags=10, autolag="AIC")
    except Exception:
        adf_stat, adf_p = np.nan, np.nan
    # ADF on returns
    ret = np.diff(np.log(ts + 1e-10))
    try:
        adf_ret_stat, adf_ret_p, *_ = adfuller(ret, maxlags=10, autolag="AIC")
    except Exception:
        adf_ret_stat, adf_ret_p = np.nan, np.nan
    h   = hurst(ts)
    vr, vz = variance_ratio(ts, q=5)
    rw_rows.append({"product": prod, "group": P2G.get(prod,""),
                    "adf_price_stat": adf_stat, "adf_price_p": adf_p,
                    "adf_ret_p": adf_ret_p,
                    "hurst": h, "vr_q5": vr, "vr_z": vz})

rw_df = pd.DataFrame(rw_rows).set_index("product")
rw_df.to_csv(OUT + "rw_tests.csv", float_format="%.5f")

# ── plot: 3-panel summary ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(22, 9), facecolor=DARK)
for ax in axes:
    ax.set_facecolor(MID); ax.spines[:].set_color("#333")

# ADF p-value (price level)
rw_sorted = rw_df.sort_values("adf_price_p")
clr_rw = [GCOLORS.get(P2G.get(p,""),"#888") for p in rw_sorted.index]
axes[0].barh(rw_sorted.index, -np.log10(rw_sorted["adf_price_p"].clip(1e-10,1)),
             color=clr_rw, alpha=0.85)
axes[0].axvline(-np.log10(0.05), color=ACCENT, lw=1.2, ls="--", label="p=0.05")
axes[0].set_xlabel("-log₁₀(ADF p-value) [price level]", color="white", fontsize=8)
axes[0].set_title("ADF Test on Price Level\n(higher = more stationary)", color="white", fontsize=9)
axes[0].tick_params(labelsize=6.5, colors="white")
axes[0].legend(fontsize=7, facecolor="#0f3460", edgecolor="#444", labelcolor="white")

# Hurst exponent
rw_h = rw_df.dropna(subset=["hurst"]).sort_values("hurst")
clr_h = [GCOLORS.get(P2G.get(p,""),"#888") for p in rw_h.index]
axes[1].barh(rw_h.index, rw_h["hurst"], color=clr_h, alpha=0.85)
axes[1].axvline(0.5, color=ACCENT, lw=1.2, ls="--", label="H=0.5 (RW)")
axes[1].axvline(0.4, color="#e74c3c", lw=0.8, ls=":", label="H<0.5 (MR)")
axes[1].axvline(0.6, color="#2ecc71", lw=0.8, ls=":", label="H>0.5 (trend)")
axes[1].set_xlabel("Hurst Exponent", color="white", fontsize=8)
axes[1].set_title("Hurst Exponent\n(<0.5=mean-reverting, >0.5=trending)", color="white", fontsize=9)
axes[1].tick_params(labelsize=6.5, colors="white")
axes[1].legend(fontsize=7, facecolor="#0f3460", edgecolor="#444", labelcolor="white")

# Variance Ratio z-stat
rw_vz = rw_df.dropna(subset=["vr_z"]).sort_values("vr_z")
clr_vz = [GCOLORS.get(P2G.get(p,""),"#888") for p in rw_vz.index]
axes[2].barh(rw_vz.index, rw_vz["vr_z"], color=clr_vz, alpha=0.85)
axes[2].axvline(1.96,  color=ACCENT, lw=1.2, ls="--", label="±1.96 (5%)")
axes[2].axvline(-1.96, color=ACCENT, lw=1.2, ls="--")
axes[2].axvline(0, color="#666", lw=0.5)
axes[2].set_xlabel("VR z-stat (q=5)", color="white", fontsize=8)
axes[2].set_title("Variance Ratio z-stat\n(>0=trending, <0=mean-reverting)", color="white", fontsize=9)
axes[2].tick_params(labelsize=6.5, colors="white")
axes[2].legend(fontsize=7, facecolor="#0f3460", edgecolor="#444", labelcolor="white")

fig.suptitle("Random Walk Detection Tests — All Products", color="white", fontsize=13)
plt.tight_layout()
plt.savefig(OUT + "rw_tests_plots.png", dpi=120, bbox_inches="tight", facecolor=DARK)
plt.close()
print("   done — rw_tests.csv + rw_tests_plots.png")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Trend detection: rolling slope significance
# ─────────────────────────────────────────────────────────────────────────────
print("9. Trend detection (rolling slope)...")
WINDOW = 10_000  # ticks
trend_rows = []
for prod in ALL_PRODUCTS:
    if prod not in pivot.columns:
        continue
    ts = pivot[prod].dropna()
    if len(ts) < WINDOW:
        continue
    n_wins = len(ts) // WINDOW
    slopes, pvals = [], []
    for i in range(n_wins):
        chunk = ts.iloc[i*WINDOW:(i+1)*WINDOW].values
        if np.any(np.isnan(chunk)):
            continue
        x = np.arange(len(chunk), dtype=float)
        sl, _, _, pv, _ = stats.linregress(x, chunk)
        slopes.append(sl)
        pvals.append(pv)
    if not slopes:
        continue
    slopes = np.array(slopes)
    pvals  = np.array(pvals)
    trend_rows.append({"product": prod,
                       "pct_sig_up":   (slopes[pvals<0.05]>0).mean(),
                       "pct_sig_down": (slopes[pvals<0.05]<0).mean(),
                       "mean_slope":   slopes.mean(),
                       "pct_sig":      (np.array(pvals)<0.05).mean()})

td_df = pd.DataFrame(trend_rows).set_index("product").sort_values("pct_sig", ascending=False)
td_df.to_csv(OUT + "trend_tests.csv", float_format="%.5f")

fig, axes = plt.subplots(1, 2, figsize=(20, 9), facecolor=DARK)
for ax in axes: ax.set_facecolor(MID); ax.spines[:].set_color("#333")

clr_td = [GCOLORS.get(P2G.get(p,""),"#888") for p in td_df.index]
axes[0].barh(td_df.index, td_df["pct_sig"]*100, color=clr_td, alpha=0.85)
axes[0].axvline(5, color=ACCENT, lw=1.2, ls="--", label="5% baseline")
axes[0].set_xlabel("% Windows with Sig. Trend (p<0.05)", color="white")
axes[0].set_title("Trend Detection: % Trending Windows", color="white", fontsize=10)
axes[0].tick_params(labelsize=6.5, colors="white")
axes[0].legend(fontsize=7, facecolor="#0f3460", edgecolor="#444", labelcolor="white")

# stacked up/down
up_   = td_df["pct_sig_up"]*100
down_ = td_df["pct_sig_down"]*100
prods_td = td_df.index.tolist()
y_pos = np.arange(len(prods_td))
axes[1].barh(y_pos, up_,   color="#2ecc71", alpha=0.8, label="Up trend")
axes[1].barh(y_pos, -down_, color="#e74c3c", alpha=0.8, label="Down trend")
axes[1].axvline(0, color="#666", lw=0.5)
axes[1].set_yticks(y_pos)
axes[1].set_yticklabels(prods_td, fontsize=6.5, color="white")
axes[1].set_xlabel("% Windows (+ up / - down)", color="white")
axes[1].set_title("Up vs Down Trending Windows", color="white", fontsize=10)
axes[1].tick_params(colors="#aaa")
axes[1].legend(fontsize=8, facecolor="#0f3460", edgecolor="#444", labelcolor="white")

fig.suptitle("Trend Detection — Rolling Linear Regression", color="white", fontsize=13)
plt.tight_layout()
plt.savefig(OUT + "trend_detection.png", dpi=120, bbox_inches="tight", facecolor=DARK)
plt.close()
print("   done")

# ─────────────────────────────────────────────────────────────────────────────
# 10. 50×50 Return Correlation Matrix
# ─────────────────────────────────────────────────────────────────────────────
print("10. Return correlation matrix (50×50)...")
ret_cols = [p for p in ALL_PRODUCTS if p in returns.columns]
ret_corr = returns[ret_cols].corr()

fig, ax = plt.subplots(figsize=(20, 18), facecolor=DARK)
ax.set_facecolor(MID)
norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
im = ax.imshow(ret_corr.values, cmap="RdBu_r", norm=norm, aspect="auto")
ax.set_xticks(range(len(ret_cols)))
ax.set_xticklabels(ret_cols, rotation=90, fontsize=5.5, color="white")
ax.set_yticks(range(len(ret_cols)))
ax.set_yticklabels(ret_cols, fontsize=5.5, color="white")
ax.set_title("Return Correlation Matrix (all non-VEV products)", color="white", fontsize=13)
cb = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
cb.ax.tick_params(colors="white"); cb.set_label("Pearson r", color="white")

# group boundary lines
group_prods_ordered = []
for g in GROUPS:
    for p in GROUPS[g]:
        if p in ret_cols:
            group_prods_ordered.append(p)
# draw separators
pos = 0
for g in GROUPS:
    cnt = sum(1 for p in GROUPS[g] if p in ret_cols)
    if cnt == 0:
        continue
    ax.axhline(pos - 0.5, color="#aaa", lw=0.6, alpha=0.5)
    ax.axvline(pos - 0.5, color="#aaa", lw=0.6, alpha=0.5)
    pos += cnt

plt.tight_layout()
plt.savefig(OUT + "return_corr_matrix.png", dpi=130, bbox_inches="tight", facecolor=DARK)
plt.close()
print("   done")

# ─────────────────────────────────────────────────────────────────────────────
# 11. 50×50 Delta (price change) Correlation Matrix
# ─────────────────────────────────────────────────────────────────────────────
print("11. Delta correlation matrix (50×50)...")
delt_cols = [p for p in ALL_PRODUCTS if p in deltas.columns]
delt_corr = deltas[delt_cols].corr()

fig, ax = plt.subplots(figsize=(20, 18), facecolor=DARK)
ax.set_facecolor(MID)
im = ax.imshow(delt_corr.values, cmap="RdBu_r", norm=norm, aspect="auto")
ax.set_xticks(range(len(delt_cols)))
ax.set_xticklabels(delt_cols, rotation=90, fontsize=5.5, color="white")
ax.set_yticks(range(len(delt_cols)))
ax.set_yticklabels(delt_cols, fontsize=5.5, color="white")
ax.set_title("Delta (ΔPrice) Correlation Matrix (all non-VEV products)", color="white", fontsize=13)
cb = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
cb.ax.tick_params(colors="white"); cb.set_label("Pearson r", color="white")
pos = 0
for g in GROUPS:
    cnt = sum(1 for p in GROUPS[g] if p in delt_cols)
    if cnt == 0: continue
    ax.axhline(pos - 0.5, color="#aaa", lw=0.6, alpha=0.5)
    ax.axvline(pos - 0.5, color="#aaa", lw=0.6, alpha=0.5)
    pos += cnt

plt.tight_layout()
plt.savefig(OUT + "delta_corr_matrix.png", dpi=130, bbox_inches="tight", facecolor=DARK)
plt.close()
print("   done")

# ─────────────────────────────────────────────────────────────────────────────
# 12. BONUS: Trade count heatmap over time (product × time bin)
# ─────────────────────────────────────────────────────────────────────────────
print("12. Trade count heatmap...")
BIN2 = 100_000
trades["bin2"] = (trades["gts"] // BIN2) * BIN2
tc_heat = trades[trades["product"].isin(ALL_PRODUCTS)].groupby(["product","bin2"]).size().unstack(fill_value=0)
tc_heat = tc_heat.loc[[p for p in ALL_PRODUCTS if p in tc_heat.index]]

fig, ax = plt.subplots(figsize=(20, 14), facecolor=DARK)
ax.set_facecolor(MID)
im = ax.imshow(tc_heat.values, aspect="auto", cmap="plasma",
               norm=plt.matplotlib.colors.LogNorm(vmin=0.5, vmax=tc_heat.values.max()))
ax.set_yticks(range(len(tc_heat.index)))
ax.set_yticklabels(tc_heat.index, fontsize=5.5, color="white")
n_xticks = min(10, len(tc_heat.columns))
tick_idx = np.linspace(0, len(tc_heat.columns)-1, n_xticks, dtype=int)
ax.set_xticks(tick_idx)
ax.set_xticklabels([f"{tc_heat.columns[i]/1e6:.1f}M" for i in tick_idx], color="white", fontsize=8)
ax.set_xlabel("Time (M ticks)", color="white")
ax.set_title("Trade Count Heatmap — Product × Time", color="white", fontsize=12)
cb = plt.colorbar(im, ax=ax, fraction=0.015)
cb.ax.tick_params(colors="white"); cb.set_label("Trade Count (log)", color="white")
plt.tight_layout()
plt.savefig(OUT + "trade_count_heatmap.png", dpi=120, bbox_inches="tight", facecolor=DARK)
plt.close()
print("   done")

print(f"\nAll outputs saved to: {OUT}")
print("Files:")
for f in sorted(os.listdir(OUT)):
    print(f"  {f}")
