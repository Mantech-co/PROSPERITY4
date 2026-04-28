"""
Round 5 — Full market analysis
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
from statsmodels.tsa.stattools import acf, adfuller
from scipy import stats
from itertools import combinations
import os

OUT = "/media/manukrishnan/Mk/prosperity_4/analysis_output"
os.makedirs(OUT, exist_ok=True)

sns.set_theme(style="darkgrid", palette="tab10")
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

GROUPS = {
    "Galaxy Sounds": ["GALAXY_SOUNDS_DARK_MATTER", "GALAXY_SOUNDS_BLACK_HOLES",
                      "GALAXY_SOUNDS_PLANETARY_RINGS", "GALAXY_SOUNDS_SOLAR_WINDS",
                      "GALAXY_SOUNDS_SOLAR_FLAMES"],
    "Sleep Pods":    ["SLEEP_POD_SUEDE", "SLEEP_POD_LAMB_WOOL", "SLEEP_POD_POLYESTER",
                      "SLEEP_POD_NYLON", "SLEEP_POD_COTTON"],
    "Microchips":    ["MICROCHIP_CIRCLE", "MICROCHIP_OVAL", "MICROCHIP_SQUARE",
                      "MICROCHIP_RECTANGLE", "MICROCHIP_TRIANGLE"],
    "Pebbles":       ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"],
    "Robots":        ["ROBOT_VACUUMING", "ROBOT_MOPPING", "ROBOT_DISHES",
                      "ROBOT_LAUNDRY", "ROBOT_IRONING"],
    "UV Visors":     ["UV_VISOR_YELLOW", "UV_VISOR_AMBER", "UV_VISOR_ORANGE",
                      "UV_VISOR_RED", "UV_VISOR_MAGENTA"],
    "Translators":   ["TRANSLATOR_SPACE_GRAY", "TRANSLATOR_ASTRO_BLACK",
                      "TRANSLATOR_ECLIPSE_CHARCOAL", "TRANSLATOR_GRAPHITE_MIST",
                      "TRANSLATOR_VOID_BLUE"],
    "Panels":        ["PANEL_1X2", "PANEL_2X2", "PANEL_1X4", "PANEL_2X4", "PANEL_4X4"],
    "Oxygen Shakes": ["OXYGEN_SHAKE_MORNING_BREATH", "OXYGEN_SHAKE_EVENING_BREATH",
                      "OXYGEN_SHAKE_MINT", "OXYGEN_SHAKE_CHOCOLATE", "OXYGEN_SHAKE_GARLIC"],
    "Snack Packs":   ["SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA", "SNACKPACK_PISTACHIO",
                      "SNACKPACK_STRAWBERRY", "SNACKPACK_RASPBERRY"],
}

PRODUCT_TO_GROUP = {p: g for g, ps in GROUPS.items() for p in ps}

# ─── Load data ──────────────────────────────────────────────────────────────

print("Loading prices ...")
prices = pd.concat([
    pd.read_csv(f"/media/manukrishnan/Mk/prosperity_4/data/prices_round_5_day_{d}.csv", sep=";")
    for d in [2, 3, 4]
], ignore_index=True)
prices["global_ts"] = (prices["day"] - 2) * 1_000_000 + prices["timestamp"]
prices.sort_values(["product", "global_ts"], inplace=True)
prices.reset_index(drop=True, inplace=True)

print("Loading trades ...")
trades = pd.concat([
    pd.read_csv(f"/media/manukrishnan/Mk/prosperity_4/data/trades_round_5_day_{d}.csv", sep=";")
    for d in [2, 3, 4]
], ignore_index=True)
trades["day"] = trades.groupby(trades.index // 1).ngroup()  # placeholder; re-derive below

# re-attach day from file
parts = []
for d in [2, 3, 4]:
    t = pd.read_csv(f"/media/manukrishnan/Mk/prosperity_4/data/trades_round_5_day_{d}.csv", sep=";")
    t["day"] = d
    parts.append(t)
trades = pd.concat(parts, ignore_index=True)
trades["global_ts"] = (trades["day"] - 2) * 1_000_000 + trades["timestamp"]
trades.sort_values(["symbol", "global_ts"], inplace=True)
trades.reset_index(drop=True, inplace=True)
trades.rename(columns={"symbol": "product"}, inplace=True)

ALL_PRODUCTS = sorted(prices["product"].unique())
print(f"Products: {len(ALL_PRODUCTS)}")

# ─── Pivot mid-price ────────────────────────────────────────────────────────

print("Building mid-price pivot ...")
pivot = prices.pivot_table(index="global_ts", columns="product", values="mid_price")
pivot.sort_index(inplace=True)
pivot.ffill(inplace=True)

# ─── EMA helpers ─────────────────────────────────────────────────────────────

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1: Mid-price time series — per group (10 subplots, one per group)
# ─────────────────────────────────────────────────────────────────────────────

print("Plot 1: Price time series per group ...")
fig, axes = plt.subplots(5, 2, figsize=(20, 25))
axes = axes.flatten()
for ax, (gname, gprods) in zip(axes, GROUPS.items()):
    gprods_avail = [p for p in gprods if p in pivot.columns]
    for p in gprods_avail:
        ax.plot(pivot.index / 1e6, pivot[p], label=p.split("_")[-1], lw=0.7, alpha=0.85)
    ax.set_title(gname, fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (M ticks)")
    ax.set_ylabel("Mid Price")
    ax.legend(fontsize=7, ncol=2)
fig.suptitle("Mid-Price Time Series by Product Group", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/01_price_timeseries_by_group.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2: Spread (ask1 - bid1) per group
# ─────────────────────────────────────────────────────────────────────────────

print("Plot 2: Bid-ask spread per group ...")
prices["spread"] = prices["ask_price_1"] - prices["bid_price_1"]
spread_mean = prices.groupby("product")["spread"].mean().rename("mean_spread")
spread_std  = prices.groupby("product")["spread"].std().rename("std_spread")
spread_df = pd.concat([spread_mean, spread_std], axis=1)
spread_df["group"] = spread_df.index.map(PRODUCT_TO_GROUP)

fig, ax = plt.subplots(figsize=(18, 7))
colors = plt.cm.tab10(np.linspace(0, 1, len(GROUPS)))
group_color = {g: c for g, c in zip(GROUPS, colors)}
bar_colors = [group_color[spread_df.loc[p, "group"]] for p in spread_df.index]
bars = ax.bar(spread_df.index, spread_df["mean_spread"], color=bar_colors, alpha=0.85,
              yerr=spread_df["std_spread"], capsize=3)
ax.set_xticks(range(len(spread_df)))
ax.set_xticklabels(spread_df.index, rotation=90, fontsize=7)
ax.set_ylabel("Mean Bid-Ask Spread")
ax.set_title("Mean Bid-Ask Spread ± Std per Product", fontweight="bold")
from matplotlib.patches import Patch
legend_els = [Patch(color=group_color[g], label=g) for g in GROUPS]
ax.legend(handles=legend_els, fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(f"{OUT}/02_spread_per_product.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3: Full 50x50 mid-price correlation heatmap
# ─────────────────────────────────────────────────────────────────────────────

print("Plot 3: Full correlation heatmap ...")
corr = pivot[ALL_PRODUCTS].corr()
fig, ax = plt.subplots(figsize=(22, 20))
norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
im = ax.imshow(corr.values, cmap="RdYlGn", norm=norm, aspect="auto")
ax.set_xticks(range(len(ALL_PRODUCTS)))
ax.set_yticks(range(len(ALL_PRODUCTS)))
ax.set_xticklabels(ALL_PRODUCTS, rotation=90, fontsize=5.5)
ax.set_yticklabels(ALL_PRODUCTS, fontsize=5.5)
plt.colorbar(im, ax=ax, fraction=0.03)
ax.set_title("50×50 Mid-Price Correlation Heatmap", fontweight="bold", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/03_full_correlation_heatmap.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4: Intra-group correlation heatmaps (10 groups, 2 rows × 5 cols)
# ─────────────────────────────────────────────────────────────────────────────

print("Plot 4: Intra-group correlation heatmaps ...")
fig, axes = plt.subplots(2, 5, figsize=(28, 10))
axes = axes.flatten()
for ax, (gname, gprods) in zip(axes, GROUPS.items()):
    gprods_avail = [p for p in gprods if p in pivot.columns]
    gc = pivot[gprods_avail].corr()
    short_labels = [p.replace(gname.upper().replace(" ", "_") + "_", "").replace("GALAXY_SOUNDS_", "")
                    .replace("SLEEP_POD_", "").replace("MICROCHIP_", "").replace("PEBBLES_", "")
                    .replace("ROBOT_", "").replace("UV_VISOR_", "").replace("TRANSLATOR_", "")
                    .replace("PANEL_", "").replace("OXYGEN_SHAKE_", "").replace("SNACKPACK_", "")
                    for p in gprods_avail]
    mask = np.zeros_like(gc, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    sns.heatmap(gc, ax=ax, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=-1, vmax=1, center=0,
                xticklabels=short_labels, yticklabels=short_labels,
                annot_kws={"size": 7}, linewidths=0.5, square=True, cbar=False)
    ax.set_title(gname, fontweight="bold", fontsize=10)
fig.suptitle("Intra-Group Correlation Heatmaps", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/04_intragroup_correlation_heatmaps.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5: Price RETURNS correlation heatmap (50×50)
# ─────────────────────────────────────────────────────────────────────────────

print("Plot 5: Returns correlation heatmap ...")
returns = pivot[ALL_PRODUCTS].pct_change().dropna()
ret_corr = returns.corr()
fig, ax = plt.subplots(figsize=(22, 20))
im = ax.imshow(ret_corr.values, cmap="RdYlGn", norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), aspect="auto")
ax.set_xticks(range(len(ALL_PRODUCTS)))
ax.set_yticks(range(len(ALL_PRODUCTS)))
ax.set_xticklabels(ALL_PRODUCTS, rotation=90, fontsize=5.5)
ax.set_yticklabels(ALL_PRODUCTS, fontsize=5.5)
plt.colorbar(im, ax=ax, fraction=0.03)
ax.set_title("50×50 Price RETURNS Correlation Heatmap", fontweight="bold", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/05_returns_correlation_heatmap.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 6: EMA correlation (short vs long EMA cross-correlation per group)
# ─────────────────────────────────────────────────────────────────────────────

print("Plot 6: EMA correlation heatmaps ...")
SHORT, LONG = 20, 200
ema_spread = {}
for p in ALL_PRODUCTS:
    if p in pivot.columns:
        s = pivot[p].dropna()
        ema_spread[p] = ema(s, SHORT) - ema(s, LONG)

ema_df = pd.DataFrame(ema_spread)
ema_corr = ema_df.corr()

fig, axes = plt.subplots(2, 5, figsize=(28, 10))
axes = axes.flatten()
for ax, (gname, gprods) in zip(axes, GROUPS.items()):
    gprods_avail = [p for p in gprods if p in ema_df.columns]
    gc = ema_df[gprods_avail].corr()
    short_labels = [p.split("_")[-1] for p in gprods_avail]
    sns.heatmap(gc, ax=ax, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, center=0,
                xticklabels=short_labels, yticklabels=short_labels,
                annot_kws={"size": 7}, linewidths=0.5, square=True, cbar=False)
    ax.set_title(f"{gname}\nEMA({SHORT}) - EMA({LONG}) corr", fontweight="bold", fontsize=9)
fig.suptitle("EMA Spread Correlation (EMA20 - EMA200) by Group", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/06_ema_correlation_heatmaps.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 7: Full EMA correlation 50×50
# ─────────────────────────────────────────────────────────────────────────────

print("Plot 7: Full EMA correlation 50x50 ...")
fig, ax = plt.subplots(figsize=(22, 20))
im = ax.imshow(ema_corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(ALL_PRODUCTS)))
ax.set_yticks(range(len(ALL_PRODUCTS)))
ax.set_xticklabels(ALL_PRODUCTS, rotation=90, fontsize=5.5)
ax.set_yticklabels(ALL_PRODUCTS, fontsize=5.5)
plt.colorbar(im, ax=ax, fraction=0.03)
ax.set_title("50×50 EMA-Spread Correlation Heatmap", fontweight="bold", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/07_ema_full_correlation_heatmap.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 8: ACF for one representative per group (lags 0-50)
# ─────────────────────────────────────────────────────────────────────────────

print("Plot 8: ACF of mid-price returns per group representative ...")
reps = {g: ps[0] for g, ps in GROUPS.items()}
fig, axes = plt.subplots(2, 5, figsize=(28, 10))
axes = axes.flatten()
NLAGS = 50
for ax, (gname, prod) in zip(axes, reps.items()):
    if prod not in pivot.columns:
        ax.set_visible(False)
        continue
    r = pivot[prod].pct_change().dropna()
    acf_vals, ci = acf(r, nlags=NLAGS, alpha=0.05, fft=True)
    lags = np.arange(NLAGS + 1)
    ax.bar(lags, acf_vals, color="steelblue", alpha=0.7, width=0.8)
    ax.fill_between(lags, ci[:, 0] - acf_vals, ci[:, 1] - acf_vals, alpha=0.3, color="orange")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"{gname}\n({prod.split('_')[-2]}_{prod.split('_')[-1]})", fontsize=8, fontweight="bold")
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
fig.suptitle("ACF of Mid-Price Returns (per group representative)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/08_acf_returns_per_group.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 9: ADF test — stationarity summary
# ─────────────────────────────────────────────────────────────────────────────

print("Plot 9: ADF stationarity test ...")
adf_results = []
for p in ALL_PRODUCTS:
    if p not in pivot.columns:
        continue
    s = pivot[p].dropna()
    try:
        adf_stat, pval, _, _, _, _ = adfuller(s, maxlag=20, autolag="AIC")
        adf_results.append({"product": p, "adf_stat": adf_stat, "pval": pval,
                             "stationary": pval < 0.05, "group": PRODUCT_TO_GROUP.get(p, "")})
    except Exception:
        pass

adf_df = pd.DataFrame(adf_results).set_index("product")

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
ax = axes[0]
colors_adf = ["green" if s else "red" for s in adf_df["stationary"]]
ax.barh(adf_df.index, -adf_df["adf_stat"], color=colors_adf, alpha=0.8)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("−ADF Statistic (larger = more stationary)")
ax.set_title("ADF Test Statistic per Product\n(green = stationary at 5%)", fontweight="bold")
ax.tick_params(axis="y", labelsize=6)

ax2 = axes[1]
ax2.barh(adf_df.index, adf_df["pval"], color=colors_adf, alpha=0.8)
ax2.axvline(0.05, color="orange", lw=1.5, linestyle="--", label="p=0.05")
ax2.set_xlabel("ADF p-value")
ax2.set_title("ADF p-value per Product\n(green = reject unit root)", fontweight="bold")
ax2.tick_params(axis="y", labelsize=6)
ax2.legend()

fig.tight_layout()
fig.savefig(f"{OUT}/09_adf_stationarity.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 10: Mean Reversion speed — half-life via OU estimation
# ─────────────────────────────────────────────────────────────────────────────

print("Plot 10: Mean reversion half-life ...")
def ou_halflife(series):
    s = series.dropna()
    y = s.diff().dropna()
    x = s.shift(1).dropna()
    x = x.loc[y.index]
    slope, intercept, r, p, se = stats.linregress(x, y)
    if slope >= 0:
        return np.nan
    return -np.log(2) / slope

hl_data = []
for p in ALL_PRODUCTS:
    if p not in pivot.columns:
        continue
    hl = ou_halflife(pivot[p])
    hl_data.append({"product": p, "half_life": hl, "group": PRODUCT_TO_GROUP.get(p, "")})

hl_df = pd.DataFrame(hl_data).set_index("product").dropna()
hl_df.sort_values("half_life", inplace=True)

fig, ax = plt.subplots(figsize=(18, 8))
bar_colors_hl = [group_color[hl_df.loc[p, "group"]] for p in hl_df.index]
ax.bar(hl_df.index, hl_df["half_life"], color=bar_colors_hl, alpha=0.85)
ax.set_xticks(range(len(hl_df)))
ax.set_xticklabels(hl_df.index, rotation=90, fontsize=7)
ax.set_ylabel("Half-Life (ticks)")
ax.set_title("Mean Reversion Half-Life (OU Estimate) per Product", fontweight="bold")
legend_els = [Patch(color=group_color[g], label=g) for g in GROUPS if g in hl_df["group"].values]
ax.legend(handles=legend_els, fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(f"{OUT}/10_mean_reversion_halflife.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 11: Price deviation from rolling mean (Z-score) — per group
# ─────────────────────────────────────────────────────────────────────────────

print("Plot 11: Rolling Z-score per group ...")
WIN = 500
fig, axes = plt.subplots(5, 2, figsize=(20, 25))
axes = axes.flatten()
for ax, (gname, gprods) in zip(axes, GROUPS.items()):
    gprods_avail = [p for p in gprods if p in pivot.columns]
    for p in gprods_avail:
        s = pivot[p]
        roll_mean = s.rolling(WIN).mean()
        roll_std  = s.rolling(WIN).std()
        z = (s - roll_mean) / roll_std
        ax.plot(pivot.index / 1e6, z, label=p.split("_")[-1], lw=0.6, alpha=0.8)
    ax.axhline(2, color="red", lw=0.8, linestyle="--", alpha=0.6)
    ax.axhline(-2, color="red", lw=0.8, linestyle="--", alpha=0.6)
    ax.axhline(0, color="white", lw=0.5)
    ax.set_title(gname, fontsize=11, fontweight="bold")
    ax.set_ylabel("Z-Score")
    ax.set_xlabel("Time (M ticks)")
    ax.legend(fontsize=7, ncol=2)
fig.suptitle(f"Rolling Z-Score (window={WIN}) of Mid-Price", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/11_rolling_zscore_per_group.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 12: EMA ribbons (5 spans) per group
# ─────────────────────────────────────────────────────────────────────────────

print("Plot 12: EMA ribbons ...")
EMA_SPANS = [10, 50, 100, 500, 1000]
fig, axes = plt.subplots(5, 2, figsize=(20, 28))
axes = axes.flatten()
for ax, (gname, gprods) in zip(axes, GROUPS.items()):
    rep = next((p for p in gprods if p in pivot.columns), None)
    if rep is None:
        ax.set_visible(False)
        continue
    s = pivot[rep]
    ax.plot(pivot.index / 1e6, s, color="white", lw=0.4, alpha=0.5, label="Price")
    cmap = plt.cm.plasma(np.linspace(0.2, 0.95, len(EMA_SPANS)))
    for span, c in zip(EMA_SPANS, cmap):
        e = ema(s, span)
        ax.plot(pivot.index / 1e6, e, color=c, lw=0.9, label=f"EMA{span}")
    ax.set_title(f"{gname} — {rep.split('_')[-1]}", fontsize=10, fontweight="bold")
    ax.set_ylabel("Price")
    ax.set_xlabel("Time (M ticks)")
    ax.legend(fontsize=7, ncol=3)
fig.suptitle("EMA Ribbon (10/50/100/500/1000) — Group Representatives", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/12_ema_ribbons_per_group.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 13: Volatility (rolling std of returns) per group
# ─────────────────────────────────────────────────────────────────────────────

print("Plot 13: Volatility per group ...")
VOL_WIN = 200
fig, axes = plt.subplots(5, 2, figsize=(20, 25))
axes = axes.flatten()
for ax, (gname, gprods) in zip(axes, GROUPS.items()):
    gprods_avail = [p for p in gprods if p in pivot.columns]
    for p in gprods_avail:
        r = pivot[p].pct_change()
        vol = r.rolling(VOL_WIN).std()
        ax.plot(pivot.index / 1e6, vol, label=p.split("_")[-1], lw=0.7, alpha=0.85)
    ax.set_title(gname, fontsize=11, fontweight="bold")
    ax.set_ylabel(f"Rolling Std of Returns (w={VOL_WIN})")
    ax.set_xlabel("Time (M ticks)")
    ax.legend(fontsize=7, ncol=2)
fig.suptitle("Rolling Volatility of Mid-Price Returns", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/13_volatility_per_group.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 14: Intra-group spread (product - group mean) — potential pairs
# ─────────────────────────────────────────────────────────────────────────────

print("Plot 14: Intra-group spread ...")
fig, axes = plt.subplots(5, 2, figsize=(20, 25))
axes = axes.flatten()
for ax, (gname, gprods) in zip(axes, GROUPS.items()):
    gprods_avail = [p for p in gprods if p in pivot.columns]
    group_mean = pivot[gprods_avail].mean(axis=1)
    for p in gprods_avail:
        diff = pivot[p] - group_mean
        ax.plot(pivot.index / 1e6, diff, label=p.split("_")[-1], lw=0.6, alpha=0.85)
    ax.axhline(0, color="white", lw=0.5)
    ax.set_title(gname, fontsize=11, fontweight="bold")
    ax.set_ylabel("Price - Group Mean")
    ax.set_xlabel("Time (M ticks)")
    ax.legend(fontsize=7, ncol=2)
fig.suptitle("Intra-Group Price Deviation from Group Mean", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/14_intragroup_spread.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 15: Pairwise scatter — all combos within each group (color = group)
# ─────────────────────────────────────────────────────────────────────────────

print("Plot 15: Pairwise scatter within groups ...")
fig, axes = plt.subplots(5, 2, figsize=(20, 25))
axes = axes.flatten()
for ax, (gname, gprods) in zip(axes, GROUPS.items()):
    gprods_avail = [p for p in gprods if p in pivot.columns]
    if len(gprods_avail) < 2:
        ax.set_visible(False)
        continue
    p1, p2 = gprods_avail[0], gprods_avail[1]
    x = pivot[p1].dropna()
    y = pivot[p2].reindex(x.index).dropna()
    x = x.loc[y.index]
    # color by day segment
    seg = (x.index // 1_000_000).astype(int)
    scatter = ax.scatter(x, y, c=seg, cmap="viridis", s=0.3, alpha=0.6)
    ax.set_xlabel(p1.split("_")[-1])
    ax.set_ylabel(p2.split("_")[-1])
    ax.set_title(f"{gname}\n{p1.split('_')[-1]} vs {p2.split('_')[-1]}", fontsize=9, fontweight="bold")
    plt.colorbar(scatter, ax=ax, label="Day segment")
fig.suptitle("Pairwise Price Scatter (first two products per group, colour=day)", fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/15_pairwise_scatter_per_group.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 16: Returns distribution per group — KDE
# ─────────────────────────────────────────────────────────────────────────────

print("Plot 16: Returns KDE ...")
fig, axes = plt.subplots(5, 2, figsize=(20, 25))
axes = axes.flatten()
for ax, (gname, gprods) in zip(axes, GROUPS.items()):
    gprods_avail = [p for p in gprods if p in pivot.columns]
    for p in gprods_avail:
        r = pivot[p].pct_change().dropna()
        r.plot.kde(ax=ax, label=p.split("_")[-1], lw=1.2)
    ax.set_xlim(-0.005, 0.005)
    ax.set_title(gname, fontsize=11, fontweight="bold")
    ax.set_xlabel("Return")
    ax.legend(fontsize=7, ncol=2)
fig.suptitle("Return Distribution (KDE) per Group", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/16_returns_kde_per_group.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# TRADES ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

print("\nTrades analysis ...")
trades["group"] = trades["product"].map(PRODUCT_TO_GROUP)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 17: Trade volume over time per group
# ─────────────────────────────────────────────────────────────────────────────

print("Plot 17: Trade volume over time ...")
fig, axes = plt.subplots(5, 2, figsize=(20, 25))
axes = axes.flatten()
for ax, (gname, gprods) in zip(axes, GROUPS.items()):
    gdata = trades[trades["product"].isin(gprods)]
    for p in gprods:
        pdata = gdata[gdata["product"] == p].copy()
        if pdata.empty:
            continue
        pdata = pdata.set_index("global_ts")["quantity"].groupby(lambda x: (x // 10000) * 10000).sum()
        ax.plot(pdata.index / 1e6, pdata.values, label=p.split("_")[-1], lw=0.7, alpha=0.85)
    ax.set_title(gname, fontsize=11, fontweight="bold")
    ax.set_ylabel("Volume (10k-tick bins)")
    ax.set_xlabel("Time (M ticks)")
    ax.legend(fontsize=7, ncol=2)
fig.suptitle("Trade Volume Over Time (10k-tick bins)", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/17_trade_volume_over_time.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 18: Total volume & trade count per product
# ─────────────────────────────────────────────────────────────────────────────

print("Plot 18: Total volume and trade count ...")
vol_summary = trades.groupby("product").agg(
    total_volume=("quantity", "sum"),
    trade_count=("quantity", "count"),
    avg_size=("quantity", "mean"),
).reset_index()
vol_summary["group"] = vol_summary["product"].map(PRODUCT_TO_GROUP)
vol_summary.sort_values("total_volume", ascending=True, inplace=True)

fig, axes = plt.subplots(1, 3, figsize=(24, 9))
vc = [group_color[vol_summary.loc[i, "group"]] for i in vol_summary.index]

axes[0].barh(vol_summary["product"], vol_summary["total_volume"], color=vc, alpha=0.85)
axes[0].set_xlabel("Total Volume")
axes[0].set_title("Total Trade Volume", fontweight="bold")
axes[0].tick_params(axis="y", labelsize=6)

axes[1].barh(vol_summary["product"], vol_summary["trade_count"], color=vc, alpha=0.85)
axes[1].set_xlabel("Trade Count")
axes[1].set_title("Number of Trades", fontweight="bold")
axes[1].tick_params(axis="y", labelsize=6)

axes[2].barh(vol_summary["product"], vol_summary["avg_size"], color=vc, alpha=0.85)
axes[2].set_xlabel("Avg Trade Size")
axes[2].set_title("Average Trade Size", fontweight="bold")
axes[2].tick_params(axis="y", labelsize=6)

legend_els = [Patch(color=group_color[g], label=g) for g in GROUPS]
axes[0].legend(handles=legend_els, fontsize=7, ncol=1, loc="lower right")

fig.tight_layout()
fig.savefig(f"{OUT}/18_volume_summary.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 19: Inter-arrival time (IAT) per product
# ─────────────────────────────────────────────────────────────────────────────

print("Plot 19: Inter-arrival times ...")
iat_stats = []
for p in ALL_PRODUCTS:
    pdata = trades[trades["product"] == p].sort_values("global_ts")
    if len(pdata) < 2:
        continue
    iat = pdata["global_ts"].diff().dropna()
    iat_stats.append({
        "product": p,
        "mean_iat": iat.mean(),
        "median_iat": iat.median(),
        "std_iat": iat.std(),
        "min_iat": iat.min(),
        "group": PRODUCT_TO_GROUP.get(p, ""),
    })
iat_df = pd.DataFrame(iat_stats).set_index("product")
iat_df.sort_values("mean_iat", inplace=True)

fig, axes = plt.subplots(1, 2, figsize=(20, 9))
ic = [group_color[iat_df.loc[p, "group"]] for p in iat_df.index]
axes[0].barh(iat_df.index, iat_df["mean_iat"], xerr=iat_df["std_iat"],
             color=ic, alpha=0.85, capsize=3)
axes[0].set_xlabel("Mean IAT (ticks)")
axes[0].set_title("Mean Inter-Arrival Time ± Std per Product", fontweight="bold")
axes[0].tick_params(axis="y", labelsize=6)

axes[1].barh(iat_df.index, iat_df["median_iat"], color=ic, alpha=0.85)
axes[1].set_xlabel("Median IAT (ticks)")
axes[1].set_title("Median Inter-Arrival Time per Product", fontweight="bold")
axes[1].tick_params(axis="y", labelsize=6)

legend_els = [Patch(color=group_color[g], label=g) for g in GROUPS]
axes[0].legend(handles=legend_els, fontsize=7, ncol=1, loc="lower right")

fig.tight_layout()
fig.savefig(f"{OUT}/19_interarrival_time.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 20: IAT distribution per group — boxplot
# ─────────────────────────────────────────────────────────────────────────────

print("Plot 20: IAT boxplots per group ...")
iat_records = []
for p in ALL_PRODUCTS:
    pdata = trades[trades["product"] == p].sort_values("global_ts")
    if len(pdata) < 2:
        continue
    iat = pdata["global_ts"].diff().dropna().values
    g = PRODUCT_TO_GROUP.get(p, "")
    for v in iat:
        iat_records.append({"product": p, "iat": v, "group": g, "short": p.split("_")[-1]})

iat_long = pd.DataFrame(iat_records)

fig, axes = plt.subplots(2, 5, figsize=(28, 12))
axes = axes.flatten()
for ax, (gname, gprods) in zip(axes, GROUPS.items()):
    gdata = iat_long[iat_long["group"] == gname]
    if gdata.empty:
        ax.set_visible(False)
        continue
    gdata.boxplot(column="iat", by="short", ax=ax, showfliers=False,
                  patch_artist=True,
                  boxprops=dict(facecolor=group_color[gname], alpha=0.7))
    ax.set_title(gname, fontweight="bold", fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("IAT (ticks)")
    plt.setp(ax.get_xticklabels(), rotation=30, fontsize=7)
    ax.title.set_position([0.5, 1.0])
fig.suptitle("Inter-Arrival Time Distributions per Group (no outliers)", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT}/20_iat_boxplot_per_group.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 21: Trade price vs mid-price — trade impact
# ─────────────────────────────────────────────────────────────────────────────

print("Plot 21: Trade price vs mid-price ...")
fig, axes = plt.subplots(5, 2, figsize=(20, 25))
axes = axes.flatten()
for ax, (gname, gprods) in zip(axes, GROUPS.items()):
    rep = next((p for p in gprods if p in pivot.columns and not trades[trades["product"] == p].empty), None)
    if rep is None:
        ax.set_visible(False)
        continue
    piv_s = pivot[rep].reset_index()
    piv_s.columns = ["global_ts", "mid_price"]
    tr = trades[trades["product"] == rep][["global_ts", "price", "quantity"]].copy()
    ax.plot(piv_s["global_ts"] / 1e6, piv_s["mid_price"], lw=0.5, color="cyan", alpha=0.6, label="Mid")
    sc = ax.scatter(tr["global_ts"] / 1e6, tr["price"],
                    s=tr["quantity"] * 0.5 + 1, alpha=0.5, c="orange", zorder=5, label="Trades")
    ax.set_title(f"{gname} — {rep.split('_')[-1]}", fontsize=9, fontweight="bold")
    ax.set_ylabel("Price")
    ax.set_xlabel("Time (M ticks)")
    ax.legend(fontsize=7)
fig.suptitle("Trade Price vs Mid-Price (dot size = quantity)", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/21_trade_price_vs_midprice.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 22: Volume-weighted average price (VWAP) vs mid
# ─────────────────────────────────────────────────────────────────────────────

print("Plot 22: VWAP vs mid-price ...")
BIN = 10_000
fig, axes = plt.subplots(5, 2, figsize=(20, 25))
axes = axes.flatten()
for ax, (gname, gprods) in zip(axes, GROUPS.items()):
    rep = next((p for p in gprods if p in pivot.columns and not trades[trades["product"] == p].empty), None)
    if rep is None:
        ax.set_visible(False)
        continue
    tr = trades[trades["product"] == rep].copy()
    tr["bin"] = (tr["global_ts"] // BIN) * BIN
    vwap = tr.groupby("bin").apply(lambda d: (d["price"] * d["quantity"]).sum() / d["quantity"].sum())
    mid_bin = pivot[rep].resample(BIN).mean() if False else pivot[rep].groupby((pivot.index // BIN) * BIN).mean()
    mid_bin = mid_bin.reindex(vwap.index)
    ax.plot(vwap.index / 1e6, vwap.values, label="VWAP", lw=0.8, color="orange")
    ax.plot(mid_bin.index / 1e6, mid_bin.values, label="Mid", lw=0.8, color="cyan", alpha=0.7)
    ax.set_title(f"{gname} — {rep.split('_')[-1]}", fontsize=9, fontweight="bold")
    ax.set_ylabel("Price")
    ax.set_xlabel("Time (M ticks)")
    ax.legend(fontsize=7)
fig.suptitle("VWAP vs Mid-Price (10k-tick bins)", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/22_vwap_vs_mid.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 23: Buyer/Seller activity heatmap (who trades what)
# ─────────────────────────────────────────────────────────────────────────────

print("Plot 23: Buyer/Seller heatmap ...")
buyer_vol = trades[trades["buyer"].notna() & (trades["buyer"] != "")].groupby(["buyer", "product"])["quantity"].sum().unstack(fill_value=0)
seller_vol = trades[trades["seller"].notna() & (trades["seller"] != "")].groupby(["seller", "product"])["quantity"].sum().unstack(fill_value=0)

fig, axes = plt.subplots(1, 2, figsize=(26, max(6, len(buyer_vol) * 0.35 + 2)))
if not buyer_vol.empty:
    sns.heatmap(buyer_vol, ax=axes[0], cmap="Blues", cbar=True,
                linewidths=0.2, xticklabels=True, yticklabels=True)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90, fontsize=5)
    axes[0].set_yticklabels(axes[0].get_yticklabels(), fontsize=7)
    axes[0].set_title("Buyer Volume by Product", fontweight="bold")
else:
    axes[0].text(0.5, 0.5, "No buyer data", ha="center", va="center")

if not seller_vol.empty:
    sns.heatmap(seller_vol, ax=axes[1], cmap="Reds", cbar=True,
                linewidths=0.2, xticklabels=True, yticklabels=True)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=90, fontsize=5)
    axes[1].set_yticklabels(axes[1].get_yticklabels(), fontsize=7)
    axes[1].set_title("Seller Volume by Product", fontweight="bold")
else:
    axes[1].text(0.5, 0.5, "No seller data", ha="center", va="center")

fig.suptitle("Buyer / Seller Volume Heatmap", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/23_buyer_seller_heatmap.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 24: Trade size distribution — violin plots per group
# ─────────────────────────────────────────────────────────────────────────────

print("Plot 24: Trade size violin per group ...")
fig, axes = plt.subplots(2, 5, figsize=(28, 12))
axes = axes.flatten()
for ax, (gname, gprods) in zip(axes, GROUPS.items()):
    gdata = trades[trades["product"].isin(gprods)].copy()
    if gdata.empty:
        ax.set_visible(False)
        continue
    gdata["short"] = gdata["product"].str.split("_").str[-1]
    order = sorted(gdata["short"].unique())
    sns.violinplot(data=gdata, x="short", y="quantity", ax=ax, order=order,
                   palette="muted", inner="quartile", cut=0)
    ax.set_title(gname, fontweight="bold", fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("Trade Quantity")
    ax.tick_params(axis="x", rotation=30, labelsize=7)
fig.suptitle("Trade Size Distribution (Violin) per Group", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/24_trade_size_violin.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 25: Cumulative volume by day per group
# ─────────────────────────────────────────────────────────────────────────────

print("Plot 25: Cumulative volume by day ...")
fig, axes = plt.subplots(5, 2, figsize=(20, 25))
axes = axes.flatten()
for ax, (gname, gprods) in zip(axes, GROUPS.items()):
    for p in gprods:
        pdata = trades[trades["product"] == p].sort_values("global_ts")
        if pdata.empty:
            continue
        cum = pdata["quantity"].cumsum()
        ax.plot(pdata["global_ts"] / 1e6, cum, label=p.split("_")[-1], lw=0.9)
    ax.set_title(gname, fontsize=11, fontweight="bold")
    ax.set_ylabel("Cumulative Volume")
    ax.set_xlabel("Time (M ticks)")
    ax.legend(fontsize=7, ncol=2)
fig.suptitle("Cumulative Trade Volume by Product", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/25_cumulative_volume.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 26: Price impact — trade size vs price change
# ─────────────────────────────────────────────────────────────────────────────

print("Plot 26: Price impact ...")
fig, axes = plt.subplots(5, 2, figsize=(20, 25))
axes = axes.flatten()
for ax, (gname, gprods) in zip(axes, GROUPS.items()):
    rep = next((p for p in gprods if p in pivot.columns and not trades[trades["product"] == p].empty), None)
    if rep is None:
        ax.set_visible(False)
        continue
    tr = trades[trades["product"] == rep].sort_values("global_ts").copy()
    tr["price_change"] = tr["price"].diff().abs()
    tr = tr.dropna(subset=["price_change"])
    ax.scatter(tr["quantity"], tr["price_change"], alpha=0.3, s=8, color=group_color[gname])
    # fit line
    if len(tr) > 5:
        slope, intercept, r, p_val, _ = stats.linregress(tr["quantity"], tr["price_change"])
        xx = np.linspace(tr["quantity"].min(), tr["quantity"].max(), 100)
        ax.plot(xx, slope * xx + intercept, "r--", lw=1.2, label=f"r={r:.2f}")
        ax.legend(fontsize=8)
    ax.set_title(f"{gname} — {rep.split('_')[-1]}", fontsize=9, fontweight="bold")
    ax.set_xlabel("Trade Size")
    ax.set_ylabel("|Price Change|")
fig.suptitle("Price Impact: Trade Size vs |Price Change|", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/26_price_impact.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 27: Cross-group EMA correlation heatmap (group-averaged EMA spread)
# ─────────────────────────────────────────────────────────────────────────────

print("Plot 27: Group-level EMA correlation ...")
group_ema = {}
for gname, gprods in GROUPS.items():
    avail = [p for p in gprods if p in ema_df.columns]
    if avail:
        group_ema[gname] = ema_df[avail].mean(axis=1)

group_ema_df = pd.DataFrame(group_ema)
group_ema_corr = group_ema_df.corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(group_ema_corr, ax=ax, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, center=0, linewidths=0.5, square=True,
            annot_kws={"size": 10})
ax.set_title("Group-Level EMA Spread Correlation", fontweight="bold", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/27_group_ema_correlation.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 28: ADT (average daily trading) heatmap
# ─────────────────────────────────────────────────────────────────────────────

print("Plot 28: ADT heatmap ...")
adt = trades.groupby(["day", "product"])["quantity"].sum().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(20, 4))
sns.heatmap(adt, ax=ax, cmap="YlOrRd", linewidths=0.3,
            xticklabels=True, yticklabels=True, cbar_kws={"label": "Volume"})
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=6)
ax.set_title("Average Daily Trading (ADT) — Volume per Day per Product", fontweight="bold", fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT}/28_adt_heatmap.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 29: Price level heatmap across time (product × time bin)
# ─────────────────────────────────────────────────────────────────────────────

print("Plot 29: Price level heatmap ...")
NBIN = 100
price_bins = pd.cut(pivot.index, bins=NBIN, labels=False)
price_binned = pivot.groupby(price_bins).mean()

# normalize each product
normed = (price_binned - price_binned.mean()) / price_binned.std()

fig, ax = plt.subplots(figsize=(22, 14))
im = ax.imshow(normed.T.values, aspect="auto", cmap="RdYlGn",
               vmin=-3, vmax=3, interpolation="nearest")
ax.set_yticks(range(len(ALL_PRODUCTS)))
ax.set_yticklabels(ALL_PRODUCTS, fontsize=5.5)
ax.set_xlabel("Time Bin (0-100)")
ax.set_title("Normalized Mid-Price Heatmap (product × time)", fontweight="bold", fontsize=13)
plt.colorbar(im, ax=ax, fraction=0.02, label="Z-Score")
fig.tight_layout()
fig.savefig(f"{OUT}/29_price_level_heatmap.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 30: Top correlated pairs (returns)
# ─────────────────────────────────────────────────────────────────────────────

print("Plot 30: Top correlated pairs ...")
pairs = []
for (p1, p2) in combinations(ALL_PRODUCTS, 2):
    if p1 in ret_corr.columns and p2 in ret_corr.columns:
        pairs.append((p1, p2, ret_corr.loc[p1, p2]))
pairs_df = pd.DataFrame(pairs, columns=["p1", "p2", "corr"]).sort_values("corr", ascending=False)
top_pos = pairs_df.head(20)
top_neg = pairs_df.tail(20)

fig, axes = plt.subplots(1, 2, figsize=(20, 9))
for ax, df, title, color in [
    (axes[0], top_pos, "Top 20 Positively Correlated Pairs (Returns)", "steelblue"),
    (axes[1], top_neg.sort_values("corr"), "Top 20 Negatively Correlated Pairs (Returns)", "salmon"),
]:
    labels = [f"{r.p1.split('_')[0][:4]}..{r.p1.split('_')[-1][:4]} | {r.p2.split('_')[0][:4]}..{r.p2.split('_')[-1][:4]}"
              for _, r in df.iterrows()]
    ax.barh(labels, df["corr"].values, color=color, alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Return Correlation")
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.tick_params(axis="y", labelsize=7)
fig.tight_layout()
fig.savefig(f"{OUT}/30_top_correlated_pairs.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 31: Spread stationarity — price diff within pairs
# ─────────────────────────────────────────────────────────────────────────────

print("Plot 31: Intra-group pair spreads (potential pairs trading) ...")
fig, axes = plt.subplots(5, 2, figsize=(20, 25))
axes = axes.flatten()
for ax, (gname, gprods) in zip(axes, GROUPS.items()):
    avail = [p for p in gprods if p in pivot.columns]
    if len(avail) < 2:
        ax.set_visible(False)
        continue
    colors_pair = plt.cm.tab10(np.linspace(0, 1, len(list(combinations(avail, 2)))))
    for (p1, p2), c in zip(combinations(avail, 2), colors_pair):
        spread = pivot[p1] - pivot[p2]
        ax.plot(pivot.index / 1e6, spread, lw=0.5, alpha=0.7, color=c,
                label=f"{p1.split('_')[-1]}-{p2.split('_')[-1]}")
    ax.axhline(0, color="white", lw=0.5)
    ax.set_title(gname, fontsize=11, fontweight="bold")
    ax.set_ylabel("Price Spread")
    ax.set_xlabel("Time (M ticks)")
    ax.legend(fontsize=6, ncol=2)
fig.suptitle("All Intra-Group Pair Spreads (potential pairs trading)", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/31_intragroup_pair_spreads.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# TEXT SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print("\nGenerating summary stats ...")

summary_lines = []
summary_lines.append("=" * 80)
summary_lines.append("ROUND 5 MARKET ANALYSIS SUMMARY")
summary_lines.append("=" * 80)
summary_lines.append("")

summary_lines.append("── ADF STATIONARITY ──")
stat_prods = adf_df[adf_df["stationary"]].index.tolist()
nonstat_prods = adf_df[~adf_df["stationary"]].index.tolist()
summary_lines.append(f"Stationary (p<0.05):     {len(stat_prods)} products")
summary_lines.append(f"Non-stationary:          {len(nonstat_prods)} products")
summary_lines.append(f"Stationary: {stat_prods}")
summary_lines.append(f"Non-stationary: {nonstat_prods}")
summary_lines.append("")

summary_lines.append("── MEAN REVERSION HALF-LIVES (top 10 fastest) ──")
for prod, row in hl_df.head(10).iterrows():
    summary_lines.append(f"  {prod:<45} {row['half_life']:.1f} ticks")
summary_lines.append("")

summary_lines.append("── MEAN REVERSION HALF-LIVES (top 10 slowest) ──")
for prod, row in hl_df.tail(10).iterrows():
    summary_lines.append(f"  {prod:<45} {row['half_life']:.1f} ticks")
summary_lines.append("")

summary_lines.append("── SPREADS ──")
summary_lines.append(spread_df[["mean_spread", "std_spread"]].sort_values("mean_spread").to_string())
summary_lines.append("")

summary_lines.append("── VOLUME SUMMARY ──")
vs = vol_summary.sort_values("total_volume", ascending=False)
summary_lines.append(vs[["product", "total_volume", "trade_count", "avg_size"]].to_string(index=False))
summary_lines.append("")

summary_lines.append("── INTER-ARRIVAL TIME (sorted by mean IAT) ──")
summary_lines.append(iat_df[["mean_iat", "median_iat", "std_iat"]].to_string())
summary_lines.append("")

summary_lines.append("── TOP 20 CORRELATED PAIRS (Returns) ──")
for _, row in top_pos.iterrows():
    summary_lines.append(f"  {str(row.p1):<40} {str(row.p2):<40} r={float(row['corr']):.4f}")
summary_lines.append("")

summary_lines.append("── TOP 20 ANTI-CORRELATED PAIRS (Returns) ──")
for _, row in top_neg.iterrows():
    summary_lines.append(f"  {str(row.p1):<40} {str(row.p2):<40} r={float(row['corr']):.4f}")
summary_lines.append("")

summary_lines.append("── INTRA-GROUP RETURN CORRELATIONS (mean off-diagonal) ──")
for gname, gprods in GROUPS.items():
    avail = [p for p in gprods if p in ret_corr.columns]
    if len(avail) < 2:
        continue
    gc_arr = ret_corr.loc[avail, avail].values.copy()
    np.fill_diagonal(gc_arr, np.nan)
    mean_corr = np.nanmean(gc_arr)
    summary_lines.append(f"  {gname:<20} mean intra-corr = {mean_corr:.4f}")
summary_lines.append("")

with open(f"{OUT}/SUMMARY.txt", "w") as f:
    f.write("\n".join(summary_lines))

print(f"\nAll done. Plots saved to {OUT}/")
print("Files:")
for fn in sorted(os.listdir(OUT)):
    print(f"  {fn}")
