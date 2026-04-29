"""
Predictive power of order-book & trade features on future returns.

Features built:
  - OBI_L1       : (bid_vol1 - ask_vol1) / (bid_vol1 + ask_vol1)
  - OBI_L3       : weighted 3-level imbalance
  - SPREAD       : ask1 - bid1
  - SPREAD_PCT   : spread / mid
  - DEPTH_BID    : total bid volume (3 levels)
  - DEPTH_ASK    : total ask volume
  - DEPTH_IMBAL  : (depth_bid - depth_ask) / (depth_bid + depth_ask)
  - MID_RETURN   : current 1-step return (momentum/reversal)
  - TRADE_FREQ   : trades per window
  - TRADE_VOL    : total trade volume per window
  - AVG_TRADE_SZ : avg trade size per window
  - BUYER_IMBAL  : (buyer_vol - seller_vol) / total_vol  (trade direction)

Target: FORWARD_RET  (mid price return over next N ticks)

Plots:
  1. Feature correlation with forward return — bar per product (heatmap)
  2. Conditional mean return by OBI quintile — all groups
  3. Conditional mean return by spread quintile
  4. Scatter: OBI vs forward return (top-4 products by |corr|)
  5. Trade frequency vs next-period return scatter
  6. Buyer imbalance vs forward return
  7. Rolling predictive correlation over time (OBI)
  8. Binned return by depth imbalance
  9. Feature correlation matrix (features × features)
 10. IC (information coefficient) over days: OBI vs fwd return
"""

import os, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy import stats

OUT = "/media/manukrishnan/Mk/prosperity_4/analysis_output/deep/"
os.makedirs(OUT, exist_ok=True)

DARK = "#1a1a2e"; MID = "#16213e"; ACCENT = "#f39c12"; RED = "#e74c3c"; GREEN = "#2ecc71"
plt.rcParams.update({"figure.facecolor": DARK, "axes.facecolor": MID,
                     "text.color": "white", "axes.labelcolor": "#aaa",
                     "xtick.color": "#aaa", "ytick.color": "#aaa",
                     "axes.edgecolor": "#444", "grid.color": "#333", "font.size": 8})

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
GCOLORS = dict(zip(GROUPS, plt.cm.tab10(np.linspace(0, 1, 10))))
FORWARD_STEPS = 1   # predict 1-step ahead mid return

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading...")
pframes, tframes = [], []
for d in [2, 3, 4]:
    pf = pd.read_csv(f"/media/manukrishnan/Mk/prosperity_4/data/prices_round_5_day_{d}.csv", sep=";")
    pf["day"] = d
    pframes.append(pf)
    tf = pd.read_csv(f"/media/manukrishnan/Mk/prosperity_4/data/trades_round_5_day_{d}.csv", sep=";")
    tf["day"] = d
    tframes.append(tf)

prices = pd.concat(pframes, ignore_index=True)
prices["gts"] = (prices["day"] - 2) * 1_000_000 + prices["timestamp"]
prices.sort_values(["product", "gts"], inplace=True)

trades = pd.concat(tframes, ignore_index=True)
trades["gts"] = (trades["day"] - 2) * 1_000_000 + trades["timestamp"]
trades.rename(columns={"symbol": "product"}, inplace=True)

ALL_PRODUCTS = sorted(p for p in prices["product"].unique()
                      if not p.startswith("VEV_") and p in P2G)

# ── Build feature frames per product ─────────────────────────────────────────
print("Building features...")
FEAT_NAMES = ["OBI_L1","OBI_L3","SPREAD","SPREAD_PCT","DEPTH_BID","DEPTH_ASK",
              "DEPTH_IMBAL","MID_RETURN","TRADE_FREQ","TRADE_VOL",
              "AVG_TRADE_SZ","BUYER_IMBAL"]

TRADE_WIN = 100   # timestamp window to aggregate trades per price row

all_feat = {}   # product -> DataFrame with features + FORWARD_RET

for prod in ALL_PRODUCTS:
    pb = prices[prices["product"] == prod].copy().reset_index(drop=True)
    if len(pb) < 50:
        continue

    # fill NaN volumes with 0
    for col in ["bid_volume_1","bid_volume_2","bid_volume_3",
                "ask_volume_1","ask_volume_2","ask_volume_3"]:
        pb[col] = pb[col].fillna(0)

    bid1 = pb["bid_volume_1"]; ask1 = pb["ask_volume_1"]
    bid_tot = pb[["bid_volume_1","bid_volume_2","bid_volume_3"]].sum(axis=1)
    ask_tot = pb[["ask_volume_1","ask_volume_2","ask_volume_3"]].sum(axis=1)

    denom1 = (bid1 + ask1).replace(0, np.nan)
    denom3 = (bid_tot + ask_tot).replace(0, np.nan)

    # L1 OBI
    obi1 = (bid1 - ask1) / denom1

    # L3 weighted OBI (closer levels weighted more)
    w = np.array([3, 2, 1])
    bid_w = (pb["bid_volume_1"]*w[0] + pb["bid_volume_2"]*w[1] + pb["bid_volume_3"]*w[2])
    ask_w = (pb["ask_volume_1"]*w[0] + pb["ask_volume_2"]*w[1] + pb["ask_volume_3"]*w[2])
    obi3 = (bid_w - ask_w) / (bid_w + ask_w).replace(0, np.nan)

    spread     = pb["ask_price_1"] - pb["bid_price_1"]
    spread_pct = spread / pb["mid_price"].replace(0, np.nan)
    mid_ret    = pb["mid_price"].pct_change()

    # trade features: join trades to price timestamps
    tb = trades[trades["product"] == prod].copy()

    # buyer imbalance per timestamp window
    def agg_trades(ts_arr, qt_arr, buyer_arr, seller_arr, price_ts):
        freq  = np.zeros(len(price_ts))
        vol   = np.zeros(len(price_ts))
        b_vol = np.zeros(len(price_ts))
        s_vol = np.zeros(len(price_ts))
        if len(ts_arr) == 0:
            return freq, vol, b_vol, s_vol
        # vectorised bin assignment
        bins = np.searchsorted(price_ts, ts_arr, side="right") - 1
        valid = (bins >= 0) & (bins < len(price_ts))
        bins = bins[valid]
        qv   = qt_arr[valid]
        bv   = buyer_arr[valid]
        sv   = seller_arr[valid]
        np.add.at(freq,  bins, 1)
        np.add.at(vol,   bins, qv)
        np.add.at(b_vol, bins, bv)
        np.add.at(s_vol, bins, sv)
        return freq, vol, b_vol, s_vol

    buyer_vol  = (~tb["buyer"].isna()).astype(float) * tb["quantity"]
    seller_vol = (~tb["seller"].isna()).astype(float) * tb["quantity"]
    freq, vol, bv, sv = agg_trades(
        tb["gts"].values, tb["quantity"].values,
        buyer_vol.values, seller_vol.values,
        pb["gts"].values)

    avg_sz    = np.where(freq > 0, vol / freq, 0)
    tot_bs    = bv + sv
    buyer_imb = np.where(tot_bs > 0, (bv - sv) / tot_bs, 0)

    feat = pd.DataFrame({
        "gts":        pb["gts"].values,
        "day":        pb["day"].values,
        "OBI_L1":     obi1.values,
        "OBI_L3":     obi3.values,
        "SPREAD":     spread.values,
        "SPREAD_PCT": spread_pct.values,
        "DEPTH_BID":  bid_tot.values,
        "DEPTH_ASK":  ask_tot.values,
        "DEPTH_IMBAL":(bid_tot - ask_tot).values / denom3.values,
        "MID_RETURN": mid_ret.values,
        "TRADE_FREQ": freq,
        "TRADE_VOL":  vol,
        "AVG_TRADE_SZ": avg_sz,
        "BUYER_IMBAL": buyer_imb,
    })
    feat["FORWARD_RET"] = feat["MID_RETURN"].shift(-FORWARD_STEPS)
    feat.dropna(subset=["FORWARD_RET","OBI_L1"], inplace=True)
    feat["product"] = prod
    feat["group"]   = P2G.get(prod, "")
    all_feat[prod]  = feat

print(f"  Built features for {len(all_feat)} products")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Feature × product Pearson-r heatmap with forward return
# ─────────────────────────────────────────────────────────────────────────────
print("1. Feature-return correlation heatmap...")
corr_rows = {}
for prod, df in all_feat.items():
    row = {}
    for f in FEAT_NAMES:
        if f not in df.columns:
            row[f] = np.nan; continue
        x = df[f].replace([np.inf,-np.inf], np.nan).dropna()
        y = df.loc[x.index, "FORWARD_RET"].dropna()
        idx = x.index.intersection(y.index)
        if len(idx) < 30:
            row[f] = np.nan; continue
        r, _ = stats.pearsonr(x[idx], y[idx])
        row[f] = r
    corr_rows[prod] = row

corr_df = pd.DataFrame(corr_rows).T   # products × features
# sort products by group
prod_order = [p for g in GROUPS for p in GROUPS[g] if p in corr_df.index]
corr_df = corr_df.loc[prod_order]

fig, ax = plt.subplots(figsize=(18, 14), facecolor=DARK)
ax.set_facecolor(MID)
norm = TwoSlopeNorm(vmin=-0.15, vcenter=0, vmax=0.15)
im = ax.imshow(corr_df.values, cmap="RdBu_r", norm=norm, aspect="auto")
ax.set_xticks(range(len(FEAT_NAMES)))
ax.set_xticklabels(FEAT_NAMES, rotation=40, ha="right", fontsize=9, color="white")
ax.set_yticks(range(len(prod_order)))
ax.set_yticklabels(prod_order, fontsize=6.5, color="white")
ax.set_title(f"Pearson r — Feature vs {FORWARD_STEPS}-step Forward Return", color="white", fontsize=13)
cb = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02)
cb.ax.tick_params(colors="white"); cb.set_label("Pearson r", color="white")
# group separators
pos = 0
for g in GROUPS:
    cnt = sum(1 for p in GROUPS[g] if p in prod_order)
    if cnt == 0: continue
    ax.axhline(pos - 0.5, color="#aaa", lw=0.7, alpha=0.6)
    pos += cnt
plt.tight_layout()
plt.savefig(OUT + "pred_01_feature_corr_heatmap.png", dpi=120, bbox_inches="tight", facecolor=DARK)
plt.close()
print("  done")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Conditional mean return by OBI quintile — per group
# ─────────────────────────────────────────────────────────────────────────────
print("2. OBI quintile conditional return...")
all_df = pd.concat(all_feat.values(), ignore_index=True)

fig, axes = plt.subplots(2, 5, figsize=(22, 10), facecolor=DARK)
axes = axes.flatten()
for idx, (gname, gprods) in enumerate(GROUPS.items()):
    ax = axes[idx]
    ax.set_facecolor(MID); ax.spines[:].set_color("#333")
    gc = GCOLORS[gname]
    gdf = all_df[all_df["group"] == gname].copy()
    if len(gdf) < 50:
        ax.set_visible(False); continue

    gdf["obi_q"] = pd.qcut(gdf["OBI_L1"], q=5, labels=False, duplicates="drop")
    grouped = gdf.groupby("obi_q")["FORWARD_RET"].agg(["mean","sem"])
    x = grouped.index.values.astype(float)
    y = grouped["mean"].values * 1e4   # bps
    yerr = grouped["sem"].values * 1e4

    clr = [RED if v < 0 else GREEN for v in y]
    ax.bar(x, y, color=clr, alpha=0.85, width=0.7, yerr=yerr,
           error_kw=dict(ecolor=ACCENT, lw=1.5, capsize=4))
    ax.axhline(0, color="#666", lw=0.8)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["Q1\n(sell)", "Q2", "Q3", "Q4", "Q5\n(buy)"], fontsize=7, color="white")
    ax.set_ylabel("Fwd Return (bps)", color="white", fontsize=8)
    ax.set_title(gname, color="white", fontsize=9)
    ax.tick_params(colors="#aaa")

fig.suptitle(f"Conditional Mean Forward Return by OBI_L1 Quintile", color="white", fontsize=13)
plt.tight_layout()
plt.savefig(OUT + "pred_02_obi_quintile_return.png", dpi=120, bbox_inches="tight", facecolor=DARK)
plt.close()
print("  done")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Conditional mean return by spread quintile — per group
# ─────────────────────────────────────────────────────────────────────────────
print("3. Spread quintile conditional return...")
fig, axes = plt.subplots(2, 5, figsize=(22, 10), facecolor=DARK)
axes = axes.flatten()
for idx, (gname, gprods) in enumerate(GROUPS.items()):
    ax = axes[idx]
    ax.set_facecolor(MID); ax.spines[:].set_color("#333")
    gdf = all_df[all_df["group"] == gname].copy()
    if len(gdf) < 50:
        ax.set_visible(False); continue

    gdf["sp_q"] = pd.qcut(gdf["SPREAD"], q=5, labels=False, duplicates="drop")
    grouped = gdf.groupby("sp_q")["FORWARD_RET"].agg(["mean","sem","std","count"])
    x = grouped.index.values.astype(float)
    y = grouped["mean"].values * 1e4
    yerr = grouped["sem"].values * 1e4
    clr = [RED if v < 0 else GREEN for v in y]
    ax.bar(x, y, color=clr, alpha=0.85, width=0.7, yerr=yerr,
           error_kw=dict(ecolor=ACCENT, lw=1.5, capsize=4))
    ax.axhline(0, color="#666", lw=0.8)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["Narrow","","Mid","","Wide"], fontsize=7, color="white")
    ax.set_ylabel("Fwd Return (bps)", color="white", fontsize=8)
    ax.set_title(gname, color="white", fontsize=9)
    ax.tick_params(colors="#aaa")

fig.suptitle("Conditional Mean Forward Return by Spread Quintile", color="white", fontsize=13)
plt.tight_layout()
plt.savefig(OUT + "pred_03_spread_quintile_return.png", dpi=120, bbox_inches="tight", facecolor=DARK)
plt.close()
print("  done")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Scatter: OBI vs forward return — top 6 most predictive products
# ─────────────────────────────────────────────────────────────────────────────
print("4. OBI scatter vs forward return (top 6)...")
obi_corr = {p: corr_rows[p].get("OBI_L1", np.nan) for p in corr_rows}
top6 = sorted([p for p in obi_corr if not np.isnan(obi_corr[p])],
               key=lambda p: abs(obi_corr[p]), reverse=True)[:6]

fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor=DARK)
axes = axes.flatten()
for ax, prod in zip(axes, top6):
    ax.set_facecolor(MID); ax.spines[:].set_color("#333")
    df = all_feat[prod]
    x = df["OBI_L1"].clip(-1, 1)
    y = df["FORWARD_RET"].clip(-0.005, 0.005) * 1e4
    gc = GCOLORS.get(P2G.get(prod, ""), "#888")
    ax.scatter(x, y, s=0.4, alpha=0.15, color=gc, rasterized=True)
    # binned mean overlay
    bins = pd.cut(x, bins=20)
    bm = y.groupby(bins).mean()
    bx = [iv.mid for iv in bm.index]
    ax.plot(bx, bm.values, color=ACCENT, lw=2, zorder=5)
    ax.axhline(0, color="#666", lw=0.7); ax.axvline(0, color="#666", lw=0.7)
    r = obi_corr[prod]
    ax.set_title(f"{prod.split('_')[0]}…{prod.split('_')[-1]}\nr={r:.4f}", color="white", fontsize=8)
    ax.set_xlabel("OBI_L1", color="#aaa"); ax.set_ylabel("Fwd Return (bps)", color="#aaa")
    ax.tick_params(colors="#aaa", labelsize=7)

fig.suptitle("OBI_L1 vs Forward Return — Top 6 Predictive Products\n(orange = binned mean)",
             color="white", fontsize=12)
plt.tight_layout()
plt.savefig(OUT + "pred_04_obi_scatter.png", dpi=120, bbox_inches="tight", facecolor=DARK)
plt.close()
print("  done")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Trade frequency vs forward return (scatter + binned mean)
# ─────────────────────────────────────────────────────────────────────────────
print("5. Trade frequency vs forward return...")
fig, axes = plt.subplots(2, 5, figsize=(22, 10), facecolor=DARK)
axes = axes.flatten()
for idx, (gname, gprods) in enumerate(GROUPS.items()):
    ax = axes[idx]
    ax.set_facecolor(MID); ax.spines[:].set_color("#333")
    gc = GCOLORS[gname]
    gdf = all_df[(all_df["group"] == gname) & (all_df["TRADE_FREQ"] > 0)].copy()
    if len(gdf) < 30:
        ax.text(0.5, 0.5, "no trades", ha="center", va="center",
                transform=ax.transAxes, color="#888")
        ax.set_title(gname, color="white", fontsize=9); continue

    x = gdf["TRADE_FREQ"]
    y = gdf["FORWARD_RET"] * 1e4
    ax.scatter(x, y.clip(-10, 10), s=1.5, alpha=0.2, color=gc, rasterized=True)
    bins = pd.qcut(x, q=10, duplicates="drop")
    bm = y.groupby(bins).mean()
    bx = [iv.mid for iv in bm.index]
    ax.plot(bx, bm.values, color=ACCENT, lw=2, marker="o", ms=4, zorder=5)
    ax.axhline(0, color="#666", lw=0.7)
    r, pv = stats.pearsonr(x.clip(0, x.quantile(0.99)), y.clip(-10, 10))
    ax.set_title(f"{gname}\nr={r:.3f}", color="white", fontsize=8)
    ax.set_xlabel("Trade Freq", color="#aaa", fontsize=7)
    ax.set_ylabel("Fwd Ret (bps)", color="#aaa", fontsize=7)
    ax.tick_params(colors="#aaa", labelsize=6)

fig.suptitle("Trade Frequency vs Forward Return (orange = binned mean)", color="white", fontsize=13)
plt.tight_layout()
plt.savefig(OUT + "pred_05_trade_freq_return.png", dpi=120, bbox_inches="tight", facecolor=DARK)
plt.close()
print("  done")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Buyer imbalance vs forward return
# ─────────────────────────────────────────────────────────────────────────────
print("6. Buyer imbalance vs forward return...")
fig, axes = plt.subplots(2, 5, figsize=(22, 10), facecolor=DARK)
axes = axes.flatten()
for idx, (gname, gprods) in enumerate(GROUPS.items()):
    ax = axes[idx]
    ax.set_facecolor(MID); ax.spines[:].set_color("#333")
    gc = GCOLORS[gname]
    gdf = all_df[(all_df["group"] == gname) & (all_df["TRADE_FREQ"] > 0)].copy()
    if len(gdf) < 30:
        ax.set_visible(False); continue

    gdf["bi_q"] = pd.qcut(gdf["BUYER_IMBAL"], q=5, labels=False, duplicates="drop")
    grouped = gdf.groupby("bi_q")["FORWARD_RET"].agg(["mean","sem"])
    x = grouped.index.values.astype(float)
    y = grouped["mean"].values * 1e4
    yerr = grouped["sem"].values * 1e4
    clr = [RED if v < 0 else GREEN for v in y]
    ax.bar(x, y, color=clr, alpha=0.85, width=0.7, yerr=yerr,
           error_kw=dict(ecolor=ACCENT, lw=1.5, capsize=4))
    ax.axhline(0, color="#666", lw=0.8)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["Sell","","Neut","","Buy"], fontsize=7, color="white")
    ax.set_ylabel("Fwd Return (bps)", color="white", fontsize=8)
    ax.set_title(gname, color="white", fontsize=9)
    ax.tick_params(colors="#aaa")

fig.suptitle("Conditional Forward Return by Buyer Imbalance Quintile", color="white", fontsize=13)
plt.tight_layout()
plt.savefig(OUT + "pred_06_buyer_imbal_return.png", dpi=120, bbox_inches="tight", facecolor=DARK)
plt.close()
print("  done")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Rolling IC (OBI_L1 → fwd return) over time — per group
# ─────────────────────────────────────────────────────────────────────────────
print("7. Rolling IC over time...")
ROLL_WIN = 500   # rows per rolling window

fig, axes = plt.subplots(2, 5, figsize=(22, 10), facecolor=DARK)
axes = axes.flatten()
for idx, (gname, gprods) in enumerate(GROUPS.items()):
    ax = axes[idx]
    ax.set_facecolor(MID); ax.spines[:].set_color("#333")
    gc = GCOLORS[gname]
    avail = [p for p in gprods if p in all_feat]
    if not avail:
        ax.set_visible(False); continue

    for prod in avail:
        df = all_feat[prod].dropna(subset=["OBI_L1","FORWARD_RET"]).reset_index(drop=True)
        if len(df) < ROLL_WIN * 2:
            continue
        ics = []
        ts_mid = []
        for start in range(0, len(df) - ROLL_WIN, ROLL_WIN // 2):
            chunk = df.iloc[start:start+ROLL_WIN]
            r, _ = stats.pearsonr(chunk["OBI_L1"], chunk["FORWARD_RET"])
            ics.append(r)
            ts_mid.append(chunk["gts"].median() / 1e6)
        ax.plot(ts_mid, ics, lw=1, alpha=0.75, label=prod.split("_")[-1])

    ax.axhline(0, color="#666", lw=0.8)
    ax.set_xlabel("Time (M ticks)", color="#aaa", fontsize=7)
    ax.set_ylabel("IC (Pearson r)", color="#aaa", fontsize=7)
    ax.set_title(gname, color="white", fontsize=9)
    ax.tick_params(colors="#aaa", labelsize=6)
    ax.legend(fontsize=5.5, facecolor="#0f3460", edgecolor="#444", labelcolor="white")

fig.suptitle("Rolling IC: OBI_L1 → Forward Return (window=500)", color="white", fontsize=13)
plt.tight_layout()
plt.savefig(OUT + "pred_07_rolling_ic_obi.png", dpi=120, bbox_inches="tight", facecolor=DARK)
plt.close()
print("  done")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Depth imbalance vs forward return — binned bar
# ─────────────────────────────────────────────────────────────────────────────
print("8. Depth imbalance vs forward return...")
fig, axes = plt.subplots(2, 5, figsize=(22, 10), facecolor=DARK)
axes = axes.flatten()
for idx, (gname, gprods) in enumerate(GROUPS.items()):
    ax = axes[idx]
    ax.set_facecolor(MID); ax.spines[:].set_color("#333")
    gdf = all_df[all_df["group"] == gname].copy().dropna(subset=["DEPTH_IMBAL"])
    if len(gdf) < 50:
        ax.set_visible(False); continue

    gdf["di_q"] = pd.qcut(gdf["DEPTH_IMBAL"], q=5, labels=False, duplicates="drop")
    grouped = gdf.groupby("di_q")["FORWARD_RET"].agg(["mean","sem"])
    x = grouped.index.values.astype(float)
    y = grouped["mean"].values * 1e4
    yerr = grouped["sem"].values * 1e4
    clr = [RED if v < 0 else GREEN for v in y]
    ax.bar(x, y, color=clr, alpha=0.85, width=0.7, yerr=yerr,
           error_kw=dict(ecolor=ACCENT, lw=1.5, capsize=4))
    ax.axhline(0, color="#666", lw=0.8)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["Ask\nheavy","","Neut","","Bid\nheavy"], fontsize=6.5, color="white")
    ax.set_ylabel("Fwd Return (bps)", color="white", fontsize=8)
    ax.set_title(gname, color="white", fontsize=9)
    ax.tick_params(colors="#aaa")

fig.suptitle("Conditional Forward Return by Depth Imbalance (3-level) Quintile", color="white", fontsize=13)
plt.tight_layout()
plt.savefig(OUT + "pred_08_depth_imbal_return.png", dpi=120, bbox_inches="tight", facecolor=DARK)
plt.close()
print("  done")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Feature intercorrelation matrix (avg across products)
# ─────────────────────────────────────────────────────────────────────────────
print("9. Feature intercorrelation matrix...")
feat_corrs = []
for prod, df in all_feat.items():
    sub = df[FEAT_NAMES].replace([np.inf,-np.inf], np.nan).dropna()
    if len(sub) < 30:
        continue
    feat_corrs.append(sub.corr().values)

avg_feat_corr = np.nanmean(feat_corrs, axis=0)
feat_df = pd.DataFrame(avg_feat_corr, index=FEAT_NAMES, columns=FEAT_NAMES)

fig, ax = plt.subplots(figsize=(12, 10), facecolor=DARK)
ax.set_facecolor(MID)
norm2 = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
im = ax.imshow(feat_df.values, cmap="RdBu_r", norm=norm2)
ax.set_xticks(range(len(FEAT_NAMES)))
ax.set_xticklabels(FEAT_NAMES, rotation=45, ha="right", fontsize=9, color="white")
ax.set_yticks(range(len(FEAT_NAMES)))
ax.set_yticklabels(FEAT_NAMES, fontsize=9, color="white")
for i in range(len(FEAT_NAMES)):
    for j in range(len(FEAT_NAMES)):
        ax.text(j, i, f"{feat_df.values[i,j]:.2f}", ha="center", va="center",
                fontsize=6.5, color="black" if abs(feat_df.values[i,j]) > 0.5 else "white")
ax.set_title("Feature Intercorrelation (avg across all products)", color="white", fontsize=12)
cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.ax.tick_params(colors="white"); cb.set_label("Pearson r", color="white")
plt.tight_layout()
plt.savefig(OUT + "pred_09_feature_intercorr.png", dpi=120, bbox_inches="tight", facecolor=DARK)
plt.close()
print("  done")

# ─────────────────────────────────────────────────────────────────────────────
# 10. IC by day: OBI_L1 vs fwd return, per product (day 2/3/4)
# ─────────────────────────────────────────────────────────────────────────────
print("10. IC by day...")
ic_day = {}
for prod, df in all_feat.items():
    ic_day[prod] = {}
    for d in [2, 3, 4]:
        sub = df[df["day"] == d].dropna(subset=["OBI_L1","FORWARD_RET"])
        if len(sub) < 30:
            ic_day[prod][d] = np.nan; continue
        r, _ = stats.pearsonr(sub["OBI_L1"], sub["FORWARD_RET"])
        ic_day[prod][d] = r

ic_day_df = pd.DataFrame(ic_day).T
ic_day_df.columns = ["Day2","Day3","Day4"]
ic_day_df["group"] = ic_day_df.index.map(P2G)
ic_day_df_sorted = ic_day_df.loc[prod_order].dropna(how="all")

fig, axes = plt.subplots(1, 3, figsize=(20, 10), facecolor=DARK)
for ax_i, (day_col, day_label) in enumerate(zip(["Day2","Day3","Day4"],["Day 2","Day 3","Day 4"])):
    ax = axes[ax_i]
    ax.set_facecolor(MID); ax.spines[:].set_color("#333")
    vals = ic_day_df_sorted[day_col]
    clr  = [GCOLORS.get(P2G.get(p,""),"#888") for p in ic_day_df_sorted.index]
    ax.barh(ic_day_df_sorted.index, vals, color=clr, alpha=0.85)
    ax.axvline(0, color="#666", lw=0.8)
    ci = 1.96 / np.sqrt(500)
    ax.axvline( ci, color=ACCENT, lw=1, ls="--")
    ax.axvline(-ci, color=ACCENT, lw=1, ls="--")
    ax.set_xlabel("IC (Pearson r)", color="white")
    ax.set_title(f"OBI_L1 IC — {day_label}", color="white", fontsize=10)
    ax.tick_params(labelsize=6, colors="white")

legend_els = [Patch(color=GCOLORS[g], label=g) for g in GROUPS]
axes[0].legend(handles=legend_els, fontsize=7, facecolor="#0f3460",
               edgecolor="#444", labelcolor="white")
fig.suptitle("Information Coefficient (OBI_L1 → Fwd Return) by Day", color="white", fontsize=13)
plt.tight_layout()
plt.savefig(OUT + "pred_10_ic_by_day.png", dpi=120, bbox_inches="tight", facecolor=DARK)
plt.close()
print("  done")

# ─────────────────────────────────────────────────────────────────────────────
# 11. Multi-horizon IC: OBI_L1 vs fwd return at lags 1,2,5,10,20
# ─────────────────────────────────────────────────────────────────────────────
print("11. Multi-horizon IC decay...")
horizons = [1, 2, 5, 10, 20]
horizon_ic = {h: {} for h in horizons}

for prod, base_df in all_feat.items():
    pb = prices[prices["product"] == prod].copy().reset_index(drop=True)
    if len(pb) < 100: continue

    for col in ["bid_volume_1","bid_volume_2","bid_volume_3",
                "ask_volume_1","ask_volume_2","ask_volume_3"]:
        pb[col] = pb[col].fillna(0)
    bid1 = pb["bid_volume_1"]; ask1 = pb["ask_volume_1"]
    denom1 = (bid1 + ask1).replace(0, np.nan)
    obi1 = ((bid1 - ask1) / denom1).fillna(0).values
    mid_ret = pb["mid_price"].pct_change().values

    for h in horizons:
        fwd = pd.Series(mid_ret).shift(-h).values
        valid = ~(np.isnan(obi1) | np.isnan(fwd))
        if valid.sum() < 30: continue
        r, _ = stats.pearsonr(obi1[valid], fwd[valid])
        horizon_ic[h][prod] = r

fig, ax = plt.subplots(figsize=(14, 7), facecolor=DARK)
ax.set_facecolor(MID); ax.spines[:].set_color("#333")
for gname, gprods in GROUPS.items():
    gc = GCOLORS[gname]
    avail = [p for p in gprods if all(p in horizon_ic[h] for h in horizons)]
    if not avail: continue
    avg_ic = [np.mean([horizon_ic[h][p] for p in avail]) for h in horizons]
    ax.plot(horizons, avg_ic, color=gc, lw=2, marker="o", ms=5, label=gname)
    ax.fill_between(horizons,
                    [np.mean([horizon_ic[h][p] for p in avail]) - np.std([horizon_ic[h][p] for p in avail]) for h in horizons],
                    [np.mean([horizon_ic[h][p] for p in avail]) + np.std([horizon_ic[h][p] for p in avail]) for h in horizons],
                    color=gc, alpha=0.12)

ax.axhline(0, color="#666", lw=0.8)
ax.set_xticks(horizons)
ax.set_xticklabels([f"h={h}" for h in horizons], color="white")
ax.set_xlabel("Forecast Horizon (steps)", color="white")
ax.set_ylabel("IC (Pearson r)", color="white")
ax.set_title("IC Decay: OBI_L1 → Forward Return at Multiple Horizons", color="white", fontsize=12)
ax.legend(fontsize=8, facecolor="#0f3460", edgecolor="#444", labelcolor="white")
ax.tick_params(colors="#aaa")
plt.tight_layout()
plt.savefig(OUT + "pred_11_ic_horizon_decay.png", dpi=120, bbox_inches="tight", facecolor=DARK)
plt.close()
print("  done")

# ─────────────────────────────────────────────────────────────────────────────
# 12. OBI_L1 + SPREAD joint: 2D binned return (heatmap per group)
# ─────────────────────────────────────────────────────────────────────────────
print("12. 2D OBI × Spread conditional return heatmap...")
fig, axes = plt.subplots(2, 5, figsize=(24, 11), facecolor=DARK)
axes = axes.flatten()
for idx, (gname, gprods) in enumerate(GROUPS.items()):
    ax = axes[idx]
    ax.set_facecolor(MID); ax.spines[:].set_color("#333")
    gdf = all_df[all_df["group"] == gname].copy().dropna(subset=["OBI_L1","SPREAD","FORWARD_RET"])
    if len(gdf) < 100:
        ax.set_visible(False); continue

    gdf["obi_b"] = pd.qcut(gdf["OBI_L1"], q=5, labels=False, duplicates="drop")
    gdf["sp_b"]  = pd.qcut(gdf["SPREAD"],  q=5, labels=False, duplicates="drop")
    heat = gdf.groupby(["obi_b","sp_b"])["FORWARD_RET"].mean().unstack() * 1e4
    heat = heat.reindex(index=range(5), columns=range(5))

    norm3 = TwoSlopeNorm(vmin=heat.min().min() or -0.01,
                          vcenter=0,
                          vmax=heat.max().max() or 0.01)
    im = ax.imshow(heat.values, cmap="RdYlGn", aspect="auto", norm=norm3)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(["Narr","","Mid","","Wide"], fontsize=6.5, color="white")
    ax.set_yticklabels(["Sell","","Neut","","Buy"], fontsize=6.5, color="white")
    ax.set_xlabel("Spread →", color="#aaa", fontsize=7)
    ax.set_ylabel("OBI →", color="#aaa", fontsize=7)
    ax.set_title(gname, color="white", fontsize=8.5)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(colors="white", labelsize=6)

fig.suptitle("2D Conditional Forward Return: OBI_L1 × Spread (bps)", color="white", fontsize=13)
plt.tight_layout()
plt.savefig(OUT + "pred_12_obi_spread_2d.png", dpi=120, bbox_inches="tight", facecolor=DARK)
plt.close()
print("  done")

print(f"\nAll predictive plots saved to {OUT}")
for f in sorted(os.listdir(OUT)):
    if f.startswith("pred_"):
        print(f"  {f}")
