"""
Day-wise trend analysis: what changes between days 2, 3, 4?
  1. Mid-price level per product per day (normalized)
  2. Return distribution shift per day — KDE overlay
  3. Volatility (std of returns) per product per day
  4. OBI mean per product per day
  5. Spread mean per product per day
  6. Trade frequency per product per day
  7. Day-over-day return correlation (do products follow same pattern?)
  8. Day-specific anomalies: z-score of daily mean vs all-day mean
  9. Intraday price path per group per day (aligned 0..1M ticks)
 10. Cumulative return per product per day
 11. Feature predictive IC per day (already done for OBI — now all features)
 12. Buyer imbalance drift per day
"""

import os, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde

OUT = "/media/manukrishnan/Mk/prosperity_4/analysis_output/deep/"
os.makedirs(OUT, exist_ok=True)

DARK="#1a1a2e"; MID="#16213e"; ACCENT="#f39c12"; RED="#e74c3c"; GREEN="#2ecc71"
DAYS = [2, 3, 4]
DAY_COLORS = {2:"#3498db", 3:"#e67e22", 4:"#9b59b6"}
DAY_LABELS  = {2:"Day 2", 3:"Day 3", 4:"Day 4"}

plt.rcParams.update({"figure.facecolor":DARK,"axes.facecolor":MID,
                     "text.color":"white","axes.labelcolor":"#aaa",
                     "xtick.color":"#aaa","ytick.color":"#aaa",
                     "axes.edgecolor":"#444","grid.color":"#333","font.size":8})

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
P2G = {p:g for g,ps in GROUPS.items() for p in ps}
GCOLORS = dict(zip(GROUPS, plt.cm.tab10(np.linspace(0,1,10))))

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading...")
pf_all, tf_all = [], []
for d in DAYS:
    pf = pd.read_csv(f"/media/manukrishnan/Mk/prosperity_4/data/prices_round_5_day_{d}.csv", sep=";")
    pf["day"] = d
    pf_all.append(pf)
    tf = pd.read_csv(f"/media/manukrishnan/Mk/prosperity_4/data/trades_round_5_day_{d}.csv", sep=";")
    tf["day"] = d
    tf_all.append(tf)

prices = pd.concat(pf_all, ignore_index=True)
trades = pd.concat(tf_all, ignore_index=True)
trades.rename(columns={"symbol":"product"}, inplace=True)

for col in ["bid_volume_1","bid_volume_2","bid_volume_3",
            "ask_volume_1","ask_volume_2","ask_volume_3"]:
    prices[col] = prices[col].fillna(0)

prices["spread"]    = prices["ask_price_1"] - prices["bid_price_1"]
prices["obi_l1"]    = ((prices["bid_volume_1"] - prices["ask_volume_1"]) /
                        (prices["bid_volume_1"] + prices["ask_volume_1"]).replace(0,np.nan))
prices["depth_bid"] = prices[["bid_volume_1","bid_volume_2","bid_volume_3"]].sum(axis=1)
prices["depth_ask"] = prices[["ask_volume_1","ask_volume_2","ask_volume_3"]].sum(axis=1)

ALL_PRODUCTS = sorted(p for p in prices["product"].unique()
                      if not p.startswith("VEV_") and p in P2G)
prod_order = [p for g in GROUPS for p in GROUPS[g] if p in ALL_PRODUCTS]

prices.sort_values(["product","day","timestamp"], inplace=True)
prices["ret"] = prices.groupby(["product","day"])["mid_price"].pct_change()

# ─────────────────────────────────────────────────────────────────────────────
# 1. Intraday price path per group per day
# ─────────────────────────────────────────────────────────────────────────────
print("1. Intraday price paths...")
fig, axes = plt.subplots(5, 2, figsize=(22, 28), facecolor=DARK)
axes = axes.flatten()
for idx, (gname, gprods) in enumerate(GROUPS.items()):
    ax = axes[idx]; ax.set_facecolor(MID); ax.spines[:].set_color("#333")
    avail = [p for p in gprods if p in ALL_PRODUCTS]
    if not avail: continue
    prod = avail[0]   # representative
    for d in DAYS:
        sub = prices[(prices["product"]==prod) & (prices["day"]==d)]
        if sub.empty: continue
        mp = sub["mid_price"].values
        mp_norm = (mp - mp[0]) / mp[0] * 100   # % deviation from open
        ax.plot(sub["timestamp"].values / 1e6, mp_norm,
                color=DAY_COLORS[d], lw=0.9, label=DAY_LABELS[d], alpha=0.9)
    ax.axhline(0, color="#555", lw=0.5)
    ax.set_title(f"{gname} ({prod.split('_')[-1]})", color="white", fontsize=9)
    ax.set_xlabel("Timestamp (M)", color="#aaa", fontsize=7)
    ax.set_ylabel("% from open", color="#aaa", fontsize=7)
    ax.tick_params(colors="#aaa", labelsize=6)
    ax.legend(fontsize=7, facecolor="#0f3460", edgecolor="#444", labelcolor="white")
fig.suptitle("Intraday Price Path per Day (% from open, 1 rep product per group)",
             color="white", fontsize=13)
plt.tight_layout()
plt.savefig(OUT+"day_01_intraday_paths.png", dpi=110, bbox_inches="tight", facecolor=DARK)
plt.close(); print("  done")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Cumulative return per product per day — all groups
# ─────────────────────────────────────────────────────────────────────────────
print("2. Cumulative returns per day...")
fig, axes = plt.subplots(5, 2, figsize=(22, 28), facecolor=DARK)
axes = axes.flatten()
for idx, (gname, gprods) in enumerate(GROUPS.items()):
    ax = axes[idx]; ax.set_facecolor(MID); ax.spines[:].set_color("#333")
    avail = [p for p in gprods if p in ALL_PRODUCTS]
    for prod in avail:
        pname = prod.split("_")[-1]
        for d in DAYS:
            sub = prices[(prices["product"]==prod) & (prices["day"]==d)].copy()
            if sub.empty: continue
            cumret = (1 + sub["ret"].fillna(0)).cumprod() - 1
            ax.plot(sub["timestamp"].values/1e6, cumret.values * 100,
                    color=DAY_COLORS[d], lw=0.8, alpha=0.7,
                    ls=["-","--","-."][avail.index(prod) % 3])
    ax.axhline(0, color="#555", lw=0.5)
    ax.set_title(gname, color="white", fontsize=9)
    ax.set_xlabel("Timestamp (M)", color="#aaa", fontsize=7)
    ax.set_ylabel("Cumulative Return %", color="#aaa", fontsize=7)
    ax.tick_params(colors="#aaa", labelsize=6)
    day_patches = [Patch(color=DAY_COLORS[d], label=DAY_LABELS[d]) for d in DAYS]
    ax.legend(handles=day_patches, fontsize=7, facecolor="#0f3460",
              edgecolor="#444", labelcolor="white")
fig.suptitle("Cumulative Return per Day per Group (solid/dashed/dotted = different products)",
             color="white", fontsize=13)
plt.tight_layout()
plt.savefig(OUT+"day_02_cumulative_returns.png", dpi=110, bbox_inches="tight", facecolor=DARK)
plt.close(); print("  done")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Return distribution shift per day — KDE overlay per group
# ─────────────────────────────────────────────────────────────────────────────
print("3. Return distribution shift by day...")
fig, axes = plt.subplots(2, 5, figsize=(22, 10), facecolor=DARK)
axes = axes.flatten()
for idx, (gname, gprods) in enumerate(GROUPS.items()):
    ax = axes[idx]; ax.set_facecolor(MID); ax.spines[:].set_color("#333")
    avail = [p for p in gprods if p in ALL_PRODUCTS]
    for d in DAYS:
        ret_all = prices[(prices["product"].isin(avail)) & (prices["day"]==d)]["ret"].dropna()
        ret_clip = ret_all.clip(-0.004, 0.004)
        if len(ret_clip) < 20: continue
        kde = gaussian_kde(ret_clip, bw_method=0.3)
        xs = np.linspace(-0.004, 0.004, 300)
        ax.plot(xs, kde(xs), color=DAY_COLORS[d], lw=2, label=DAY_LABELS[d])
        ax.fill_between(xs, kde(xs), alpha=0.12, color=DAY_COLORS[d])
    ax.axvline(0, color="#555", lw=0.6)
    ax.set_title(gname, color="white", fontsize=8.5)
    ax.set_xlabel("Return", color="#aaa", fontsize=7)
    ax.tick_params(colors="#aaa", labelsize=6)
    ax.legend(fontsize=7, facecolor="#0f3460", edgecolor="#444", labelcolor="white")
fig.suptitle("Return Distribution Shift Across Days (group-level KDE)", color="white", fontsize=13)
plt.tight_layout()
plt.savefig(OUT+"day_03_return_dist_shift.png", dpi=120, bbox_inches="tight", facecolor=DARK)
plt.close(); print("  done")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Volatility heatmap: product × day
# ─────────────────────────────────────────────────────────────────────────────
print("4. Volatility heatmap (product × day)...")
vol_mat = np.zeros((len(prod_order), 3))
for i, prod in enumerate(prod_order):
    for j, d in enumerate(DAYS):
        sub = prices[(prices["product"]==prod) & (prices["day"]==d)]["ret"].dropna()
        vol_mat[i, j] = sub.std() * 1e4   # bps

fig, ax = plt.subplots(figsize=(8, 16), facecolor=DARK)
ax.set_facecolor(MID)
im = ax.imshow(vol_mat, aspect="auto", cmap="YlOrRd")
ax.set_xticks([0,1,2]); ax.set_xticklabels(["Day 2","Day 3","Day 4"], color="white", fontsize=10)
ax.set_yticks(range(len(prod_order)))
ax.set_yticklabels(prod_order, fontsize=6.5, color="white")
for i in range(len(prod_order)):
    for j in range(3):
        ax.text(j, i, f"{vol_mat[i,j]:.2f}", ha="center", va="center",
                fontsize=5.5, color="black" if vol_mat[i,j]>vol_mat.max()*0.6 else "white")
ax.set_title("Volatility (σ of returns, bps)\nProduct × Day", color="white", fontsize=12)
cb = plt.colorbar(im, ax=ax, fraction=0.02); cb.ax.tick_params(colors="white")
cb.set_label("Volatility (bps)", color="white")
# group separators
pos = 0
for g in GROUPS:
    cnt = sum(1 for p in GROUPS[g] if p in prod_order)
    if cnt == 0: continue
    ax.axhline(pos - 0.5, color="#aaa", lw=0.8, alpha=0.6)
    pos += cnt
plt.tight_layout()
plt.savefig(OUT+"day_04_vol_heatmap.png", dpi=120, bbox_inches="tight", facecolor=DARK)
plt.close(); print("  done")

# ─────────────────────────────────────────────────────────────────────────────
# 5. OBI mean heatmap: product × day
# ─────────────────────────────────────────────────────────────────────────────
print("5. OBI mean heatmap...")
obi_mat = np.zeros((len(prod_order), 3))
for i, prod in enumerate(prod_order):
    for j, d in enumerate(DAYS):
        sub = prices[(prices["product"]==prod) & (prices["day"]==d)]["obi_l1"].dropna()
        obi_mat[i, j] = sub.mean() if len(sub) > 0 else 0

fig, ax = plt.subplots(figsize=(8, 16), facecolor=DARK)
ax.set_facecolor(MID)
norm = TwoSlopeNorm(vmin=obi_mat.min(), vcenter=0, vmax=obi_mat.max())
im = ax.imshow(obi_mat, aspect="auto", cmap="RdBu_r", norm=norm)
ax.set_xticks([0,1,2]); ax.set_xticklabels(["Day 2","Day 3","Day 4"], color="white", fontsize=10)
ax.set_yticks(range(len(prod_order)))
ax.set_yticklabels(prod_order, fontsize=6.5, color="white")
for i in range(len(prod_order)):
    for j in range(3):
        ax.text(j, i, f"{obi_mat[i,j]:.3f}", ha="center", va="center", fontsize=5.5, color="white")
ax.set_title("Mean OBI_L1 — Product × Day\n(blue=bid heavy, red=ask heavy)", color="white", fontsize=11)
cb = plt.colorbar(im, ax=ax, fraction=0.02); cb.ax.tick_params(colors="white")
pos = 0
for g in GROUPS:
    cnt = sum(1 for p in GROUPS[g] if p in prod_order)
    if cnt == 0: continue
    ax.axhline(pos - 0.5, color="#aaa", lw=0.8, alpha=0.6); pos += cnt
plt.tight_layout()
plt.savefig(OUT+"day_05_obi_heatmap.png", dpi=120, bbox_inches="tight", facecolor=DARK)
plt.close(); print("  done")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Spread mean heatmap: product × day
# ─────────────────────────────────────────────────────────────────────────────
print("6. Spread heatmap...")
sp_mat = np.zeros((len(prod_order), 3))
for i, prod in enumerate(prod_order):
    for j, d in enumerate(DAYS):
        sub = prices[(prices["product"]==prod) & (prices["day"]==d)]["spread"].dropna()
        sp_mat[i, j] = sub.mean() if len(sub) > 0 else 0

fig, ax = plt.subplots(figsize=(8, 16), facecolor=DARK)
ax.set_facecolor(MID)
im = ax.imshow(sp_mat, aspect="auto", cmap="plasma")
ax.set_xticks([0,1,2]); ax.set_xticklabels(["Day 2","Day 3","Day 4"], color="white", fontsize=10)
ax.set_yticks(range(len(prod_order)))
ax.set_yticklabels(prod_order, fontsize=6.5, color="white")
for i in range(len(prod_order)):
    for j in range(3):
        ax.text(j, i, f"{sp_mat[i,j]:.1f}", ha="center", va="center", fontsize=5.5, color="white")
ax.set_title("Mean Spread — Product × Day", color="white", fontsize=12)
cb = plt.colorbar(im, ax=ax, fraction=0.02); cb.ax.tick_params(colors="white")
pos = 0
for g in GROUPS:
    cnt = sum(1 for p in GROUPS[g] if p in prod_order)
    if cnt == 0: continue
    ax.axhline(pos - 0.5, color="#aaa", lw=0.8, alpha=0.6); pos += cnt
plt.tight_layout()
plt.savefig(OUT+"day_06_spread_heatmap.png", dpi=120, bbox_inches="tight", facecolor=DARK)
plt.close(); print("  done")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Trade frequency & volume heatmap: product × day
# ─────────────────────────────────────────────────────────────────────────────
print("7. Trade frequency heatmap...")
freq_mat = np.zeros((len(prod_order), 3))
vol_t_mat = np.zeros((len(prod_order), 3))
for i, prod in enumerate(prod_order):
    for j, d in enumerate(DAYS):
        sub = trades[(trades["product"]==prod) & (trades["day"]==d)]
        freq_mat[i, j]  = len(sub)
        vol_t_mat[i, j] = sub["quantity"].sum() if len(sub) > 0 else 0

fig, axes = plt.subplots(1, 2, figsize=(16, 16), facecolor=DARK)
for ax, mat, title, cmap in [
    (axes[0], freq_mat,  "Trade Count — Product × Day", "viridis"),
    (axes[1], vol_t_mat, "Trade Volume — Product × Day", "cividis"),
]:
    ax.set_facecolor(MID)
    im = ax.imshow(mat, aspect="auto", cmap=cmap)
    ax.set_xticks([0,1,2]); ax.set_xticklabels(["Day 2","Day 3","Day 4"], color="white", fontsize=10)
    ax.set_yticks(range(len(prod_order)))
    ax.set_yticklabels(prod_order, fontsize=6.5, color="white")
    for i in range(len(prod_order)):
        for j in range(3):
            ax.text(j, i, f"{int(mat[i,j])}", ha="center", va="center",
                    fontsize=5, color="white")
    ax.set_title(title, color="white", fontsize=11)
    cb = plt.colorbar(im, ax=ax, fraction=0.02); cb.ax.tick_params(colors="white")
    pos = 0
    for g in GROUPS:
        cnt = sum(1 for p in GROUPS[g] if p in prod_order)
        if cnt == 0: continue
        ax.axhline(pos - 0.5, color="#aaa", lw=0.8, alpha=0.6); pos += cnt
plt.tight_layout()
plt.savefig(OUT+"day_07_trade_freq_vol_heatmap.png", dpi=120, bbox_inches="tight", facecolor=DARK)
plt.close(); print("  done")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Day-specific anomaly z-score (which products behave differently on each day)
# ─────────────────────────────────────────────────────────────────────────────
print("8. Day anomaly z-scores...")
def day_zscore_mat(mat):
    """Z-score each row (product) across its 3 day values."""
    mu  = mat.mean(axis=1, keepdims=True)
    sig = mat.std(axis=1, keepdims=True); sig[sig==0] = 1
    return (mat - mu) / sig

for mat, fname, title in [
    (vol_mat,   "day_08a_vol_anomaly.png",   "Volatility Day Anomaly (z-score vs own mean)"),
    (obi_mat,   "day_08b_obi_anomaly.png",   "OBI Day Anomaly"),
    (sp_mat,    "day_08c_spread_anomaly.png", "Spread Day Anomaly"),
    (freq_mat,  "day_08d_freq_anomaly.png",   "Trade Count Day Anomaly"),
]:
    zmat = day_zscore_mat(mat)
    fig, ax = plt.subplots(figsize=(7, 16), facecolor=DARK)
    ax.set_facecolor(MID)
    norm2 = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    im = ax.imshow(zmat, aspect="auto", cmap="RdBu_r", norm=norm2)
    ax.set_xticks([0,1,2]); ax.set_xticklabels(["Day 2","Day 3","Day 4"], color="white", fontsize=10)
    ax.set_yticks(range(len(prod_order)))
    ax.set_yticklabels(prod_order, fontsize=6.5, color="white")
    for i in range(len(prod_order)):
        for j in range(3):
            ax.text(j, i, f"{zmat[i,j]:.1f}", ha="center", va="center",
                    fontsize=5.5, color="black" if abs(zmat[i,j])>1.2 else "white")
    ax.set_title(title, color="white", fontsize=11)
    cb = plt.colorbar(im, ax=ax, fraction=0.02); cb.ax.tick_params(colors="white")
    cb.set_label("Z-score", color="white")
    pos = 0
    for g in GROUPS:
        cnt = sum(1 for p in GROUPS[g] if p in prod_order)
        if cnt == 0: continue
        ax.axhline(pos - 0.5, color="#aaa", lw=0.8, alpha=0.6); pos += cnt
    plt.tight_layout()
    plt.savefig(OUT+fname, dpi=120, bbox_inches="tight", facecolor=DARK)
    plt.close()
print("  done (4 anomaly plots)")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Day-over-day return correlation: do products move together differently per day?
# ─────────────────────────────────────────────────────────────────────────────
print("9. Day-over-day return correlation...")
fig, axes = plt.subplots(1, 3, figsize=(22, 8), facecolor=DARK)
for ax, d in zip(axes, DAYS):
    ax.set_facecolor(MID)
    sub = prices[prices["day"]==d]
    piv = sub.pivot_table(index="timestamp", columns="product", values="ret")
    piv = piv[[c for c in prod_order if c in piv.columns]]
    corr = piv.corr()
    norm3 = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax.imshow(corr.values, cmap="RdBu_r", norm=norm3, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=4.5, color="white")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=4.5, color="white")
    ax.set_title(f"Return Correlation — Day {d}", color="white", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.03).ax.tick_params(colors="white")
    pos = 0
    for g in GROUPS:
        cnt = sum(1 for p in GROUPS[g] if p in corr.columns)
        if cnt == 0: continue
        ax.axhline(pos-0.5, color="#aaa", lw=0.5, alpha=0.5)
        ax.axvline(pos-0.5, color="#aaa", lw=0.5, alpha=0.5)
        pos += cnt
fig.suptitle("Return Correlation Matrix per Day (does co-movement change?)", color="white", fontsize=13)
plt.tight_layout()
plt.savefig(OUT+"day_09_daily_corr_matrix.png", dpi=110, bbox_inches="tight", facecolor=DARK)
plt.close(); print("  done")

# ─────────────────────────────────────────────────────────────────────────────
# 10. OBI rolling mean per day — intraday OBI drift (does it trend?)
# ─────────────────────────────────────────────────────────────────────────────
print("10. Intraday OBI drift per day...")
fig, axes = plt.subplots(5, 2, figsize=(22, 28), facecolor=DARK)
axes = axes.flatten()
ROLL_TS = 50   # rows
for idx, (gname, gprods) in enumerate(GROUPS.items()):
    ax = axes[idx]; ax.set_facecolor(MID); ax.spines[:].set_color("#333")
    avail = [p for p in gprods if p in ALL_PRODUCTS]
    prod = avail[0] if avail else None
    if prod is None: continue
    for d in DAYS:
        sub = prices[(prices["product"]==prod) & (prices["day"]==d)].copy()
        if sub.empty: continue
        obi_roll = sub["obi_l1"].rolling(ROLL_TS, min_periods=10).mean()
        ax.plot(sub["timestamp"].values/1e6, obi_roll.values,
                color=DAY_COLORS[d], lw=1.2, label=DAY_LABELS[d], alpha=0.9)
    ax.axhline(0, color="#555", lw=0.7)
    ax.set_title(f"{gname}\n({prod.split('_')[-1]})", color="white", fontsize=8.5)
    ax.set_xlabel("Timestamp (M)", color="#aaa", fontsize=7)
    ax.set_ylabel("Rolling OBI (50-row)", color="#aaa", fontsize=7)
    ax.tick_params(colors="#aaa", labelsize=6)
    ax.legend(fontsize=7, facecolor="#0f3460", edgecolor="#444", labelcolor="white")
fig.suptitle("Intraday OBI Drift per Day (does order imbalance trend intraday?)",
             color="white", fontsize=13)
plt.tight_layout()
plt.savefig(OUT+"day_10_obi_drift.png", dpi=110, bbox_inches="tight", facecolor=DARK)
plt.close(); print("  done")

# ─────────────────────────────────────────────────────────────────────────────
# 11. All-feature predictive IC per day — grouped bar per feature
# ─────────────────────────────────────────────────────────────────────────────
print("11. Per-feature IC by day...")
FEAT_NAMES = ["obi_l1","spread","depth_bid","depth_ask"]
feat_labels = {"obi_l1":"OBI_L1","spread":"SPREAD",
               "depth_bid":"DEPTH_BID","depth_ask":"DEPTH_ASK"}

# add forward return
prices.sort_values(["product","day","timestamp"], inplace=True)
prices["fwd_ret"] = prices.groupby(["product","day"])["ret"].shift(-1)

ic_feat_day = {f: {d: [] for d in DAYS} for f in FEAT_NAMES}
for prod in ALL_PRODUCTS:
    for d in DAYS:
        sub = prices[(prices["product"]==prod)&(prices["day"]==d)].dropna(
            subset=FEAT_NAMES+["fwd_ret"])
        if len(sub) < 30: continue
        for f in FEAT_NAMES:
            r, _ = stats.pearsonr(sub[f].replace([np.inf,-np.inf], np.nan).fillna(0),
                                  sub["fwd_ret"])
            ic_feat_day[f][d].append(r)

fig, axes = plt.subplots(1, len(FEAT_NAMES), figsize=(18, 7), facecolor=DARK)
width = 0.25
x = np.arange(len(DAYS))
for ax, f in zip(axes, FEAT_NAMES):
    ax.set_facecolor(MID); ax.spines[:].set_color("#333")
    means = [np.mean(ic_feat_day[f][d]) if ic_feat_day[f][d] else 0 for d in DAYS]
    sems  = [np.std(ic_feat_day[f][d])/max(len(ic_feat_day[f][d])**0.5,1) for d in DAYS]
    clr = [DAY_COLORS[d] for d in DAYS]
    ax.bar(x, means, color=clr, alpha=0.85, width=0.6,
           yerr=sems, error_kw=dict(ecolor=ACCENT, lw=1.5, capsize=4))
    ax.axhline(0, color="#666", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([DAY_LABELS[d] for d in DAYS], color="white", fontsize=8)
    ax.set_title(feat_labels[f], color="white", fontsize=10)
    ax.set_ylabel("Mean IC", color="white", fontsize=8)
    ax.tick_params(colors="#aaa")
fig.suptitle("Predictive IC per Feature per Day (avg across products)", color="white", fontsize=13)
plt.tight_layout()
plt.savefig(OUT+"day_11_feature_ic_by_day.png", dpi=120, bbox_inches="tight", facecolor=DARK)
plt.close(); print("  done")

# ─────────────────────────────────────────────────────────────────────────────
# 12. Return distribution shift: violin per product per day (top anomalies)
# ─────────────────────────────────────────────────────────────────────────────
print("12. Return violin per day for most anomalous products...")
# pick top 12 products with highest day-to-day volatility variance
vol_var = np.var(vol_mat, axis=1)
top12_idx = np.argsort(vol_var)[::-1][:12]
top12_prods = [prod_order[i] for i in top12_idx]

fig, axes = plt.subplots(3, 4, figsize=(22, 16), facecolor=DARK)
axes = axes.flatten()
for ax, prod in zip(axes, top12_prods):
    ax.set_facecolor(MID); ax.spines[:].set_color("#333")
    data = [prices[(prices["product"]==prod)&(prices["day"]==d)]["ret"]
            .dropna().clip(-0.003,0.003).values for d in DAYS]
    parts = ax.violinplot(data, positions=[0,1,2], showmedians=True,
                          showextrema=True, widths=0.65)
    for pc, d in zip(parts["bodies"], DAYS):
        pc.set_facecolor(DAY_COLORS[d]); pc.set_alpha(0.75)
    parts["cmedians"].set_color(ACCENT)
    for key in ["cmins","cmaxes","cbars"]: parts[key].set_color("#888")
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(["D2","D3","D4"], color="white", fontsize=8)
    ax.set_title(f"{prod.split('_')[0]}…{prod.split('_')[-1]}", color="white", fontsize=8)
    ax.axhline(0, color="#555", lw=0.5)
    ax.tick_params(colors="#aaa", labelsize=7)

fig.suptitle("Return Distribution per Day — Top 12 Products by Day-to-Day Vol Variance",
             color="white", fontsize=12)
plt.tight_layout()
plt.savefig(OUT+"day_12_return_violin_anomalous.png", dpi=120, bbox_inches="tight", facecolor=DARK)
plt.close(); print("  done")

print(f"\nAll day-wise plots saved to {OUT}")
for f in sorted(os.listdir(OUT)):
    if f.startswith("day_"):
        print(f"  {f}")
