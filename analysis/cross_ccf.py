"""
Cross-product & cross-class phase-lagged correlation (CCF).
For each pair (within-group and across-group), compute CCF at lags -20..+20.
Outputs:
  - Within-group CCF grid per group
  - Cross-class CCF heatmap (group representative pairs)
  - Max-lag heatmap: which lag gives max |CCF| for every product pair
  - Lead/lag network: who leads whom
"""

import os, warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags

OUT = "/media/manukrishnan/Mk/prosperity_4/analysis_output/deep/"
os.makedirs(OUT, exist_ok=True)

DARK = "#1a1a2e"; MID = "#16213e"; ACCENT = "#f39c12"; CI_COLOR = "#e74c3c"
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
GCOLORS = dict(zip(GROUPS, plt.cm.tab10(np.linspace(0,1,10))))
MAX_LAG = 20

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading...")
pframes = []
for d in [2, 3, 4]:
    pf = pd.read_csv(f"/media/manukrishnan/Mk/prosperity_4/data/prices_round_5_day_{d}.csv", sep=";")
    pf["day"] = d
    pframes.append(pf)
prices = pd.concat(pframes, ignore_index=True)
prices["gts"] = (prices["day"] - 2) * 1_000_000 + prices["timestamp"]
prices.sort_values(["product","gts"], inplace=True)

ALL_PRODUCTS = sorted(p for p in prices["product"].unique() if not p.startswith("VEV_"))
pivot = prices[prices["product"].isin(ALL_PRODUCTS)].pivot_table(
    index="gts", columns="product", values="mid_price")
pivot.sort_index(inplace=True)
pivot.ffill(inplace=True)

# Use returns (zero-mean, unit comparable)
returns = pivot.diff().dropna(how="all")
returns.fillna(0, inplace=True)

def ccf_vec(x, y, max_lag):
    """CCF at lags -max_lag..+max_lag using scipy FFT correlate."""
    x = (x - x.mean()).astype(np.float32)
    y = (y - y.mean()).astype(np.float32)
    sx = np.std(x) or 1e-10
    sy = np.std(y) or 1e-10
    n = len(x)
    full = correlate(x / sx, y / sy, mode="full") / n
    all_lags = correlation_lags(n, n, mode="full")
    mask = (all_lags >= -max_lag) & (all_lags <= max_lag)
    return all_lags[mask], full[mask]

ci95 = 1.96 / np.sqrt(len(returns))
lags_axis = np.arange(-MAX_LAG, MAX_LAG + 1)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Within-group CCF grids
# ─────────────────────────────────────────────────────────────────────────────
print("1. Within-group CCF grids...")
for gname, gprods in GROUPS.items():
    avail = [p for p in gprods if p in returns.columns]
    if len(avail) < 2:
        continue
    pairs = [(p1, p2) for i,p1 in enumerate(avail) for j,p2 in enumerate(avail) if i != j]
    n = len(avail)
    gc = GCOLORS[gname]
    fig, axes = plt.subplots(n, n, figsize=(n*3.2, n*2.8), facecolor=DARK)
    fig.suptitle(f"{gname} — Within-Group CCF (lag -20..+20)", color="white", fontsize=11, y=1.005)

    for i, p1 in enumerate(avail):
        for j, p2 in enumerate(avail):
            ax = axes[i][j]
            ax.set_facecolor(MID); ax.spines[:].set_color("#333")
            ax.tick_params(labelsize=5, colors="#aaa")
            if i == j:
                # diagonal: ACF of this product
                x_arr = returns[p1].values
                _, acf_vals = ccf_vec(x_arr, x_arr, MAX_LAG)
                clr = [CI_COLOR if abs(v) > ci95 else "#3498db" for v in acf_vals]
                ax.bar(lags_axis, acf_vals, color=clr, width=0.7, alpha=0.85)
                ax.set_title(p1.split("_")[-1], color="white", fontsize=7, pad=2)
            else:
                x_arr = returns[p1].values
                y_arr = returns[p2].values
                _, cv = ccf_vec(x_arr, y_arr, MAX_LAG)
                clr = [CI_COLOR if abs(v) > ci95 else gc for v in cv]
                ax.bar(lags_axis, cv, color=clr, width=0.7, alpha=0.85)
                # mark max
                peak_lag = lags_axis[np.argmax(np.abs(cv))]
                peak_val = cv[np.argmax(np.abs(cv))]
                ax.text(0.97, 0.93 if peak_val>0 else 0.07,
                        f"peak@{peak_lag}", transform=ax.transAxes,
                        ha="right", fontsize=5, color=ACCENT)
            ax.axhline(ci95,  color=ACCENT, lw=0.7, ls="--", alpha=0.6)
            ax.axhline(-ci95, color=ACCENT, lw=0.7, ls="--", alpha=0.6)
            ax.axhline(0, color="#555", lw=0.4)
            ax.axvline(0, color="#555", lw=0.4)
            ax.set_xlim(-MAX_LAG-1, MAX_LAG+1)
            ax.set_ylim(-0.4, 0.4)
            if i == n-1:
                ax.set_xlabel(avail[j].split("_")[-1], fontsize=6, color="white")
            if j == 0:
                ax.set_ylabel(avail[i].split("_")[-1], fontsize=6, color="white")

    plt.tight_layout()
    slug = gname.lower().replace(" ","_")
    plt.savefig(OUT + f"ccf_intragroup_{slug}.png", dpi=110, bbox_inches="tight", facecolor=DARK)
    plt.close()
print("   done")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Cross-class CCF: one representative per group, 10×10 grid
# ─────────────────────────────────────────────────────────────────────────────
print("2. Cross-class CCF (rep per group)...")
group_names = list(GROUPS.keys())
reps = {}
for g, prods in GROUPS.items():
    for p in prods:
        if p in returns.columns:
            reps[g] = p
            break

n_g = len(group_names)
fig, axes = plt.subplots(n_g, n_g, figsize=(n_g*3.2, n_g*2.8), facecolor=DARK)
fig.suptitle("Cross-Class CCF (group representatives, lag -20..+20)", color="white", fontsize=11, y=1.005)

for i, g1 in enumerate(group_names):
    for j, g2 in enumerate(group_names):
        ax = axes[i][j]
        ax.set_facecolor(MID); ax.spines[:].set_color("#333")
        ax.tick_params(labelsize=5, colors="#aaa")
        if g1 not in reps or g2 not in reps:
            ax.set_visible(False)
            continue
        p1, p2 = reps[g1], reps[g2]
        x_arr = returns[p1].values
        y_arr = returns[p2].values
        _, cv = ccf_vec(x_arr, y_arr, MAX_LAG)
        gc = GCOLORS[g1] if i != j else "#3498db"
        clr = [CI_COLOR if abs(v) > ci95 else gc for v in cv]
        ax.bar(lags_axis, cv, color=clr, width=0.7, alpha=0.85)
        ax.axhline(ci95,  color=ACCENT, lw=0.6, ls="--", alpha=0.6)
        ax.axhline(-ci95, color=ACCENT, lw=0.6, ls="--", alpha=0.6)
        ax.axhline(0, color="#555", lw=0.4)
        ax.axvline(0, color="#555", lw=0.4)
        ax.set_xlim(-MAX_LAG-1, MAX_LAG+1)
        ax.set_ylim(-0.6, 0.6)
        if i == n_g-1:
            ax.set_xlabel(g2.split()[0], fontsize=6, color="white")
        if j == 0:
            ax.set_ylabel(g1.split()[0], fontsize=6, color="white")
        peak_lag = lags_axis[np.argmax(np.abs(cv))]
        ax.text(0.97, 0.92, f"@{peak_lag}", transform=ax.transAxes,
                ha="right", fontsize=5, color=ACCENT)

plt.tight_layout()
plt.savefig(OUT + "ccf_cross_class.png", dpi=110, bbox_inches="tight", facecolor=DARK)
plt.close()
print("   done")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Max-|CCF| heatmap + peak-lag heatmap (all 50×50)
# ─────────────────────────────────────────────────────────────────────────────
print("3. All-pairs max CCF heatmap (50×50)...")
PRODS = [p for p in ALL_PRODUCTS if p in returns.columns]
N = len(PRODS)
max_ccf  = np.zeros((N, N))
peak_lag = np.zeros((N, N), dtype=int)

ret_arr = returns[PRODS].values.astype(np.float32)  # (T, N)
# z-score each series
ret_z = ret_arr - ret_arr.mean(axis=0)
stds = ret_z.std(axis=0); stds[stds == 0] = 1e-10
ret_z /= stds

T = ret_z.shape[0]

for i in range(N):
    x = ret_z[:, i]
    for j in range(N):
        y = ret_z[:, j]
        # full correlation via scipy (uses FFT internally for long arrays)
        full_corr = correlate(x, y, mode="full") / T
        all_lags  = correlation_lags(len(x), len(y), mode="full")
        # restrict to ±MAX_LAG
        mask = (all_lags >= -MAX_LAG) & (all_lags <= MAX_LAG)
        sub_corr = full_corr[mask]
        sub_lags = all_lags[mask]
        best_idx = np.argmax(np.abs(sub_corr))
        max_ccf[i, j]  = sub_corr[best_idx]
        peak_lag[i, j] = int(sub_lags[best_idx])

# Max CCF heatmap
fig, ax = plt.subplots(figsize=(20, 18), facecolor=DARK)
ax.set_facecolor(MID)
norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
im = ax.imshow(max_ccf, cmap="RdBu_r", norm=norm, aspect="auto")
ax.set_xticks(range(N)); ax.set_xticklabels(PRODS, rotation=90, fontsize=5.5, color="white")
ax.set_yticks(range(N)); ax.set_yticklabels(PRODS, fontsize=5.5, color="white")
ax.set_title(f"Max |CCF| at Optimal Lag ∈ [-{MAX_LAG},{MAX_LAG}] — All Pairs", color="white", fontsize=12)
cb = plt.colorbar(im, ax=ax, fraction=0.015)
cb.ax.tick_params(colors="white"); cb.set_label("Max CCF", color="white")
# group separators
pos = 0
for g in GROUPS:
    cnt = sum(1 for p in GROUPS[g] if p in PRODS)
    if cnt == 0: continue
    ax.axhline(pos - 0.5, color="#aaa", lw=0.6, alpha=0.5)
    ax.axvline(pos - 0.5, color="#aaa", lw=0.6, alpha=0.5)
    pos += cnt
plt.tight_layout()
plt.savefig(OUT + "max_ccf_heatmap.png", dpi=120, bbox_inches="tight", facecolor=DARK)
plt.close()

# Peak lag heatmap
fig, ax = plt.subplots(figsize=(20, 18), facecolor=DARK)
ax.set_facecolor(MID)
im = ax.imshow(peak_lag, cmap="coolwarm", vmin=-MAX_LAG, vmax=MAX_LAG, aspect="auto")
ax.set_xticks(range(N)); ax.set_xticklabels(PRODS, rotation=90, fontsize=5.5, color="white")
ax.set_yticks(range(N)); ax.set_yticklabels(PRODS, fontsize=5.5, color="white")
ax.set_title(f"Peak Lag Heatmap (lag of max |CCF|, blue=leads row, red=lags row)", color="white", fontsize=11)
cb = plt.colorbar(im, ax=ax, fraction=0.015)
cb.ax.tick_params(colors="white"); cb.set_label("Peak Lag", color="white")
pos = 0
for g in GROUPS:
    cnt = sum(1 for p in GROUPS[g] if p in PRODS)
    if cnt == 0: continue
    ax.axhline(pos - 0.5, color="#aaa", lw=0.6, alpha=0.5)
    ax.axvline(pos - 0.5, color="#aaa", lw=0.6, alpha=0.5)
    pos += cnt
plt.tight_layout()
plt.savefig(OUT + "peak_lag_heatmap.png", dpi=120, bbox_inches="tight", facecolor=DARK)
plt.close()
print("   done")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Net lead/lag score per product (does this product lead or lag others?)
# ─────────────────────────────────────────────────────────────────────────────
print("4. Lead/lag net score...")
# For pair (i, j): if peak_lag[i,j] > 0 then j leads i  (y leads x when optimal lag is +k)
# Net lead score for product i = average peak_lag over all j (positive = i tends to lag others)
net_lag = peak_lag.mean(axis=1)   # positive = i tends to lag others = others lead i
net_lead_score = -net_lag          # positive = i tends to LEAD others

lead_df = pd.Series(net_lead_score, index=PRODS).sort_values(ascending=False)
lead_clr = [GCOLORS.get(P2G.get(p,""),"#888") for p in lead_df.index]

fig, ax = plt.subplots(figsize=(12, 9), facecolor=DARK)
ax.set_facecolor(MID); ax.spines[:].set_color("#333")
ax.barh(lead_df.index, lead_df.values, color=lead_clr, alpha=0.85)
ax.axvline(0, color="#666", lw=0.8)
ax.set_xlabel("Net Lead Score (positive = leads others)", color="white")
ax.set_title("Product Lead/Lag Score (avg peak lag across all pairs)", color="white", fontsize=11)
ax.tick_params(labelsize=6.5, colors="white")
from matplotlib.patches import Patch
legend_els = [Patch(color=GCOLORS[g], label=g) for g in GROUPS]
ax.legend(handles=legend_els, fontsize=7, facecolor="#0f3460", edgecolor="#444", labelcolor="white")
plt.tight_layout()
plt.savefig(OUT + "lead_lag_score.png", dpi=120, bbox_inches="tight", facecolor=DARK)
plt.close()

# CSV
lead_df.to_csv(OUT + "lead_lag_score.csv", header=["net_lead_score"])
print("   done")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Asymmetry map: CCF(i,j) vs CCF(j,i) — who leads?
# ─────────────────────────────────────────────────────────────────────────────
print("5. Asymmetry (lag-1 CCF) heatmap...")
# lag1_fwd[i,j] = corr(x_t, y_{t+1}) — how much col j leads row i
lag1_fwd = (ret_z[:-1].T @ ret_z[1:]) / (T - 1)  # (N, N)

fig, ax = plt.subplots(figsize=(20, 18), facecolor=DARK)
ax.set_facecolor(MID)
norm2 = TwoSlopeNorm(vmin=-0.3, vcenter=0, vmax=0.3)
im = ax.imshow(lag1_fwd, cmap="PiYG", norm=norm2, aspect="auto")
ax.set_xticks(range(N)); ax.set_xticklabels(PRODS, rotation=90, fontsize=5.5, color="white")
ax.set_yticks(range(N)); ax.set_yticklabels(PRODS, fontsize=5.5, color="white")
ax.set_title("Lag-1 CCF[i,j]: corr(x_t, y_{t+1})  — column leads row", color="white", fontsize=12)
cb = plt.colorbar(im, ax=ax, fraction=0.015)
cb.ax.tick_params(colors="white"); cb.set_label("CCF at lag +1", color="white")
pos = 0
for g in GROUPS:
    cnt = sum(1 for p in GROUPS[g] if p in PRODS)
    if cnt == 0: continue
    ax.axhline(pos - 0.5, color="#aaa", lw=0.6, alpha=0.5)
    ax.axvline(pos - 0.5, color="#aaa", lw=0.6, alpha=0.5)
    pos += cnt
plt.tight_layout()
plt.savefig(OUT + "lag1_ccf_heatmap.png", dpi=120, bbox_inches="tight", facecolor=DARK)
plt.close()
print("   done")

print(f"\nAll CCF outputs in: {OUT}")
for f in sorted(os.listdir(OUT)):
    if "ccf" in f or "lag" in f or "lead" in f:
        print(f"  {f}")
