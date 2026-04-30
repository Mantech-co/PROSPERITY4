"""
Dynamic phase-lag analysis via rolling CCF + DTW path cost.
For every product group:
  - rolling window CCF → peak lag over time (animated/static heatmap)
  - DTW warp path visualisation for representative pairs
  - static summary: peak-lag timeline per group
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm

try:
    from dtaidistance import dtw as dtaidtw
    HAS_DTW = True
except ImportError:
    HAS_DTW = False
    print("dtaidistance not found; DTW section skipped")

OUT = "/media/manukrishnan/Mk/prosperity_4/analysis/phase_lag_dynamic/"
os.makedirs(OUT, exist_ok=True)

DARK = "#0d0d1a"; MID = "#12122a"; ACCENT = "#f39c12"; CI_COLOR = "#e74c3c"
plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": MID,
    "text.color": "white", "axes.labelcolor": "#ccc",
    "xtick.color": "#aaa", "ytick.color": "#aaa",
    "axes.edgecolor": "#444", "grid.color": "#2a2a4a",
    "font.size": 8, "axes.titlesize": 9,
})

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
MAX_LAG   = 30
WIN       = 5_000   # rolling window (timestamps)
STEP      = 1_000   # step size
GCOLORS   = {g: cm.tab10(i / 10) for i, g in enumerate(GROUPS)}

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading prices...")
frames = []
DATA = "/media/manukrishnan/Mk/prosperity_4/data/round5/"
for d in [2, 3, 4]:
    pf = pd.read_csv(f"{DATA}prices_round_5_day_{d}.csv", sep=";")
    pf["day"] = d
    frames.append(pf)
prices = pd.concat(frames, ignore_index=True)
prices["gts"] = (prices["day"] - 2) * 1_000_000 + prices["timestamp"]
prices.sort_values(["product", "gts"], inplace=True)

ALL = sorted(p for p in prices["product"].unique() if not p.startswith("VEV_"))
pivot = prices[prices["product"].isin(ALL)].pivot_table(
    index="gts", columns="product", values="mid_price"
)
pivot.sort_index(inplace=True)
pivot.ffill(inplace=True)

returns = pivot.diff().fillna(0)
gts_idx = pivot.index.values
print(f"  {len(ALL)} products, {len(gts_idx)} timestamps")


def ccf_peak_lag(x: np.ndarray, y: np.ndarray, max_lag: int) -> tuple[int, float]:
    x = x - x.mean(); y = y - y.mean()
    sx = x.std() or 1e-10; sy = y.std() or 1e-10
    n  = len(x)
    full  = correlate(x / sx, y / sy, mode="full") / n
    lags  = correlation_lags(n, n, mode="full")
    mask  = (lags >= -max_lag) & (lags <= max_lag)
    sub_lags, sub_corr = lags[mask], full[mask]
    best  = np.argmax(np.abs(sub_corr))
    return int(sub_lags[best]), float(sub_corr[best])


def rolling_peak_lag(xa: np.ndarray, ya: np.ndarray,
                     win: int, step: int, max_lag: int):
    centers, peak_lags, peak_corrs = [], [], []
    T = len(xa)
    for start in range(0, T - win, step):
        end = start + win
        xw, yw = xa[start:end], ya[start:end]
        lag, cor = ccf_peak_lag(xw, yw, max_lag)
        centers.append(start + win // 2)
        peak_lags.append(lag)
        peak_corrs.append(cor)
    return np.array(centers), np.array(peak_lags), np.array(peak_corrs)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Per-group rolling peak-lag timeline (all pairs per group)
# ═════════════════════════════════════════════════════════════════════════════
print("1. Rolling peak-lag timelines per group...")

for gname, gprods in GROUPS.items():
    avail = [p for p in gprods if p in returns.columns]
    if len(avail) < 2:
        continue

    from itertools import combinations
    pairs = list(combinations(avail, 2))
    np_pairs = len(pairs)
    ncols = min(3, np_pairs)
    nrows = (np_pairs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5.5, nrows * 3.2),
                             facecolor=DARK)
    axes = np.array(axes).ravel() if np_pairs > 1 else [axes]
    fig.suptitle(f"{gname} — Rolling Peak Lag (win={WIN}, step={STEP})",
                 color="white", fontsize=11)

    gc = GCOLORS[gname]
    for k, (p1, p2) in enumerate(pairs):
        ax = axes[k]
        ax.set_facecolor(MID)
        xa = returns[p1].values.astype(np.float32)
        ya = returns[p2].values.astype(np.float32)
        centers, lags_t, corrs_t = rolling_peak_lag(xa, ya, WIN, STEP, MAX_LAG)

        # map centers → gts ticks for axis labels
        ts_centers = gts_idx[centers]

        # colour by correlation sign/magnitude
        colors_bar = [ACCENT if c > 0 else CI_COLOR for c in corrs_t]
        ax.bar(range(len(lags_t)), lags_t, color=colors_bar,
               alpha=0.75, width=0.85, zorder=2)
        ax.axhline(0, color="#888", lw=0.6)

        # Mark day boundaries
        day_bounds = [0, 1_000_000, 2_000_000]
        for db in day_bounds:
            if db > ts_centers[0] and db < ts_centers[-1]:
                xpos = np.searchsorted(ts_centers, db)
                ax.axvline(xpos, color="#555", lw=1, ls="--", alpha=0.6)

        ax.set_ylim(-MAX_LAG - 2, MAX_LAG + 2)
        ax.set_ylabel("Peak Lag (ts)", fontsize=7)
        ax.set_xlabel("Window Index", fontsize=7)
        ax.set_title(f"{p1.split('_')[-1]} → {p2.split('_')[-1]}", color="white", fontsize=7, pad=2)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3, axis="y")

        # annotate mean lag
        mean_lag = np.mean(lags_t)
        ax.text(0.98, 0.93, f"μ={mean_lag:.1f}", transform=ax.transAxes,
                ha="right", fontsize=7, color=ACCENT)

    for k in range(np_pairs, len(axes)):
        axes[k].set_visible(False)

    plt.tight_layout()
    slug = gname.lower().replace(" ", "_")
    plt.savefig(OUT + f"rolling_lag_{slug}.png", dpi=110,
                bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"   {gname} done")

# ═════════════════════════════════════════════════════════════════════════════
# 2. Phase-lag heatmap over time (per group, N×N grid of lag timeseries)
# ═════════════════════════════════════════════════════════════════════════════
print("2. Phase-lag dynamic heatmap per group...")

for gname, gprods in GROUPS.items():
    avail = [p for p in gprods if p in returns.columns]
    if len(avail) < 2:
        continue
    n = len(avail)

    # compute rolling peak-lag for all directed pairs
    n_windows = len(range(0, len(gts_idx) - WIN, STEP))
    lag_grid = np.full((n, n, n_windows), np.nan)

    for i, p1 in enumerate(avail):
        for j, p2 in enumerate(avail):
            if i == j:
                continue
            xa = returns[p1].values.astype(np.float32)
            ya = returns[p2].values.astype(np.float32)
            centers, lags_t, _ = rolling_peak_lag(xa, ya, WIN, STEP, MAX_LAG)
            lag_grid[i, j, :len(lags_t)] = lags_t

    fig, axes = plt.subplots(n, n, figsize=(n * 3.8, n * 2.4), facecolor=DARK)
    fig.suptitle(f"{gname} — Dynamic Phase-Lag Heatmap (lag over time)",
                 color="white", fontsize=11)

    norm = TwoSlopeNorm(vmin=-MAX_LAG, vcenter=0, vmax=MAX_LAG)

    for i, p1 in enumerate(avail):
        for j, p2 in enumerate(avail):
            ax = axes[i][j]
            ax.set_facecolor(MID)
            ax.spines[:].set_color("#333")
            ax.tick_params(labelsize=5)
            if i == j:
                # diagonal: rolling autocorrelation at lag=1
                xa = returns[p1].values.astype(np.float32)
                centers, lags_t, corrs_t = rolling_peak_lag(xa, xa, WIN, STEP, MAX_LAG)
                ax.plot(range(len(corrs_t)), corrs_t, color=ACCENT, lw=1)
                ax.set_ylim(-1, 1)
                ax.axhline(0, color="#555", lw=0.5)
                ax.set_title(p1.split("_")[-1], color="white", fontsize=6, pad=1)
            else:
                series = lag_grid[i, j]
                valid = series[~np.isnan(series)]
                if len(valid) == 0:
                    ax.set_visible(False)
                    continue
                # colour each point by lag value
                xs = np.arange(len(valid))
                c_vals = plt.cm.coolwarm(norm(valid))
                ax.scatter(xs, valid, c=c_vals, s=6, zorder=2)
                ax.plot(xs, valid, color="#555", lw=0.4, zorder=1)
                ax.axhline(0, color="#888", lw=0.5)
                ax.set_ylim(-MAX_LAG - 2, MAX_LAG + 2)
                mean_l = np.mean(valid)
                ax.text(0.98, 0.93 if mean_l > 0 else 0.07,
                        f"μ={mean_l:.1f}", transform=ax.transAxes,
                        ha="right", fontsize=5, color=ACCENT)
                ax.set_title(f"{p1.split('_')[-1]}→{p2.split('_')[-1]}",
                             color="white", fontsize=5, pad=1)

            if i == n - 1:
                ax.set_xlabel("Window", fontsize=5, color="#aaa")
            if j == 0:
                ax.set_ylabel("Lag", fontsize=5, color="#aaa")

    plt.tight_layout()
    slug = gname.lower().replace(" ", "_")
    plt.savefig(OUT + f"dynamic_lag_grid_{slug}.png", dpi=110,
                bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"   {gname} done")

# ═════════════════════════════════════════════════════════════════════════════
# 3. Summary: mean phase lag per group (cross-group overview)
# ═════════════════════════════════════════════════════════════════════════════
print("3. Cross-group mean phase lag summary...")

group_reps = {}
for g, prods in GROUPS.items():
    for p in prods:
        if p in returns.columns:
            group_reps[g] = p
            break

gnames = list(group_reps.keys())
G = len(gnames)
mean_lag_mat = np.zeros((G, G))
std_lag_mat  = np.zeros((G, G))

for i, g1 in enumerate(gnames):
    for j, g2 in enumerate(gnames):
        if i == j:
            continue
        p1, p2 = group_reps[g1], group_reps[g2]
        xa = returns[p1].values.astype(np.float32)
        ya = returns[p2].values.astype(np.float32)
        _, lags_t, _ = rolling_peak_lag(xa, ya, WIN, STEP, MAX_LAG)
        mean_lag_mat[i, j] = np.mean(lags_t)
        std_lag_mat[i, j]  = np.std(lags_t)

fig, axes = plt.subplots(1, 2, figsize=(20, 8), facecolor=DARK)

for ax, mat, title, cmap in zip(
    axes,
    [mean_lag_mat, std_lag_mat],
    ["Mean Rolling Peak Lag (cross-group reps)",
     "Std of Rolling Peak Lag (cross-group reps)"],
    ["coolwarm", "YlOrRd"],
):
    ax.set_facecolor(MID)
    if cmap == "coolwarm":
        norm = TwoSlopeNorm(vmin=-MAX_LAG, vcenter=0, vmax=MAX_LAG)
        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
    else:
        im = ax.imshow(mat, cmap=cmap, aspect="auto")

    ax.set_xticks(range(G))
    ax.set_xticklabels([g.split()[0] for g in gnames], rotation=45, ha="right", fontsize=8, color="white")
    ax.set_yticks(range(G))
    ax.set_yticklabels([g.split()[0] for g in gnames], fontsize=8, color="white")
    ax.set_title(title, color="white", fontsize=10)
    cb = plt.colorbar(im, ax=ax, fraction=0.04)
    cb.ax.tick_params(colors="white")

    for ii in range(G):
        for jj in range(G):
            ax.text(jj, ii, f"{mat[ii,jj]:.1f}", ha="center", va="center",
                    fontsize=6.5, color="white")

plt.tight_layout()
plt.savefig(OUT + "cross_group_mean_lag.png", dpi=120,
            bbox_inches="tight", facecolor=DARK)
plt.close()
print("   done")

# ═════════════════════════════════════════════════════════════════════════════
# 4. Intra-group lag stability: violin of peak-lag distribution per pair
# ═════════════════════════════════════════════════════════════════════════════
print("4. Intra-group lag stability violins...")

from itertools import combinations

fig_rows = len(GROUPS)
fig, axes = plt.subplots(fig_rows, 1, figsize=(18, fig_rows * 3.5), facecolor=DARK)
fig.suptitle("Phase Lag Stability — Violin of Rolling Peak Lag per Pair",
             color="white", fontsize=12)

for ax, (gname, gprods) in zip(axes, GROUPS.items()):
    ax.set_facecolor(MID)
    ax.spines[:].set_color("#333")
    avail = [p for p in gprods if p in returns.columns]
    if len(avail) < 2:
        ax.set_visible(False)
        continue

    pairs = list(combinations(avail, 2))
    all_lags = []
    pair_labels = []
    for p1, p2 in pairs:
        xa = returns[p1].values.astype(np.float32)
        ya = returns[p2].values.astype(np.float32)
        _, lags_t, _ = rolling_peak_lag(xa, ya, WIN, STEP, MAX_LAG)
        all_lags.append(lags_t)
        pair_labels.append(f"{p1.split('_')[-1]}\n{p2.split('_')[-1]}")

    vp = ax.violinplot(all_lags, positions=range(len(pairs)),
                       showmedians=True, showextrema=True)
    gc = GCOLORS[gname]
    for body in vp["bodies"]:
        body.set_facecolor(gc)
        body.set_alpha(0.65)
    vp["cmedians"].set_color(ACCENT)
    vp["cmins"].set_color("#888")
    vp["cmaxes"].set_color("#888")
    vp["cbars"].set_color("#888")

    ax.axhline(0, color="#666", lw=0.7, ls="--")
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pair_labels, fontsize=6, color="white")
    ax.set_ylabel("Peak Lag", fontsize=7, color="#aaa")
    ax.set_title(gname, color="white", fontsize=9, pad=3)
    ax.set_ylim(-MAX_LAG - 2, MAX_LAG + 2)
    ax.tick_params(labelsize=6)

plt.tight_layout()
plt.savefig(OUT + "lag_stability_violins.png", dpi=110,
            bbox_inches="tight", facecolor=DARK)
plt.close()
print("   done")

# ═════════════════════════════════════════════════════════════════════════════
# 5. DTW warp paths for representative intra-group pairs
# ═════════════════════════════════════════════════════════════════════════════
if HAS_DTW:
    print("5. DTW warp path visualisations...")
    SAMPLE = 2000  # use a short slice to keep DTW tractable

    fig, axes = plt.subplots(len(GROUPS), 1, figsize=(16, len(GROUPS) * 3.2), facecolor=DARK)
    fig.suptitle("DTW Warp Path Cost over Time (sliding window)",
                 color="white", fontsize=12)

    for ax, (gname, gprods) in zip(axes, GROUPS.items()):
        ax.set_facecolor(MID)
        ax.spines[:].set_color("#333")
        avail = [p for p in gprods if p in returns.columns]
        if len(avail) < 2:
            ax.set_visible(False)
            continue
        p1, p2 = avail[0], avail[1]
        xa = returns[p1].values.astype(np.float32)
        ya = returns[p2].values.astype(np.float32)

        dtw_costs, centers = [], []
        for start in range(0, len(xa) - SAMPLE, SAMPLE // 2):
            end = start + SAMPLE
            xw = xa[start:end]; yw = ya[start:end]
            sx = xw.std() or 1e-6; sy = yw.std() or 1e-6
            cost = dtaidtw.distance_fast(
                (xw / sx).astype(np.float64),
                (yw / sy).astype(np.float64),
                window=MAX_LAG, use_pruning=True)
            dtw_costs.append(cost)
            centers.append(start + SAMPLE // 2)

        ts_c = gts_idx[centers]
        ax.plot(range(len(dtw_costs)), dtw_costs, color=GCOLORS[gname], lw=1.5)
        ax.fill_between(range(len(dtw_costs)), dtw_costs,
                        color=GCOLORS[gname], alpha=0.25)
        for db in [1_000_000, 2_000_000]:
            if db > ts_c[0] and db < ts_c[-1]:
                xpos = np.searchsorted(ts_c, db)
                ax.axvline(xpos, color="#555", lw=1, ls="--", alpha=0.7)
        ax.set_ylabel("DTW cost", fontsize=7, color="#aaa")
        ax.set_title(f"{gname}: {p1.split('_')[-1]} vs {p2.split('_')[-1]}",
                     color="white", fontsize=8, pad=2)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(OUT + "dtw_cost_timeline.png", dpi=110,
                bbox_inches="tight", facecolor=DARK)
    plt.close()
    print("   done")
else:
    print("5. DTW skipped (install dtaidistance for DTW section)")

# ═════════════════════════════════════════════════════════════════════════════
# 6. All-products lag leader board (static snapshot)
# ═════════════════════════════════════════════════════════════════════════════
print("6. All-products lead/lag leaderboard...")

PRODS = [p for p in ALL if p in returns.columns]
N = len(PRODS)
P2G = {p: g for g, ps in GROUPS.items() for p in ps}

ret_arr = returns[PRODS].values.astype(np.float32)
ret_z = ret_arr - ret_arr.mean(0)
stds = ret_z.std(0); stds[stds == 0] = 1e-10
ret_z /= stds
T = ret_z.shape[0]

peak_lag_full = np.zeros((N, N), dtype=np.int8)
for i in range(N):
    x = ret_z[:, i]
    for j in range(N):
        if i == j:
            continue
        y = ret_z[:, j]
        full_c = correlate(x, y, mode="full") / T
        all_l  = correlation_lags(T, T, mode="full")
        mask   = (all_l >= -MAX_LAG) & (all_l <= MAX_LAG)
        best   = np.argmax(np.abs(full_c[mask]))
        peak_lag_full[i, j] = int(all_l[mask][best])

net_lead = -peak_lag_full.mean(1)
lead_ser = pd.Series(net_lead, index=PRODS).sort_values(ascending=False)
lead_clr = [GCOLORS.get(P2G.get(p, ""), "#888") for p in lead_ser.index]

fig, ax = plt.subplots(figsize=(10, 16), facecolor=DARK)
ax.set_facecolor(MID)
ax.spines[:].set_color("#333")
bars = ax.barh(lead_ser.index, lead_ser.values, color=lead_clr, alpha=0.85)
ax.axvline(0, color="#666", lw=0.8)
ax.set_xlabel("Net Lead Score (+ve = leads others)", color="white", fontsize=9)
ax.set_title("All-Products Phase Lead/Lag Leaderboard\n(avg peak CCF lag across all pairs)",
             color="white", fontsize=10)
ax.tick_params(labelsize=6.5, colors="white")
from matplotlib.patches import Patch
legend_els = [Patch(color=GCOLORS[g], label=g) for g in GROUPS]
ax.legend(handles=legend_els, fontsize=7, facecolor="#0a0a1e", edgecolor="#444",
          labelcolor="white", loc="lower right")
plt.tight_layout()
plt.savefig(OUT + "all_products_lead_lag_leaderboard.png", dpi=120,
            bbox_inches="tight", facecolor=DARK)
plt.close()

lead_ser.to_csv(OUT + "lead_lag_score.csv", header=["net_lead_score"])
print("   done")

print(f"\nAll outputs in: {OUT}")
for f in sorted(os.listdir(OUT)):
    print(f"  {f}")
