"""
Round 5 — Deep inter-product & trade-relation analysis
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm, LogNorm
from matplotlib.patches import Patch
import seaborn as sns
from statsmodels.tsa.stattools import acf, ccf, coint, grangercausalitytests
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from itertools import combinations, product as iproduct
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
colors10 = plt.cm.tab10(np.linspace(0, 1, 10))
group_color = {g: colors10[i] for i, g in enumerate(GROUPS)}

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading ...")
prices = pd.concat([
    pd.read_csv(f"/media/manukrishnan/Mk/prosperity_4/data/prices_round_5_day_{d}.csv", sep=";")
    for d in [2, 3, 4]], ignore_index=True)
prices["global_ts"] = (prices["day"] - 2) * 1_000_000 + prices["timestamp"]
prices.sort_values(["product", "global_ts"], inplace=True)

parts = []
for d in [2, 3, 4]:
    t = pd.read_csv(f"/media/manukrishnan/Mk/prosperity_4/data/trades_round_5_day_{d}.csv", sep=";")
    t["day"] = d
    parts.append(t)
trades = pd.concat(parts, ignore_index=True)
trades["global_ts"] = (trades["day"] - 2) * 1_000_000 + trades["timestamp"]
trades.sort_values(["symbol", "global_ts"], inplace=True)
trades.rename(columns={"symbol": "product"}, inplace=True)
trades["group"] = trades["product"].map(PRODUCT_TO_GROUP)

ALL_PRODUCTS = sorted(prices["product"].unique())

pivot = prices.pivot_table(index="global_ts", columns="product", values="mid_price")
pivot.sort_index(inplace=True)
pivot.ffill(inplace=True)

returns = pivot[ALL_PRODUCTS].pct_change().dropna()
ret_corr = returns.corr()

print(f"Loaded: {len(prices)} price rows, {len(trades)} trade rows")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 32: Hierarchical clustering dendrogram of return correlations
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 32: Dendrogram ...")
dist = 1 - ret_corr.values
dist = (dist + dist.T) / 2
np.fill_diagonal(dist, 0)
dist = np.clip(dist, 0, None)
Z = linkage(squareform(dist), method="ward")

fig, ax = plt.subplots(figsize=(20, 9))
labels_short = [p.replace("GALAXY_SOUNDS_", "GS_").replace("SLEEP_POD_", "SP_")
                 .replace("MICROCHIP_", "MC_").replace("PEBBLES_", "PB_")
                 .replace("ROBOT_", "RB_").replace("UV_VISOR_", "UV_")
                 .replace("TRANSLATOR_", "TR_").replace("PANEL_", "PN_")
                 .replace("OXYGEN_SHAKE_", "OX_").replace("SNACKPACK_", "SN_")
                 for p in ALL_PRODUCTS]
leaf_colors = [group_color[PRODUCT_TO_GROUP[p]] for p in ALL_PRODUCTS]
dn = dendrogram(Z, labels=labels_short, ax=ax, leaf_rotation=90,
                color_threshold=0.6 * max(Z[:, 2]))
ax.set_title("Hierarchical Clustering of Products by Return Correlation", fontweight="bold", fontsize=13)
ax.set_ylabel("Ward Distance")
legend_els = [Patch(color=group_color[g], label=g) for g in GROUPS]
ax.legend(handles=legend_els, fontsize=7, ncol=2, loc="upper right")
fig.tight_layout()
fig.savefig(f"{OUT}/32_dendrogram_clustering.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 33: PCA — first 3 PCs of returns, scatter coloured by group
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 33: PCA of products ...")
ret_clean = returns[ALL_PRODUCTS].dropna(axis=1)
prods_clean = list(ret_clean.columns)
scaler = StandardScaler()
# fit on (T, 50) → components_ shape (n_components, 50)
X_scaled = scaler.fit_transform(ret_clean.values)
pca = PCA(n_components=6)
pca.fit(X_scaled)
# position of each product in PC space = component loadings transposed
prod_coords = pca.components_.T  # (50, 6)

fig, axes = plt.subplots(1, 3, figsize=(21, 7))
pairs_pca = [(0, 1), (0, 2), (1, 2)]
for ax, (i, j) in zip(axes, pairs_pca):
    for gname, gprods in GROUPS.items():
        idxs = [prods_clean.index(p) for p in gprods if p in prods_clean]
        ax.scatter(prod_coords[idxs, i], prod_coords[idxs, j], label=gname,
                   color=group_color[gname], s=60, alpha=0.85, zorder=5)
        for idx in idxs:
            ax.annotate(prods_clean[idx].split("_")[-1],
                        (prod_coords[idx, i], prod_coords[idx, j]),
                        fontsize=5, alpha=0.7, textcoords="offset points", xytext=(2, 2))
    ax.set_xlabel(f"PC{i+1} ({pca.explained_variance_ratio_[i]*100:.1f}%)")
    ax.set_ylabel(f"PC{j+1} ({pca.explained_variance_ratio_[j]*100:.1f}%)")
    ax.set_title(f"PCA: PC{i+1} vs PC{j+1}", fontweight="bold")
axes[0].legend(fontsize=7, ncol=2)
fig.suptitle("PCA of 50 Products — Loadings in Return-Variance Space", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/33_pca_products.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 34: PCA explained variance + PC loadings heatmap
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 34: PCA loadings ...")
pca_full = PCA(n_components=10)
pca_full.fit(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(22, 8))
axes[0].bar(range(1, 11), pca_full.explained_variance_ratio_ * 100, color="steelblue", alpha=0.8)
axes[0].plot(range(1, 11), np.cumsum(pca_full.explained_variance_ratio_) * 100,
             "ro-", lw=1.5, label="Cumulative")
axes[0].set_xlabel("PC")
axes[0].set_ylabel("Explained Variance (%)")
axes[0].set_title("PCA Scree Plot", fontweight="bold")
axes[0].legend()

# loadings: shape (n_components, n_products)
loadings = pd.DataFrame(pca_full.components_[:6, :],
                         columns=prods_clean,
                         index=[f"PC{i+1}" for i in range(6)])
sns.heatmap(loadings, ax=axes[1], cmap="RdBu_r", center=0,
            xticklabels=True, yticklabels=True,
            linewidths=0.3, cbar_kws={"label": "Loading"})
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=90, fontsize=5.5)
axes[1].set_title("PCA Loadings (PC1–PC6)", fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/34_pca_loadings.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 35: Lead-lag cross-correlation for key pairs (CCF)
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 35: Lead-lag CCF ...")
KEY_PAIRS = [
    ("SNACKPACK_PISTACHIO", "SNACKPACK_STRAWBERRY"),
    ("SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA"),
    ("SNACKPACK_RASPBERRY", "SNACKPACK_STRAWBERRY"),
    ("PEBBLES_XL", "PEBBLES_M"),
    ("PEBBLES_XL", "PEBBLES_XS"),
    ("GALAXY_SOUNDS_DARK_MATTER", "GALAXY_SOUNDS_BLACK_HOLES"),
    ("MICROCHIP_OVAL", "MICROCHIP_CIRCLE"),
    ("UV_VISOR_RED", "UV_VISOR_YELLOW"),
    ("ROBOT_DISHES", "ROBOT_IRONING"),
    ("OXYGEN_SHAKE_GARLIC", "OXYGEN_SHAKE_MINT"),
    ("PANEL_4X4", "PANEL_1X2"),
    ("TRANSLATOR_VOID_BLUE", "TRANSLATOR_SPACE_GRAY"),
]
NLAGS_CCF = 30
fig, axes = plt.subplots(3, 4, figsize=(24, 14))
axes = axes.flatten()
for ax, (p1, p2) in zip(axes, KEY_PAIRS):
    r1 = returns[p1].values if p1 in returns else np.zeros(len(returns))
    r2 = returns[p2].values if p2 in returns else np.zeros(len(returns))
    ccf_vals = [np.corrcoef(r1[:-lag], r2[lag:])[0, 1] if lag > 0
                else np.corrcoef(r1, r2)[0, 1]
                for lag in range(NLAGS_CCF + 1)]
    ccf_neg = [np.corrcoef(r2[:-lag], r1[lag:])[0, 1] if lag > 0
               else np.corrcoef(r1, r2)[0, 1]
               for lag in range(NLAGS_CCF + 1)]
    lags_full = list(range(-NLAGS_CCF, NLAGS_CCF + 1))
    ccf_full = list(reversed(ccf_neg[1:])) + ccf_vals
    conf = 1.96 / np.sqrt(len(r1))
    ax.bar(lags_full, ccf_full, width=0.8, alpha=0.7, color="steelblue")
    ax.axhline(conf, color="orange", lw=1, linestyle="--")
    ax.axhline(-conf, color="orange", lw=1, linestyle="--")
    ax.axhline(0, color="white", lw=0.5)
    ax.axvline(0, color="red", lw=0.8, linestyle="--", alpha=0.5)
    sn1 = p1.split("_")[-1][:6]
    sn2 = p2.split("_")[-1][:6]
    ax.set_title(f"{sn1} → {sn2}", fontweight="bold", fontsize=9)
    ax.set_xlabel("Lag (→ means p1 leads p2)")
    ax.set_ylabel("CCF")
fig.suptitle("Lead-Lag Cross-Correlation Between Key Product Pairs", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/35_lead_lag_ccf.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 36: Rolling correlation over time — key pairs
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 36: Rolling correlations ...")
ROLL_CORR_WIN = 5000
INTERESTING_PAIRS = [
    ("SNACKPACK_PISTACHIO", "SNACKPACK_STRAWBERRY"),
    ("SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA"),
    ("SNACKPACK_RASPBERRY", "SNACKPACK_STRAWBERRY"),
    ("PEBBLES_XL", "PEBBLES_M"),
    ("PEBBLES_XL", "PEBBLES_XS"),
    ("PEBBLES_M", "PEBBLES_S"),
    ("GALAXY_SOUNDS_DARK_MATTER", "GALAXY_SOUNDS_SOLAR_FLAMES"),
    ("MICROCHIP_OVAL", "MICROCHIP_CIRCLE"),
]

fig, axes = plt.subplots(4, 2, figsize=(20, 18))
axes = axes.flatten()
for ax, (p1, p2) in zip(axes, INTERESTING_PAIRS):
    r1 = returns[p1]
    r2 = returns[p2]
    roll_corr = r1.rolling(ROLL_CORR_WIN).corr(r2)
    ax.plot(pivot.index[1:] / 1e6, roll_corr.values, lw=0.8, color=group_color.get(PRODUCT_TO_GROUP.get(p1, ""), "steelblue"))
    ax.axhline(0, color="white", lw=0.5)
    ax.fill_between(pivot.index[1:] / 1e6, roll_corr.values, 0, alpha=0.2,
                    color="green" if roll_corr.mean() > 0 else "red")
    sn1 = p1.split("_")[-1]
    sn2 = p2.split("_")[-1]
    ax.set_title(f"{sn1} ↔ {sn2} (win={ROLL_CORR_WIN})", fontweight="bold", fontsize=9)
    ax.set_ylabel("Rolling Corr")
    ax.set_xlabel("Time (M ticks)")
    ax.set_ylim(-1.1, 1.1)
fig.suptitle("Rolling Correlation Over Time — Key Pairs", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/36_rolling_correlation_pairs.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 37: Cointegration p-value heatmap (Engle-Granger, intra-group)
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 37: Cointegration heatmap ...")
fig, axes = plt.subplots(2, 5, figsize=(28, 10))
axes = axes.flatten()
for ax, (gname, gprods) in zip(axes, GROUPS.items()):
    avail = [p for p in gprods if p in pivot.columns]
    n = len(avail)
    coint_mat = np.ones((n, n))
    for i, p1 in enumerate(avail):
        for j, p2 in enumerate(avail):
            if i == j:
                coint_mat[i, j] = 1.0
                continue
            try:
                _, pval, _ = coint(pivot[p1].dropna(), pivot[p2].dropna())
                coint_mat[i, j] = pval
            except Exception:
                coint_mat[i, j] = 1.0
    short = [p.split("_")[-1] for p in avail]
    df_coint = pd.DataFrame(coint_mat, index=short, columns=short)
    sns.heatmap(df_coint, ax=ax, annot=True, fmt=".2f", cmap="RdYlGn_r",
                vmin=0, vmax=0.2, center=0.05,
                xticklabels=short, yticklabels=short,
                annot_kws={"size": 7}, linewidths=0.5, square=True, cbar=False)
    ax.set_title(f"{gname}\nEG Coint p-val", fontweight="bold", fontsize=9)
fig.suptitle("Engle-Granger Cointegration p-values (green=cointegrated)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/37_cointegration_heatmap.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 38: Cointegrated spread Z-score for top cointegrated pairs
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 38: Cointegrated spread Z-scores ...")
COINT_PAIRS = [
    ("SNACKPACK_PISTACHIO", "SNACKPACK_STRAWBERRY"),
    ("SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA"),
    ("SNACKPACK_RASPBERRY", "SNACKPACK_STRAWBERRY"),
    ("PEBBLES_XL", "PEBBLES_M"),
    ("PEBBLES_XL", "PEBBLES_XS"),
    ("PEBBLES_L", "PEBBLES_S"),
    ("GALAXY_SOUNDS_DARK_MATTER", "GALAXY_SOUNDS_SOLAR_FLAMES"),
    ("GALAXY_SOUNDS_BLACK_HOLES", "GALAXY_SOUNDS_SOLAR_WINDS"),
    ("MICROCHIP_OVAL", "MICROCHIP_CIRCLE"),
    ("MICROCHIP_TRIANGLE", "MICROCHIP_RECTANGLE"),
    ("ROBOT_DISHES", "ROBOT_IRONING"),
    ("TRANSLATOR_VOID_BLUE", "TRANSLATOR_SPACE_GRAY"),
]

fig, axes = plt.subplots(4, 3, figsize=(24, 18))
axes = axes.flatten()
WIN_Z = 1000
for ax, (p1, p2) in zip(axes, COINT_PAIRS):
    if p1 not in pivot.columns or p2 not in pivot.columns:
        ax.set_visible(False)
        continue
    # OLS hedge ratio
    from numpy.polynomial import polynomial as P
    y = pivot[p1].values
    x = pivot[p2].values
    valid = ~(np.isnan(y) | np.isnan(x))
    hr = np.polyfit(x[valid], y[valid], 1)[0]
    spread = pivot[p1] - hr * pivot[p2]
    z = (spread - spread.rolling(WIN_Z).mean()) / spread.rolling(WIN_Z).std()
    ax.plot(pivot.index / 1e6, z, lw=0.5, color=group_color.get(PRODUCT_TO_GROUP.get(p1, ""), "steelblue"))
    ax.axhline(2, color="red", lw=1, linestyle="--", alpha=0.7)
    ax.axhline(-2, color="green", lw=1, linestyle="--", alpha=0.7)
    ax.axhline(0, color="white", lw=0.5)
    sn1 = p1.split("_")[-1][:8]
    sn2 = p2.split("_")[-1][:8]
    ax.set_title(f"{sn1} − {hr:.2f}×{sn2}", fontweight="bold", fontsize=8)
    ax.set_ylabel("Z-score")
    ax.set_xlabel("Time (M ticks)")
    ax.set_ylim(-5, 5)
fig.suptitle(f"Cointegrated Spread Z-score (win={WIN_Z}) — Trading Signals", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/38_coint_spread_zscore.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 39: Granger causality matrix (intra-group, lag=5)
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 39: Granger causality ...")
GRANGER_LAG = 5
fig, axes = plt.subplots(2, 5, figsize=(28, 10))
axes = axes.flatten()
for ax, (gname, gprods) in zip(axes, GROUPS.items()):
    avail = [p for p in gprods if p in returns.columns]
    n = len(avail)
    gc_mat = np.ones((n, n))
    for i, p1 in enumerate(avail):
        for j, p2 in enumerate(avail):
            if i == j:
                gc_mat[i, j] = 1.0
                continue
            try:
                data = pd.concat([returns[p2], returns[p1]], axis=1).dropna()
                res = grangercausalitytests(data, maxlag=GRANGER_LAG, verbose=False)
                pvals = [res[lag][0]["ssr_ftest"][1] for lag in range(1, GRANGER_LAG + 1)]
                gc_mat[i, j] = min(pvals)
            except Exception:
                gc_mat[i, j] = 1.0
    short = [p.split("_")[-1] for p in avail]
    df_gc = pd.DataFrame(gc_mat, index=short, columns=short)
    sns.heatmap(df_gc, ax=ax, annot=True, fmt=".2f", cmap="RdYlGn_r",
                vmin=0, vmax=0.2, center=0.05,
                xticklabels=short, yticklabels=short,
                annot_kws={"size": 7}, linewidths=0.5, square=True, cbar=False)
    ax.set_title(f"{gname}\n[row] Granger-causes [col]", fontweight="bold", fontsize=8)
fig.suptitle(f"Granger Causality p-values (lag≤{GRANGER_LAG}, green=significant)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/39_granger_causality.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 40: Price ratio (p1/p2) per group — detect structural divergence
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 40: Price ratios ...")
fig, axes = plt.subplots(5, 2, figsize=(20, 25))
axes = axes.flatten()
for ax, (gname, gprods) in zip(axes, GROUPS.items()):
    avail = [p for p in gprods if p in pivot.columns]
    if len(avail) < 2:
        ax.set_visible(False)
        continue
    base = avail[0]
    for p in avail[1:]:
        ratio = pivot[p] / pivot[base]
        ax.plot(pivot.index / 1e6, ratio, label=f"{p.split('_')[-1]}/{base.split('_')[-1]}", lw=0.7)
    ax.axhline(1, color="white", lw=0.5, linestyle="--")
    ax.set_title(gname, fontweight="bold", fontsize=11)
    ax.set_ylabel("Price Ratio")
    ax.set_xlabel("Time (M ticks)")
    ax.legend(fontsize=7, ncol=2)
fig.suptitle("Price Ratios Within Groups (relative to first product)", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/40_price_ratios.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 41: Cross-product return scatter matrix — Snack Packs (most interesting)
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 41: Snack pack pair scatter matrix ...")
snack = GROUPS["Snack Packs"]
snack_avail = [p for p in snack if p in returns.columns]
n = len(snack_avail)
fig, axes = plt.subplots(n, n, figsize=(16, 16))
for i, p1 in enumerate(snack_avail):
    for j, p2 in enumerate(snack_avail):
        ax = axes[i][j]
        if i == j:
            returns[p1].plot.kde(ax=ax, color="steelblue", lw=1.5)
            ax.set_ylabel("")
        else:
            x = returns[p2].values
            y = returns[p1].values
            ax.scatter(x, y, s=0.3, alpha=0.3, color="steelblue")
            slope, intercept, r, _, _ = stats.linregress(x, y)
            xx = np.array([x.min(), x.max()])
            ax.plot(xx, slope * xx + intercept, "r-", lw=1)
            ax.text(0.05, 0.92, f"r={r:.2f}", transform=ax.transAxes, fontsize=7, color="orange")
        if i == n - 1:
            ax.set_xlabel(p2.split("_")[-1], fontsize=8)
        if j == 0:
            ax.set_ylabel(p1.split("_")[-1], fontsize=8)
        ax.tick_params(labelsize=5)
fig.suptitle("Snack Pack — Return Pair Scatter Matrix", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/41_snackpack_scatter_matrix.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 42: Cross-product return scatter matrix — Pebbles
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 42: Pebbles scatter matrix ...")
peb = GROUPS["Pebbles"]
peb_avail = [p for p in peb if p in returns.columns]
n = len(peb_avail)
fig, axes = plt.subplots(n, n, figsize=(16, 16))
for i, p1 in enumerate(peb_avail):
    for j, p2 in enumerate(peb_avail):
        ax = axes[i][j]
        if i == j:
            returns[p1].plot.kde(ax=ax, color="salmon", lw=1.5)
            ax.set_ylabel("")
        else:
            x = returns[p2].values
            y = returns[p1].values
            ax.scatter(x, y, s=0.3, alpha=0.3, color="salmon")
            slope, intercept, r, _, _ = stats.linregress(x, y)
            xx = np.array([x.min(), x.max()])
            ax.plot(xx, slope * xx + intercept, "b-", lw=1)
            ax.text(0.05, 0.92, f"r={r:.2f}", transform=ax.transAxes, fontsize=7, color="orange")
        if i == n - 1:
            ax.set_xlabel(p2.split("_")[-1], fontsize=8)
        if j == 0:
            ax.set_ylabel(p1.split("_")[-1], fontsize=8)
        ax.tick_params(labelsize=5)
fig.suptitle("Pebbles — Return Pair Scatter Matrix", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/42_pebbles_scatter_matrix.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 43: Partial correlation matrix (remove first PC factor)
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 43: Partial correlation (market-factor removed) ...")
ret_arr = returns[ALL_PRODUCTS].values
market = ret_arr.mean(axis=1, keepdims=True)
residuals = ret_arr - market
resid_df = pd.DataFrame(residuals, columns=ALL_PRODUCTS)
partial_corr = resid_df.corr()

fig, ax = plt.subplots(figsize=(22, 20))
im = ax.imshow(partial_corr.values, cmap="RdYlGn",
               norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), aspect="auto")
ax.set_xticks(range(len(ALL_PRODUCTS)))
ax.set_yticks(range(len(ALL_PRODUCTS)))
ax.set_xticklabels(ALL_PRODUCTS, rotation=90, fontsize=5.5)
ax.set_yticklabels(ALL_PRODUCTS, fontsize=5.5)
plt.colorbar(im, ax=ax, fraction=0.03)
ax.set_title("Partial Correlation (market factor removed) — 50×50", fontweight="bold", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/43_partial_correlation_heatmap.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 44: Trade synchronisation — co-occurrence heatmap
# (do trades in product A and B happen at same timestamp?)
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 44: Trade synchronisation ...")
trade_ts_sets = {p: set(trades[trades["product"] == p]["global_ts"].values) for p in ALL_PRODUCTS}
sync_mat = np.zeros((len(ALL_PRODUCTS), len(ALL_PRODUCTS)))
for i, p1 in enumerate(ALL_PRODUCTS):
    for j, p2 in enumerate(ALL_PRODUCTS):
        if i > j:
            sync_mat[i, j] = sync_mat[j, i]
            continue
        union = trade_ts_sets[p1] | trade_ts_sets[p2]
        inter = trade_ts_sets[p1] & trade_ts_sets[p2]
        sync_mat[i, j] = len(inter) / len(union) if union else 0

sync_df = pd.DataFrame(sync_mat, index=ALL_PRODUCTS, columns=ALL_PRODUCTS)
fig, ax = plt.subplots(figsize=(22, 20))
im = ax.imshow(sync_mat, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(ALL_PRODUCTS)))
ax.set_yticks(range(len(ALL_PRODUCTS)))
ax.set_xticklabels(ALL_PRODUCTS, rotation=90, fontsize=5.5)
ax.set_yticklabels(ALL_PRODUCTS, fontsize=5.5)
plt.colorbar(im, ax=ax, fraction=0.03, label="Jaccard Similarity")
ax.set_title("Trade Timestamp Co-occurrence (Jaccard) — Do they trade together?", fontweight="bold", fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT}/44_trade_synchronisation.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 45: Trade burst detection — count per 1000-tick window, all products
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 45: Trade burst heatmap ...")
BIN = 1000
max_ts = int(trades["global_ts"].max())
bins = np.arange(0, max_ts + BIN, BIN)
trade_counts = {}
for p in ALL_PRODUCTS:
    pdata = trades[trades["product"] == p]["global_ts"].values
    counts, _ = np.histogram(pdata, bins=bins)
    trade_counts[p] = counts

tc_df = pd.DataFrame(trade_counts, index=bins[:-1])
tc_df = tc_df.T  # shape (50, T)
tc_df_smooth = tc_df.rolling(50, axis=1).mean()

fig, ax = plt.subplots(figsize=(24, 14))
im = ax.imshow(tc_df_smooth.values, aspect="auto", cmap="hot",
               vmin=0, vmax=np.percentile(tc_df_smooth.values[~np.isnan(tc_df_smooth.values)], 99),
               interpolation="nearest")
ax.set_yticks(range(len(ALL_PRODUCTS)))
ax.set_yticklabels(ALL_PRODUCTS, fontsize=5.5)
n_xticks = 10
xtick_pos = np.linspace(0, tc_df_smooth.shape[1] - 1, n_xticks, dtype=int)
ax.set_xticks(xtick_pos)
ax.set_xticklabels([f"{bins[i]/1e6:.1f}M" for i in xtick_pos], rotation=45)
ax.set_xlabel("Time")
plt.colorbar(im, ax=ax, fraction=0.02, label="Trade Count (smoothed)")
ax.set_title("Trade Activity Heatmap (1k-tick bins, smoothed) — Burst Detection", fontweight="bold", fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT}/45_trade_burst_heatmap.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 46: Price return AFTER trade — immediate impact by product
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 46: Post-trade price impact ...")
HORIZON = 100  # ticks after trade
post_impact = {}
for p in ALL_PRODUCTS:
    if p not in pivot.columns:
        continue
    tr = trades[trades["product"] == p]["global_ts"].values
    impacts = []
    for ts in tr:
        idx_after = pivot.index.searchsorted(ts + HORIZON)
        idx_at = pivot.index.searchsorted(ts)
        if idx_at >= len(pivot) or idx_after >= len(pivot):
            continue
        p_at = pivot[p].iloc[idx_at]
        p_after = pivot[p].iloc[idx_after]
        if p_at != 0 and not np.isnan(p_at) and not np.isnan(p_after):
            impacts.append((p_after - p_at) / p_at)
    post_impact[p] = impacts

fig, axes = plt.subplots(5, 2, figsize=(20, 25))
axes = axes.flatten()
for ax, (gname, gprods) in zip(axes, GROUPS.items()):
    for p in gprods:
        if p not in post_impact or not post_impact[p]:
            continue
        imp = np.array(post_impact[p])
        imp = imp[np.abs(imp) < np.percentile(np.abs(imp), 99)]  # clip outliers
        ax.hist(imp * 1e4, bins=50, alpha=0.5, label=p.split("_")[-1], density=True)
    ax.axvline(0, color="white", lw=0.8)
    ax.set_title(gname, fontsize=11, fontweight="bold")
    ax.set_xlabel(f"Return (bps) over {HORIZON} ticks post-trade")
    ax.legend(fontsize=7, ncol=2)
fig.suptitle(f"Price Return Distribution {HORIZON} Ticks After Trade", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/46_post_trade_impact.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 47: Trade price deviation from mid — signed (above/below mid)
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 47: Trade price deviation from mid ...")
dev_data = []
for p in ALL_PRODUCTS:
    if p not in pivot.columns:
        continue
    tr = trades[trades["product"] == p].copy()
    for _, row in tr.iterrows():
        idx = pivot.index.searchsorted(row["global_ts"])
        if idx >= len(pivot):
            continue
        mid = pivot[p].iloc[idx]
        if not np.isnan(mid) and mid != 0:
            dev = (row["price"] - mid) / mid * 1e4
            dev_data.append({"product": p, "group": PRODUCT_TO_GROUP.get(p, ""), "dev_bps": dev,
                             "quantity": row["quantity"]})

dev_df = pd.DataFrame(dev_data)

fig, axes = plt.subplots(2, 5, figsize=(28, 12))
axes = axes.flatten()
for ax, (gname, gprods) in zip(axes, GROUPS.items()):
    gdata = dev_df[dev_df["group"] == gname]
    if gdata.empty:
        ax.set_visible(False)
        continue
    gdata2 = gdata.copy()
    gdata2["short"] = gdata2["product"].str.split("_").str[-1]
    order = sorted(gdata2["short"].unique())
    sns.boxplot(data=gdata2, x="short", y="dev_bps", ax=ax, order=order,
                palette="coolwarm", showfliers=False)
    ax.axhline(0, color="white", lw=0.8, linestyle="--")
    ax.set_title(gname, fontweight="bold", fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("Trade price − mid (bps)")
    ax.tick_params(axis="x", rotation=30, labelsize=7)
fig.suptitle("Trade Price Deviation from Mid-Price (bps)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/47_trade_deviation_from_mid.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 48: Volume-weighted price impact (Kyle's lambda proxy) per product
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 48: Kyle lambda ...")
kyle_data = []
for p in ALL_PRODUCTS:
    if p not in pivot.columns:
        continue
    tr = trades[trades["product"] == p].sort_values("global_ts").copy()
    if len(tr) < 10:
        continue
    tr["price_chg"] = tr["price"].diff().abs()
    tr = tr.dropna(subset=["price_chg"])
    slope, intercept, r, pval, _ = stats.linregress(tr["quantity"], tr["price_chg"])
    kyle_data.append({"product": p, "lambda": slope, "r2": r**2,
                      "group": PRODUCT_TO_GROUP.get(p, "")})

kyle_df = pd.DataFrame(kyle_data).set_index("product").sort_values("lambda")

fig, axes = plt.subplots(1, 2, figsize=(20, 9))
kc = [group_color[kyle_df.loc[p, "group"]] for p in kyle_df.index]
axes[0].barh(kyle_df.index, kyle_df["lambda"], color=kc, alpha=0.85)
axes[0].set_xlabel("Kyle λ (slope: volume → |price change|)")
axes[0].set_title("Kyle's Lambda — Market Impact per Product", fontweight="bold")
axes[0].tick_params(axis="y", labelsize=6)
axes[0].axvline(0, color="white", lw=0.5)

axes[1].barh(kyle_df.index, kyle_df["r2"], color=kc, alpha=0.85)
axes[1].set_xlabel("R² of volume → |price change| regression")
axes[1].set_title("R² of Price Impact Regression", fontweight="bold")
axes[1].tick_params(axis="y", labelsize=6)

legend_els = [Patch(color=group_color[g], label=g) for g in GROUPS]
axes[0].legend(handles=legend_els, fontsize=7, ncol=1)
fig.tight_layout()
fig.savefig(f"{OUT}/48_kyle_lambda.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 49: Trade ACF — is trade arrival clustered? (Hawkes-like)
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 49: Trade arrival ACF ...")
fig, axes = plt.subplots(2, 5, figsize=(28, 12))
axes = axes.flatten()
BIN_TRADE_ACF = 500
for ax, (gname, gprods) in zip(axes, GROUPS.items()):
    rep = next((p for p in gprods if not trades[trades["product"] == p].empty), None)
    if rep is None:
        ax.set_visible(False)
        continue
    pdata = trades[trades["product"] == rep].sort_values("global_ts")
    max_ts2 = int(pdata["global_ts"].max())
    bins2 = np.arange(0, max_ts2 + BIN_TRADE_ACF, BIN_TRADE_ACF)
    counts2, _ = np.histogram(pdata["global_ts"].values, bins=bins2)
    acf_vals, ci = acf(counts2, nlags=40, alpha=0.05, fft=True)
    lags2 = np.arange(41)
    ax.bar(lags2, acf_vals, color=group_color[gname], alpha=0.8, width=0.8)
    ax.fill_between(lags2, ci[:, 0] - acf_vals, ci[:, 1] - acf_vals, alpha=0.3, color="orange")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"{gname}\n{rep.split('_')[-1]}", fontweight="bold", fontsize=9)
    ax.set_xlabel(f"Lag ({BIN_TRADE_ACF}-tick bins)")
    ax.set_ylabel("ACF")
fig.suptitle("Trade Arrival ACF — Clustering / Hawkes Process Detection", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/49_trade_arrival_acf.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 50: Group-level cross-trade synchronisation (macro view)
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 50: Group trade synchronisation ...")
group_ts_sets = {}
for gname, gprods in GROUPS.items():
    ts_set = set()
    for p in gprods:
        ts_set |= trade_ts_sets.get(p, set())
    group_ts_sets[gname] = ts_set

gnames = list(GROUPS.keys())
n_g = len(gnames)
gsync_mat = np.zeros((n_g, n_g))
for i, g1 in enumerate(gnames):
    for j, g2 in enumerate(gnames):
        union = group_ts_sets[g1] | group_ts_sets[g2]
        inter = group_ts_sets[g1] & group_ts_sets[g2]
        gsync_mat[i, j] = len(inter) / len(union) if union else 0

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(pd.DataFrame(gsync_mat, index=gnames, columns=gnames),
            ax=ax, annot=True, fmt=".2f", cmap="YlOrRd",
            linewidths=0.5, square=True, cbar_kws={"label": "Jaccard"})
ax.set_title("Group-Level Trade Timestamp Synchronisation (Jaccard)", fontweight="bold", fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT}/50_group_trade_synchronisation.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 51: Intra-day trade rhythm — histogram of trade times mod 100k
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 51: Intra-day trade rhythm ...")
trades["intraday_ts"] = trades["timestamp"] % 1_000_000
fig, axes = plt.subplots(5, 2, figsize=(20, 25))
axes = axes.flatten()
for ax, (gname, gprods) in zip(axes, GROUPS.items()):
    gdata = trades[trades["product"].isin(gprods)]
    for p in gprods:
        pdata = gdata[gdata["product"] == p]["intraday_ts"].values
        if len(pdata) == 0:
            continue
        ax.hist(pdata, bins=50, alpha=0.4, density=True, label=p.split("_")[-1])
    ax.set_title(gname, fontsize=11, fontweight="bold")
    ax.set_xlabel("Timestamp within day")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7, ncol=2)
fig.suptitle("Intra-Day Trade Rhythm (when within day do trades cluster?)", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/51_intraday_trade_rhythm.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 52: Volume profile — volume at each price level (per group)
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 52: Volume profile ...")
fig, axes = plt.subplots(5, 2, figsize=(20, 25))
axes = axes.flatten()
for ax, (gname, gprods) in zip(axes, GROUPS.items()):
    for p in gprods:
        pdata = trades[trades["product"] == p]
        if pdata.empty:
            continue
        ax.barh(pdata.groupby("price")["quantity"].sum().index,
                pdata.groupby("price")["quantity"].sum().values,
                height=0.8, alpha=0.4, label=p.split("_")[-1])
    ax.set_title(gname, fontsize=11, fontweight="bold")
    ax.set_xlabel("Total Volume")
    ax.set_ylabel("Price Level")
    ax.legend(fontsize=7, ncol=2)
fig.suptitle("Volume Profile (volume traded at each price level)", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/52_volume_profile.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 53: Bid depth & ask depth imbalance over time
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 53: Order book imbalance ...")
prices["bid_depth"] = prices["bid_volume_1"].fillna(0) + prices["bid_volume_2"].fillna(0) + prices["bid_volume_3"].fillna(0)
prices["ask_depth"] = prices["ask_volume_1"].fillna(0) + prices["ask_volume_2"].fillna(0) + prices["ask_volume_3"].fillna(0)
prices["obi"] = (prices["bid_depth"] - prices["ask_depth"]) / (prices["bid_depth"] + prices["ask_depth"] + 1e-9)

fig, axes = plt.subplots(5, 2, figsize=(20, 25))
axes = axes.flatten()
for ax, (gname, gprods) in zip(axes, GROUPS.items()):
    for p in gprods:
        pdata = prices[prices["product"] == p].set_index("global_ts")["obi"]
        pdata_smooth = pdata.rolling(500).mean()
        ax.plot(pdata.index / 1e6, pdata_smooth.values, label=p.split("_")[-1], lw=0.7)
    ax.axhline(0, color="white", lw=0.5)
    ax.set_title(gname, fontsize=11, fontweight="bold")
    ax.set_ylabel("OBI (bid−ask)/(bid+ask)")
    ax.set_xlabel("Time (M ticks)")
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(-1, 1)
fig.suptitle("Order Book Imbalance (OBI) Over Time", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/53_order_book_imbalance.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 54: OBI → next return predictability per product
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 54: OBI predictive power ...")
obi_pred = []
for p in ALL_PRODUCTS:
    if p not in pivot.columns:
        continue
    pdata = prices[prices["product"] == p].set_index("global_ts")[["obi", "mid_price"]].sort_index()
    pdata["next_ret"] = pdata["mid_price"].pct_change().shift(-1)
    pdata = pdata.dropna()
    if len(pdata) < 100:
        continue
    slope, intercept, r, pval, _ = stats.linregress(pdata["obi"], pdata["next_ret"])
    obi_pred.append({"product": p, "r": r, "r2": r**2, "slope": slope,
                     "pval": pval, "group": PRODUCT_TO_GROUP.get(p, "")})

obi_pred_df = pd.DataFrame(obi_pred).set_index("product").sort_values("r")

fig, axes = plt.subplots(1, 2, figsize=(20, 9))
oc = [group_color[obi_pred_df.loc[p, "group"]] for p in obi_pred_df.index]
axes[0].barh(obi_pred_df.index, obi_pred_df["r"], color=oc, alpha=0.85)
axes[0].axvline(0, color="white", lw=0.8)
axes[0].set_xlabel("Pearson r (OBI → next return)")
axes[0].set_title("OBI Predictive Power per Product", fontweight="bold")
axes[0].tick_params(axis="y", labelsize=6)

axes[1].barh(obi_pred_df.index, -np.log10(obi_pred_df["pval"].clip(1e-300)), color=oc, alpha=0.85)
axes[1].axvline(-np.log10(0.05), color="orange", lw=1.5, linestyle="--", label="p=0.05")
axes[1].set_xlabel("−log₁₀(p-value)")
axes[1].set_title("Statistical Significance of OBI Signal", fontweight="bold")
axes[1].tick_params(axis="y", labelsize=6)
axes[1].legend()

legend_els = [Patch(color=group_color[g], label=g) for g in GROUPS]
axes[0].legend(handles=legend_els, fontsize=7, ncol=1)
fig.tight_layout()
fig.savefig(f"{OUT}/54_obi_predictive_power.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 55: Spread auto-correlation — does spread persistence differ by group?
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 55: Spread autocorrelation ...")
prices["spread"] = prices["ask_price_1"] - prices["bid_price_1"]
fig, axes = plt.subplots(2, 5, figsize=(28, 12))
axes = axes.flatten()
for ax, (gname, gprods) in zip(axes, GROUPS.items()):
    rep = next((p for p in gprods if p in pivot.columns), None)
    if rep is None:
        ax.set_visible(False)
        continue
    s = prices[prices["product"] == rep]["spread"].dropna()
    acf_vals, ci = acf(s, nlags=50, alpha=0.05, fft=True)
    lags = np.arange(51)
    ax.bar(lags, acf_vals, color=group_color[gname], alpha=0.8, width=0.8)
    ax.fill_between(lags, ci[:, 0] - acf_vals, ci[:, 1] - acf_vals, alpha=0.3, color="orange")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"{gname}\n{rep.split('_')[-1]}", fontweight="bold", fontsize=9)
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
fig.suptitle("Bid-Ask Spread Autocorrelation per Group", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/55_spread_acf.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 56: Lag-1 return autocorrelation per product (momentum vs reversal)
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 56: Lag-1 return autocorrelation ...")
lag1_ac = {}
for p in ALL_PRODUCTS:
    if p not in returns.columns:
        continue
    r = returns[p].dropna()
    lag1_ac[p] = r.autocorr(lag=1)

lag1_df = pd.Series(lag1_ac).sort_values()
fig, ax = plt.subplots(figsize=(18, 8))
bar_colors_ac = [group_color[PRODUCT_TO_GROUP.get(p, list(GROUPS.keys())[0])] for p in lag1_df.index]
ax.bar(lag1_df.index, lag1_df.values, color=bar_colors_ac, alpha=0.85)
ax.axhline(0, color="white", lw=0.8)
ax.set_xticks(range(len(lag1_df)))
ax.set_xticklabels(lag1_df.index, rotation=90, fontsize=7)
ax.set_ylabel("Lag-1 Autocorrelation of Returns")
ax.set_title("Lag-1 Return Autocorrelation (negative=reversal, positive=momentum)", fontweight="bold")
legend_els = [Patch(color=group_color[g], label=g) for g in GROUPS]
ax.legend(handles=legend_els, fontsize=8, ncol=2)
conf95 = 1.96 / np.sqrt(len(returns))
ax.axhline(conf95, color="orange", lw=1.2, linestyle="--", label="95% CI")
ax.axhline(-conf95, color="orange", lw=1.2, linestyle="--")
fig.tight_layout()
fig.savefig(f"{OUT}/56_lag1_return_autocorr.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 57: Trade flow — volume in 10k-tick windows corr across products
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 57: Trade flow correlation ...")
BIN_F = 10_000
flow = {}
for p in ALL_PRODUCTS:
    pdata = trades[trades["product"] == p][["global_ts", "quantity"]].copy()
    pdata["bin"] = (pdata["global_ts"] // BIN_F) * BIN_F
    vol_by_bin = pdata.groupby("bin")["quantity"].sum()
    flow[p] = vol_by_bin

flow_df = pd.DataFrame(flow).fillna(0)
flow_corr = flow_df.corr()

fig, ax = plt.subplots(figsize=(22, 20))
im = ax.imshow(flow_corr.values, cmap="RdYlGn",
               norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), aspect="auto")
ax.set_xticks(range(len(ALL_PRODUCTS)))
ax.set_yticks(range(len(ALL_PRODUCTS)))
ax.set_xticklabels(ALL_PRODUCTS, rotation=90, fontsize=5.5)
ax.set_yticklabels(ALL_PRODUCTS, fontsize=5.5)
plt.colorbar(im, ax=ax, fraction=0.03)
ax.set_title("Trade Volume Flow Correlation (10k-tick bins) — 50×50", fontweight="bold", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/57_trade_flow_correlation.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 58: Group-level trade flow correlation
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 58: Group-level flow correlation ...")
group_flow = {}
for gname, gprods in GROUPS.items():
    avail = [p for p in gprods if p in flow_df.columns]
    if avail:
        group_flow[gname] = flow_df[avail].sum(axis=1)
group_flow_df = pd.DataFrame(group_flow)
gflow_corr = group_flow_df.corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(gflow_corr, ax=ax, annot=True, fmt=".2f", cmap="RdYlGn",
            vmin=-1, vmax=1, center=0, linewidths=0.5, square=True,
            annot_kws={"size": 10})
ax.set_title("Group-Level Trade Flow Correlation", fontweight="bold", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/58_group_flow_correlation.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 59: Multi-horizon return correlations (compare 1t, 10t, 100t, 500t)
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 59: Multi-horizon correlations ...")
HORIZONS = [1, 10, 100, 500]
fig, axes = plt.subplots(2, 2, figsize=(22, 18))
axes = axes.flatten()
for ax, h in zip(axes, HORIZONS):
    ret_h = pivot[ALL_PRODUCTS].pct_change(h).dropna()
    corr_h = ret_h.corr()
    im = ax.imshow(corr_h.values, cmap="RdYlGn",
                   norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), aspect="auto")
    ax.set_xticks(range(len(ALL_PRODUCTS)))
    ax.set_yticks(range(len(ALL_PRODUCTS)))
    ax.set_xticklabels(ALL_PRODUCTS, rotation=90, fontsize=4.5)
    ax.set_yticklabels(ALL_PRODUCTS, fontsize=4.5)
    plt.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title(f"{h}-tick Return Correlation", fontweight="bold", fontsize=11)
fig.suptitle("Return Correlations at Multiple Horizons", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/59_multihorizon_correlation.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 60: Intra-group rolling correlation between all pairs (all 10 combos shown)
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 60: Intra-group all-pairs rolling correlation ...")
fig, axes = plt.subplots(5, 2, figsize=(20, 28))
axes = axes.flatten()
ROLL_WIN2 = 10_000
for ax, (gname, gprods) in zip(axes, GROUPS.items()):
    avail = [p for p in gprods if p in returns.columns]
    combos = list(combinations(avail, 2))
    palette = plt.cm.tab10(np.linspace(0, 1, len(combos)))
    for (p1, p2), c in zip(combos, palette):
        rc = returns[p1].rolling(ROLL_WIN2).corr(returns[p2])
        ax.plot(pivot.index[1:] / 1e6, rc.values, lw=0.5, alpha=0.7, color=c,
                label=f"{p1.split('_')[-1][:4]}↔{p2.split('_')[-1][:4]}")
    ax.axhline(0, color="white", lw=0.5)
    ax.set_title(gname, fontsize=11, fontweight="bold")
    ax.set_ylabel("Rolling Corr")
    ax.set_xlabel("Time (M ticks)")
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=6, ncol=2)
fig.suptitle(f"All Intra-Group Rolling Correlations (win={ROLL_WIN2})", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/60_intragroup_all_rolling_correlations.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 61: Bid-ask spread vs trade volume scatter — market quality
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 61: Spread vs volume ...")
spread_mean2 = prices.groupby("product")["spread"].mean()
vol_mean2 = vol_summary.set_index("product")["total_volume"]
sv_df = pd.concat([spread_mean2, vol_mean2], axis=1).dropna()
sv_df.columns = ["spread", "volume"]
sv_df["group"] = sv_df.index.map(PRODUCT_TO_GROUP)

fig, ax = plt.subplots(figsize=(12, 9))
for gname, gdata in sv_df.groupby("group"):
    ax.scatter(gdata["volume"], gdata["spread"], label=gname,
               color=group_color[gname], s=80, alpha=0.85, zorder=5)
    for p, row in gdata.iterrows():
        ax.annotate(p.split("_")[-1], (row["volume"], row["spread"]),
                    fontsize=6, alpha=0.7, textcoords="offset points", xytext=(3, 2))
slope, intercept, r, _, _ = stats.linregress(sv_df["volume"], sv_df["spread"])
xx = np.linspace(sv_df["volume"].min(), sv_df["volume"].max(), 100)
ax.plot(xx, slope * xx + intercept, "white", lw=1.5, linestyle="--", label=f"r={r:.2f}")
ax.set_xlabel("Total Trade Volume")
ax.set_ylabel("Mean Bid-Ask Spread")
ax.set_title("Market Quality: Spread vs Trade Volume", fontweight="bold", fontsize=12)
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(f"{OUT}/61_spread_vs_volume.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 62: Return volatility vs trade frequency scatter
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 62: Volatility vs trade frequency ...")
ret_vol = returns[ALL_PRODUCTS].std().rename("ret_vol")
trade_freq = trades.groupby("product")["global_ts"].count().rename("trade_freq")
vf_df = pd.concat([ret_vol, trade_freq], axis=1).dropna()
vf_df["group"] = vf_df.index.map(PRODUCT_TO_GROUP)

fig, ax = plt.subplots(figsize=(12, 9))
for gname, gdata in vf_df.groupby("group"):
    ax.scatter(gdata["trade_freq"], gdata["ret_vol"] * 1e4, label=gname,
               color=group_color[gname], s=80, alpha=0.85, zorder=5)
    for p, row in gdata.iterrows():
        ax.annotate(p.split("_")[-1], (row["trade_freq"], row["ret_vol"] * 1e4),
                    fontsize=6, alpha=0.7, textcoords="offset points", xytext=(3, 2))
ax.set_xlabel("Trade Frequency (count)")
ax.set_ylabel("Return Volatility (bps)")
ax.set_title("Volatility vs Trade Frequency", fontweight="bold", fontsize=12)
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(f"{OUT}/62_volatility_vs_trade_frequency.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 63: Price trend by day — did any product trend persistently?
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 63: Per-day price trends ...")
fig, axes = plt.subplots(3, 1, figsize=(20, 18))
for ax, day in zip(axes, [2, 3, 4]):
    day_prices = prices[prices["day"] == day].pivot_table(
        index="timestamp", columns="product", values="mid_price")
    day_prices.ffill(inplace=True)
    first = day_prices.iloc[0]
    normed = day_prices / first - 1
    for gname, gprods in GROUPS.items():
        avail = [p for p in gprods if p in normed.columns]
        group_norm = normed[avail].mean(axis=1)
        ax.plot(day_prices.index / 1e6, group_norm * 100, label=gname,
                color=group_color[gname], lw=1.2, alpha=0.85)
    ax.axhline(0, color="white", lw=0.5)
    ax.set_title(f"Day {day} — Group Average Price Change from Open (%)", fontweight="bold", fontsize=11)
    ax.set_ylabel("% Change from Open")
    ax.set_xlabel("Time within day (M ticks)")
    ax.legend(fontsize=7, ncol=5)
fig.suptitle("Per-Day Group Price Trends", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/63_perday_price_trends.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 64: Snack Pack spread pairs — both directions shown clearly
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 64: Snack pack spread decomposition ...")
snack_pairs = list(combinations(GROUPS["Snack Packs"], 2))
fig, axes = plt.subplots(5, 2, figsize=(20, 25))
axes = axes.flatten()
for ax, (p1, p2) in zip(axes, snack_pairs):
    if p1 not in pivot.columns or p2 not in pivot.columns:
        ax.set_visible(False)
        continue
    spread = pivot[p1] - pivot[p2]
    roll_mean = spread.rolling(500).mean()
    roll_std = spread.rolling(500).std()
    z = (spread - roll_mean) / roll_std
    ax.plot(pivot.index / 1e6, spread, lw=0.5, color="cyan", alpha=0.5, label="Spread")
    ax.plot(pivot.index / 1e6, roll_mean, lw=1.2, color="yellow", label="Mean")
    ax2 = ax.twinx()
    ax2.plot(pivot.index / 1e6, z, lw=0.6, color="orange", alpha=0.7, label="Z")
    ax2.axhline(2, color="red", lw=0.8, linestyle="--")
    ax2.axhline(-2, color="green", lw=0.8, linestyle="--")
    ax2.set_ylabel("Z-score", color="orange")
    ax2.set_ylim(-6, 6)
    ax.set_title(f"{p1.split('_')[-1]} − {p2.split('_')[-1]}", fontweight="bold", fontsize=9)
    ax.set_ylabel("Spread")
    ax.set_xlabel("Time (M ticks)")
    ax.legend(fontsize=7, loc="upper left")
    ax2.legend(fontsize=7, loc="upper right")
fig.suptitle("All Snack Pack Pair Spreads with Rolling Z-score", fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/64_snackpack_all_pair_spreads.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 65: Trade size vs next price return (predictive)
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 65: Trade size vs next price return ...")
fig, axes = plt.subplots(5, 2, figsize=(20, 25))
axes = axes.flatten()
HORIZON_R = 50
for ax, (gname, gprods) in zip(axes, GROUPS.items()):
    rep = next((p for p in gprods if p in pivot.columns and len(trades[trades["product"] == p]) > 20), None)
    if rep is None:
        ax.set_visible(False)
        continue
    tr = trades[trades["product"] == rep].sort_values("global_ts").copy()
    next_rets = []
    for _, row in tr.iterrows():
        idx_at = pivot.index.searchsorted(row["global_ts"])
        idx_fwd = pivot.index.searchsorted(row["global_ts"] + HORIZON_R)
        if idx_at >= len(pivot) or idx_fwd >= len(pivot):
            next_rets.append(np.nan)
            continue
        p_at = pivot[rep].iloc[idx_at]
        p_fwd = pivot[rep].iloc[idx_fwd]
        next_rets.append((p_fwd - p_at) / p_at * 1e4 if p_at else np.nan)
    tr["next_ret"] = next_rets
    tr = tr.dropna(subset=["next_ret"])
    ax.scatter(tr["quantity"], tr["next_ret"], alpha=0.3, s=8, color=group_color[gname])
    if len(tr) > 5:
        slope, intercept, r, pv, _ = stats.linregress(tr["quantity"], tr["next_ret"])
        xx = np.array([tr["quantity"].min(), tr["quantity"].max()])
        ax.plot(xx, slope * xx + intercept, "r--", lw=1.5, label=f"r={r:.2f}, p={pv:.3f}")
        ax.legend(fontsize=8)
    ax.axhline(0, color="white", lw=0.5)
    ax.set_title(f"{gname} — {rep.split('_')[-1]}", fontsize=9, fontweight="bold")
    ax.set_xlabel("Trade Size")
    ax.set_ylabel(f"Return {HORIZON_R} ticks later (bps)")
fig.suptitle(f"Trade Size Predictiveness of {HORIZON_R}-tick Forward Return", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/65_trade_size_vs_forward_return.png", bbox_inches="tight")
plt.close(fig)

print(f"\nDone. Total plots in {OUT}:")
for fn in sorted(f for f in os.listdir(OUT) if f.endswith(".png")):
    print(f"  {fn}")
