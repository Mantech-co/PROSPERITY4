"""
Volatility clustering analysis — ARCH effects, squared-return ACF, regime detection
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import het_arch
from scipy import stats
import os

OUT = "/media/manukrishnan/Mk/prosperity_4/analysis_output"
sns.set_theme(style="darkgrid")
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

print("Loading ...")
prices = pd.concat([
    pd.read_csv(f"/media/manukrishnan/Mk/prosperity_4/data/prices_round_5_day_{d}.csv", sep=";")
    for d in [2, 3, 4]], ignore_index=True)
prices["global_ts"] = (prices["day"] - 2) * 1_000_000 + prices["timestamp"]
prices.sort_values(["product", "global_ts"], inplace=True)

pivot = prices.pivot_table(index="global_ts", columns="product", values="mid_price")
pivot.sort_index(inplace=True)
pivot.ffill(inplace=True)
ALL_PRODUCTS = sorted(pivot.columns)
returns = pivot[ALL_PRODUCTS].pct_change().dropna()

# ─────────────────────────────────────────────────────────────────────────────
# V1: Absolute return time series — volatility clustering visible as bursts
# ─────────────────────────────────────────────────────────────────────────────
print("V1: Abs return time series per group ...")
fig, axes = plt.subplots(5, 2, figsize=(20, 25))
axes = axes.flatten()
for ax, (gname, gprods) in zip(axes, GROUPS.items()):
    avail = [p for p in gprods if p in returns.columns]
    for p in avail:
        abs_ret = returns[p].abs() * 1e4
        smooth = abs_ret.rolling(200).mean()
        ax.plot(returns.index / 1e6, smooth, label=p.split("_")[-1], lw=0.8, alpha=0.85)
    ax.set_title(gname, fontweight="bold", fontsize=11)
    ax.set_ylabel("|Return| rolling mean (bps)")
    ax.set_xlabel("Time (M ticks)")
    ax.legend(fontsize=7, ncol=2)
fig.suptitle("Volatility Clustering — Rolling Mean |Return| (bps)", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/V1_abs_return_timeseries.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# V2: Squared return ACF — ARCH signature (if clustered, ACF of r² is positive)
# ─────────────────────────────────────────────────────────────────────────────
print("V2: Squared return ACF per group ...")
fig, axes = plt.subplots(5, 2, figsize=(20, 28))
axes = axes.flatten()
NLAGS = 60
for ax, (gname, gprods) in zip(axes, GROUPS.items()):
    avail = [p for p in gprods if p in returns.columns]
    for p in avail:
        r2 = returns[p].dropna() ** 2
        acf_vals = acf(r2, nlags=NLAGS, fft=True)
        ax.plot(range(NLAGS + 1), acf_vals, label=p.split("_")[-1], lw=0.9, alpha=0.85)
    conf = 1.96 / np.sqrt(len(returns))
    ax.axhline(conf, color="orange", lw=1, linestyle="--", alpha=0.7)
    ax.axhline(-conf, color="orange", lw=1, linestyle="--", alpha=0.7)
    ax.axhline(0, color="white", lw=0.5)
    ax.set_title(gname, fontweight="bold", fontsize=11)
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF(r²)")
    ax.legend(fontsize=7, ncol=2)
fig.suptitle("ACF of Squared Returns — ARCH/Volatility Clustering Signature", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/V2_squared_return_acf.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# V3: ACF of absolute returns (more robust clustering test)
# ─────────────────────────────────────────────────────────────────────────────
print("V3: Absolute return ACF per group ...")
fig, axes = plt.subplots(5, 2, figsize=(20, 28))
axes = axes.flatten()
for ax, (gname, gprods) in zip(axes, GROUPS.items()):
    avail = [p for p in gprods if p in returns.columns]
    for p in avail:
        absr = returns[p].dropna().abs()
        acf_vals = acf(absr, nlags=NLAGS, fft=True)
        ax.plot(range(NLAGS + 1), acf_vals, label=p.split("_")[-1], lw=0.9, alpha=0.85)
    conf = 1.96 / np.sqrt(len(returns))
    ax.axhline(conf, color="orange", lw=1, linestyle="--", alpha=0.7)
    ax.axhline(0, color="white", lw=0.5)
    ax.set_title(gname, fontweight="bold", fontsize=11)
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF(|r|)")
    ax.legend(fontsize=7, ncol=2)
fig.suptitle("ACF of |Returns| — Long-Memory Volatility Test", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/V3_abs_return_acf.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# V4: ARCH-LM test p-values per product (bar chart)
# ─────────────────────────────────────────────────────────────────────────────
print("V4: ARCH-LM test ...")
arch_results = []
for p in ALL_PRODUCTS:
    r = returns[p].dropna().values
    try:
        lm_stat, lm_pval, f_stat, f_pval = het_arch(r, nlags=10)
        arch_results.append({"product": p, "lm_pval": lm_pval, "f_pval": f_pval,
                              "lm_stat": lm_stat, "group": PRODUCT_TO_GROUP.get(p, "")})
    except Exception:
        arch_results.append({"product": p, "lm_pval": 1.0, "f_pval": 1.0,
                              "lm_stat": 0.0, "group": PRODUCT_TO_GROUP.get(p, "")})

arch_df = pd.DataFrame(arch_results).set_index("product").sort_values("lm_pval")

fig, axes = plt.subplots(1, 2, figsize=(22, 9))
ac = [group_color[arch_df.loc[p, "group"]] for p in arch_df.index]

bars = axes[0].barh(arch_df.index, -np.log10(arch_df["lm_pval"].clip(1e-300)),
                    color=ac, alpha=0.85)
axes[0].axvline(-np.log10(0.05), color="orange", lw=1.5, linestyle="--", label="p=0.05")
axes[0].axvline(-np.log10(0.01), color="red", lw=1.5, linestyle="--", label="p=0.01")
axes[0].set_xlabel("−log₁₀(p-value)")
axes[0].set_title("ARCH-LM Test — Evidence of Volatility Clustering", fontweight="bold")
axes[0].tick_params(axis="y", labelsize=6)
axes[0].legend()

# shade clustered vs not
for i, p in enumerate(arch_df.index):
    if arch_df.loc[p, "lm_pval"] < 0.01:
        axes[0].get_yticklabels()[i].set_color("red")

axes[1].barh(arch_df.index, arch_df["lm_stat"], color=ac, alpha=0.85)
axes[1].set_xlabel("LM Statistic")
axes[1].set_title("ARCH-LM Statistic (higher = stronger clustering)", fontweight="bold")
axes[1].tick_params(axis="y", labelsize=6)

legend_els = [Patch(color=group_color[g], label=g) for g in GROUPS]
axes[0].legend(handles=legend_els + [
    Patch(color="orange", label="p=0.05"),
    Patch(color="red", label="p=0.01")], fontsize=7, ncol=1, loc="lower right")

fig.tight_layout()
fig.savefig(f"{OUT}/V4_arch_lm_test.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# V5: Rolling volatility — per product overlay (50 lines, 1 axis per group)
# ─────────────────────────────────────────────────────────────────────────────
print("V5: Rolling vol scatter ...")
WIN = 500
fig, axes = plt.subplots(5, 2, figsize=(20, 25))
axes = axes.flatten()
for ax, (gname, gprods) in zip(axes, GROUPS.items()):
    avail = [p for p in gprods if p in returns.columns]
    vol_data = {}
    for p in avail:
        vol_data[p] = (returns[p].rolling(WIN).std() * 1e4)
    # scatter: vol at t vs vol at t+WIN (persistence)
    for p in avail:
        v = vol_data[p].dropna().values
        ax.scatter(v[:-1], v[1:], s=1.5, alpha=0.2, label=p.split("_")[-1])
    ax.set_xlabel("Vol(t) (bps)")
    ax.set_ylabel("Vol(t+1 window) (bps)")
    ax.set_title(gname, fontweight="bold", fontsize=11)
    ax.legend(fontsize=7, ncol=2, markerscale=4)
fig.suptitle("Volatility Scatter: Vol(t) vs Vol(t+1 window) — Persistence Check", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/V5_vol_persistence_scatter.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# V6: Volatility cross-correlation — do products spike together?
# ─────────────────────────────────────────────────────────────────────────────
print("V6: Volatility cross-correlation heatmap ...")
vol_df = returns[ALL_PRODUCTS].rolling(WIN).std().dropna()
vol_corr = vol_df.corr()

from matplotlib.colors import TwoSlopeNorm
fig, ax = plt.subplots(figsize=(22, 20))
im = ax.imshow(vol_corr.values, cmap="RdYlGn",
               norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), aspect="auto")
ax.set_xticks(range(len(ALL_PRODUCTS)))
ax.set_yticks(range(len(ALL_PRODUCTS)))
ax.set_xticklabels(ALL_PRODUCTS, rotation=90, fontsize=5.5)
ax.set_yticklabels(ALL_PRODUCTS, fontsize=5.5)
plt.colorbar(im, ax=ax, fraction=0.03)
ax.set_title(f"Rolling Volatility Correlation (win={WIN}) — Do Products Cluster Together?", fontweight="bold", fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT}/V6_vol_crosscorrelation.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# V7: Volatility regime detection — threshold-based (high/low) per product
# ─────────────────────────────────────────────────────────────────────────────
print("V7: Volatility regimes ...")
fig, axes = plt.subplots(5, 2, figsize=(20, 25))
axes = axes.flatten()
for ax, (gname, gprods) in zip(axes, GROUPS.items()):
    rep = next((p for p in gprods if p in returns.columns), None)
    if rep is None:
        ax.set_visible(False)
        continue
    r = returns[rep]
    rv = r.rolling(WIN).std() * 1e4
    rv = rv.dropna()
    thresh = rv.median()
    high_vol = rv > thresh
    ts = rv.index / 1e6
    ax.plot(ts, rv, lw=0.7, color="cyan", alpha=0.8, label=f"RV {rep.split('_')[-1]}")
    ax.fill_between(ts, rv.where(high_vol), thresh, alpha=0.35, color="red", label="High vol")
    ax.fill_between(ts, rv.where(~high_vol), thresh, alpha=0.25, color="green", label="Low vol")
    ax.axhline(thresh, color="yellow", lw=1, linestyle="--", label=f"Median={thresh:.2f}")
    ax.set_title(f"{gname} — {rep.split('_')[-1]}", fontweight="bold", fontsize=10)
    ax.set_ylabel("Rolling Vol (bps)")
    ax.set_xlabel("Time (M ticks)")
    ax.legend(fontsize=7, ncol=2)
fig.suptitle("Volatility Regime Detection — High/Low Vol Periods", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/V7_vol_regimes.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# V8: QQ-plots of returns — fat tails (evidence of vol clustering)
# ─────────────────────────────────────────────────────────────────────────────
print("V8: QQ plots ...")
fig, axes = plt.subplots(5, 2, figsize=(20, 25))
axes = axes.flatten()
for ax, (gname, gprods) in zip(axes, GROUPS.items()):
    rep = next((p for p in gprods if p in returns.columns), None)
    if rep is None:
        ax.set_visible(False)
        continue
    r = returns[rep].dropna().values
    (osm, osr), (slope, intercept, r_val) = stats.probplot(r, dist="norm")
    ax.scatter(osm, osr, s=2, alpha=0.4, color=group_color[gname])
    ax.plot(osm, slope * np.array(osm) + intercept, "r-", lw=1.5, label=f"r={r_val:.3f}")
    kurt = stats.kurtosis(r)
    skew = stats.skew(r)
    ax.text(0.05, 0.92, f"Kurt={kurt:.2f}  Skew={skew:.2f}", transform=ax.transAxes,
            fontsize=8, color="orange")
    ax.set_title(f"{gname} — {rep.split('_')[-1]}", fontweight="bold", fontsize=10)
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")
    ax.legend(fontsize=7)
fig.suptitle("QQ Plots of Returns vs Normal — Fat Tails = Volatility Clustering", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/V8_qq_plots.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# V9: Kurtosis & skewness bar chart per product
# ─────────────────────────────────────────────────────────────────────────────
print("V9: Kurtosis and skewness ...")
kurt_skew = []
for p in ALL_PRODUCTS:
    if p not in returns.columns:
        continue
    r = returns[p].dropna().values
    kurt_skew.append({"product": p, "kurtosis": stats.kurtosis(r),
                      "skewness": stats.skew(r), "group": PRODUCT_TO_GROUP.get(p, "")})
ks_df = pd.DataFrame(kurt_skew).set_index("product")

fig, axes = plt.subplots(2, 1, figsize=(20, 14))
kc = [group_color[ks_df.loc[p, "group"]] for p in ks_df.index]
axes[0].bar(ks_df.index, ks_df["kurtosis"], color=kc, alpha=0.85)
axes[0].axhline(0, color="white", lw=0.5)
axes[0].axhline(3, color="orange", lw=1, linestyle="--", label="Normal kurtosis (excess=0)")
axes[0].set_xticks(range(len(ks_df)))
axes[0].set_xticklabels(ks_df.index, rotation=90, fontsize=7)
axes[0].set_ylabel("Excess Kurtosis")
axes[0].set_title("Excess Kurtosis per Product (>0 = fat tails → volatility clustering)", fontweight="bold")
axes[0].legend()

axes[1].bar(ks_df.index, ks_df["skewness"], color=kc, alpha=0.85)
axes[1].axhline(0, color="white", lw=0.5)
axes[1].set_xticks(range(len(ks_df)))
axes[1].set_xticklabels(ks_df.index, rotation=90, fontsize=7)
axes[1].set_ylabel("Skewness")
axes[1].set_title("Skewness per Product", fontweight="bold")

legend_els = [Patch(color=group_color[g], label=g) for g in GROUPS]
axes[0].legend(handles=legend_els, fontsize=8, ncol=2, loc="upper right")
fig.tight_layout()
fig.savefig(f"{OUT}/V9_kurtosis_skewness.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# V10: Volatility-of-volatility (VoV) per product — second-order clustering
# ─────────────────────────────────────────────────────────────────────────────
print("V10: Volatility-of-volatility ...")
vov_data = []
for p in ALL_PRODUCTS:
    if p not in returns.columns:
        continue
    rv = returns[p].rolling(WIN).std().dropna() * 1e4
    vov = rv.std()
    vol_mean = rv.mean()
    vov_data.append({"product": p, "vov": vov, "vol_mean": vol_mean,
                     "vov_ratio": vov / vol_mean if vol_mean > 0 else 0,
                     "group": PRODUCT_TO_GROUP.get(p, "")})

vov_df = pd.DataFrame(vov_data).set_index("product").sort_values("vov_ratio")

fig, axes = plt.subplots(1, 2, figsize=(22, 9))
vc = [group_color[vov_df.loc[p, "group"]] for p in vov_df.index]
axes[0].barh(vov_df.index, vov_df["vov_ratio"], color=vc, alpha=0.85)
axes[0].set_xlabel("VoV Ratio (std(vol)/mean(vol))")
axes[0].set_title("Volatility-of-Volatility Ratio — Second-Order Clustering", fontweight="bold")
axes[0].tick_params(axis="y", labelsize=6)

axes[1].scatter(vov_df["vol_mean"], vov_df["vov"], c=vc, s=60, alpha=0.85, zorder=5)
for p, row in vov_df.iterrows():
    axes[1].annotate(p.split("_")[-1], (row["vol_mean"], row["vov"]),
                     fontsize=6, alpha=0.7, textcoords="offset points", xytext=(2, 2))
axes[1].set_xlabel("Mean Volatility (bps)")
axes[1].set_ylabel("Vol-of-Vol (bps)")
axes[1].set_title("Mean Vol vs Vol-of-Vol Scatter", fontweight="bold")

legend_els = [Patch(color=group_color[g], label=g) for g in GROUPS]
axes[0].legend(handles=legend_els, fontsize=7, ncol=1)
fig.tight_layout()
fig.savefig(f"{OUT}/V10_vol_of_vol.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# V11: Rolling vol scatter across products — which products co-spike?
# ─────────────────────────────────────────────────────────────────────────────
print("V11: Group rolling vol overlay ...")
fig, axes = plt.subplots(5, 2, figsize=(20, 25))
axes = axes.flatten()
for ax, (gname, gprods) in zip(axes, GROUPS.items()):
    avail = [p for p in gprods if p in returns.columns]
    for p in avail:
        rv = returns[p].rolling(WIN).std() * 1e4
        ax.plot(returns.index / 1e6, rv, label=p.split("_")[-1], lw=0.7, alpha=0.85)
    ax.set_title(gname, fontweight="bold", fontsize=11)
    ax.set_ylabel("Rolling σ (bps)")
    ax.set_xlabel("Time (M ticks)")
    ax.legend(fontsize=7, ncol=2)
fig.suptitle(f"Rolling Volatility (win={WIN}) Over Time — Co-Spike Detection", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/V11_rolling_vol_per_group.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# V12: Return distribution: normal vs actual overlay (all 50, grid)
# ─────────────────────────────────────────────────────────────────────────────
print("V12: Return distributions with normal overlay ...")
n_prods = len(ALL_PRODUCTS)
ncols = 5
nrows = (n_prods + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(22, nrows * 3))
axes = axes.flatten()
for ax, p in zip(axes, ALL_PRODUCTS):
    r = returns[p].dropna().values * 1e4
    ax.hist(r, bins=60, density=True, alpha=0.6,
            color=group_color.get(PRODUCT_TO_GROUP.get(p, ""), "steelblue"), label="Actual")
    mu, sigma = r.mean(), r.std()
    xx = np.linspace(r.min(), r.max(), 200)
    ax.plot(xx, stats.norm.pdf(xx, mu, sigma), "r-", lw=1.2, label="Normal")
    kurt = stats.kurtosis(r)
    ax.set_title(f"{p.split('_')[-1][:8]}\nkurt={kurt:.1f}", fontsize=7, fontweight="bold")
    ax.set_xlim(-5, 5)
    ax.tick_params(labelsize=5)
for ax in axes[n_prods:]:
    ax.set_visible(False)
fig.suptitle("Return Distributions vs Normal (all 50 products)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/V12_return_distributions_all.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# Print ARCH test summary
# ─────────────────────────────────────────────────────────────────────────────
arch_df_sorted = arch_df.sort_values("lm_pval")
print("\n── ARCH-LM TEST SUMMARY ──")
print(f"Products with ARCH effects (p<0.05): {(arch_df['lm_pval']<0.05).sum()}")
print(f"Products with ARCH effects (p<0.01): {(arch_df['lm_pval']<0.01).sum()}")
print("\nTop 10 most clustered:")
print(arch_df_sorted[["lm_stat", "lm_pval", "group"]].head(10).to_string())
print("\nTop 10 least clustered:")
print(arch_df_sorted[["lm_stat", "lm_pval", "group"]].tail(10).to_string())

print("\n── KURTOSIS SUMMARY ──")
ks_sorted = ks_df.sort_values("kurtosis", ascending=False)
print(ks_sorted[["kurtosis", "skewness", "group"]].head(15).to_string())

print(f"\nAll done. Plots: {OUT}/V1 … V12")
