"""
Autocorrelation of mid-price deltas (first differences) across all products.
Subsamples to MAX_OBS consecutive observations per product for speed.
Outputs saved to analysis_output/delta_autocorr/
"""

import glob
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")

OUT = "/media/manukrishnan/Mk/prosperity_4/analysis_output/delta_autocorr/"
os.makedirs(OUT, exist_ok=True)

DATA_DIRS = [
    "/media/manukrishnan/Mk/prosperity_4/data",
    "/media/manukrishnan/Mk/prosperity_4/data/round4",
    "/media/manukrishnan/Mk/prosperity_4/data/old",
    "/media/manukrishnan/Mk/prosperity_4/round_4",
]
MAX_LAG = 40
CONF = 1.96
MAX_OBS = 10_000   # subsample cap per product for LB speed

# ── load ──────────────────────────────────────────────────────────────────────
frames = []
seen = set()
for d in DATA_DIRS:
    for f in sorted(glob.glob(os.path.join(d, "prices_*.csv"))):
        key = os.path.basename(f)
        if key in seen:
            continue
        seen.add(key)
        parts = os.path.splitext(key)[0].split("_")
        try:
            rnd, day_raw = int(parts[2]), int(parts[4])
        except (IndexError, ValueError):
            rnd, day_raw = 0, 0
        df = pd.read_csv(f, sep=";")
        df["gts"] = (rnd * 10 + day_raw) * 1_000_000 + df["timestamp"]
        frames.append(df)

prices = pd.concat(frames, ignore_index=True)
if "mid_price" not in prices.columns or prices["mid_price"].isna().mean() > 0.5:
    prices["mid_price"] = (prices["bid_price_1"] + prices["ask_price_1"]) / 2
else:
    mask = prices["mid_price"].isna()
    prices.loc[mask, "mid_price"] = (
        prices.loc[mask, "bid_price_1"] + prices.loc[mask, "ask_price_1"]
    ) / 2

PRODUCTS = sorted(p for p in prices["product"].unique() if not p.startswith("VEV_"))

# pre-compute deltas
print(f"Computing deltas for {len(PRODUCTS)} products...")
deltas = {}
for prod in PRODUCTS:
    sub = prices.loc[prices["product"] == prod].sort_values("gts")
    d = np.diff(sub["mid_price"].values)
    if len(d) >= MAX_LAG + 10:
        deltas[prod] = d

PRODUCTS = list(deltas.keys())
print(f"Valid products: {len(PRODUCTS)}")


def acf_vec(x, max_lag):
    x = x - x.mean()
    n = len(x)
    var = np.dot(x, x) / n
    if var == 0:
        return np.zeros(max_lag)
    return np.array([np.dot(x[:n - k], x[k:]) / (n * var) for k in range(1, max_lag + 1)])


print("Computing ACF and Ljung-Box...")
acf_matrix = {}
lb_matrix = {}
ci_per = {}

for prod in PRODUCTS:
    d = deltas[prod]
    # use full series for ACF
    r = acf_vec(d, MAX_LAG)
    acf_matrix[prod] = r
    ci_per[prod] = CONF / np.sqrt(len(d))
    # subsample for LB
    d_sub = d[:MAX_OBS]
    try:
        lb = acorr_ljungbox(d_sub, lags=MAX_LAG, return_df=True)
        lb_matrix[prod] = lb["lb_pvalue"].values
    except Exception:
        lb_matrix[prod] = np.ones(MAX_LAG)

lags = np.arange(1, MAX_LAG + 1)
n = len(PRODUCTS)
ci_avg = np.mean(list(ci_per.values()))

# ── plot 1: ACF grid ──────────────────────────────────────────────────────────
print("Plotting ACF grid...")
cols = 5
rows = int(np.ceil(n / cols))
fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 2.8), facecolor="#1a1a2e")
fig.suptitle("ACF of Mid-Price Deltas", color="white", fontsize=15, y=1.005)
axes_flat = axes.flatten() if n > 1 else [axes]

for i, prod in enumerate(PRODUCTS):
    ax = axes_flat[i]
    ax.set_facecolor("#16213e")
    r = acf_matrix[prod]
    ci = ci_per[prod]
    clr = ["#e74c3c" if abs(v) > ci else "#3498db" for v in r]
    ax.bar(lags, r, color=clr, width=0.7, alpha=0.85)
    ax.axhline(ci, color="#f39c12", lw=0.8, ls="--")
    ax.axhline(-ci, color="#f39c12", lw=0.8, ls="--")
    ax.axhline(0, color="#666", lw=0.5)
    ax.set_title(prod, color="white", fontsize=7, pad=2)
    ax.set_xlim(0, MAX_LAG + 1)
    ax.set_ylim(-0.3, 0.3)
    ax.tick_params(colors="#aaa", labelsize=6)
    ax.spines[:].set_color("#333")

for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)

plt.tight_layout()
plt.savefig(OUT + "acf_grid.png", dpi=120, bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
print("  acf_grid.png")

# ── plot 2: ACF heatmap ───────────────────────────────────────────────────────
print("Plotting ACF heatmap...")
acf_arr = np.array([acf_matrix[p] for p in PRODUCTS])

fig, ax = plt.subplots(figsize=(14, max(5, n * 0.28 + 2)), facecolor="#1a1a2e")
ax.set_facecolor("#16213e")
im = ax.imshow(acf_arr, aspect="auto", cmap="RdBu_r", vmin=-0.3, vmax=0.3)
ax.set_xticks(np.arange(MAX_LAG))
ax.set_xticklabels(lags, fontsize=7, color="white")
ax.set_yticks(np.arange(n))
ax.set_yticklabels(PRODUCTS, fontsize=7, color="white")
ax.set_xlabel("Lag", color="#aaa")
ax.set_title("ACF Heatmap — Mid-Price Deltas", color="white", fontsize=13)
cb = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02)
cb.ax.tick_params(colors="white")
cb.set_label("ACF", color="white")
plt.tight_layout()
plt.savefig(OUT + "acf_heatmap.png", dpi=120, bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
print("  acf_heatmap.png")

# ── plot 3: Ljung-Box heatmap ─────────────────────────────────────────────────
print("Plotting Ljung-Box heatmap...")
lb_arr = np.array([lb_matrix[p] for p in PRODUCTS])
log_p = -np.log10(np.clip(lb_arr, 1e-10, 1.0))

fig, ax = plt.subplots(figsize=(14, max(5, n * 0.28 + 2)), facecolor="#1a1a2e")
ax.set_facecolor("#16213e")
im = ax.imshow(log_p, aspect="auto", cmap="inferno", vmin=0, vmax=10)
ax.set_xticks(np.arange(MAX_LAG))
ax.set_xticklabels(lags, fontsize=7, color="white")
ax.set_yticks(np.arange(n))
ax.set_yticklabels(PRODUCTS, fontsize=7, color="white")
ax.set_xlabel("Lag", color="#aaa")
ax.set_title(f"Ljung-Box  -log₁₀(p)  [subsample {MAX_OBS:,}]  bright=predictable", color="white", fontsize=12)
cb = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02)
cb.ax.tick_params(colors="white")
cb.set_label("-log₁₀(p)", color="white")
plt.tight_layout()
plt.savefig(OUT + "ljungbox_heatmap.png", dpi=120, bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
print("  ljungbox_heatmap.png")

# ── plot 4: lag-1 ACF bar sorted ─────────────────────────────────────────────
print("Plotting lag-1 ACF bar...")
lag1 = {p: acf_matrix[p][0] for p in PRODUCTS}
sorted_prods = sorted(lag1, key=lambda p: lag1[p])
vals = [lag1[p] for p in sorted_prods]
ci_vals = [ci_per[p] for p in sorted_prods]
clr = ["#e74c3c" if abs(v) > c else "#3498db" for v, c in zip(vals, ci_vals)]

fig, ax = plt.subplots(figsize=(10, max(4, n * 0.25 + 1)), facecolor="#1a1a2e")
ax.set_facecolor("#16213e")
ax.barh(sorted_prods, vals, color=clr, alpha=0.85)
ax.axvline(ci_avg, color="#f39c12", lw=1, ls="--", label="95% CI (avg)")
ax.axvline(-ci_avg, color="#f39c12", lw=1, ls="--")
ax.axvline(0, color="#666", lw=0.5)
ax.set_xlabel("Lag-1 ACF", color="#aaa")
ax.set_title("Lag-1 ACF of Price Deltas (sorted)", color="white", fontsize=13)
ax.tick_params(colors="white", labelsize=7)
ax.spines[:].set_color("#333")
ax.legend(facecolor="#0f3460", labelcolor="white", edgecolor="#444", fontsize=8)
plt.tight_layout()
plt.savefig(OUT + "lag1_acf_bar.png", dpi=120, bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
print("  lag1_acf_bar.png")

# ── plot 5: top-5 detail ──────────────────────────────────────────────────────
print("Plotting top-5 detail...")
top5 = sorted(lag1, key=lambda p: abs(lag1[p]), reverse=True)[:5]
fig, axes = plt.subplots(2, 5, figsize=(18, 7), facecolor="#1a1a2e")
fig.suptitle("Top-5 |lag-1 ACF|: Delta Distribution & ACF", color="white", fontsize=12)

for col, prod in enumerate(top5):
    d = deltas[prod]
    r = acf_matrix[prod]
    ci = ci_per[prod]

    ax = axes[0, col]
    ax.set_facecolor("#16213e")
    ax.hist(d, bins=60, color="#3498db", alpha=0.8, edgecolor="none")
    ax.axvline(0, color="#f39c12", lw=1)
    ax.set_title(f"{prod}\nlag1={lag1[prod]:.4f}", color="white", fontsize=7)
    ax.tick_params(colors="#aaa", labelsize=6)
    ax.spines[:].set_color("#333")

    ax = axes[1, col]
    ax.set_facecolor("#16213e")
    clr = ["#e74c3c" if abs(v) > ci else "#3498db" for v in r]
    ax.bar(lags, r, color=clr, width=0.7, alpha=0.85)
    ax.axhline(ci, color="#f39c12", lw=0.8, ls="--")
    ax.axhline(-ci, color="#f39c12", lw=0.8, ls="--")
    ax.axhline(0, color="#666", lw=0.5)
    ax.set_xlim(0, MAX_LAG + 1)
    ax.tick_params(colors="#aaa", labelsize=6)
    ax.spines[:].set_color("#333")

plt.tight_layout()
plt.savefig(OUT + "top5_detail.png", dpi=120, bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
print("  top5_detail.png")

# ── summary CSV ───────────────────────────────────────────────────────────────
rows = []
for prod in PRODUCTS:
    d = deltas[prod]
    r = acf_matrix[prod]
    ci = ci_per[prod]
    lb_p = lb_matrix.get(prod, np.ones(MAX_LAG))
    rows.append({
        "product": prod,
        "n_obs": len(d),
        "mean_delta": round(d.mean(), 6),
        "std_delta": round(d.std(), 4),
        "lag1_acf": round(r[0], 6),
        "lag2_acf": round(r[1], 6),
        "lag5_acf": round(r[4], 6),
        "max_abs_acf": round(np.max(np.abs(r)), 6),
        "lag_of_max": int(np.argmax(np.abs(r)) + 1),
        "n_sig_lags": int(np.sum(np.abs(r) > ci)),
        "lb_p_lag10": round(lb_p[9], 6),
    })

summary = pd.DataFrame(rows).sort_values("max_abs_acf", ascending=False)
summary.to_csv(OUT + "summary.csv", index=False)
print("\nsummary.csv")
print(summary[["product", "lag1_acf", "max_abs_acf", "n_sig_lags", "lb_p_lag10"]].to_string(index=False))
