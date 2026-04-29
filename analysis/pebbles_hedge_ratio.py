import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT_PNG  = "/media/manukrishnan/Mk/prosperity_4/analysis_output/50_pebbles_hedge_ratios.png"
OUT_TXT  = "/media/manukrishnan/Mk/prosperity_4/analysis_output/SUMMARY.txt"

DARK = "#1a1a2e"; MID = "#16213e"; ACCENT = "#f39c12"; ADJ = "#00ff99"

# ── data ──────────────────────────────────────────────────────────────────────
price_frames = []
for d in [2, 3, 4]:
    pf = pd.read_csv(f"/media/manukrishnan/Mk/prosperity_4/data/prices_round_5_day_{d}.csv", sep=";")
    pf["day"] = d
    price_frames.append(pf)

prices = pd.concat(price_frames, ignore_index=True)
prices["gts"] = (prices["day"] - 2) * 1_000_000 + prices["timestamp"]
prices.sort_values(["product", "gts"], inplace=True)

PRODS = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]
LABELS = ["XS", "S", "M", "L", "XL"]

pivot = prices[prices["product"].isin(PRODS)].pivot_table(
    index="gts", columns="product", values="mid_price")
pivot.sort_index(inplace=True)
pivot.ffill(inplace=True)
returns = pivot.pct_change().dropna(how="all")

# ── user-adjusted slopes (from interactive session) ───────────────────────────
# key: (row_idx, col_idx) → adjusted slope
# row=i means y=PRODS[i], col=j means x=PRODS[j]
USER_SLOPES = {
    (0, 4): -0.7197,   # XS vs XL (adjusted)
    (1, 4): -0.5019,   # S  vs XL (adjusted)
    (2, 4): -0.4356,   # M  vs XL (adjusted)
    (3, 4): -0.4735,   # L  vs XL (adjusted)
}

# ── compute OLS for every pair + store data ────────────────────────────────────
ols = {}; pair_data = {}
n = len(PRODS)
for i, p1 in enumerate(PRODS):
    for j, p2 in enumerate(PRODS):
        if i == j: continue
        x = returns[p2].dropna()
        y = returns[p1].reindex(x.index)
        valid = x.notna() & y.notna()
        xv, yv = x[valid].values, y[valid].values
        if len(xv) > 5:
            m, b, r, *_ = stats.linregress(xv, yv)
            ols[(i, j)] = (m, b, r)
            pair_data[(i, j)] = (xv, yv)

# ── hedge ratios: β of R_i ~ R_XL  (XL is col 4) ─────────────────────────────
XL_IDX = 4
hedge = {}
for i, label in enumerate(LABELS):
    if i == XL_IDX: continue
    ols_slope = ols[(i, XL_IDX)][0]
    adj_slope  = USER_SLOPES.get((i, XL_IDX), ols_slope)
    hedge[label] = {
        "ols_beta":  ols_slope,
        "adj_beta":  adj_slope,
        "r":         ols[(i, XL_IDX)][2],
        # hedge: to neutralise XL risk in pebble_i, go long |adj_beta| XL per 1 pebble_i
        "hedge_ratio": abs(adj_slope),
    }

# ── print & update SUMMARY.txt ────────────────────────────────────────────────
section = """
── PEBBLES HEDGE RATIOS (XL as hedge instrument) ──
  Hedge: R_i ≈ α + β·R_XL  →  long 1 pebble_i, long |β| XL to neutralise XL-factor
  β_adj = user-adjusted slope; β_ols = OLS slope

  {:4s}  {:>10s}  {:>10s}  {:>12s}  {:>8s}
""".format("Prod", "β_ols", "β_adj", "hedge_ratio", "r")

for label, d in hedge.items():
    section += "  {:4s}  {:>10.4f}  {:>10.4f}  {:>12.4f}  {:>8.4f}\n".format(
        label, d["ols_beta"], d["adj_beta"], d["hedge_ratio"], d["r"])

print(section)

with open(OUT_TXT, "r") as f:
    txt = f.read()

MARKER = "── PEBBLES HEDGE RATIOS"
if MARKER in txt:
    # replace existing block (up to next ──)
    start = txt.index(MARKER) - 1
    rest  = txt[start:]
    next_block = rest.find("\n──", 2)
    if next_block == -1:
        txt = txt[:start] + section
    else:
        txt = txt[:start] + section + rest[next_block:]
else:
    txt = txt.rstrip("\n") + "\n" + section

with open(OUT_TXT, "w") as f:
    f.write(txt)
print("SUMMARY.txt updated")

# ── plot: 5×5 scatter matrix, XL pairs use adjusted slopes ────────────────────
plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": MID,
    "text.color": "white", "axes.labelcolor": "#aaa",
    "xtick.color": "#aaa", "ytick.color": "#aaa",
    "axes.edgecolor": "#444", "font.size": 8,
})

fig, axes = plt.subplots(n, n, figsize=(n * 3, n * 3), facecolor=DARK)
fig.suptitle("Pebbles — Return Pair Scatter Matrix  (green = user-adjusted hedge line)",
             color="white", fontsize=12, y=1.005)

for i in range(n):
    for j in range(n):
        ax = axes[i][j]
        ax.set_facecolor(MID)
        ax.tick_params(labelsize=6, colors="#aaa")
        ax.spines[:].set_color("#333")

        if i == j:
            ret = returns[PRODS[i]].dropna().values
            ax.hist(ret, bins=50, color="#e74c3c", alpha=0.8, edgecolor="none")
            ax.set_xlabel(LABELS[i], fontsize=7, color="white")
        else:
            if (i, j) in pair_data:
                xv, yv = pair_data[(i, j)]
                ax.scatter(xv, yv, s=0.5, alpha=0.3, color="#e8a0a0", rasterized=True)

                ols_m, ols_b, r_val = ols[(i, j)]
                xx = np.array([xv.min(), xv.max()])

                # is this an XL pair?
                is_xl_pair = (j == XL_IDX or i == XL_IDX)
                adj_key = (i, j)

                if adj_key in USER_SLOPES:
                    # draw OLS faint + adjusted bright
                    ax.plot(xx, ols_m * xx + ols_b, color=ACCENT, lw=0.8, alpha=0.4, ls="--")
                    adj_m = USER_SLOPES[adj_key]
                    adj_b = ols[(i, j)][1]
                    ax.plot(xx, adj_m * xx + adj_b, color=ADJ, lw=1.2)
                    hr = abs(adj_m)
                    ax.text(0.05, 0.90,
                            f"r={r_val:.2f}\nβ_adj={adj_m:.3f}\nHR={hr:.3f}",
                            transform=ax.transAxes, fontsize=5.5, color=ADJ, va="top")
                elif i == XL_IDX and (j, XL_IDX) in USER_SLOPES:
                    # XL as y-axis: show OLS only, note inverse
                    ax.plot(xx, ols_m * xx + ols_b, color=ACCENT, lw=1.0)
                    ax.text(0.05, 0.90, f"r={r_val:.2f}\nβ={ols_m:.3f}",
                            transform=ax.transAxes, fontsize=5.5, color=ACCENT, va="top")
                else:
                    ax.plot(xx, ols_m * xx + ols_b, color=ACCENT, lw=1.0)
                    ax.text(0.05, 0.92, f"r={r_val:.2f}",
                            transform=ax.transAxes, fontsize=6.5, color=ACCENT)

        if j == 0:
            ax.set_ylabel(LABELS[i], fontsize=7, color="white")
        if i == n - 1:
            ax.set_xlabel(LABELS[j], fontsize=7, color="white")

# legend
from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], color=ACCENT, lw=1.2, ls="--", label="OLS fit"),
    Line2D([0], [0], color=ADJ,   lw=1.2,            label="Adjusted (hedge) fit"),
]
fig.legend(handles=handles, loc="lower center", ncol=2,
           facecolor="#0f3460", edgecolor="#555", labelcolor="white",
           fontsize=9, bbox_to_anchor=(0.5, -0.01))

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=120, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"Saved {OUT_PNG}")
