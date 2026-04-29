import warnings; warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
import pandas as pd
from scipy import stats

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
pivot = prices[prices["product"].isin(PRODS)].pivot_table(
    index="gts", columns="product", values="mid_price")
pivot.sort_index(inplace=True)
pivot.ffill(inplace=True)
returns = pivot.pct_change().dropna(how="all")

labels = [p.split("_")[-1] for p in PRODS]
n = len(PRODS)

# ── compute OLS slopes/intercepts and store data per pair ─────────────────────
ols_slopes = {}
ols_intercepts = {}
pair_data = {}

for i, p1 in enumerate(PRODS):
    for j, p2 in enumerate(PRODS):
        if i == j:
            continue
        x = returns[p2].dropna()
        y = returns[p1].reindex(x.index)
        valid = x.notna() & y.notna()
        xv, yv = x[valid].values, y[valid].values
        if len(xv) > 5:
            m, b, r, *_ = stats.linregress(xv, yv)
            ols_slopes[(i, j)] = m
            ols_intercepts[(i, j)] = b
            pair_data[(i, j)] = (xv, yv, r)

# ── state: custom slopes (start from OLS) ────────────────────────────────────
custom_slopes = {k: v for k, v in ols_slopes.items()}
custom_intercepts = {k: v for k, v in ols_intercepts.items()}

selected = [0, 1]  # (row, col) of selected off-diagonal cell

# ── build figure ──────────────────────────────────────────────────────────────
DARK = "#1a1a2e"; MID = "#16213e"; ACCENT = "#f39c12"; SEL = "#00ff99"

fig = plt.figure(figsize=(18, 14), facecolor=DARK)
fig.suptitle("Pebbles — Return Pair Scatter Matrix  (click cell → adjust slope)",
             color="white", fontsize=12, y=0.98)

# scatter grid occupies top 80% of figure
grid_top = 0.94; grid_bot = 0.20
gs = fig.add_gridspec(n, n,
                      left=0.06, right=0.98,
                      top=grid_top, bottom=grid_bot,
                      hspace=0.35, wspace=0.35)

scatter_axes = {}
line_artists = {}
text_artists = {}

def draw_cell(i, j, highlight=False):
    ax = scatter_axes.get((i, j))
    if ax is None:
        ax = fig.add_subplot(gs[i, j])
        scatter_axes[(i, j)] = ax

    ax.clear()
    ax.set_facecolor("#2a2a4e" if highlight else MID)
    ax.tick_params(labelsize=5, colors="#aaa")
    for sp in ax.spines.values():
        sp.set_color(SEL if highlight else "#333")
        sp.set_linewidth(2 if highlight else 0.8)

    if i == j:
        ret = returns[PRODS[i]].dropna().values
        ax.hist(ret, bins=50, color="#e74c3c", alpha=0.8, edgecolor="none")
        ax.set_xlabel(labels[i], fontsize=6, color="white")
    else:
        if (i, j) in pair_data:
            xv, yv, r = pair_data[(i, j)]
            ax.scatter(xv, yv, s=0.4, alpha=0.25, color="#e8a0a0", rasterized=True)
            m = custom_slopes[(i, j)]
            b = custom_intercepts[(i, j)]
            xx = np.array([xv.min(), xv.max()])
            ln, = ax.plot(xx, m * xx + b, color=ACCENT, lw=1.2)
            line_artists[(i, j)] = ln
            # show both OLS r and current slope
            txt = ax.text(0.05, 0.90, f"r={r:.2f}\nm={m:.2f}", transform=ax.transAxes,
                          fontsize=5.5, color=ACCENT, va="top")
            text_artists[(i, j)] = txt

    if j == 0:
        ax.set_ylabel(labels[i], fontsize=6, color="white")
    if i == n - 1:
        ax.set_xlabel(labels[j], fontsize=6, color="white")

# initial draw
for i in range(n):
    for j in range(n):
        draw_cell(i, j, highlight=(i == selected[0] and j == selected[1]))

# ── slider panel ──────────────────────────────────────────────────────────────
ax_slope = fig.add_axes([0.15, 0.11, 0.55, 0.025], facecolor="#222244")
ax_inter = fig.add_axes([0.15, 0.07, 0.55, 0.025], facecolor="#222244")
ax_info  = fig.add_axes([0.15, 0.03, 0.55, 0.025]); ax_info.axis("off")

si, sj = selected
init_m = custom_slopes.get((si, sj), 0.0)
init_b = custom_intercepts.get((si, sj), 0.0)

sl_slope = Slider(ax_slope, "Slope", -5.0, 5.0, valinit=init_m, color=ACCENT)
sl_inter = Slider(ax_inter, "Intercept", -0.005, 0.005, valinit=init_b, color="#3498db")

info_txt = ax_info.text(0.0, 0.5,
    f"Selected: {labels[si]} vs {labels[sj]}  |  OLS slope={ols_slopes.get((si,sj),0):.4f}",
    color="white", fontsize=9, va="center")

# reset button
ax_reset = fig.add_axes([0.76, 0.07, 0.08, 0.045])
btn_reset = Button(ax_reset, "Reset OLS", color="#333366", hovercolor="#555599")

ax_done = fig.add_axes([0.86, 0.07, 0.08, 0.045])
btn_done = Button(ax_done, "Print & Close", color="#1a4a1a", hovercolor="#2a7a2a")

def update_line(val=None):
    si, sj = selected
    if (si, sj) not in pair_data:
        return
    m = sl_slope.val
    b = sl_inter.val
    custom_slopes[(si, sj)] = m
    custom_intercepts[(si, sj)] = b
    xv, yv, r = pair_data[(si, sj)]
    xx = np.array([xv.min(), xv.max()])
    if (si, sj) in line_artists:
        line_artists[(si, sj)].set_ydata(m * xx + b)
    if (si, sj) in text_artists:
        text_artists[(si, sj)].set_text(f"r={r:.2f}\nm={m:.2f}")
    fig.canvas.draw_idle()

sl_slope.on_changed(update_line)
sl_inter.on_changed(update_line)

def on_click(event):
    if event.inaxes is None:
        return
    for (i, j), ax in scatter_axes.items():
        if event.inaxes is ax and i != j:
            selected[0], selected[1] = i, j
            si, sj = i, j
            # redraw all cells (update highlight)
            for ii in range(n):
                for jj in range(n):
                    draw_cell(ii, jj, highlight=(ii == si and jj == sj))
            # update sliders
            sl_slope.set_val(custom_slopes.get((si, sj), 0.0))
            sl_inter.set_val(custom_intercepts.get((si, sj), 0.0))
            ols_m = ols_slopes.get((si, sj), 0.0)
            info_txt.set_text(
                f"Selected: {labels[si]} vs {labels[sj]}  |  OLS slope={ols_m:.4f}")
            fig.canvas.draw_idle()
            break

def on_reset(event):
    si, sj = selected
    if (si, sj) in ols_slopes:
        sl_slope.set_val(ols_slopes[(si, sj)])
        sl_inter.set_val(ols_intercepts[(si, sj)])

def on_done(event):
    print("\n=== Selected slopes (row_label vs col_label : slope, intercept) ===")
    for (i, j), m in sorted(custom_slopes.items()):
        b = custom_intercepts[(i, j)]
        ols_m = ols_slopes.get((i, j), float("nan"))
        changed = "*" if abs(m - ols_m) > 1e-6 else " "
        print(f"  {changed} {labels[i]:4s} vs {labels[j]:4s} : slope={m:8.4f}  intercept={b:.6f}  (OLS={ols_m:.4f})")
    plt.close("all")

btn_reset.on_clicked(on_reset)
btn_done.on_clicked(on_done)
fig.canvas.mpl_connect("button_press_event", on_click)

plt.show()
